import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

# Matches the <document>...</document> block used by older stored messages.
_DOCUMENT_RE = re.compile(r"<document>(.*?)</document>(.*)", re.S)

_SYSTEM_PROMPT = (
    "You are a precise, factual AI assistant specialized in extracting values and "
    "metadata from documents. When a document is provided, base every answer strictly "
    "on the document's content - never invent or guess values that are not present. "
    "Answer directly and concisely. If a requested value is missing, say it is not found."
)


class LocalAIEngine:
    def __init__(self):
        self.model = None
        self.current_model_path = None
        self.chat_handler = None
        self.has_mmproj = False
        # Approximate character budget for a single message so the attached
        # document fits inside the context window alongside history/response.
        # Recalculated precisely once a model is loaded (see load_model).
        self.max_prompt_chars = int(os.getenv("MAX_PROMPT_CHARS", "8000"))

    def load_model(self, model_filename, n_ctx=4096, n_threads=4):
        if not LLAMA_AVAILABLE:
            raise ImportError("llama-cpp-python is not installed. Please run 'pip install -r requirements.txt'.")

        model_path = MODELS_DIR / model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        if self.current_model_path == str(model_path) and self.model is not None:
            return  # Already loaded

        # Unload previous model if any
        if self.model is not None:
            del self.model
            self.model = None
        self.chat_handler = None
        self.has_mmproj = False

        # Auto-adjust context window for 2048-trained models (e.g. moondream2, tinyllama)
        fname_lower = model_filename.lower()
        if "moondream" in fname_lower or "tinyllama" in fname_lower:
            n_ctx = min(n_ctx, 2048)

        mmproj_path = None
        if "moondream" in fname_lower:
            potential_mmproj = MODELS_DIR / "moondream2-mmproj-f16.gguf"
            if potential_mmproj.exists():
                mmproj_path = potential_mmproj

        # Register a multimodal chat handler when a vision projector (mmproj)
        # is available so the model can natively "see" attached images.
        if mmproj_path is not None:
            try:
                from llama_cpp.llama_chat_format import MoondreamChatHandler
                self.chat_handler = MoondreamChatHandler(
                    clip_model_path=str(mmproj_path),
                    verbose=False,
                )
                self.has_mmproj = True
            except Exception:
                self.chat_handler = None
                self.has_mmproj = False

        # Load model with context window and multimodal handler if available
        self.model = Llama(
            model_path=str(model_path),
            chat_handler=self.chat_handler,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False
        )
        self.current_model_path = str(model_path)

        # Derive a safe per-message character budget from the *actual* context
        # window so we never exceed it (prevents "Requested tokens exceed
        # context window" errors). Documents larger than this are read in packets.
        self.max_prompt_chars = self._context_char_budget(self._context_len())

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------
    def _context_len(self):
        """Return the active context window size in tokens (best effort)."""
        try:
            return int(self.model.n_ctx())
        except Exception:
            return 2048

    def _context_char_budget(self, n_ctx, max_tokens=1024):
        """Conservative character budget for prompt content for a given context.

        Reserves room for the completion and chat-format overhead, then converts
        the remaining token budget to characters using a ~3 chars/token estimate.
        """
        reserve = max(128, min(max_tokens, n_ctx // 2))
        usable_tokens = max(128, n_ctx - reserve - 64)
        return max(600, usable_tokens * 3)

    def count_tokens(self, text):
        if self.model is None:
            return len(text.split()) * 2
        try:
            return len(self.model.tokenize(text.encode("utf-8")))
        except Exception:
            return len(text.split()) * 2

    @staticmethod
    def _system_message(format_prompt):
        if format_prompt:
            return {"role": "system", "content": _SYSTEM_PROMPT}
        return None

    @staticmethod
    def _build_prompt(messages, continuation_text=None):
        """Construct a universal ChatML prompt string."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if continuation_text is not None:
            parts.append(f"<|im_start|>assistant\n{continuation_text}<|im_end|>")
            parts.append("<|im_start|>user\nPlease continue exactly where you left off, without repeating.<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def _fit_messages(self, messages, budget_tokens):
        """Return a copy of ``messages`` trimmed so its prompt fits ``budget_tokens``.

        Older history turns are dropped first; the final message's content is
        then binary-search truncated to exactly fit the remaining budget.
        """
        messages = [dict(m) for m in messages]
        system = [m for m in messages if m.get("role") == "system"]
        others = [m for m in messages if m.get("role") != "system"]

        # Drop oldest turns while keeping at least the final message.
        while len(others) > 1 and self.count_tokens(self._build_prompt(system + others)) > budget_tokens:
            others.pop(0)

        if others:
            last = others[-1]
            note = "\n\n[Content truncated to fit the model context window]"
            overhead = self.count_tokens(
                self._build_prompt(system + others[:-1] + [{"role": last.get("role", "user"), "content": ""}])
            )
            available = max(32, budget_tokens - overhead - self.count_tokens(note))
            content = last.get("content", "") or ""
            if self.count_tokens(content) > available:
                lo, hi = 0, len(content)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if self.count_tokens(content[:mid]) <= available:
                        lo = mid
                    else:
                        hi = mid - 1
                last["content"] = content[:lo] + note

        return system + others

    @staticmethod
    def _split_document(content):
        """Split a legacy GUI message into its embedded document body and request."""
        if not content:
            return None
        match = _DOCUMENT_RE.search(content)
        if not match:
            return None
        return match.group(1).strip(), match.group(2).strip()

    @staticmethod
    def _chunk_text(text, size, overlap=200):
        """Split ``text`` into overlapping chunks of at most ``size`` characters."""
        chunks = []
        start = 0
        total = len(text)
        while start < total:
            end = min(total, start + size)
            chunks.append(text[start:end])
            if end >= total:
                break
            start = max(end - overlap, start + 1)
        return chunks

    def chunk_document(self, text, size=None):
        """Split a document into coherent, paragraph-aligned sections.

        Sections are kept under the retrieval budget so they can be stored in
        the session database and selectively loaded to answer questions.
        """
        text = text or ""
        if size is None:
            size = max(600, int(self._context_char_budget(self._context_len()) * 0.6))

        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # Break over-long paragraphs (e.g. OCR output) into hard slices.
            while len(para) > size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(para[:size])
                para = para[size:]
            if not para:
                continue
            if not current:
                current = para
            elif len(current) + len(para) + 2 <= size:
                current = current + "\n\n" + para
            else:
                chunks.append(current)
                current = para
        if current:
            chunks.append(current)

        if not chunks and text.strip():
            chunks = self._chunk_text(text, size)
        return chunks

    @staticmethod
    def _document_prompt(question, context, partial=False):
        prefix = (
            "Below is part of the user's attached document. Use it as the only source "
            "of truth.\n"
            if partial else
            "Below is the full content of the user's attached document. Use it as the "
            "only source of truth.\n"
        )
        return (
            prefix
            + "<document>\n" + context + "\n</document>\n\n"
            + "Request: " + question + "\n\n"
            + "Instructions:\n"
            + "- Answer using only facts found in the document above.\n"
            + "- Report each requested value exactly as written, with its label.\n"
            + "- Do not guess, and do not add information that is not in the document.\n"
            + "- Only say a value is not found if it is truly absent from the text above.\n"
            + "Answer:"
        )

    # ------------------------------------------------------------------
    # Single-pass inference with continuation packets
    # ------------------------------------------------------------------
    def _run_single(self, current_messages, n_ctx, max_tokens, temperature, stream_callback, stop_event):
        """Run inference on messages, fitting them to the context window.

        Returns ``(full_text, prompt_str)``. Continuation packets are only used
        when the completion hit the token limit *and* the follow-up prompt still
        fits, so the context window is never exceeded.
        """
        budget = max(64, n_ctx - max_tokens - 64)
        current_messages = self._fit_messages(current_messages, budget)

        full_text = ""
        prompt_str = self._build_prompt(current_messages)
        max_packets = 3

        for packet_count in range(max_packets):
            if stop_event and stop_event.is_set():
                break

            continuation = full_text if packet_count > 0 else None
            prompt_str = self._build_prompt(current_messages, continuation_text=continuation)
            if self.count_tokens(prompt_str) > budget:
                break

            response = self.model.create_completion(
                prompt=prompt_str,
                max_tokens=max_tokens,
                temperature=temperature,
                repeat_penalty=1.1,
                stop=["<|im_end|>", "<|im_start|>"],
                stream=True,
            )

            packet_text = ""
            finish_reason = None
            for chunk in response:
                if stop_event and stop_event.is_set():
                    break
                choices = chunk.get("choices", [])
                if choices:
                    choice = choices[0]
                    text_chunk = choice.get("text", "")
                    finish_reason = choice.get("finish_reason")
                    if text_chunk:
                        packet_text += text_chunk
                        full_text += text_chunk
                        if stream_callback:
                            stream_callback(text_chunk)

            if finish_reason != "length" or not packet_text:
                break

        return full_text, prompt_str

    def _make_usage(self, full_text, prompt_str):
        prompt_tokens = self.count_tokens(prompt_str or "")
        completion_tokens = self.count_tokens(full_text or "")
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    # ------------------------------------------------------------------
    # Document answering
    # ------------------------------------------------------------------
    def _answer_document(self, question, chunks, system, history, n_ctx, max_tokens,
                         temperature, stream_callback, stop_event, status_callback):
        question = (question or "").strip() or "Extract all the key values and metadata from this document."
        budget = max(64, n_ctx - max_tokens - 64)

        # Read the document directly whenever the whole thing fits the window.
        # This gives the model complete visibility (no section filtering), which
        # is what makes value/metadata extraction reliable.
        full_document = "\n\n".join(chunks)
        prefixes = [
            ([system] if system else []) + history,
            ([system] if system else []),
        ]
        for prefix in prefixes:
            messages = prefix + [
                {"role": "user", "content": self._document_prompt(question, full_document)}
            ]
            if self.count_tokens(self._build_prompt(messages)) <= budget:
                full_text, prompt_str = self._run_single(
                    messages, n_ctx, max_tokens, temperature, stream_callback, stop_event
                )
                return full_text, self._make_usage(full_text, prompt_str)

        # Document is larger than the context window: read every section in
        # packets, then synthesize one answer from all of the findings.
        return self._map_reduce_document(
            question, chunks, system, n_ctx, max_tokens, temperature,
            stream_callback, stop_event, status_callback,
        )

    def _map_reduce_document(self, question, chunks, system, n_ctx, max_tokens,
                             temperature, stream_callback, stop_event, status_callback):
        budget = max(600, self._context_char_budget(n_ctx, max_tokens) - 500)

        batches = []
        current = []
        used = 0
        for chunk in chunks:
            if used + len(chunk) > budget and current:
                batches.append(current)
                current = []
                used = 0
            current.append(chunk)
            used += len(chunk)
        if current:
            batches.append(current)

        partials = []
        for idx, batch in enumerate(batches):
            if stop_event and stop_event.is_set():
                break
            if status_callback:
                status_callback(f"Reading document ({idx + 1}/{len(batches)})...")
            context = "\n\n".join(batch)
            messages = ([system] if system else []) + [
                {"role": "user", "content": self._document_prompt(question, context, partial=True)}
            ]
            # Silent pass: accumulate findings, never flood the chat view.
            text, _ = self._run_single(messages, n_ctx, max_tokens, temperature, None, stop_event)
            if text and text.strip():
                partials.append(text.strip())

        if status_callback:
            status_callback("Writing final answer...")

        combined = "\n".join(partials)
        synthesis = (
            "A document was analyzed in parts. Here are the findings from each part:\n\n"
            + combined
            + "\n\nQuestion: " + question
            + "\n\nWrite one clear, final answer to the question using only the facts above. "
            "Do not mention parts, sections, or the analysis process. Answer directly."
        )
        messages = ([system] if system else []) + [{"role": "user", "content": synthesis}]
        full_text, prompt_str = self._run_single(
            messages, n_ctx, max_tokens, temperature, stream_callback, stop_event
        )
        return full_text, self._make_usage(full_text, prompt_str)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def generate_response(self, messages, max_tokens=1024, temperature=0.1, stream_callback=None,
                          stop_event=None, format_prompt=True, image_path=None,
                          document_chunks=None, document_name=None, status_callback=None):
        """Generate a response and return text along with accurate token usage stats."""
        if self.model is None:
            raise RuntimeError("No model is currently loaded. Please select and load a model first.")

        if image_path and self.has_mmproj:
            return self._generate_multimodal(
                messages, max_tokens, temperature, stream_callback, stop_event,
                format_prompt, image_path, document_chunks,
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        cleaned_messages = [m for m in messages if m.get("role") != "system"]
        if len(cleaned_messages) > 6:
            cleaned_messages = cleaned_messages[-6:]

        n_ctx = self._context_len()
        # Keep the completion within half the window so the prompt always fits.
        max_tokens = max(64, min(max_tokens, n_ctx // 2))

        question = ""
        for msg in reversed(cleaned_messages):
            if msg.get("role") == "user":
                question = msg.get("content", "") or ""
                break

        system = self._system_message(format_prompt)
        history = cleaned_messages[:-1] if cleaned_messages else []

        # Prefer sections stored in the session database; fall back to a legacy
        # embedded <document> block for messages saved by older versions.
        chunks = [c for c in (document_chunks or []) if c and c.strip()]
        if not chunks:
            legacy = self._split_document(question)
            if legacy:
                chunks = self.chunk_document(legacy[0])
                question = legacy[1] or ""

        if chunks:
            try:
                return self._answer_document(
                    question, chunks, system, history, n_ctx, max_tokens,
                    temperature, stream_callback, stop_event, status_callback,
                )
            except Exception as e:
                raise RuntimeError(f"Inference error: {str(e)}")

        current_messages = ([system] if system else []) + cleaned_messages
        try:
            full_text, prompt_str = self._run_single(
                current_messages, n_ctx, max_tokens, temperature, stream_callback, stop_event
            )
            return full_text, self._make_usage(full_text, prompt_str)
        except Exception as e:
            raise RuntimeError(f"Inference error: {str(e)}")

    def _generate_multimodal(self, messages, max_tokens, temperature, stream_callback,
                             stop_event, format_prompt, image_path, document_chunks=None):
        """Run native vision inference: pass the raw image + the (OCR-supplemented) text to the model."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        cleaned_messages = [m for m in messages if m.get("role") != "system"]
        if not cleaned_messages:
            return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        n_ctx = self._context_len()
        max_tokens = max(64, min(max_tokens, n_ctx // 2))

        last_user_text = ""
        for msg in reversed(cleaned_messages):
            if msg.get("role") == "user":
                last_user_text = msg.get("content", "") or ""
                break

        chunks = [c for c in (document_chunks or []) if c and c.strip()]
        if chunks:
            context = "\n\n".join(chunks)
            if len(context) > self.max_prompt_chars:
                context = context[:self.max_prompt_chars]
            last_user_text = (
                "OCR/metadata extracted from the attached image:\n"
                + context + "\n\n" + last_user_text
            )
        elif len(last_user_text) > self.max_prompt_chars:
            last_user_text = last_user_text[:self.max_prompt_chars] + "\n[Content truncated to fit AI context window]"

        try:
            image_uri = Path(image_path).resolve().as_uri()
        except Exception:
            image_uri = str(image_path)

        user_content = [
            {"type": "text", "text": last_user_text},
            {"type": "image_url", "image_url": {"url": image_uri}},
        ]

        chat_messages = []
        if format_prompt:
            chat_messages.append({
                "role": "system",
                "content": (
                    "You are a precise, factual AI assistant that can see images. "
                    "Analyze the attached image and answer strictly based on what is visible, "
                    "using any provided OCR/metadata as supplementary context. "
                    "Answer directly and concisely."
                ),
            })
        chat_messages.append({"role": "user", "content": user_content})

        full_text = ""
        try:
            response = self.model.create_chat_completion(
                messages=chat_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in response:
                if stop_event and stop_event.is_set():
                    break
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        full_text += content
                        if stream_callback:
                            stream_callback(content)
        except Exception as e:
            raise RuntimeError(f"Vision inference error: {str(e)}")

        prompt_tokens = self.count_tokens(last_user_text)
        completion_tokens = self.count_tokens(full_text)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return full_text, usage

    def enhance_prompt(self, user_prompt):
        """Use the active model to format and convert user input into the optimal extraction prompt."""
        if self.model is None or not user_prompt.strip():
            return user_prompt

        formatting_system = (
            "You are an expert prompt engineer for document extraction AI. "
            "Convert the user's raw request into a precise, highly effective prompt for extracting values and metadata. "
            "Return ONLY the refined prompt text, without any conversational filler, explanations, or quotes."
        )
        prompt_str = f"<|im_start|>system\n{formatting_system}<|im_end|>\n<|im_start|>user\nUser Request: {user_prompt}<|im_end|>\n<|im_start|>assistant\n"

        try:
            res = self.model.create_completion(
                prompt=prompt_str,
                max_tokens=150,
                temperature=0.1,
                stop=["<|im_end|>", "<|im_start|>"]
            )
            choices = res.get("choices", [])
            if choices:
                refined = choices[0].get("text", "").strip()
                if refined:
                    return refined
        except Exception:
            pass
        return user_prompt

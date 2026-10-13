import os
import threading
from pathlib import Path
from tkinter import filedialog
import customtkinter as ctk
from dotenv import load_dotenv
from src.downloader import RECOMMENDED_MODELS, get_local_models, download_model
from src.engine import LocalAIEngine
from src.database import ChatDatabase
from src.file_reader import extract_content, ScanCancelled, FileTooLargeError, IMAGE_EXTENSIONS

load_dotenv()

class LocalAIGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Pure Python Local AI - Value & Metadata Extractor")
        self.geometry("1050x720")
        
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.engine = LocalAIEngine()
        self.db = ChatDatabase()
        self.stop_event = threading.Event()
        self.scan_stop_event = threading.Event()
        self.download_stop_event = threading.Event()
        self.attached_file_path = None
        self.attached_file_content = None
        self.pending_image_path = None
        self.pending_document_chunks = None
        self.pending_document_name = None
        self.file_scanning = False
        self.model_downloading = False
        self._last_scan_percent = -1
        self.is_generating = False
        self.generation_active = False
        self.generation_id = 0
        self.model_loading = False
        self.booting = True
        self.loading_id = 0
        self.loader_job = None
        self.loading_frames = ["Thinking. ⏳", "Thinking.. ⌛", "Thinking... ⏳", "Thinking.... ⌛"]
        self.loading_frame_idx = 0

        # Handle window close event to wipe temporary DB data
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Configure grid layout: fixed-width sidebar (col 0), fluid chat (col 1)
        self.grid_columnconfigure(0, weight=0, minsize=300)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---------------- SIDEBAR ----------------
        self.sidebar_frame = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(11, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Local AI Extractor", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 15))

        # Download Model Section
        self.dl_label = ctk.CTkLabel(self.sidebar_frame, text="Download Model:", anchor="w")
        self.dl_label.grid(row=1, column=0, padx=20, pady=(5, 0), sticky="w")
        
        self.model_choice_var = ctk.StringVar(value=list(RECOMMENDED_MODELS.keys())[0])
        self.model_dropdown = ctk.CTkOptionMenu(self.sidebar_frame, variable=self.model_choice_var, values=list(RECOMMENDED_MODELS.keys()), width=230, dynamic_resizing=False)
        self.model_dropdown.grid(row=2, column=0, padx=20, pady=(5, 10), sticky="ew")

        self.download_btn = ctk.CTkButton(self.sidebar_frame, text="Download Selected Model", command=self.start_download_thread)
        self.download_btn.grid(row=3, column=0, padx=20, pady=(0, 5), sticky="ew")

        # Progress Bar for Downloading
        self.progress_bar = ctk.CTkProgressBar(self.sidebar_frame)
        self.progress_bar.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.progress_bar.set(0)

        # Select & Autoload Model Section
        self.load_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.load_frame.grid(row=5, column=0, padx=20, pady=(5, 15), sticky="nsew")
        self.load_frame.grid_columnconfigure(0, weight=1)

        self.local_label = ctk.CTkLabel(self.load_frame, text="Select Model (Autoloads):", anchor="w")
        self.local_label.pack(anchor="w", pady=(0, 5))

        local_models = get_local_models()
        self.local_model_var = ctk.StringVar(value=local_models[0] if local_models else "No models found")
        self.local_dropdown = ctk.CTkOptionMenu(
            self.load_frame, 
            variable=self.local_model_var, 
            values=local_models if local_models else ["No models found"],
            command=self.on_model_selected,
            width=230,
            dynamic_resizing=False
        )
        self.local_dropdown.pack(fill="x", pady=(0, 10))

        self.refresh_btn = ctk.CTkButton(self.load_frame, text="Refresh List", command=self.refresh_local_models, fg_color="transparent", border_width=1, text_color=("gray10", "gray90"))
        self.refresh_btn.pack(fill="x", pady=(0, 5))

        # Memory / Chat Management (Moved to bottom of sidebar)
        self.mem_label = ctk.CTkLabel(self.sidebar_frame, text="Chat & Memory (SQLite DB):", anchor="w")
        self.mem_label.grid(row=6, column=0, padx=20, pady=(5, 0), sticky="w")
        
        self.clear_mem_btn = ctk.CTkButton(self.sidebar_frame, text="Clear Chat Memory", command=self.clear_memory, fg_color="#b83232", hover_color="#922525")
        self.clear_mem_btn.grid(row=7, column=0, padx=20, pady=(5, 5), sticky="ew")

        self.export_chat_btn = ctk.CTkButton(self.sidebar_frame, text="Export Chat History", command=self.export_chat, fg_color="gray", hover_color="darkgray")
        self.export_chat_btn.grid(row=8, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Prompt Formatting Switch
        self.prompt_format_var = ctk.BooleanVar(value=True)
        self.prompt_format_switch = ctk.CTkSwitch(
            self.sidebar_frame, 
            text="AI Prompt Formatting", 
            variable=self.prompt_format_var, 
            onvalue=True, 
            offvalue=False
        )
        self.prompt_format_switch.grid(row=9, column=0, padx=20, pady=(5, 10), sticky="w")

        # Status Message Block at bottom of sidebar (responsive, full width)
        self.status_frame = ctk.CTkFrame(self.sidebar_frame, corner_radius=6, fg_color=("gray85", "gray20"))
        self.status_frame.grid(row=10, column=0, padx=20, pady=(5, 15), sticky="ew")
        self.status_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Status: Ready (DB Active)",
            font=ctk.CTkFont(size=12),
            text_color=("gray25", "gray80"),
            anchor="w",
            justify="left",
            wraplength=220,
        )
        self.status_label.grid(row=0, column=0, padx=10, pady=8, sticky="ew")

        # Keep the status text wrapping fluidly as the sidebar/frame resizes
        self.status_frame.bind("<Configure>", self._on_status_resize)

        # ---------------- MAIN CHAT AREA ----------------
        self.chat_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.chat_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.chat_frame.grid_columnconfigure(0, weight=1)
        self.chat_frame.grid_rowconfigure(0, weight=1)

        # Chat Text Display
        self.chat_textbox = ctk.CTkTextbox(self.chat_frame, wrap="word", font=ctk.CTkFont(size=14))
        self.chat_textbox.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        self.chat_textbox.configure(state="disabled")

        # Attachment status display frame
        self.attachment_frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        self.attachment_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        self.attachment_label = ctk.CTkLabel(self.attachment_frame, text="No file attached - attach 1 file, then type what values to extract", font=ctk.CTkFont(size=12, slant="italic"), text_color="gray")
        self.attachment_label.pack(side="left", padx=2)

        # Input Frame
        self.input_frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        self.input_frame.grid(row=2, column=0, sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.msg_entry = ctk.CTkEntry(self.input_frame, placeholder_text="Ask to extract values, e.g. 'Get the invoice no, vendor, and total'...", height=40, font=ctk.CTkFont(size=14))
        self.msg_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.msg_entry.bind("<Return>", lambda event: self.send_message())

        self.attach_btn = ctk.CTkButton(self.input_frame, text="📎 Attach", width=80, height=40, command=self.attach_file, fg_color="gray", hover_color="darkgray")
        self.attach_btn.grid(row=0, column=1, sticky="e", padx=(0, 5))

        self.send_btn = ctk.CTkButton(self.input_frame, text="Send", width=75, height=40, command=self.send_message)
        self.send_btn.grid(row=0, column=2, sticky="e", padx=(0, 5))

        self.terminate_btn = ctk.CTkButton(self.input_frame, text="Terminate", width=75, height=40, command=self.terminate_current, fg_color="red", hover_color="darkred", state="normal")
        self.terminate_btn.grid(row=0, column=3, sticky="e")

        # Load session from SQLite DB on startup & autoload the first model in the list
        self.load_session_from_db()
        # Lock every interactive control until a model has finished autoloading.
        self._refresh_control_states()
        self.update_status("Starting up - loading model, please wait...")
        self.after(300, self.autoload_first_model)

    def _refresh_control_states(self):
        """Enable/disable controls based on whether any blocking task is running.

        The chat block, input and action buttons stay disabled while a model is
        autoloading/downloading, a file is being scanned, or a response is being
        generated; the Terminate button stays available to cancel the work.
        """
        busy = (
            self.booting
            or self.model_loading
            or self.model_downloading
            or self.file_scanning
            or self.generation_active
        )
        state = "disabled" if busy else "normal"
        for widget in (
            self.msg_entry,
            self.attach_btn,
            self.send_btn,
            self.local_dropdown,
            self.model_dropdown,
            self.download_btn,
            self.refresh_btn,
            self.clear_mem_btn,
            self.export_chat_btn,
            self.prompt_format_switch,
        ):
            try:
                widget.configure(state=state)
            except Exception:
                pass
        try:
            self.terminate_btn.configure(state="normal" if busy else "disabled")
        except Exception:
            pass

    def update_status(self, text):
        self.status_label.configure(text=f"Status: {text}")

    def _on_status_resize(self, event):
        # Match wrap width to the actual rendered frame width so text flows fluidly.
        width = max(80, event.width - 24)
        self.status_label.configure(wraplength=width)

    def load_session_from_db(self):
        try:
            messages = self.db.get_messages(limit=50)
            self.render_messages(messages)
            if messages:
                self.update_status("Restored active session from SQLite DB.")
        except Exception as e:
            print(f"Error loading DB session: {e}")

    def render_messages(self, messages):
        self.chat_textbox.configure(state="normal")
        self.chat_textbox.delete("1.0", "end")
        self.chat_textbox.configure(state="disabled")
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                self.append_chat_user(content)
            elif role == "assistant":
                self.append_chat_assistant_header()
                self.append_chat_chunk(content)
                self.append_chat_newline()

    def refresh_chat_view(self):
        """Re-render the conversation from the DB, dropping any terminated turn."""
        try:
            self.render_messages(self.db.get_messages(limit=50))
        except Exception as e:
            print(f"Error refreshing chat view: {e}")

    def clear_memory(self):
        self.db.clear()
        self.attached_file_path = None
        self.attached_file_content = None
        self.pending_document_chunks = None
        self.pending_document_name = None
        self.chat_textbox.configure(state="normal")
        self.chat_textbox.delete("1.0", "end")
        self.chat_textbox.configure(state="disabled")
        self.attachment_label.configure(text="No file attached - attach 1 file, then type what values to extract")
        self.update_status("Chat memory and attached document cleared.")

    def export_chat(self):
        messages = self.db.get_messages(limit=500)
        if not messages:
            self.update_status("No chat history to export.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt"), ("JSON Files", "*.json")])
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    for msg in messages:
                        role = msg.get("role", "user").capitalize()
                        content = msg.get("content", "")
                        f.write(f"{role}: {content}\n\n")
                self.update_status(f"Chat exported to {Path(file_path).name}")
            except Exception as e:
                self.update_status(f"Export failed: {e}")

    def attach_file(self):
        file_types = [
            ("All Supported Files", "*.txt;*.py;*.md;*.json;*.csv;*.log;*.pdf;*.docx;*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.webp"),
            ("PDF & Word Docs", "*.pdf;*.docx"),
            ("Text/Code Files", "*.txt;*.py;*.md;*.json;*.csv;*.log"),
            ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.webp"),
            ("All Files", "*.*")
        ]
        file_path = filedialog.askopenfilename(title="Attach File, PDF, Word, or Image", filetypes=file_types)
        if not file_path:
            return

        self.scan_stop_event.clear()
        self._last_scan_percent = -1
        self.file_scanning = True
        self._refresh_control_states()
        self.terminate_btn.configure(text="Stop Read")
        self.attachment_label.configure(text=f"Reading {Path(file_path).name} (0%)...")
        self.update_status(f"Reading {Path(file_path).name} locally (0%)...")
        threading.Thread(target=self._scan_file_worker, args=(file_path,), daemon=True).start()

    def _scan_progress(self, file_name, percent, stage):
        self.after(0, self._update_scan_progress, file_name, percent, stage)

    def _update_scan_progress(self, file_name, percent, stage):
        if not self.file_scanning:
            return
        pct = int(round(percent))
        if pct == self._last_scan_percent:
            return
        self._last_scan_percent = pct
        self.attachment_label.configure(text=f"Reading {file_name} ({pct}%) - {stage}")
        self.update_status(f"Reading {file_name} ({pct}%) - {stage}")

    def _scan_file_worker(self, file_path):
        name = Path(file_path).name
        try:
            result = extract_content(
                file_path,
                progress_callback=lambda pct, stage: self._scan_progress(name, pct, stage),
                stop_event=self.scan_stop_event,
            )
            if self.scan_stop_event.is_set():
                self.after(0, self._on_scan_complete, file_path, None, None, True)
            else:
                self.after(0, self._on_scan_complete, file_path, result, None, False)
        except ScanCancelled:
            self.after(0, self._on_scan_complete, file_path, None, None, True)
        except FileTooLargeError as e:
            self.after(0, self._on_scan_complete, file_path, None, str(e), False)
        except Exception as e:
            self.after(0, self._on_scan_complete, file_path, None, str(e), False)

    def _on_scan_complete(self, file_path, result, error, cancelled=False):
        self.file_scanning = False
        self._refresh_control_states()
        self.terminate_btn.configure(text="Terminate")
        name = Path(file_path).name

        if cancelled:
            self.attached_file_path = None
            self.attached_file_content = None
            self.attachment_label.configure(text="File read terminated by user")
            self.update_status("File read terminated by user.")
            return

        if error:
            self.attached_file_path = None
            self.attached_file_content = None
            self.attachment_label.configure(text=f"Could not read {name}")
            self.update_status(f"{error}")
            return

        text = (result or {}).get("text", "") or ""
        kind = (result or {}).get("kind", "file")
        meta = (result or {}).get("meta", {}) or {}

        if not text.strip():
            self.attached_file_path = None
            self.attached_file_content = None
            self.attachment_label.configure(text=f"No readable text found in {name}")
            self.update_status(f"No readable text found in {name}.")
            return

        self.attached_file_path = file_path
        self.attached_file_content = text

        # Index the document into the session database so questions can be
        # answered (and follow-ups asked) from the most relevant sections.
        try:
            sections = self.engine.chunk_document(text)
            self.db.set_document(name, sections)
            section_note = f", indexed into {len(sections)} sections"
        except Exception as e:
            sections = []
            section_note = ""
            print(f"Error indexing document: {e}")

        if kind == "image":
            label = f"Scanned Image (OCR): {name} ({len(text)} chars{section_note})"
        elif kind == "pdf":
            label = f"Scanned PDF: {name} ({meta.get('text_pages', 0)} text + {meta.get('ocr_pages', 0)} OCR pages, {len(text)} chars{section_note})"
        elif kind == "docx":
            label = f"Scanned Word Doc: {name} ({len(text)} chars{section_note})"
        else:
            label = f"Attached File: {name} ({len(text)} chars{section_note})"

        self.attachment_label.configure(text=label)
        self.update_status(
            f"Ready: {name} indexed into {len(sections)} sections. Ask what you need - "
            f"the most relevant content will be used to answer."
        )

    def refresh_local_models(self):
        local_models = get_local_models()
        if local_models:
            self.local_dropdown.configure(values=local_models)
            self.local_model_var.set(local_models[0])
            self.on_model_selected(local_models[0])
            self.update_status("Model list refreshed & autoloaded.")
        else:
            self.local_dropdown.configure(values=["No models found"])
            self.local_model_var.set("No models found")
            self.booting = False
            self.model_loading = False
            self._refresh_control_states()
            self.update_status("No local models found in /models.")

    def start_download_thread(self):
        selected_key = self.model_choice_var.get()
        self.download_stop_event.clear()
        self.model_downloading = True
        self._refresh_control_states()
        self.progress_bar.set(0)
        self.update_status("Downloading model... input disabled until ready.")
        threading.Thread(target=self.download_worker, args=(selected_key,), daemon=True).start()

    def download_worker(self, model_key):
        def progress_cb(percent, downloaded, total, speed, eta):
            if self.download_stop_event.is_set():
                return
            d_mb = downloaded / (1024 * 1024)
            t_mb = total / (1024 * 1024)
            speed_mb = speed / (1024 * 1024)
            
            if total > 0:
                status_str = f"Downloading: {percent:.1f}% ({d_mb:.1f} MB / {t_mb:.1f} MB) @ {speed_mb:.2f} MB/s - ETA: {int(eta)}s"
            else:
                status_str = f"Downloading: {d_mb:.1f} MB @ {speed_mb:.2f} MB/s"

            self.after(0, lambda: self.progress_bar.set(percent / 100))
            self.after(0, lambda: self.update_status(status_str))

        try:
            download_model(model_key, progress_callback=progress_cb, stop_event=self.download_stop_event)
            if self.download_stop_event.is_set():
                self.after(0, lambda: self.update_status("Download terminated."))
            else:
                self.after(0, lambda: self.progress_bar.set(1.0))
                self.after(0, lambda: self.update_status("Download complete - loading model..."))
                self.after(0, self.refresh_local_models)
        except Exception as e:
            if not self.download_stop_event.is_set():
                self.after(0, lambda: self.update_status(f"Download failed: {str(e)}"))
        finally:
            self.model_downloading = False
            self.after(0, self._refresh_control_states)

    def on_model_selected(self, model_name):
        if not model_name or model_name == "No models found":
            return
        if self.model_loading:
            self.update_status("A model is already loading, please wait...")
            return
        self.model_loading = True
        self.loading_id += 1
        load_id = self.loading_id
        self._refresh_control_states()
        self.update_status(f"Loading {model_name}... input disabled until ready.")
        threading.Thread(target=self.load_worker, args=(model_name, load_id), daemon=True).start()

    def load_worker(self, model_name, load_id):
        try:
            self.engine.load_model(model_name)
            self.after(0, self._on_model_ready, model_name, load_id)
        except Exception as e:
            err = str(e)
            self.after(0, self._on_model_failed, err, load_id)

    def _on_model_ready(self, model_name, load_id):
        if load_id != self.loading_id:
            return
        self.model_loading = False
        self.booting = False
        self._refresh_control_states()
        self.update_status(f"Ready: {model_name}")

    def _on_model_failed(self, err, load_id):
        if load_id != self.loading_id:
            return
        self.model_loading = False
        self.booting = False
        self._refresh_control_states()
        self.update_status(f"Failed to autoload: {err}")

    def autoload_first_model(self):
        """Autoload the first available local model."""
        self.refresh_local_models()

    def send_message(self):
        user_text = self.msg_entry.get().strip()
        if self.model_loading or self.booting:
            self.update_status("Model is still loading, please wait...")
            return
        if self.file_scanning:
            self.update_status("Still scanning the attached file, please wait...")
            return
        if not user_text and not self.attached_file_path and not self.db.has_document():
            return

        if self.engine.model is None:
            if self.model_loading:
                self.append_chat_system("Model is still loading, please wait a moment and try again...")
                return
            local_models = get_local_models()
            if local_models:
                self.append_chat_system(f"Loading model '{self.local_model_var.get() or local_models[0]}', please wait a moment and resend...")
                self.on_model_selected(self.local_model_var.get() if self.local_model_var.get() in local_models else local_models[0])
            else:
                self.append_chat_system("System Error: No model loaded! Please download and select a model first.")
            return

        # The active document lives in the session database (not embedded in the
        # message), so we pass its sections to the engine for full-document reading.
        has_document = self.db.has_document()
        document_chunks = self.db.get_document_chunks() if has_document else None
        document_name = self.db.get_active_document_name() if has_document else None

        # When a document is active, keep the user's own wording for extraction.
        # Rewriting it with a small model can distort the request and produce
        # wrong values, so only run prompt-formatting for plain chat.
        effective_user_text = user_text
        if self.prompt_format_var.get() and user_text and not has_document:
            effective_user_text = self.engine.enhance_prompt(user_text)

        question = effective_user_text
        if not question and has_document:
            question = "Extract all the key values and metadata from this document."

        self.msg_entry.delete(0, "end")

        display_text = user_text
        attached_name = None
        if self.attached_file_path:
            attached_name = Path(self.attached_file_path).name
        elif document_name:
            attached_name = document_name
        if attached_name and not user_text:
            display_text = ""
        if attached_name:
            display_text = (display_text + f" [Attached: {attached_name}]").strip()

        self.append_chat_user(display_text)
        self.db.add_message("user", question)

        # Keep the raw image path so a vision-capable model can "see" it natively.
        if self.attached_file_path and Path(self.attached_file_path).suffix.lower() in IMAGE_EXTENSIONS:
            self.pending_image_path = self.attached_file_path
        else:
            self.pending_image_path = None

        self.pending_document_chunks = document_chunks
        self.pending_document_name = document_name

        self.attached_file_path = None
        self.attached_file_content = None
        self.attachment_label.configure(text="No file attached - attach 1 file, then type what values to extract")

        self.stop_event.clear()
        self.generation_id += 1
        self.generation_active = True
        self._refresh_control_states()
        self.terminate_btn.configure(text="Terminate")

        threading.Thread(target=self.generate_worker, daemon=True).start()

    def terminate_current(self):
        """Globally abort ALL running tasks (file scan, AI generation, model download, model loading) and reset app state."""
        terminated_scan = self.file_scanning
        terminated_gen = self.generation_active
        terminated_download = self.model_downloading
        terminated_load = self.model_loading

        # 1) Stop file scan
        if terminated_scan:
            self.scan_stop_event.set()
            self.file_scanning = False

        # 2) Stop AI generation
        if terminated_gen:
            self.is_generating = False
            self.generation_active = False
            self.stop_event.set()
            try:
                if self.loader_job:
                    self.after_cancel(self.loader_job)
                    self.loader_job = None
            except Exception:
                pass
            self.generation_id += 1

        # 3) Stop model download
        if terminated_download:
            self.download_stop_event.set()
            self.model_downloading = False
            self.progress_bar.set(0)

        # 4) Stop model loading
        if terminated_load:
            self.model_loading = False
            self.loading_id += 1

        # 5) Stop boot-time lock
        self.booting = False

        # 6) Clear active data (attached files, input, progress)
        self.attached_file_path = None
        self.attached_file_content = None
        self.pending_image_path = None
        self._last_scan_percent = -1
        self.msg_entry.delete(0, "end")
        self.attachment_label.configure(text="No file attached - attach 1 file, then type what values to extract")

        if terminated_gen:
            self.db.delete_last_message(role="user")
            self.refresh_chat_view()

        # 7) Reset UI controls and terminate button
        self._refresh_control_states()
        self.terminate_btn.configure(text="Terminate")

        targets = []
        if terminated_scan:
            targets.append("file read")
        if terminated_gen:
            targets.append("AI generation")
        if terminated_download:
            targets.append("model download")
        if terminated_load:
            targets.append("model loading")

        if targets:
            self.update_status(f"Terminated {', '.join(targets)}. All active data cleared.")
        else:
            self.update_status("Terminated all running processes. Ready to start again.")

    def generate_worker(self):
        self.is_generating = True
        self.generation_active = True
        self.loading_frame_idx = 0
        gen_id = self.generation_id

        self.append_chat_assistant_header()
        self.chat_textbox.configure(state="normal")
        thinking_index = self.chat_textbox.index("end-1c")
        self.chat_textbox.insert("end", self.loading_frames[0])
        self.chat_textbox.see("end")
        self.chat_textbox.configure(state="disabled")

        # When the engine reports progress (e.g. "Reading document (1/4)..."),
        # it replaces the cycling "Thinking" text in the chat block while the
        # animation keeps ticking, and is mirrored to the status bar too.
        progress_state = {"status": None}
        stream_state = {"started": False}

        def set_placeholder(text):
            self.chat_textbox.configure(state="normal")
            self.chat_textbox.delete(thinking_index, "end-1c")
            self.chat_textbox.insert(thinking_index, text)
            self.chat_textbox.see("end")
            self.chat_textbox.configure(state="disabled")

        def animate_loader():
            if self.is_generating and gen_id == self.generation_id:
                self.loading_frame_idx = (self.loading_frame_idx + 1) % len(self.loading_frames)
                frame = self.loading_frames[self.loading_frame_idx]
                status = progress_state["status"]
                display = f"{status} {frame}" if status else frame
                set_placeholder(display)
                self.loader_job = self.after(400, animate_loader)

        self.loader_job = self.after(400, animate_loader)

        full_response = ""
        first_chunk = True
        
        def stream_cb(chunk):
            nonlocal full_response, first_chunk
            if gen_id != self.generation_id:
                return
            if first_chunk:
                self.is_generating = False
                stream_state["started"] = True
                try:
                    if self.loader_job:
                        self.after_cancel(self.loader_job)
                except Exception:
                    pass
                self.chat_textbox.configure(state="normal")
                self.chat_textbox.delete(thinking_index, "end-1c")
                self.chat_textbox.configure(state="disabled")
                first_chunk = False
            full_response += chunk
            self.after(0, lambda: self.append_chat_chunk(chunk) if gen_id == self.generation_id else None)

        try:
            messages = self.db.get_messages(limit=6)
            image_path = self.pending_image_path
            self.pending_image_path = None
            document_chunks = self.pending_document_chunks
            document_name = self.pending_document_name
            self.pending_document_chunks = None
            self.pending_document_name = None

            def status_cb(text):
                if gen_id != self.generation_id:
                    return
                progress_state["status"] = text
                self.after(0, self.update_status, text)
                if not stream_state["started"]:
                    self.after(0, lambda: None if stream_state["started"] else
                               set_placeholder(f"{text} {self.loading_frames[self.loading_frame_idx]}"))

            full_resp, usage = self.engine.generate_response(
                messages,
                stream_callback=stream_cb,
                stop_event=self.stop_event,
                format_prompt=self.prompt_format_var.get(),
                image_path=image_path,
                document_chunks=document_chunks,
                document_name=document_name,
                status_callback=status_cb,
            )
            if not self.stop_event.is_set() and gen_id == self.generation_id:
                self.db.add_message("assistant", full_resp)
                token_str = f"\n[📊 Tokens Sent: {usage['prompt_tokens']} | Received: {usage['completion_tokens']} | Consumed: {usage['total_tokens']}]\n"
                self.after(0, lambda: self.append_chat_system(token_str))
                self.after(0, self.append_chat_newline)
        except Exception as e:
            err_msg = str(e)
            if not self.stop_event.is_set() and gen_id == self.generation_id:
                self.after(0, lambda: self.append_chat_system(f"\nError during generation: {err_msg}"))
        finally:
            self.is_generating = False
            self.generation_active = False
            if gen_id == self.generation_id:
                self.after(0, self._refresh_control_states)
                self.after(0, lambda: self.terminate_btn.configure(text="Terminate"))

    def on_close(self):
        """Wipe temporary SQLite DB and session data on app close."""
        try:
            self.db.close()
        except Exception:
            pass
        session_file = Path("models/chat_session.json")
        if session_file.exists():
            try:
                session_file.unlink()
            except Exception:
                pass
        self.destroy()

    def append_chat_user(self, text):
        self.chat_textbox.configure(state="normal")
        self.chat_textbox.insert("end", f"\nYou: {text}\n")
        self.chat_textbox.see("end")
        self.chat_textbox.configure(state="disabled")

    def append_chat_assistant_header(self):
        self.chat_textbox.configure(state="normal")
        self.chat_textbox.insert("end", "\nAI: ")
        self.chat_textbox.see("end")
        self.chat_textbox.configure(state="disabled")

    def append_chat_chunk(self, chunk):
        self.chat_textbox.configure(state="normal")
        self.chat_textbox.insert("end", chunk)
        self.chat_textbox.see("end")
        self.chat_textbox.configure(state="disabled")

    def append_chat_newline(self):
        self.chat_textbox.configure(state="normal")
        self.chat_textbox.insert("end", "\n")
        self.chat_textbox.see("end")
        self.chat_textbox.configure(state="disabled")

    def append_chat_system(self, text):
        self.chat_textbox.configure(state="normal")
        self.chat_textbox.insert("end", f"\n{text}\n")
        self.chat_textbox.see("end")
        self.chat_textbox.configure(state="disabled")

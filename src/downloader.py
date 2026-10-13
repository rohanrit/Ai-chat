import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RECOMMENDED_MODELS = {
    # --- Recommended default (best accuracy on any PC) ---
    "Qwen2.5-1.5B-Instruct (986 MB - RECOMMENDED: smart, fast, runs on any PC)": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    },
    # --- Lightweight (1B - 2B) ---
    "TinyLlama-1.1B-Chat (637 MB - Fastest/Lightest)": {
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    },
    "Llama-3.2-1B-Instruct (1.3 GB - Meta Llama 3.2)": {
        "url": "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    },
    # --- Code Manipulation ---
    "Qwen2.5-Coder-1.5B-Instruct (986 MB - Specialized Coding)": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    },
    "Qwen2.5-Coder-7B-Instruct (4.7 GB - Advanced Coding & Refactoring)": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    },
    # --- Vision & Image ---
    "Moondream2 (1.6 GB + mmproj - Multimodal Vision / Image Understanding)": {
        "url": "https://huggingface.co/corono1/moondream2-2b-q4_k_m-gguf/resolve/main/moondream2-text-model-q4_k_m.gguf",
        "filename": "moondream2-text-model-q4_k_m.gguf",
        "mmproj_url": "https://huggingface.co/zhimeme/moondream2-gguf/resolve/main/moondream2-mmproj-f16.gguf",
        "mmproj_filename": "moondream2-mmproj-f16.gguf"
    },
    # --- Mid-Level (3B - 4B) ---
    "Llama-3.2-3B-Instruct (2.0 GB - Mid-Level Meta)": {
        "url": "https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    },
    "Qwen2.5-3B-Instruct (1.9 GB - Mid-Level Qwen)": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf"
    },
    "Phi-3.5-mini-instruct (2.4 GB - Microsoft Phi 3.5)": {
        "url": "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf"
    },
    # --- High-Level (7B - 8B) ---
    "Llama-3.1-8B-Instruct (4.7 GB - High-Level Meta)": {
        "url": "https://huggingface.co/unsloth/Llama-3.1-8B-Instruct-GGUF/resolve/main/Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "filename": "Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    },
    "Mistral-7B-Instruct-v0.3 (4.4 GB - High-Level Mistral)": {
        "url": "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
    },
    "Qwen2.5-7B-Instruct (4.7 GB - High-Level Qwen)": {
        "url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf"
    }
}

def get_local_models():
    """Return a sorted list of locally downloaded .gguf models."""
    if not MODELS_DIR.exists():
        return []
    return sorted(
        p.name for p in MODELS_DIR.glob("*.gguf")
        if "mmproj" not in p.name.lower()
    )

def download_model(model_key, progress_callback=None, stop_event=None):
    """Download model file(s) with streaming progress callback."""
    if model_key not in RECOMMENDED_MODELS:
        raise ValueError(f"Unknown model: {model_key}")
    
    info = RECOMMENDED_MODELS[model_key]
    files_to_download = [(info["url"], MODELS_DIR / info["filename"])]
    if "mmproj_url" in info and "mmproj_filename" in info:
        files_to_download.append((info["mmproj_url"], MODELS_DIR / info["mmproj_filename"]))

    token = os.getenv("HF_TOKEN") or None
    headers = {}
    if token and token != "":
        headers["Authorization"] = f"Bearer {token}"

    total_files = len(files_to_download)
    for file_idx, (url, filepath) in enumerate(files_to_download):
        if stop_event and stop_event.is_set():
            break

        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        file_total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        start_time = time.time()
        chunk_size = 1024 * 64

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if stop_event and stop_event.is_set():
                    break
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    elapsed_time = time.time() - start_time
                    speed = downloaded_size / elapsed_time if elapsed_time > 0 else 0
                    
                    if file_total_size > 0:
                        percent = (downloaded_size / file_total_size) * 100
                        remaining_bytes = file_total_size - downloaded_size
                        eta = remaining_bytes / speed if speed > 0 else 0
                    else:
                        percent = 0
                        eta = 0
                    
                    if progress_callback:
                        overall_percent = ((file_idx * 100) + percent) / total_files
                        progress_callback(overall_percent, downloaded_size, file_total_size, speed, eta)

    return str(MODELS_DIR / info["filename"])

# Pure Python Local AI Studio (No Ollama Required)

A lightweight, high-performance desktop application built in **pure Python** that runs basic low-level AI models locally on your computer. It operates entirely in-process using direct GGUF bindings (`llama.cpp`), requiring **no background servers, daemons, or third-party apps like Ollama** to minimize system load.

---

## 🚀 Key Features

- **Zero External Dependencies**: Runs entirely in-process via C++ `llama.cpp` bindings (`llama-cpp-python`).
- **Modern Cross-Platform GUI**: Built with `CustomTkinter` supporting dark/light system themes across Windows, macOS, and Linux.
- **Direct Model Downloader**: Automatically downloads lightweight, mid-level, high-level, code-specialized, and vision GGUF models directly from Hugging Face into `/models` with live progress bars, speed metrics, and ETA.
- **Automatic Model Autoloading**: Instantly loads models into memory upon app startup, download completion, or dropdown selection (no manual load buttons required).
- **SQLite DB Temporary Session Memory**: Tracks active chat sessions and context smoothly in an in-memory SQLite database (`chat_memory.db`), which automatically wipes clean when you close the app.
- **Local Value / Metadata Extraction**: Attach a single file, then ask in plain English for the values you want (e.g. *"Get the invoice number, vendor and total"*). The selected local model returns the raw extracted answer, grounded strictly in the document. Works with **any** model in the dropdown — no schema, no JSON toggle, no batch processing required.
- **File, PDF, Word & Image Attachments**: Attach PDFs (`.pdf`), Word Documents (`.docx`), text/code files (`.txt`, `.py`, `.md`, `.json`, `.csv`), or images (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`) to inject context. Images and scanned PDFs are read with a **fully bundled, offline OCR engine** (RapidOCR + PyMuPDF) requiring **no external binary or system install**.
- **Animated Thinking Loader**: Displays an active animated indicator (`Thinking... ⏳`) while the AI generates responses in real time.
- **Token Usage Statistics**: Displays exact token counts (`[📊 Tokens Sent | Received | Consumed]`) below every chat response block.
- **Auto-Packet Continuation**: Automatically stitches responses together across chunks if token limits are reached, ensuring complete and uninterrupted answers.
- **Live File Progress**: While a file is being read/OCR'd, the attachment bar shows a live percentage and stage (e.g. `Reading invoice.pdf (45%) - page 3/6`), so the UI never appears frozen.
- **Unified Terminate**: The **Terminate** button instantly aborts whichever task is running — file reading/OCR, AI generation, or both — and clears all active/pending data (partial reads, pending attachment, and the dangling in-progress turn).
- **Stop / Terminate Generation**: Instantly halt AI text generation mid-stream.

---

## 🛠️ Tech Stack & Libraries Used

- **Language**: Python 3.10 – 3.14
- **GUI Framework**: `customtkinter` (Modern Tkinter wrapper)
- **Inference Engine**: `llama-cpp-python` (Direct GGUF model runner)
- **Model Downloader**: `huggingface_hub` & `requests`
- **Database**: Built-in Python `sqlite3` (Temporary session memory)
- **Document Parsers**: `pypdf` (PDF text) & `python-docx` (Word doc parsing)
- **Local OCR (Images & Scanned PDFs)**: `rapidocr-onnxruntime` + `onnxruntime` (models bundled, offline) with `PyMuPDF` page rendering — no Tesseract, no poppler, no online services
- **Configuration**: `python-dotenv`

---

## 📦 Curated Free Local AI Models Available in App

### Lightweight (1B - 2B)
- **TinyLlama-1.1B-Chat** (~637 MB) — Ultra-fast on low-RAM systems.
- **Qwen2.5-1.5B-Instruct** (~986 MB) — Smart and fast general assistant.
- **Llama-3.2-1B-Instruct** (~1.3 GB) — Meta's optimized instruction model.

### Code Manipulation
- **Qwen2.5-Coder-1.5B-Instruct** (~986 MB) — Specialized code generation & debugging.
- **Qwen2.5-Coder-7B-Instruct** (~4.7 GB) — Advanced coding, refactoring, and logic.

### Vision & Image Understanding
- **Moondream2** (~829 MB) — Lightweight vision and image analysis.

### Mid-Level (3B - 4B)
- **Llama-3.2-3B-Instruct** (~2.0 GB) — Enhanced Meta reasoning model.
- **Qwen2.5-3B-Instruct** (~1.9 GB) — Powerful mid-tier logic and coding.
- **Phi-3.5-mini-instruct** (~2.4 GB) — Microsoft high-performance reasoning.

### High-Level (7B - 8B)
- **Llama-3.1-8B-Instruct** (~4.7 GB) — Meta enterprise-grade open model.
- **Mistral-7B-Instruct-v0.3** (~4.4 GB) — Top-tier open-source instruction model.
- **Qwen2.5-7B-Instruct** (~4.7 GB) — Exceptional multi-domain reasoning model.

---

## ⚙️ Installation & Setup Steps

### 1. Clone or Download the Repository
Clone the repository and navigate into the project directory:
```bash
git clone https://github.com/rohanrit/ailocalchat.git
cd ailocalchat
```

### 2. Run the Automated Installer
Run the setup script to install all GUI, document parsers, downloader, and prebuilt inference engine dependencies:
```bash
python install.py
```
*(Note: The installer automatically pulls prebuilt binary wheels for `llama-cpp-python`, requiring no local C++ compiler or CMake).*

---

## 🖥️ Running the App

Start the desktop application:
```bash
python main.py
```

### How to Use:
1. **Download a Model**: Select a model from the dropdown in the sidebar and click **Download Selected Model**. **Qwen2.5-1.5B-Instruct** is recommended — it is small enough to run on any PC while being highly accurate at value extraction. Progress, download speed, and ETA display in real time.
2. **Autoload**: Once downloaded, the model automatically loads into memory. You can switch models anytime using the model selector dropdown.
3. **Extract Values**: Click **📎 Attach**, pick **one** file, then type a plain-English request in the input bar such as *"Extract the invoice number, vendor and total amount"* and press **Send**. The selected model streams back the raw extracted answer, grounded in the document.
4. **Attach Documents**: Click **📎 Attach** to include PDFs (`.pdf`), Word files (`.docx`), code, text, or images for the AI to analyze. Images and image-only (scanned) PDF pages are read locally via the bundled OCR engine — fully offline, with no external tools required.
5. **Token Metrics**: Review token consumption stats displayed below each AI response block.
6. **Memory**: Active session memory is stored temporarily in SQLite and automatically wiped clean when you close the app window.

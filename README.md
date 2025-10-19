# Local-LLM
A simple way to run local Large Language Models (LLMs) offline using [Ollama](https://ollama.com/).

---
## 🛠️ Install Ollama

Ollama makes it super easy to run models locally.

1. Go to 👉 [https://ollama.com/download](https://ollama.com/download)
2. Download and install for your OS (Windows, macOS, or Linux).


---

## 📁 Project Structure & File Contents

```
README.md                # This file. Project overview and instructions.
run_llm.py               # Main script to interact with the local LLM.
textbook_llm.py          # Script for querying the textbook using the LLM.
textbook.txt             # Text version of the textbook used for context.
db/                      # Database folder for storing embeddings and data.
   chroma.sqlite3         # SQLite database for vector storage (Chroma).
   a77b493b-.../          # Chroma DB internal files.
textbook_create/         # Utilities for creating the textbook text file.
   pdf_to_text.py         # Script to convert PDF textbook to text.
   textbook-pdf/          # Folder to store original textbook PDFs.
```

- **run_llm.py**: Main entry point for running LLM queries.
- **textbook_llm.py**: Specialized for textbook-based Q&A.
- **textbook.txt**: The context source for textbook_llm.py.
- **db/**: Stores vector database files for fast retrieval.
- **textbook_create/**: Tools for converting and managing textbook files.


## 🧠 Run a Model

After installing Ollama, open your terminal and run:

```powershell
# First time: download models
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text
# Or choose other models per your preference
```

You’ll now have a local chatbot that works entirely offline.

---

## 🚀 Setup & Usage Guide

1. **Clone this repository**
   ```sh
   git clone https://github.com/ISNE11/Local-LLM.git
   cd Local-LLM
   ```

3. **Install Python dependencies**

   ```powershell
   # Create and activate a venv (Windows PowerShell)
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   # Install deps
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is missing, install any needed packages manually as you run the scripts.)*

4. **Start Ollama**
   - Make sure Ollama is running and the desired model is available.
   - Make sure the python files are set to use the correct model name.

5. **Run the main script**
   ```powershell
   python run_llm.py
   ```
   or
    ```powershell
    python textbook_llm.py
    ```

### Optional: custom Ollama host

If Ollama runs on a different machine or port, set `OLLAMA_HOST`:

```powershell
$env:OLLAMA_HOST = "http://127.0.0.1:11434"   # default
# or, e.g. remote docker/another PC
$env:OLLAMA_HOST = "http://192.168.1.50:11434"
```

The scripts will use `OLLAMA_HOST` automatically for both the LLM and embeddings.

---

## Troubleshooting

- Error: `[WinError 10061] No connection could be made ...` or `httpx.ConnectError`  
   → Ollama isn't reachable. Make sure it's running and the host/port are correct. Try opening the Ollama app on Windows or run `ollama serve`. Also ensure the model is pulled: `ollama pull deepseek-r1:8b`.

- Error: `ModuleNotFoundError: No module named 'rich'`  
   → Install dependencies: `pip install -r requirements.txt`. The scripts include a fallback if `rich` isn't installed, but installing it gives nicer output.

- First query is slow  
   → The vector DB and model may be initializing or downloading on first run. Subsequent runs will be much faster.

6. **Interact with your local LLM!**

---

## 📄 Notes
- This repo assumes you have Python 3.8+ installed.
- Ollama must be running in the background for the scripts to connect to the local LLM.
- For more models and advanced usage, see the [Ollama documentation](https://ollama.com/library).

---
![MIT License](https://img.shields.io/badge/License-MIT-green.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

Local-LLM is an open source project that makes it easy to run Large Language Models (LLMs) entirely offline on your own machine using [Ollama](https://ollama.com/). This project is designed for privacy, speed, and full local control—no cloud required!

This project is licensed under the [MIT License](./LICENSE).
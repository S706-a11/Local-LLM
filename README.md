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

```sh
ollama pull mistral
ollama run mistral
```

You’ll now have a local chatbot that works entirely offline.

---

## 🚀 Setup & Usage Guide

1. **Clone this repository**
   ```sh
   git clone https://github.com/ISNE11/Local-LLM.git
   cd Local-LLM
   ```

2. **(Optional) Create a virtual environment**
   ```sh
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```sh
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is missing, install any needed packages manually as you run the scripts.)*

4. **Start Ollama**
   - Make sure Ollama is running and the desired model (e.g., `mistral`) is available.

5. **Run the main script**
   ```sh
   python run_llm.py
   ```
   or
   ```sh
   python textbook_llm.py
   ```

6. **Interact with your local LLM!**

---

## 📄 Notes
- This repo assumes you have Python 3.8+ installed.
- Ollama must be running in the background for the scripts to connect to the local LLM.
- For more models and advanced usage, see the [Ollama documentation](https://ollama.com/library).

---

## 🤝 Contributing
Pull requests and issues are welcome!

---

## 📜 License
MIT License

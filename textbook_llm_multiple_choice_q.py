import warnings
import time
import re
from rich.console import Console
from rich.markdown import Markdown
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
import os

# ───────────────────────────────
# Setup
# ───────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
console = Console()

TEXTBOOK_PATH = "textbook.txt"
VECTOR_DB_DIR = "./db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "deepseek-r1:8b"

# ───────────────────────────────
# Load textbook
# ───────────────────────────────
console.print("📥 Loading textbook...", style="bold yellow")
loader = TextLoader(TEXTBOOK_PATH, encoding="utf-8")
docs = loader.load()
console.print(f"✅ Loaded {len(docs)} document(s).", style="bold green")

# ───────────────────────────────
# Split into smaller chunks
# ───────────────────────────────
console.print("✂️ Splitting documents into smaller chunks...", style="bold yellow")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(docs)
console.print(f"✅ Split into {len(texts)} chunks.", style="bold green")

# ───────────────────────────────
# Create or load vector DB
# ───────────────────────────────
emb = OllamaEmbeddings(model=EMBED_MODEL)
if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
    console.print("📂 Loading existing vector DB...", style="bold yellow")
    db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=emb)
    console.print("✅ Vector DB loaded.", style="bold green")
else:
    console.print("🧠 Creating vector DB (this may take time)...", style="bold yellow")
    db = Chroma.from_documents(texts, emb, persist_directory=VECTOR_DB_DIR)
    console.print("✅ Vector DB created!", style="bold green")

# ───────────────────────────────
# Initialize LLM
# ───────────────────────────────
console.print("🤖 Initializing DeepSeek LLM...", style="bold yellow")
llm = OllamaLLM(model=LLM_MODEL)
console.print("✅ LLM ready!\n", style="bold green")

# ───────────────────────────────
# Helper
# ───────────────────────────────
def split_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)

# ───────────────────────────────
# Main chat loop (fast MCQ mode)
# ───────────────────────────────
console.print("📘 Ready! Type your question (or 'exit' to quit).\n", style="bold cyan")

while True:
    user_query = input("You: ").strip()
    if user_query.lower() in {"exit", "quit"}:
        console.print("👋 Goodbye!", style="bold cyan")
        break

    console.print("🔍 Searching relevant textbook content...", style="bold yellow")
    docs = db.similarity_search(user_query, k=5)

    # Deduplicate chunks
    seen = set()
    unique_texts = [d.page_content for d in docs if not (d.page_content in seen or seen.add(d.page_content))]
    context = "\n".join(unique_texts)
    console.print(f"✅ Using {len(unique_texts)} chunk(s) for context.", style="bold green")

    # Build prompt for MCQ: answer first, explanation second
    prompt = f"""
You are an expert at answering multiple-choice questions based on the provided textbook content.

Textbook content:
{context}

Question: {user_query}

Instructions:
1. First, give the correct answer concisely (e.g., "Answer: B").
2. Then provide a step-by-step explanation below your answer.
"""

    # Stream LLM output
    console.print("\n🤖 DeepSeek: ", end="")
    full_response = ""
    for chunk in llm.stream(prompt):
        full_response += chunk
        print(chunk, end="", flush=True)
    print("\n")

    console.print(Markdown(full_response))

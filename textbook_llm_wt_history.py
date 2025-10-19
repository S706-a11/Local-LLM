import warnings
import time
import re
import os
import socket
try:
    from rich.console import Console
    from rich.markdown import Markdown
except Exception:
    class _PlainConsole:
        def print(self, *args, **kwargs):
            print(*args)

    class _PlainMarkdown(str):
        pass

    Console = _PlainConsole
    Markdown = _PlainMarkdown
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# ──────────────────────────────────────────────
# Setup and warnings
# ──────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
console = Console()

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
TEXTBOOK_PATH = "textbook.txt"
VECTOR_DB_DIR = "./db"
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-r1:8b")  ## Change model here
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")


def _wait_for_ollama(host: str, timeout: float = 2.0) -> bool:
    try:
        addr = host.replace("http://", "").replace("https://", "")
        if "/" in addr:
            addr = addr.split("/")[0]
        if ":" in addr:
            h, p = addr.split(":", 1)
            port = int(p)
        else:
            h, port = addr, 11434
        with socket.create_connection((h, port), timeout=timeout):
            return True
    except Exception:
        return False

# ──────────────────────────────────────────────
# Step 1: Load Textbook
# ──────────────────────────────────────────────
console.print("📥 Loading textbook...", style="bold yellow")
if not os.path.exists(TEXTBOOK_PATH):
    console.print(f"❌ File not found: {TEXTBOOK_PATH}", style="bold red")
    exit()

loader = TextLoader(TEXTBOOK_PATH, encoding="utf-8")
docs = loader.load()
console.print(f"✅ Loaded {len(docs)} document(s).", style="bold green")

# ──────────────────────────────────────────────
# Step 2: Split Textbook into Chunks
# ──────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(docs)
console.print(f"✅ Split into {len(texts)} chunks.", style="bold green")

# ──────────────────────────────────────────────
# Step 3: Load or Create Vector Database
# ──────────────────────────────────────────────
emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)

if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
    console.print("📂 Loading existing vector DB (cached)...", style="bold yellow")
    db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=emb)
    console.print("✅ Vector DB loaded from cache.", style="bold green")
else:
    console.print("🧠 Creating vector DB (first run, may take time)...", style="bold yellow")
    db = Chroma.from_documents(texts, emb, persist_directory=VECTOR_DB_DIR)
    console.print("✅ Vector DB created and saved for future runs.", style="bold green")

# ──────────────────────────────────────────────
# Step 4: Initialize LLM
# ──────────────────────────────────────────────
console.print("🤖 Initializing DeepSeek LLM...", style="bold yellow")
if not _wait_for_ollama(OLLAMA_HOST):
    console.print(
        f"❌ Can't connect to Ollama at {OLLAMA_HOST}. Ensure it's running.",
        style="bold red",
    )
    console.print(f"Tip: Pull required models first: ollama pull {LLM_MODEL} ; ollama pull {EMBED_MODEL}")
    raise SystemExit(1)

llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_HOST)
console.print("✅ LLM ready!\n", style="bold green")

# ──────────────────────────────────────────────
# Helper: Sentence Splitter
# ──────────────────────────────────────────────
def split_sentences(text):
    """Split text by sentence-ending punctuation for smoother streaming."""
    return re.split(r'(?<=[.!?]) +', text)

# ──────────────────────────────────────────────
# Step 5: Chat Loop
# ──────────────────────────────────────────────
conversation_history = []
console.print("📘 Ready! Type your question (or 'exit' to quit).\n", style="bold cyan")

while True:
    user_query = input("You: ").strip()
    if user_query.lower() in {"exit", "quit"}:
        console.print("👋 Goodbye!", style="bold cyan")
        break

    # Record question in conversation history
    conversation_history.append(f"You: {user_query}")

    # Search top 5 relevant chunks
    start_time = time.time()
    docs = db.similarity_search(user_query, k=5)
    elapsed = time.time() - start_time
    console.print(f"🔍 Found relevant chunks in {elapsed:.2f}s.", style="bold yellow")

    # Deduplicate chunks
    seen = set()
    unique_texts = [d.page_content for d in docs if not (d.page_content in seen or seen.add(d.page_content))]
    context = "\n".join(unique_texts)

    # Build LLM prompt
    prompt = f"""
Conversation history:
{chr(10).join(conversation_history[-5:])}  # Only keep last 5 turns for speed

Use the following textbook content as your primary reference.
If necessary, reason with general knowledge but prioritize textbook facts.
Avoid repeating information that was already explained earlier.

Textbook content:
{context}

Question: {user_query}
"""

    # Stream response faster
    console.print("\n🤖 DeepSeek: ", end="")
    full_response = ""

    # Start LLM response
    try:
        for chunk in llm.stream(prompt):
            full_response += chunk
            # print instantly without sentence-by-sentence delay
            print(chunk, end="", flush=True)
    except Exception as e:
        console.print(f"\n❌ Error during generation: {e}", style="bold red")
        console.print(f"Check that Ollama is running at {OLLAMA_HOST} and models are pulled.")
        continue

    print("\n")
    conversation_history.append(f"DeepSeek: {full_response}")
    console.print(Markdown(full_response))

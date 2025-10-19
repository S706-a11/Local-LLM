import warnings
import time
import re
import os
import socket
import sys
try:
    from rich.console import Console
    from rich.markdown import Markdown
except Exception:
    # Minimal fallbacks if 'rich' isn't installed
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
# Setup
# ──────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
console = Console()

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


def _ensure_models_present(host: str, required: list[str]) -> bool:
    """Check that required models exist in Ollama. Returns True if all found."""
    try:
        try:
            from ollama import Client  # type: ignore
        except Exception:
            return True  # Don't block if client import fails; connectivity check covers us.
        client = Client(host=host)
        data = client.list() or {}
        models = data.get("models", [])
        names = set()
        for m in models:
            # accept either 'model' or 'name' field depending on ollama version
            n = m.get("model") or m.get("name")
            if n:
                names.add(n)

        def has(name: str) -> bool:
            base = name.split(":")[0]
            for t in names:
                if t == name:
                    return True
                if t.split(":")[0] == base:
                    return True
            return False

        missing = [n for n in required if not has(n)]
        if missing:
            console.print(
                f"❌ Missing Ollama model(s): {', '.join(missing)}",
                style="bold red",
            )
            console.print(
                "Run the following in a terminal:",
            )
            for m in missing:
                console.print(f"  ollama pull {m}")
            return False
        return True
    except Exception:
        # If listing fails for any reason, don't block execution here.
        return True

# ──────────────────────────────────────────────
# 1️⃣ Load textbook
# ──────────────────────────────────────────────
console.print("📥 Loading textbook...", style="bold yellow")
if not os.path.exists(TEXTBOOK_PATH):
    console.print(f"❌ File not found: {TEXTBOOK_PATH}", style="bold red")
    exit()

loader = TextLoader(TEXTBOOK_PATH, encoding="utf-8")
docs = loader.load()
console.print(f"✅ Loaded {len(docs)} document(s).", style="bold green")

# ──────────────────────────────────────────────
# 2️⃣ Split into smaller chunks
# ──────────────────────────────────────────────
console.print("✂️ Splitting documents into smaller chunks...", style="bold yellow")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(docs)
console.print(f"✅ Split into {len(texts)} chunks.", style="bold green")

# ──────────────────────────────────────────────
# 3️⃣ Create or load vector database
# ──────────────────────────────────────────────
if not _wait_for_ollama(OLLAMA_HOST):
    console.print(
        f"❌ Can't connect to Ollama at {OLLAMA_HOST}. Ensure it's running (e.g., 'ollama serve').",
        style="bold red",
    )
    console.print(f"Tip: Pull required models first: ollama pull {LLM_MODEL} ; ollama pull {EMBED_MODEL}")
    raise SystemExit(1)

emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)

if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
    console.print("📂 Loading existing vector DB (cached)...", style="bold yellow")
    db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=emb)
    console.print("✅ Vector DB loaded from cache.", style="bold green")
else:
    console.print("🧠 Creating new vector DB (first run, may take time)...", style="bold yellow")
    if not _ensure_models_present(OLLAMA_HOST, [EMBED_MODEL]):
        raise SystemExit(1)
    db = Chroma.from_documents(texts, emb, persist_directory=VECTOR_DB_DIR)
    console.print("✅ Vector DB created and cached for next time!", style="bold green")

# ──────────────────────────────────────────────
# 4️⃣ Initialize LLM
# ──────────────────────────────────────────────
console.print("🤖 Initializing DeepSeek LLM...", style="bold yellow")
if not _wait_for_ollama(OLLAMA_HOST):
    console.print(
        f"❌ Can't connect to Ollama at {OLLAMA_HOST}. Ensure it's running (e.g., 'ollama serve') and reachable.",
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
# 5️⃣ Main chat loop (no history)
# ──────────────────────────────────────────────
console.print("📘 Ready! Type your question (or 'exit' to quit).\n", style="bold cyan")

while True:
    user_query = input("You: ").strip()
    if user_query.lower() in {"exit", "quit"}:
        console.print("👋 Goodbye!", style="bold cyan")
        break

    # Search relevant chunks
    start_time = time.time()
    docs = db.similarity_search(user_query, k=5)
    elapsed = time.time() - start_time
    console.print(f"🔍 Found relevant chunks in {elapsed:.2f}s.", style="bold yellow")

    # Deduplicate chunks
    seen = set()
    unique_texts = [d.page_content for d in docs if not (d.page_content in seen or seen.add(d.page_content))]
    context = "\n".join(unique_texts)
    console.print(f"✅ Using {len(unique_texts)} unique chunk(s) for context.", style="bold green")

    # Build prompt
    prompt = f"""
Use the following textbook content as reference. 
You may also reason using your general knowledge if needed,
but prioritize the textbook content. Provide clear, step-by-step explanations.

Textbook content:
{context}

Question: {user_query}
"""

    # ──────────────────────────────────────────────
    # Stream LLM output (fast streaming)
    # ──────────────────────────────────────────────
    console.print("\n🤖 DeepSeek: ", end="")
    full_response = ""

    try:
        for chunk in llm.stream(prompt):
            full_response += chunk
            print(chunk, end="", flush=True)
    except Exception as e:
        console.print(f"\n❌ Error during generation: {e}", style="bold red")
        console.print(f"Check that Ollama is running at {OLLAMA_HOST} and models are pulled.")
        continue

    print("\n")
    console.print(Markdown(full_response))

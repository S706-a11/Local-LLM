import warnings
import time
import re
import os
from rich.console import Console
from rich.markdown import Markdown
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "deepseek-r1:8b"  ## Change model here

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
emb = OllamaEmbeddings(model=EMBED_MODEL)

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
llm = OllamaLLM(model=LLM_MODEL)
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
    for chunk in llm.stream(prompt):
        full_response += chunk
        # print instantly without sentence-by-sentence delay
        print(chunk, end="", flush=True)

    print("\n")
    conversation_history.append(f"DeepSeek: {full_response}")
    console.print(Markdown(full_response))

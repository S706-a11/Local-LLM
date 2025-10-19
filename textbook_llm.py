import warnings
import time
import re
from rich.console import Console
from rich.markdown import Markdown

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import community/up-to-date LangChain packages
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

console = Console()

# 1️⃣ Load your textbook
console.print("📥 Loading textbook...", style="bold yellow")
loader = TextLoader("textbook.txt", encoding="utf-8")
docs = loader.load()
console.print(f"✅ Loaded {len(docs)} document(s).", style="bold green")

# 2️⃣ Split into smaller chunks
console.print("✂️ Splitting documents into smaller chunks...", style="bold yellow")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(docs)
console.print(f"✅ Split into {len(texts)} chunks.", style="bold green")

# 3️⃣ Create embeddings / vector DB
persist_dir = "./db"
try:
    console.print("📂 Loading existing vector DB...", style="bold yellow")
    hf_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=persist_dir, embedding_function=hf_model)
    console.print("✅ Vector DB loaded.", style="bold green")
except:
    console.print("🔢 Converting chunks into vector embeddings (this may take a while)...", style="bold yellow")
    hf_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(texts, hf_model, persist_directory=persist_dir)
    console.print("✅ Vector database created. Ready for questions!", style="bold green")

# 4️⃣ Initialize LLM
console.print("🤖 Initializing DeepSeek LLM...", style="bold yellow")
llm = OllamaLLM(model="deepseek-r1:8b")
console.print("✅ LLM ready!\n", style="bold green")

# Helper function to split text into sentences
def split_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)

# 5️⃣ Chat loop
console.print("📘 Ready! Type your question (or 'exit' to quit).\n", style="bold cyan")
while True:
    user_query = input("You: ")
    if user_query.lower() in ["exit", "quit"]:
        console.print("👋 Goodbye!", style="bold cyan")
        break

    console.print("🔍 Searching for relevant textbook content...", style="bold yellow")

    # Search top 5 relevant chunks
    docs = db.similarity_search(user_query, k=5)

    # Deduplicate chunks
    seen = set()
    unique_texts = []
    for d in docs:
        if d.page_content not in seen:
            unique_texts.append(d.page_content)
            seen.add(d.page_content)
    context = "\n".join(unique_texts)
    console.print(f"✅ Found {len(unique_texts)} relevant chunk(s).", style="bold green")

    # Build prompt for reasoning
    prompt = f"""
Use the following textbook content as reference. You may also reason using your general knowledge if needed,
but prioritize the textbook content. Think step by step and provide clear, detailed explanations.
Do NOT repeat information unnecessarily; if something is already explained, avoid restating it.

Textbook content:
{context}

Question: {user_query}
"""

    # Stream sentences like a chatbot
    console.print("\n🤖 DeepSeek: ", end="")
    full_response = ""
    for chunk in llm.stream(prompt):
        full_response += chunk
        sentences = split_sentences(chunk)
        for sentence in sentences:
            print(sentence + " ", end="", flush=True)
            time.sleep(0.05)  # Adjust typing speed here
    print("\n")

    # Render Markdown for final nicely formatted output
    console.print(Markdown(full_response))

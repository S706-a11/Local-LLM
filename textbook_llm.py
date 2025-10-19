import warnings
import time
import re
from rich.console import Console
from rich.markdown import Markdown

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

console = Console()

# 1️⃣ Load your textbook
loader = TextLoader("textbook.txt", encoding="utf-8")
docs = loader.load()

# 2️⃣ Split into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(docs)

# 3️⃣ Create embeddings
hf_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(texts, hf_model, persist_directory="./db")

# 4️⃣ Initialize LLM
llm = OllamaLLM(model="deepseek-r1:8b")

console.print("📘 Ready! Type your question (or 'exit' to quit).\n", style="bold cyan")

# Helper function to split text into sentences
def split_sentences(text):
    # This regex splits on period, question mark, exclamation mark followed by space
    return re.split(r'(?<=[.!?]) +', text)

# 5️⃣ Streaming chat loop
while True:
    user_query = input("You: ")
    if user_query.lower() in ["exit", "quit"]:
        break

    # Retrieve relevant chunks
    docs = db.similarity_search(user_query)
    context = "\n".join([d.page_content for d in docs])

    # Build prompt with reasoning
    prompt = f"""
Use the following textbook content as reference. You may also reason using your general knowledge if needed,
but prioritize the textbook content. Think step by step and provide clear, detailed explanations.

Textbook content:
{context}

Question: {user_query}
"""

    # Stream raw sentences first
    console.print("\n🤖 DeepSeek: ", end="")
    full_response = ""
    for chunk in llm.stream(prompt):
        full_response += chunk
        sentences = split_sentences(chunk)
        for sentence in sentences:
            print(sentence + " ", end="", flush=True)
            time.sleep(0.05)  # adjust typing speed here
    print("\n")

    # Once full response is complete, render proper Markdown
    console.print(Markdown(full_response))

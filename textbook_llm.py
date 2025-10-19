# Suppress any remaining UserWarnings (just in case)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Use only the community/up-to-date packages
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

# 1️⃣ Load your textbook
loader = TextLoader("textbook.txt", encoding="utf-8")
docs = loader.load()

# 2️⃣ Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = splitter.split_documents(docs)

# 3️⃣ Create embeddings using the updated HuggingFace package
hf_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(texts, hf_model, persist_directory="./db")

# 4️⃣ Initialize the updated Ollama LLM
llm = OllamaLLM(model="deepseek-r1:8b")

print("📘 Ready! Type your question (or 'exit' to quit).")

# 5️⃣ Streaming chat loop
while True:
    user_query = input("You: ")
    if user_query.lower() in ["exit", "quit"]:
        break

    # Retrieve relevant chunks
    docs = db.similarity_search(user_query)
    context = "\n".join([d.page_content for d in docs])

    # Prompt that allows reasoning
    prompt = f"""
Use the following textbook content as a reference. You may also reason using your general knowledge if needed, 
but prioritize the textbook content. Think step by step and provide clear, detailed explanations.

Textbook content:
{context}

Question: {user_query}
"""

    # Stream the response
    print("\n🤖 DeepSeek: ", end="", flush=True)
    for chunk in llm.stream(prompt):
        print(chunk, end="", flush=True)
    print("\n")

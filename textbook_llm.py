from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama

# 1️⃣ Load your textbook text
loader = TextLoader("textbook.txt")
docs = loader.load()

# 2️⃣ Split into chunks for searching
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = splitter.split_documents(docs)

# 3️⃣ Create embeddings and store them locally
embeddings = OllamaEmbeddings(model="deepseek-r1:8b")
db = Chroma.from_documents(texts, embeddings, persist_directory="./db")

# 4️⃣ Start an interactive Q&A loop
llm = Ollama(model="deepseek-r1:8b")

print("📘 Ready! Ask me anything from your textbook. Type 'exit' to quit.\n")

while True:
    query = input("Question: ")
    if query.lower() in ["exit", "quit"]:
        break

    docs = db.similarity_search(query)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"Use only the following textbook content to answer clearly:\n\n{context}\n\nQuestion: {query}"
    response = llm(prompt)
    print("\n🧠 Answer:", response, "\n")

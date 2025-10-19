from langchain_ollama import OllamaLLM

# Initialize Ollama LLM
llm = OllamaLLM(model="deepseek-r1:8b")

print("🧠 DeepSeek-R1:8B is ready. Type 'exit' to quit.\n")

while True:
    prompt = input("You: ")
    if prompt.lower() in ["exit", "quit"]:
        break

    response = llm.invoke(prompt)
    print(f"\n🤖 DeepSeek: {response}\n")

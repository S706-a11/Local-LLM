from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="deepseek-r1:8b")

print("🧠 DeepSeek-R1:8B (Streaming Mode). Type 'exit' to quit.\n")

while True:
    prompt = input("You: ")
    if prompt.lower() in ["exit", "quit"]:
        break

    print("\n🤖 DeepSeek: ", end="", flush=True)
    for chunk in llm.stream(prompt):
        print(chunk, end="", flush=True)
    print("\n")

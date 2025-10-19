import os
import sys
import socket
import time
from langchain_ollama import OllamaLLM

MODEL = os.getenv("LLM_MODEL", "deepseek-r1:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")


def _wait_for_ollama(host: str, timeout: float = 2.0) -> bool:
    """Quick TCP check that Ollama is reachable."""
    try:
        # host can be http://host:port or host:port
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


if not _wait_for_ollama(OLLAMA_HOST):
    print(
        "❌ Can't connect to Ollama at",
        OLLAMA_HOST,
        "\n• Make sure Ollama is installed and running",
        "\n  - Windows: Launch 'Ollama' app or run: ollama serve",
        "\n• Optionally set OLLAMA_HOST if using a non-default address",
        "\n• Pull the model (first time only): ollama pull",
        MODEL,
        sep=" ",
    )
    sys.exit(1)

try:
    llm = OllamaLLM(model=MODEL, base_url=OLLAMA_HOST)
except Exception as e:
    print("❌ Failed to initialize LLM:", e)
    print("Hint: Ensure the model is pulled: ollama pull", MODEL)
    sys.exit(1)

print(f"🧠 {MODEL} (Streaming Mode). Type 'exit' to quit.\n")

while True:
    try:
        prompt = input("You: ")
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if prompt.strip().lower() in ["exit", "quit"]:
        break

    print("\n🤖 DeepSeek: ", end="", flush=True)
    try:
        for chunk in llm.stream(prompt):
            print(chunk, end="", flush=True)
    except Exception as e:
        print("\n❌ Streaming error:", e)
        print("If this mentions a connection error, ensure Ollama is running and reachable at", OLLAMA_HOST)
        print("If it mentions the model, pull it: ollama pull", MODEL)
    print("\n")

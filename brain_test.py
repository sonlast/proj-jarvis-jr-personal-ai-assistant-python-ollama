import ollama

MODEL = "qwen3:8b"
SYSTEM = (
    "You are JARVIS, a calm, polite, dry-witted personal assistant. "
    "Address the user as 'sir'. Keep replies short (1 to 3 sentences). "
    "Never use emojis, markdowns, or lists, because your replies are spoken aloud."
)

history = [
    {
      "role": "system",
      "content": SYSTEM
    }
]
print("JARVIS text test. Type 'exit' to quit.")

while True:
  text = input("You: ").strip()
  if text.lower() in ("exit", "quit"):
    break
  history.append({"role": "user", "content": text})
  reply = ollama.chat(model=MODEL, messages=history,
                      think=False, keep_alive="30m")
  answer = reply["message"]["content"].strip()
  history.append({"role": "assistant", "content": answer})
  print("JARVIS:", answer)
import ollama


res = ollama.chat(
    model = "llama3.2",
    messages = [
        {
            "role": "user", 
         
            "content": "why is the sky blue?"}
    ],
    stream = True
)

for chunks in res:
    print()
print(res)
import ollama


def generate_ollama_response(prompt):

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    return response["message"]["content"]

import requests
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

messages=[{"role": "system", "content": "You are a certified fitness and nutrition coach who gives realistic meal and workout plans."}]



while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    messages.append({"role": "user", "content": user_input})

    chat_completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    reply = chat_completion.choices[0].message.content
    print("Coach:", reply)

    messages.append({"role": "assistant", "content": reply})



import requests
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))



age = input("Enter your age: ")
gender = input("Enter your gender: ")
weight = input("Enter your current weight (in lbs): ")
height = input("Enter your height (in inches): ")
goal = input("Are you hoping to (1) lose weight, (2) gain muscle, or (3) maintain weight and increase overall health? ")
time = input("When would you like to accomplish this fitness goal by, ideally? ")

match goal:
    case "1":
        weight_goal = input("Enter your weight goal (in lbs): ")
        goal = f"{age} year old {gender} trying to lose weight. User is {height} inches tall and currently weighs {weight} pounds. Their goal is to be {weight_goal} pounds by {time}. Give them a plan day by day with daily meals and snacks and calories and protein included for each food item. And give them a detailed workout plan for each day. And give additional helpful tips."
    case "2":
        weight_goal = input("Enter your weight goal (in lbs) after gaining additional muscle: ")
        goal = f"{age} year old {gender} trying to gain muscle. User is {height} inches tall and currently weighs {weight} pounds. Their goal is to be {weight_goal} pounds by {time}, after gaining muscle. Give them a plan day by day with daily meals and snacks and calories and protein included for each food item. And give them a detailed workout plan for each day. And give additional helpful tips."
    case "3":
        goal = f"{age} year old {gender} trying to maintain weight and increase overall health. User is {height} inches tall and currently weighs {weight} pounds. Their goal is to adopt this new healthy lifestyle by {time}. Give them a plan day by day with daily meals and snacks and calories and protein included for each food item. And give them a detailed workout plan for each day. And give additional helpful tips."
    case _:
        goal = f"{age} year old {gender} trying to maintain weight and increase overall health. User is {height} inches tall and currently weighs {weight} pounds. Their goal is to adopt this new healthy lifestyle by {time}. Give them a plan day by day with daily meals and snacks and calories and protein included for each food item. And give them a detailed workout plan for each day. And give additional helpful tips."

print("\nGenerating response...\n")

messages=[{"role": "system", "content": f"You are a certified fitness and nutrition coach who gives realistic meal and workout plans. Give advice centered around the user's info: ' {goal}."}]

messages.append({"role": "user", "content": goal})

chat_completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

reply = chat_completion.choices[0].message.content
print("Coach:", reply)

print("\n\n")

messages.append({"role": "assistant", "content": reply})

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    print("\nGenerating response...\n")

    messages.append({"role": "user", "content": user_input})

    chat_completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    reply = chat_completion.choices[0].message.content
    print("Coach:", reply)

    print("\n\n")

    messages.append({"role": "assistant", "content": reply})



from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import Accelerator


model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
accelerator = Accelerator()
device = accelerator.device


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#memory loop
system_prompt = (
    "You are a friendly and knowledgeable fitness coach. "
    "Respond conversationally and give positive, useful advice about health and workouts."
)
conversation = system_prompt + "\n\n"

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    #Add user message to memory
    conversation += f"User: {user_input}\nCoach:"

    #Encode the entire conversation so far
    inputs = tokenizer(conversation, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    #Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.9
    )

    #Decode the reply
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #Extract only the latest part after the most recent "Coach:"
    reply = reply.split("Coach:")[-1].strip()

    print("Coach:", reply)

    #Append the model's reply to the memory for the next turn
    conversation += f" {reply}\n"




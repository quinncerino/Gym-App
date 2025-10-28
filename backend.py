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

system_prompt2 = (
    """You are a certified nutrition coach. 
    Respond with a realistic, calorie-accurate 9-day meal plan for healthy weight loss.
    Each day should list:
    - Breakfast (~350–450 cal)
    - Lunch (~450–550 cal)
    - Dinner (~450–550 cal)
    Also give the *total daily calories* and a short motivational note.

    Format exactly as:
    Day 1:
    Breakfast:
    Lunch:
    Dinner:
    Total Calories:
    Note:
    """
)

conversation = system_prompt2 + "\n\n"

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    #Add user message to memory
    # conversation += f"User: {user_input}\nCoach:"
    conversation += (
        f"User: {user_input}\n"
        "Coach: Please respond in full detail. "
        "Do not stop until you have completed the full response.\n"
    )

    #Encode the entire conversation so far
    inputs = tokenizer(conversation, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    #Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        pad_token_id=tokenizer.eos_token_id
    )

    #Decode the reply
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #Extract only the latest part after the most recent "Coach:"
    reply = reply.split("Coach:")[-1].strip()
    print("Coach:", reply)

    #Append the model's reply to the memory for the next turn
    conversation += f" {reply}\n"

    continuation_attempts = 0
    max_continuations = 3  # limit to 3 extra generations
    cutoff_words = ("and", "with", "to", "of", "the", "Stay")
    complete = False

    #auto continue if response appears incomplete
    while not complete and continuation_attempts < max_continuations:
        inputs = tokenizer(conversation, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply = reply.split("Coach:")[-1].strip()

        if continuation_attempts == 0:
            print("Coach:", reply)
        else:
            print("Coach (cont’d):", reply)

        conversation += f"{reply}\n"

        if reply.endswith((".", "!", "?")) and not reply.endswith(cutoff_words):
            complete = True
        else:
            continuation_attempts += 1
            if continuation_attempts < max_continuations:
                conversation += "User: Please continue where you left off, finish your thought clearly.\nCoach:"
            else:
                print("Coach (done): (stopped after max continuation attempts)")
                complete = True

        # if reply.endswith((".", "!", "?")):
        #     break
        # if not (reply.endswith(cutoff_words) or len(reply.split()) < 30):
        #     break

        # followup = "Please finish your previous response clearly but do not repeat or summarize."
        # conversation += f"User: {followup}\nCoach:"
        # # re-generate again to complete
        # inputs = tokenizer(conversation, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        

        # extra = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # extra = extra.split("Coach:")[-1].strip()
        # print("Coach (cont’d):", extra)

        # reply = extra
        # conversation += f"{extra}\n"
        # continuation_attempts += 1
    

    
    




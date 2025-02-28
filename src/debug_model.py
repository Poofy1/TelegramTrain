from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import re

# Load the model and tokenizer
env = os.path.dirname(os.path.abspath(__file__))
model_name = f"{env}/checkpoints/checkpoint-8750"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def generate_text(prompt, username, max_length=4096):
    # Format the prompt with chat participants
    chat_participants_text = f"Chat Participants: <Coelacant1>, {username}\n"
    formatted_prompt = "<|user|>\n" + chat_participants_text + prompt + "<|end|>\n<|assistant|>\n"
    
    # Encode the input
    inputs = tokenizer.encode_plus(formatted_prompt, return_tensors="pt", padding=False, truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    pad_token_id = tokenizer.eos_token_id
    eos_token_id = tokenizer.encode("<|end|>")
    
    # Generate response
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        pad_token_id=pad_token_id,
        top_k=50,
        top_p=0.95,
        eos_token_id=eos_token_id,
    )
    
    # Decode and clean up the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    
    # Extract the assistant's response
    match = re.search(r'<\|assistant\|>(.*?)<\|end\|>', generated_text, re.DOTALL)
    if match:
        generated_text = match.group(1).strip()
    else:
        generated_text = ""
    
    return generated_text.strip()

def main():
    username = input("Enter your username: ")
    username = f"<{username}>"
    
    print("\nEnter your messages (type 'quit' to exit):")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        message = f"{username}:{user_input}\n"
        response = generate_text(message, username)
        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

env = os.path.dirname(os.path.abspath(__file__))
model_path = f"{env}/results/checkpoint-100"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define a function to generate text using the fine-tuned model
def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)  # Move input tensors to GPU
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Test the fine-tuned model with sample prompts
prompts = ["[INST] <<SYS>>\nRespond as if you are Peter \n<</SYS>>\n\nPoof: What is your opinion on Unity? [/INST]",
           "[INST] <<SYS>>\nRespond as if you are Peter \n<</SYS>>\n\nPoof: I love you c: [/INST]",
           ]

for prompt in prompts:
    print(f"Prompt: {prompt}")
    generated_text = generate_text(prompt)
    print(f"Generated Text: {generated_text}\n")
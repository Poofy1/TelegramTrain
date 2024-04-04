from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os, re, torch

env = os.path.dirname(os.path.abspath(__file__))
model_path = f"{env}/results/checkpoint-1100"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

respondent = "Peter"

# Define a function to generate text using the fine-tuned model
def generate_text(prompt, max_length=512):
    messages = [
        {"role": "system", "content": f"Respond as if you are {respondent}"},
        {"role": "user", "content": prompt},
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    inputs = tokenizer.encode_plus(formatted_prompt, return_tensors="pt", padding=False, truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    pad_token_id = tokenizer.eos_token_id
    eos_token_id = tokenizer.encode("<|im_end|>")[0]
    
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
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the system message and user prompt from the generated text
    pattern = re.compile(r'<\|im_start\|>system\n.*?<\|im_end\|>\n<\|im_start\|>user\n.*?<\|im_end\|>\n<\|im_start\|>assistant\n(.*?)<\|im_end\|>', re.DOTALL)
    match = pattern.search(generated_text)
    if match:
        generated_text = match.group(1).strip()
    else:
        generated_text = ""
    
    return generated_text.strip()

# Test the fine-tuned model with sample prompts
prompts = [
    ("Poof: What is your opinion on Unity?"),
    ("Poof: I love you c:"),
    ("Poof: Good morning"),
    ("Poof: Who is Brandon to you?"),
    ("Poof: What's your favorite thing to do? :3"),
    ("Poof: im gonna make sum hotdogs, want any?"),
    ("Poof: List out what kinks you enjoy"),
    ("Poof: nomnomnm\nPoof: I will eat you <3"),
    ("Poof: What's your opinion on La Crosse?"),
    ("Poof: Aww, why are you sad? ;c"),
]

for prompt in prompts:
    print(f"\nPrompt: \n{prompt}")
    generated_text = generate_text(prompt)
    print(f"\n{respondent}'s Response: \n{generated_text}\n")
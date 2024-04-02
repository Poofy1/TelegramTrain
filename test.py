from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set the path to the directory containing the model and tokenizer
#model_path = "./llama-2-7b-custom"
model_path = "D:/AI_Creation/llama/new/llama-2-7b-custom"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create the prompt
prompt = "[INST] <<SYS>>\nRespond as if you are Peter \n<</SYS>>\n\nPoof: Hey Peter, what is your opinion on Unity?. [/INST]"

# Count the number of tokens in the prompt
num_prompt_tokens = len(tokenizer.encode(prompt))

# Calculate the maximum length for the generation
max_length = 512

# Generate a response using the model and tokenizer on the specified device
gen = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Generate the text
result = gen(prompt)

# Print the generated text, removing the initial prompt
generated_text = result[0]['generated_text']
print(generated_text[len(prompt):])

import torch
import os
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers import logging as transformers_logging

# Suppress the specific warning
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.")

env = os.path.dirname(os.path.abspath(__file__))
model_cache_dir = f"{env}/models/"

# Load the pre-trained model and tokenizer
model_name = "stabilityai/stable-code-instruct-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, cache_dir=model_cache_dir, trust_remote_code=True)

# Freeze some layers
for parameter in model.model.layers[:27].parameters():
    parameter.requires_grad = False

# Tokenize the dataset
def tokenize_function(examples, max_input = 1024):
    tokenized_batches = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        
        # Truncate the user content if it's too long
        max_user_content_length = max_input - len(tokenizer.encode(instruction)) - len(tokenizer.encode(output)) - 10  # Adjust the buffer size as needed
        if len(tokenizer.encode(input_text)) > max_user_content_length:
            input_text = tokenizer.decode(tokenizer.encode(input_text)[-max_user_content_length:])

        # Prepare the messages in the format expected by the model
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output},
        ]

        # Apply the chat template to structure the prompt correctly
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

        # Tokenize the structured prompt
        tokenized_prompt = tokenizer(prompt, truncation=True, max_length=1024)
        tokenized_batches.append(tokenized_prompt)

    # Since we're assuming batch processing, we need to properly structure the tokenized output
    if len(tokenized_batches) > 0:
        # Assuming all tokenized batches have the same keys, we consolidate them into batches
        consolidated_batch = {key: [dic[key] for dic in tokenized_batches] for key in tokenized_batches[0]}
    else:
        consolidated_batch = {}

    return consolidated_batch

dataset = load_dataset("csv", data_files={"train": f"{env}/data/train.csv", "validation": f"{env}/data/val.csv"})
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir=f"{env}/checkpoints",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    num_train_epochs=30,
    weight_decay=0.025,
    push_to_hub=False,
    bf16=True,
    log_level="info",
    logging_steps=100,
    save_steps=100,
    save_total_limit=3,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Fine-tune the model
try:
    # Resume training from the latest checkpoint if available
    trainer.train(resume_from_checkpoint=True)
except ValueError as e:
    print(f"No valid checkpoint found. Starting training from scratch.")
    # Train the model from scratch
    trainer.train()

# Save the fine-tuned model
model.save_pretrained(f"{env}/fine_tuned_model")
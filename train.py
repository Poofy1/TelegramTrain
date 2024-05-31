import torch
import os, random, multiprocessing
import pandas as pd
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Suppress the specific warning
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.")

env = os.path.dirname(os.path.abspath(__file__))
model_cache_dir = f"{env}/models/"

# Load the pre-trained model and tokenizer
model_name = "stabilityai/stable-code-instruct-3b"
#model_name = "microsoft/Phi-3-mini-4k-instruct"
#model_name = "stabilityai/stablelm-2-1_6b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, cache_dir=model_cache_dir, trust_remote_code=True)

# Freeze some layers
for parameter in model.model.layers[:28].parameters():
    parameter.requires_grad = False



# Add the additional tokens to the tokenizer
key_table_path = f"{env}/data/key_table.csv"
key_table = pd.read_csv(key_table_path)
additional_tokens = key_table["token"].tolist()
tokenizer.add_special_tokens({"additional_special_tokens": additional_tokens})
model.resize_token_embeddings(len(tokenizer))


# Tokenize the dataset
def tokenize_function(examples, max_input=1024):
    tokenized_batches = []
    for i in range(len(examples["text"])):
        # Get the current message
        current_message = examples["text"][i]
        
        # Get the chat ID and validation flag
        chat_id = examples["chat_id"][i]
        
        system_prompt = f"Respond as if you are {examples['username'][i]}"
        
        # Create a context window with the closest messages
        context_window = []
        if i > 0:
            # Get the messages before the current message in the same chat
            prev_messages = [
                msg for msg, cid in zip(examples["text"][:i], examples["chat_id"][:i])
                if cid == chat_id
            ]
            
            # Determine the context window size
            max_context_size = max_input - len(tokenizer.encode(current_message)) - len(tokenizer.encode(system_prompt))
            context_size = min(max_context_size, len(prev_messages))
            
            # Get the closest messages as the context window
            context_window = prev_messages[-context_size:]
        
        # Prepare the messages in the format expected by the model
        messages = [
            {"role": "system", "content": system_prompt},
            *[{"role": "user", "content": msg} for msg in context_window],
            {"role": "user", "content": current_message},
        ]
        
        # Apply the chat template to structure the prompt correctly
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        
        # Tokenize the structured prompt
        tokenized_prompt = tokenizer(prompt, truncation=True, max_length=max_input)
        tokenized_batches.append(tokenized_prompt)
    
    # Consolidate the tokenized batches
    if len(tokenized_batches) > 0:
        consolidated_batch = {key: [dic[key] for dic in tokenized_batches] for key in tokenized_batches[0]}
    else:
        consolidated_batch = {}
    
    return consolidated_batch



dataset = load_dataset("csv", data_files={"train": f"{env}/data/train.csv"})
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

train_dataset = tokenized_datasets["train"].filter(lambda example: example["val"] == 0)
val_dataset = tokenized_datasets["train"].filter(lambda example: example["val"] == 1)


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
    gradient_accumulation_steps=7,
    gradient_checkpointing=True,
    num_train_epochs=2,
    weight_decay=1e-4,
    push_to_hub=False,
    bf16=True,
    log_level="info",
    logging_steps=250,
    save_steps=250,
    save_total_limit=3,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
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
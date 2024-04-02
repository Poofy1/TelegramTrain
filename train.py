import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
env = os.path.dirname(os.path.abspath(__file__))
model_cache_dir = f"{env}/models/"


# Tokenize the dataset
def tokenize_function(examples):
    print(examples["text"])
    return tokenizer(examples["text"], truncation=True, max_length=512)

# Load the pre-trained model and tokenizer
model_name = "stabilityai/stable-code-instruct-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, cache_dir=model_cache_dir, trust_remote_code=True)

# Freeze some layers
for parameter in model.model.layers[:24].parameters():
    parameter.requires_grad = False
    
dataset = load_dataset("csv", data_files={"train": f"{env}/data/train.csv", "validation": f"{env}/data/val.csv"})
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)



# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    bf16=True,
)


# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,  # Pass the tokenizer to the Trainer
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained(f"{env}/fine_tuned_model")
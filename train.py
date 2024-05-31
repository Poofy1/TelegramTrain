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
    prev_messages = []
    prev_chat_id = None

    for i in range(len(examples["text"])):
        # Get the current message
        current_message = examples["text"][i]

        # Get the chat ID and validation flag
        chat_id = examples["chat_id"][i]

        # Check if the chat ID has changed
        if chat_id != prev_chat_id:
            prev_messages = []
            prev_chat_id = chat_id

        # Randomly determine the context window size (between 1 and the number of previous messages)
        max_context_size = min(len(prev_messages), 25) 
        context_size = random.randint(1, max_context_size) if max_context_size > 0 else 0

        # Get the closest messages as the context window
        context_window = prev_messages[-context_size:] if context_size > 0 else []

        # Manual prompt formatting
        system_prompt = f"<|im_start|>system\nRespond as if you are {examples['username'][i]}<|im_end|>\n"
        user_text = "<|im_start|>user\n" + "\n".join(context_window) + "<|im_end|>\n"
        assistant_response = "<|im_start|>assistant\n" + current_message + "<|im_end|>\n"

        # Tokenize the system prompt, user text, and assistant response
        tokenized_system_prompt = tokenizer(system_prompt, truncation=False)
        tokenized_user_text = tokenizer(user_text, truncation=False)
        tokenized_assistant_response = tokenizer(assistant_response, truncation=False)

        # Calculate the maximum length for the user text
        max_user_length = max_input - len(tokenized_system_prompt["input_ids"]) - len(tokenized_assistant_response["input_ids"])

        # Check if there is no room for user text
        if max_user_length < 0:
            # Truncate the assistant response to make room for user text
            excess_tokens = -max_user_length + 200
            if len(tokenized_assistant_response["input_ids"]) > excess_tokens:
                tokenized_assistant_response["input_ids"] = tokenized_assistant_response["input_ids"][:-excess_tokens]
                tokenized_assistant_response["attention_mask"] = tokenized_assistant_response["attention_mask"][:-excess_tokens]
            max_user_length = max_input - len(tokenized_system_prompt["input_ids"]) - len(tokenized_assistant_response["input_ids"])
            print("Handled overflow")

        # Manually truncate the user text from the beginning
        if len(tokenized_user_text["input_ids"]) > max_user_length:
            tokenized_user_text["input_ids"] = tokenized_user_text["input_ids"][-max_user_length:]
            tokenized_user_text["attention_mask"] = tokenized_user_text["attention_mask"][-max_user_length:]

        # Combine the tokenized parts
        tokenized_prompt = {
            "input_ids": tokenized_system_prompt["input_ids"] + tokenized_user_text["input_ids"] + tokenized_assistant_response["input_ids"],
            "attention_mask": tokenized_system_prompt["attention_mask"] + tokenized_user_text["attention_mask"] + tokenized_assistant_response["attention_mask"],
        }
        tokenized_batches.append(tokenized_prompt)

        # Append the current message to the previous messages list
        prev_messages.append(current_message)

        # Decode the tokenized prompt for debugging
        if len(tokenized_prompt["input_ids"]) > max_input:
            print("-- OVERFLOW ERROR --")
            print(len(tokenized_prompt["input_ids"]))
            print(max_user_length)
            decoded_prompt = tokenizer.decode(tokenized_prompt["input_ids"])
            print("Decoded Prompt:")
            print(decoded_prompt)
            print("---")

    # Consolidate the tokenized batches
    consolidated_batch = {key: [dic[key] for dic in tokenized_batches] for key in tokenized_batches[0]} if tokenized_batches else {}

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
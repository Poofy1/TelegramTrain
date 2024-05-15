from transformers import AutoModelForCausalLM, AutoTokenizer
import os, re, torch, json
from telegram.ext import Application, MessageHandler, filters
from telegram import ReactionTypeCustomEmoji
from collections import defaultdict
from util import *
from datetime import datetime

env = os.path.dirname(os.path.abspath(__file__))
model_path = f"{env}/checkpoints/checkpoint-5250"
#model_path = f"{env}/final_models/Model3"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()



# Create a dictionary to store the conversation history for each user
conversation_history = defaultdict(list)
MAX_CHAR_LIMIT = 2000


# Define a function to generate text using the fine-tuned model
def generate_text(prompt, respondent, max_length=1024):
    system = f"Respond as if you are <{respondent}>"
    
    messages = [
        {"role": "system", "content": system},
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
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    print(generated_text)
    
    # Remove the system message, user prompt, and respondent from the generated text
    pattern = re.compile(r'<\|im_start\|>system\n.*?<\|im_end\|>\n<\|im_start\|>user\n.*?<\|im_end\|>\n<\|im_start\|>assistant\n[^:]*:(.*?)<\|im_end\|>', re.DOTALL)
    match = pattern.search(generated_text)
    if match:
        generated_text = match.group(1).strip()
    else:
        generated_text = ""

    return generated_text.strip()


# Define the message handler
async def respond(update, context):

    user_id = f'<{update.message.from_user.username}>'
    timestamp = update.message.date
    time_gap = 0.0
    # Move the prev_timestamp declaration inside the function
    prev_timestamp = context.user_data.get('prev_timestamp')

    if prev_timestamp and timestamp:
        time_gap = normalize_time_gap(timestamp - prev_timestamp)

    # Update the prev_timestamp for the current user
    context.user_data['prev_timestamp'] = timestamp
    
    if update.message.sticker:
        # Handle sticker message
        sticker_id = update.message.sticker.file_id
        message = f"{time_gap:.3f}{user_id}:<sticker-{sticker_id}>\n"
    elif update.message.photo:
        message = f"{time_gap:.3f}{user_id}:<photo>\n"
    else:
        # Handle text message
        text = update.message.text
        message = f"{time_gap:.3f}{user_id}:{text}\n"
    
    # Add the current message to the conversation history
    conversation_history[user_id].append(message)
    
    # Build the prompt by concatenating the conversation history
    prompt = ""
    for message in reversed(conversation_history[user_id]):
        if len(prompt) + len(message) <= MAX_CHAR_LIMIT:
            prompt = message + prompt
        else:
            break
    
    respondent = "mltnfox"
    generated_text = generate_text(prompt, respondent)
    
    # Add the generated response to the conversation history
    conversation_history[user_id].append(f"<{respondent}>:{generated_text}\n")
    
    
    # Check if the generated response contains a reaction
    if "<reaction-" in generated_text:
        # Extract the sticker ID from the generated response
        sticker_match = re.search(r'<reaction-(\d+)>', generated_text)
        if sticker_match:
            sticker_id = sticker_match.group(1)
            try:
                print(f"sending reaction with sticker ID: {sticker_id}")
                # Create a ReactionTypeCustomEmoji instance with the sticker ID
                reaction = ReactionTypeCustomEmoji(custom_emoji_id=sticker_id)
                # Send the reaction with the custom emoji
                await context.bot.set_message_reaction(
                    chat_id=update.message.chat_id,
                    message_id=update.message.message_id,
                    reaction=reaction,
                )
            except Exception as e:
                print(f"Failed to send reaction with sticker ID: {str(e)}")
    elif "<reaction>" in generated_text:
        # Extract the emoji from the generated response
        reaction_match = re.search(r'<reaction>(.*)', generated_text)
        if reaction_match:
            emoji = reaction_match.group(1)
            try:
                print(f"sending reaction, {emoji}")
                # Send the reaction with the emoji
                await context.bot.set_message_reaction(
                    chat_id=update.message.chat_id,
                    message_id=update.message.message_id,
                    reaction=emoji
                )
            except Exception as e:
                print(f"Failed to send reaction with emoji: {str(e)}")
                    
                
    # Check if the generated response contains a sticker
    elif "<sticker-" in generated_text:
        # Extract the sticker ID from the generated response
        sticker_match = re.search(r'<sticker-([^>]*)>', generated_text)
        if sticker_match:
            sticker_id = sticker_match.group(1)
            try:
                # Send the sticker
                await context.bot.send_sticker(
                    chat_id=update.message.chat_id,
                    sticker=sticker_id
                )
            except Exception as e:
                # If sending the sticker fails, send the "Failed sticker" message
                await context.bot.send_message(
                    chat_id=update.message.chat_id,
                    text="Failed sticker"
                )
                print(f"Failed to send sticker: {str(e)}")
    elif generated_text.strip():
        
        # Send the modified response
        await context.bot.send_message(
            chat_id=update.message.chat_id,
            text=generated_text
        )



# Load the login JSON file
with open(f'{env}/api.json') as file:
    api_credentials = json.load(file)
bot_token = api_credentials['bot_token']

# Set up the Telegram bot
def main():
    # Create the Application and pass it your bot's token
    application = Application.builder().token(bot_token).build()
    

    # Register the message handler
    message_handler = MessageHandler(filters.ALL, respond)
    application.add_handler(message_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling()

if __name__ == '__main__':
    main()
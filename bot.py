from transformers import AutoModelForCausalLM, AutoTokenizer
import os, re, torch, json
from telegram.ext import Application, MessageHandler, filters
from telegram import ReactionTypeCustomEmoji

env = os.path.dirname(os.path.abspath(__file__))
model_path = f"{env}/checkpoints/checkpoint-10600"
#model_path = f"{env}/final_models/Model3"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# Define a function to generate text using the fine-tuned model
def generate_text(prompt, respondent, max_length=1024):
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


# Load the login JSON file
with open(f'{env}/api.json') as file:
    api_credentials = json.load(file)
bot_token = api_credentials['bot_token']


# Define the message handler
async def respond(update, context):
    # Check if the message is sent by the bot itself
    if update.message.from_user.is_bot:
        return

    user_id = update.message.from_user.id
    
    if update.message.sticker:
        # Handle sticker message
        sticker_id = update.message.sticker.file_id
        prompt = f"{user_id}:(STICKER){sticker_id}\n"
    else:
        # Handle text message
        text = update.message.text
        prompt = f"{user_id}:{text}\n"
    
    print(prompt)
    respondent = "1085395295"
    generated_text = generate_text(prompt, respondent)
    #generated_text = "1085395295:(REACTION)(EMOJI_ID)5456659832494888087" 
    print(generated_text)
    
    # Check if the generated response contains a reaction
    if "(REACTION)" in generated_text and "(EMOJI_ID)" in generated_text:
        # Extract the sticker ID from the generated response
        sticker_match = re.search(r'\(EMOJI_ID\)(\d+)', generated_text)
        if sticker_match:
            sticker_id = sticker_match.group(1)
            try:
                print(f"sending reaction with sticker ID: {sticker_id}")
                # Create a ReactionTypeCustomEmoji instance with the sticker ID
                reaction = ReactionTypeCustomEmoji(custom_emoji_id=sticker_id)
                print(reaction)
                # Send the reaction with the custom emoji
                await context.bot.set_message_reaction(
                    chat_id=update.message.chat_id,
                    message_id=update.message.message_id,
                    reaction=reaction,
                )
            except Exception as e:
                print(f"Failed to send reaction with sticker ID: {str(e)}")
    elif "(REACTION)" in generated_text:
        # Extract the emoji from the generated response
        reaction_match = re.search(r'\(REACTION\)(.*)', generated_text)
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
                # Send the modified response
                await context.bot.send_message(
                    chat_id=update.message.chat_id,
                    text=generated_text
                )
                
                
    # Check if the generated response contains a sticker
    elif "(STICKER)" in generated_text:
        # Extract the sticker ID from the generated response
        sticker_match = re.search(r'\(STICKER\)(.*?)\s', generated_text)
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
    else:
        # Send the modified response
        await context.bot.send_message(
            chat_id=update.message.chat_id,
            text=generated_text
        )

# Set up the Telegram bot
def main():
    # Create the Application and pass it your bot's token
    application = Application.builder().token(bot_token).build()

    # Register the message handler
    message_handler = MessageHandler(filters.TEXT | filters.Sticker.ALL, respond)
    application.add_handler(message_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling()

if __name__ == '__main__':
    main()
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, re, torch, json, asyncio
from telegram.ext import Application, MessageHandler, filters
from telegram import ReactionTypeCustomEmoji
from collections import defaultdict

# Create a dictionary to store the conversation history for each user
conversation_history = defaultdict(list)
MAX_CHAR_LIMIT = 2000

# Define a function to generate text using the fine-tuned model
def generate_text(model, tokenizer, device, prompt, chat_participants, max_length=4096):
    
    participant_list = ", ".join(chat_participants)
    chat_participants_text = f"Chat Participants: {participant_list}\n"

    formatted_prompt = "<|user|>\n" + chat_participants_text + prompt + "<|end|>\n<|assistant|>\n"
    
    inputs = tokenizer.encode_plus(formatted_prompt, return_tensors="pt", padding=False, truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    pad_token_id = tokenizer.eos_token_id
    eos_token_id = tokenizer.encode("<|end|>")
    
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
    
    # Extract the generated response from the formatted output
    match = re.search(r'<\|assistant\|>(.*?)<\|end\|>', generated_text, re.DOTALL)
    if match:
        generated_text = match.group(1).strip()
    else:
        generated_text = ""

    return generated_text.strip()


# Define the message handler
async def respond(model, tokenizer, device, update, context):
    user_id = update.message.from_user.id
    username = f"<{update.message.from_user.username}>"
    respondent_list = [username]

    if update.message.sticker:
        sticker_id = update.message.sticker.file_id
        message = f"{username}:<sticker-{sticker_id}>\n"
    elif update.message.photo:
        message = f"{username}:<photo>\n"
    else:
        text = update.message.text
        message = f"{username}:{text}\n"
    
    conversation_history[user_id].append(message)
    

    while True:
        prompt = "".join(conversation_history[user_id][-MAX_CHAR_LIMIT:])
        generated_text = generate_text(model, tokenizer, device, prompt, respondent_list)
        if generated_text.startswith(username) or not any(generated_text.startswith(respondent) for respondent in respondent_list):
            return
        
        conversation_history[user_id].append(f"{generated_text}\n")
    
    
        if "<reaction-" in generated_text:
            sticker_match = re.search(r'<reaction-(\d+)>', generated_text)
            if sticker_match:
                sticker_id = sticker_match.group(1)
                try:
                    reaction = ReactionTypeCustomEmoji(custom_emoji_id=sticker_id)
                    await context.bot.set_message_reaction(
                        chat_id=update.message.chat_id,
                        message_id=update.message.message_id,
                        reaction=reaction,
                    )
                except Exception as e:
                    print(f"Failed to send reaction with sticker ID: {str(e)}")
        elif "<reaction>" in generated_text:
            reaction_match = re.search(r'<reaction>(.*)', generated_text)
            if reaction_match:
                emoji = reaction_match.group(1).replace(" ", "")
                try:
                    await context.bot.set_message_reaction(
                        chat_id=update.message.chat_id,
                        message_id=update.message.message_id,
                        reaction=emoji
                    )
                except Exception as e:
                    print(f"Failed to send reaction with emoji: {str(e)}")
                        
        elif "<sticker-" in generated_text:
            sticker_match = re.search(r'<sticker-([^>]*)>', generated_text)
            if sticker_match:
                sticker_id = sticker_match.group(1)
                try:
                    await context.bot.send_sticker(
                        chat_id=update.message.chat_id,
                        sticker=sticker_id
                    )
                except Exception as e:
                    await context.bot.send_message(
                        chat_id=update.message.chat_id,
                        text="Failed sticker"
                    )
                    print(f"Failed to send sticker: {str(e)}")
        elif generated_text.strip():
            await context.bot.send_message(
                chat_id=update.message.chat_id,
                text=generated_text
            )

        # Wait 30 seconds
        await asyncio.sleep(30)




# Set up the Telegram bot
def main(env):
    model_name = f"{env}/models/chatbot_checkpoints/checkpoint-8750"
    #model_path = f"{env}/final_models/Model3"

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)

    # Move the model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()



    


    # Load the login JSON file
    with open(f'{env}/api.json') as file:
        api_credentials = json.load(file)
    bot_token = api_credentials['bot_token']

    # Create the Application and pass it your bot's token
    application = Application.builder().token(bot_token).build()

    # Define a variable to store the current message handler
    current_handler = None

    async def handle_message(update, context):
        nonlocal current_handler

        # Cancel the previous message handler if it exists
        if current_handler:
            current_handler.cancel()

        # Start a new message handler
        current_handler = asyncio.create_task(respond(model, tokenizer, device, update, context))

    # Register the message handler
    message_handler = MessageHandler(filters.ALL, handle_message)
    application.add_handler(message_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling()

if __name__ == '__main__':
    main()
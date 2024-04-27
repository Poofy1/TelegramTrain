import os
import json
from tqdm import tqdm
from pyrogram import Client
from pyrogram.enums import ChatType
from pyrogram.raw.functions.messages import GetMessageReactionsList
env = os.path.dirname(os.path.abspath(__file__))

# Load the API ID and API Hash from the JSON file
with open(f'{env}/api.json') as file:
    api_credentials = json.load(file)
    api_id = api_credentials['api_id']
    api_hash = api_credentials['api_hash']

# Create a Pyrogram client
app = Client("my_account", api_id=api_id, api_hash=api_hash)

# Function to download a chat
def download_chat(chat_id, output_dir):
    with app:
        # Get the chat
        chat = app.get_chat(chat_id)

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get the chat history
        messages = app.get_chat_history(chat_id)
        total_messages = app.get_chat_history_count(chat_id)
        

        # Create a JSON file to store the chat data
        chat_data = []

        # Iterate through the messages in the chat
        for message in tqdm(messages, total=total_messages):
    
            message_data = {
                "message_id": message.id,
                "timestamp": message.date.isoformat(),
                "sender_id": message.from_user.id if message.from_user else None,
                "sender_username": message.from_user.username if message.from_user else None,
                "text": message.text,
            }
            
            if message.entities:
                custom_emojis = []
                for r in message.entities:
                    entity = {
                        "offset": r.offset,
                        "length": r.length,
                        "emoji_id": r.custom_emoji_id
                    }
                                    
                    custom_emojis.append(entity)
                    
                message_data["test_entities"] = custom_emojis
            
            if message.sticker:
                message_data["sticker_id"] = message.sticker.file_id
                
            elif message.media:
                media_type = str(message.media).replace("MessageMediaType.", "")
                message_data["media_type"] = media_type
                
                
            if message.forward_from:
                message_data["forwarded_from_id"] = message.forward_from.id
                message_data["forwarded_from_user"] = message.forward_from.username
            
            if message.reply_to_message_id:
                message_data["reply_to_message_id"] = message.reply_to_message_id
                
                
                
            if message.reactions:
                reactions = []
                reactionsList = app.invoke(GetMessageReactionsList(
                    id = message.id,
                    limit = 100,
                    peer = app.resolve_peer(chat.username)
                ))

                
                for r in reactionsList.reactions:
                    rebuiltReaction = { "user_id": r.peer_id.user_id }
                    if hasattr(r.reaction, "emoticon"):
                        rebuiltReaction["emoji"] = r.reaction.emoticon
                    if hasattr(r.reaction, "document_id"):
                        rebuiltReaction["emoji_id"] = r.reaction.document_id
                        
                    reactions.append(rebuiltReaction)
                message_data["reactions"] = reactions


            chat_data.append(message_data)

        # Save the chat data as a JSON file
        json_file = os.path.join(output_dir, "chat_data.json")
        with open(json_file, "w", encoding="utf-8") as file:
            json_string = json.dumps(chat_data, ensure_ascii=False, indent=4)
            file.write(json_string)
        print(f"Saved chat data: {json_file}")


# Function to get all chat IDs
def get_chat_ids():
    with app:
        # Get all dialogs (chats and channels)
        dialogs = app.get_dialogs()

        # Extract chat IDs and titles/usernames
        chat_info = []
        for dialog in dialogs:
            if dialog.chat.type == ChatType.PRIVATE:
                # For personal DMs, use the first name or username
                chat_title = dialog.chat.first_name or dialog.chat.username
            elif dialog.chat.type in [ChatType.GROUP, ChatType.SUPERGROUP]:
                # For groups and supergroups, use the title
                chat_title = dialog.chat.title
            elif dialog.chat.type == ChatType.CHANNEL:
                # For channels, use the title
                chat_title = dialog.chat.title
            else:
                # For other chat types, use a default title
                chat_title = "Unknown Chat Type"

            chat_info.append((dialog.chat.id, chat_title))

        return chat_info

# Get all chat IDs
chat_info = get_chat_ids()

# Print the list of chats
print("Available chats:")
for index, (chat_id, chat_title) in enumerate(chat_info, start=1):
    print(f"{index}. {chat_title} (ID: {chat_id})")

# Prompt the user to select chats to download
selected_chats = input("Enter the numbers of the chats you want to download (comma-separated): ")
selected_chat_indices = [int(index.strip()) - 1 for index in selected_chats.split(",")]

# Download selected chats
for index in selected_chat_indices:
    chat_id, chat_title = chat_info[index]
    output_dir = f"{env}/downloads/{chat_id}"
    print(f"Downloading chat: {chat_title}")
    download_chat(chat_id, output_dir)
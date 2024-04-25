import os
import json
from pyrogram import Client
from pyrogram.enums import ChatType  # Import ChatType enum
from pyrogram.raw.functions.messages import GetMessagesReactions

env = os.path.dirname(os.path.abspath(__file__))

# Load the API ID and API Hash from the JSON file
with open(f'{env}/api.json') as file:
    api_credentials = json.load(file)
    api_id = api_credentials['api_id']
    api_hash = api_credentials['api_hash']

# Create a Pyrogram client
client = Client("my_account", api_id=api_id, api_hash=api_hash)

# Function to download a chat
def download_chat(chat_id, output_dir):
    with client:
        # Get the chat
        chat = client.get_chat(chat_id)

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get the chat history
        messages = client.get_chat_history(chat_id)

        # Create a JSON file to store the chat data
        chat_data = []

        # Iterate through the messages in the chat
        for message in messages:
            print(message)
            message_data = {
                "message_id": message.id,
                "sender_id": message.from_user.id if message.from_user else None,
                "sender_username": message.from_user.username if message.from_user else None,
                "text": message.text,
                "timestamp": message.date.isoformat()
            }

            if message.media:
                if isinstance(message.media, types.Photo):
                    message_data["media"] = "photo"
                    message_data["photo"] = {
                        "file_id": message.media.file_id,
                        "file_unique_id": message.media.file_unique_id,
                        "width": message.media.width,
                        "height": message.media.height,
                        "file_size": message.media.file_size
                    }
                elif isinstance(message.media, types.Document):
                    message_data["media"] = "document"
                    message_data["document"] = {
                        "file_name": message.media.file_name,
                        "mime_type": message.media.mime_type,
                        "file_id": message.media.file_id,
                        "file_unique_id": message.media.file_unique_id,
                        "file_size": message.media.file_size
                    }
                else:
                    message_data["media"] = "other"

            if message.sticker:
                message_data["sticker_id"] = message.sticker.file_id

            if message.forward_from_chat:
                message_data["forwarded_from"] = message.forward_from_chat.id

            # Get the reaction emojis for the message
            reaction_emojis = []
            try:
                reactions = client.invoke(
                    GetMessagesReactions(
                        peer=message.chat.id,
                        id=[message.id]
                    )
                )
                for reaction in reactions.reactions:
                    reaction_emojis.append(reaction.reaction)
            except:
                pass

            if reaction_emojis:
                message_data["reaction_emojis"] = reaction_emojis

            chat_data.append(message_data)

        # Save the chat data as a JSON file
        json_file = os.path.join(output_dir, "chat_data.json")
        with open(json_file, "w") as file:
            json.dump(chat_data, file, indent=4)
        print(f"Saved chat data: {json_file}")

# Function to get all chat IDs
def get_chat_ids():
    with client:
        # Get all dialogs (chats and channels)
        dialogs = client.get_dialogs()

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
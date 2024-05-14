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



def get_chat_members(chat):
    # Get the list of members based on the chat type
    if chat.type in [ChatType.GROUP, ChatType.SUPERGROUP, ChatType.CHANNEL]:
        members = app.get_chat_members(chat_id)
    elif chat.type == ChatType.PRIVATE:
        members = [chat]
    else:
        members = []
            
    # Create a list to store member information
    member_data = []
    for member in members:
        if chat.type == ChatType.PRIVATE:
            # Add the other user's information
            member_info = {
                "user_id": member.id,
                "username": member.username,
                "joined_date": None
            }
            member_data.append(member_info)
            
            # Add your own information
            me = app.get_me()
            my_info = {
                "user_id": me.id,
                "username": me.username,
                "joined_date": None
            }
            member_data.append(my_info)
        else:
            if member.user.username:
                member_info = {
                    "user_id": member.user.id,
                    "username": member.user.username,
                    "joined_date": member.joined_date.isoformat() if member.joined_date else None
                }
            else:
                member_info = {
                    "user_id": member.user.id,
                    "username": member.user.first_name,
                    "joined_date": member.joined_date.isoformat() if member.joined_date else None
                }
            member_data.append(member_info)

    return {"members": member_data}



# Function to download a chat
def download_chat(chat_id, output_dir):
    with app:
        chat = app.get_chat(chat_id)
        messages = app.get_chat_history(chat_id)
        total_messages = app.get_chat_history_count(chat_id)
        chat_data = [get_chat_members(chat)]

        for message in tqdm(messages, total=total_messages):
            message_data = {
                "message_id": message.id,
                "timestamp": message.date.isoformat(),
                "sender_id": message.from_user.id if message.from_user else None,
                "sender_username": (
                    message.from_user.username
                    if message.from_user and message.from_user.username
                    else message.from_user.first_name
                    if message.from_user and message.from_user.first_name
                    else None
                ),
                "text": message.text,
            }

            if message.entities:
                custom_emojis = [
                    {"offset": r.offset, "length": r.length, "emoji_id": r.custom_emoji_id}
                    for r in message.entities
                    if r.custom_emoji_id
                ]
                if custom_emojis:
                    message_data["text_entities"] = custom_emojis

            if message.sticker:
                message_data["sticker_id"] = message.sticker.file_id
            elif message.media:
                message_data["media_type"] = str(message.media).replace("MessageMediaType.", "")

            if message.forward_from:
                message_data["forwarded_from_id"] = message.forward_from.id
                message_data["forwarded_from_user"] = message.forward_from.username

            if message.reply_to_message_id:
                message_data["reply_to_message_id"] = message.reply_to_message_id

            if message.reactions:
                reactions_list = app.invoke(
                    GetMessageReactionsList(
                        id=message.id,
                        limit=100,
                        peer=app.resolve_peer(chat.id)
                    )
                )
                reactions = [
                    {
                        "user_id": r.peer_id.user_id,
                        "emoji": r.reaction.emoticon if hasattr(r.reaction, "emoticon") else None,
                        "emoji_id": r.reaction.document_id if hasattr(r.reaction, "document_id") else None,
                    }
                    for r in reactions_list.reactions
                ]
                message_data["reactions"] = reactions

            chat_data.append(message_data)

        json_file = os.path.join(output_dir, f"{chat_id}.json")
        with open(json_file, "w", encoding="utf-8") as file:
            json.dump(chat_data, file, ensure_ascii=False, indent=4)
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
    output_dir = f"{env}/downloads/"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading chat: {chat_title}")
    download_chat(chat_id, output_dir)
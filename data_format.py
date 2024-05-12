from glob import glob
from tqdm import tqdm
import os, json
import pandas as pd
from collections import defaultdict
env = os.path.dirname(os.path.abspath(__file__))



def process_json_file(json_file):
    with open(json_file, encoding='utf-8') as file:
        data = json.load(file)
    
    messages = []
    usernames = defaultdict(int)
    emoji_ids = defaultdict(int)
    sticker_ids = defaultdict(int)
    
    # Process messages from oldest to newest
    for item in reversed(data):
        sender_id = item.get("sender_id")
        username = item.get("sender_username")
        text = item.get("text", "")
        media_type = item.get("media_type")
        
        if username:
            usernames[username] += 1
        
        if media_type and media_type.lower() not in ["sticker", "web_page"]:
            text = f"<{media_type.lower()}> {text}" if text else f"<{media_type.lower()}>"
        
        forwarded_from_user = item.get("forwarded_from_user")
        if forwarded_from_user:
            text = f"<forwarded>{forwarded_from_user}:{text}</forwarded>"
        
        message = {
            "message_id": item.get("message_id"),
            "timestamp": item.get("timestamp"),
            "sender_id": sender_id,
            "username": username,
            "text": text
        }
        
        sticker_id = item.get("sticker_id")
        if sticker_id:
            message["text"] = f"<sticker-{sticker_id}>"
            sticker_ids[sticker_id] += 1
        
        messages.append(message)
        
        reactions = item.get("reactions", [])
        for reaction in reactions:
            reaction_message = {
                "message_id": item.get("message_id"),
                "timestamp": item.get("timestamp"),
                "sender_id": reaction['user_id'],
                "username": username
            }
            
            if "emoji" in reaction:
                reaction_message["text"] = f"<reaction>{reaction['emoji']}"
            elif "emoji_id" in reaction:
                emoji_id = reaction['emoji_id']
                reaction_message["text"] = f"<reaction-{emoji_id}>"
                emoji_ids[emoji_id] += 1
            
            messages.append(reaction_message)
    
    messages.reverse()
    messages = [message for message in messages if message.get("text")]
    
    df = pd.DataFrame(messages)
    df["id"] = [i for i in range(1, len(df) + 1)]
    
    username_rows = [{"token": f"<{username}>", "identifier": username} for username in usernames]
    emoji_rows = [{"token": f"<emoji-{emoji_id}>", "identifier": emoji_id} for emoji_id in emoji_ids]
    sticker_rows = [{"token": f"<sticker-{sticker_id}>", "identifier": sticker_id} for sticker_id in sticker_ids]
    
    key_table = pd.DataFrame.from_records(username_rows + emoji_rows + sticker_rows, columns=["token", "identifier"])
    
    return df, key_table


def group_into_conversations(df, max_input_chars):
    conversations = []
    for row in df.itertuples(index=False):
        target_message = f"<{row.username}>:{row.text}\n"
        instruction = f"Respond as if you are <{row.username}>"
        input_messages = ""

        if row.id != df.iloc[0].id:
            prev_messages = (msg for msg in df[df['id'] < row.id].sort_values('id', ascending=False).itertuples(index=False))
            prev_message = ""
            for prev_row in prev_messages:
                prev_message = f"<{prev_row.username}>:{prev_row.text}\n"
                if len(input_messages) + len(prev_message) > max_input_chars:
                    break
                input_messages = prev_message + input_messages

            if not input_messages:
                available_chars = max_input_chars - len(str(row.username)) - 7
                truncated_message = f"<{row.username}>:...{row.text[-available_chars:]}\n"
                input_messages = truncated_message

        conversation = {
            'input': input_messages.strip(),
            'output': target_message.strip(),
            'instruction': instruction
        }
        conversations.append(conversation)

    return pd.DataFrame(conversations)



max_input_chars = 1024
val_split = .10


custom_tokens = [
    "<photo>",
    "<sticker>",
    "<reaction>",
    "<video>",
    "<forwarded>",
    "</forwarded>",
    "<reply>",
    "</reply>",
]

data_folder = os.path.join(env, 'downloads')
json_files = glob(os.path.join(data_folder, '*.json'))

train_dataframes = []
val_dataframes = []
key_tables = []

i = 0
for json_file in tqdm(json_files):
    df, key_table = process_json_file(json_file)  # Unpack the returned tuple
    conversation_df = group_into_conversations(df, max_input_chars)
    
    #i = i + 1
    #df.to_csv(f'{env}/testing{i}.csv', index=False, encoding='utf-8')
    #conversation_df.to_csv(f'{env}/convo{i}.csv', index=False, encoding='utf-8')
    
    # Split the conversation_df into training and validation sets
    num_rows = len(conversation_df)
    val_size = int(num_rows * val_split)
    train_size = num_rows - val_size
    train_df = conversation_df.iloc[:train_size]
    val_df = conversation_df.iloc[train_size:]
    train_dataframes.append(train_df)
    val_dataframes.append(val_df)
    key_tables.append(key_table)

# Combine the training and validation dataframes
train_df = pd.concat(train_dataframes, ignore_index=True)
val_df = pd.concat(val_dataframes, ignore_index=True)
train_df.to_csv(f'{env}/data/train.csv', index=False, encoding='utf-8')
val_df.to_csv(f'{env}/data/val.csv', index=False, encoding='utf-8')

# Combine the key tables from all JSON files
combined_key_table = pd.concat(key_tables, ignore_index=True)

# Append custom tokens to the combined key table
custom_token_rows = [{"token": token, "identifier": ""} for token in custom_tokens]
combined_key_table = pd.concat([combined_key_table, pd.DataFrame(custom_token_rows)], ignore_index=True)
combined_key_table = combined_key_table.drop_duplicates(subset='token', keep='first')
combined_key_table.to_csv(f'{env}/data/key_table.csv', index=False, encoding='utf-8')
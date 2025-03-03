from glob import glob
from tqdm import tqdm
from datetime import datetime
import os, json
import pandas as pd
from collections import defaultdict


def combine_image_messages(messages):
    # Group messages by user and timestamp
    grouped_messages = defaultdict(list)
    for message in messages:
        key = (message["username"], message["timestamp"])
        grouped_messages[key].append(message)
    
    # Combine images into <photos> token
    combined_messages = []
    for key, message_group in grouped_messages.items():
        if len(message_group) > 1:
            image_messages = [msg for msg in message_group if "<photo>" in msg["text"]]
            if len(image_messages) > 1:
                combined_text = image_messages[0]["text"].replace("<photo>", "<photos>")
                combined_message = {
                    "timestamp": image_messages[0]["timestamp"],
                    "username": image_messages[0]["username"],
                    "text": combined_text
                }
                non_image_messages = [msg for msg in message_group if "<photo>" not in msg["text"]]
                combined_messages.append(combined_message)
                combined_messages.extend(non_image_messages)
            else:
                combined_messages.extend(message_group)
        else:
            combined_messages.extend(message_group)
    
    return combined_messages


def process_json_file(json_file):
    with open(json_file, encoding='utf-8') as file:
        data = json.load(file)
    
    messages = []
    usernames = defaultdict(int)
    emoji_ids = defaultdict(int)
    user_id_to_username = {}
    
    # Process messages from oldest to newest
    for idx, item in enumerate(reversed(data)):
        sender_id = item.get("sender_id")
        username = item.get("sender_username")
        
        if sender_id is None:
            continue
        
        if username:
            usernames[username] += 1
            user_id_to_username[sender_id] = username
        
        text = item.get("text", "")
        media_type = item.get("media_type")
        
        if media_type and media_type.lower() not in ["sticker", "web_page"]:
            text = f"<{media_type.lower()}> {text}" if text else f"<{media_type.lower()}>"
        
        forwarded_from_user = item.get("forwarded_from_user")
        if forwarded_from_user:
            text = f"<forwarded><{forwarded_from_user}>:{text}</forwarded>"
        
        message = {
            "timestamp": item.get("timestamp"),
            "username": f"<{username}>",
            "text": f"<{username}>:{text}"
        }
        
        if item.get("sticker_id"):
            message["text"] = f"<{username}>:<sticker>"
        
        messages.append(message)
        
        reactions = item.get("reactions", [])
        for reaction in reactions:
            reaction_user_id = reaction['user_id']
            reaction_username = user_id_to_username.get(reaction_user_id)
            
            if reaction_username is None:
                continue  # Skip reactions from unknown users
            
            reaction_message = {
                "timestamp": item.get("timestamp"),
                "username": f"<{reaction_username}>"
            }
            
            if "emoji" in reaction:
                reaction_message["text"] = f"<{reaction_username}>:<reaction>{reaction['emoji']}"
            elif "emoji_id" in reaction:
                emoji_id = reaction['emoji_id']
                reaction_message["text"] = f"<{reaction_username}>:<reaction-{emoji_id}>"
                emoji_ids[emoji_id] += 1
            
            messages.append(reaction_message)
    
    messages = [message for message in messages if message.get("text")]
    
    # Combine image messages
    combined_messages = combine_image_messages(messages)
    
    df = pd.DataFrame(combined_messages)
    
    username_rows = [{"token": f"<{username}>", "identifier": username, "sender_id": sender_id} for sender_id, username in user_id_to_username.items()]
    emoji_rows = [{"token": f"<emoji-{emoji_id}>", "identifier": emoji_id} for emoji_id in emoji_ids]
    
    key_table = pd.DataFrame.from_records(username_rows + emoji_rows, columns=["token", "identifier"])
    
    return df, key_table, username_rows


def process_csv_file(csv_file, username_map):
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    messages = []
    usernames = defaultdict(int)
    user_id_to_username = {}
    
    # Convert CSV data to match JSON format
    for _, row in df.iterrows():
        username = row['Author']
        username = username_map.get(username, username)
        sender_id = row['AuthorID']
        
        if sender_id is None or pd.isna(sender_id):
            continue
            
        if username:
            usernames[username] += 1
            user_id_to_username[sender_id] = username
            
        # Convert timestamp format
        timestamp = pd.to_datetime(row['Date']).timestamp()
        
        # Process content and attachments
        text = row['Content'] if pd.notna(row['Content']) else ""
        
        # Handle sticker messages (text containing :word:)
        if text and text.startswith(':') and text.endswith(':'):
            message = {
                "timestamp": timestamp,
                "username": f"<{username}>",
                "text": f"<{username}>:<sticker>"
            }
            messages.append(message)
            continue
        
        # Handle attachments
        if pd.notna(row['Attachments']) and row['Attachments']:
            if text:
                text = f"{text} <photo>"
            else:
                text = "<photo>"
                
        # Handle call messages
        if "Started a call" in text:
            text = "<video>"
            
        message = {
            "timestamp": timestamp,
            "username": f"<{username}>",
            "text": f"<{username}>:{text}"
        }
        messages.append(message)
        
        # Process reactions if they exist
        if pd.notna(row['Reactions']) and row['Reactions']:
            reactions = row['Reactions'].split(';')  # Assuming reactions are semicolon-separated
            for reaction in reactions:
                reaction_message = {
                    "timestamp": timestamp,
                    "username": f"<{username}>",
                    "text": f"<{username}>:<reaction>{reaction}"
                }
                messages.append(reaction_message)
    
    # Convert messages to DataFrame
    df = pd.DataFrame(messages)
    
    # Create username lookup information
    username_rows = [
        {"token": f"<{username}>", "identifier": username, "sender_id": sender_id} 
        for sender_id, username in user_id_to_username.items()
    ]
    
    key_table = pd.DataFrame.from_records(
        username_rows, 
        columns=["token", "identifier"]
    )
    
    return df, key_table, username_rows
    






def main(env):
    max_input_chars = 1024
    val_split = .10


    custom_tokens = [
        "<photo>",
        "<photos>",
        "<hour>",
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
    csv_files = glob(os.path.join(data_folder, '*.csv'))

    train_dataframes = []
    val_dataframes = []
    key_tables = []
    chat_id = 0
    message_id = 0
    username_lookup = []

    # Process JSON files
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        df, key_table, username_rows = process_json_file(json_file)
        
        # Add the chat_id column to the dataframe
        df['chat_id'] = chat_id
        chat_id += 1
        
        # Add the message_id column with unique numbers, oldest messages first
        df = df.sort_values('timestamp', ascending=True)
        df['message_id'] = range(message_id, message_id + len(df))
        message_id += len(df)
        
        # Split into train and validation
        df = df.sort_values('timestamp', ascending=False)
        val_size = int(len(df) * 0.05)
        val_df = df.iloc[:val_size]
        train_df = df.iloc[val_size:]
        
        train_dataframes.append(train_df)
        val_dataframes.append(val_df)
        key_tables.append(key_table)
        
        for username_row in username_rows:
            username_lookup.append({
                "chat_id": chat_id - 1,
                "token": username_row["token"],
                "sender_id": username_row["sender_id"]
            })


    username_map = {}
    try:
        mapping_file = os.path.join(env, 'username_map.json')
        with open(mapping_file, 'r', encoding='utf-8') as f:
            username_map = json.load(f)
    except FileNotFoundError:
        username_map = {}
            
    # Process CSV files
    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        df, key_table, username_rows = process_csv_file(csv_file, username_map)
        
        # Add the chat_id column to the dataframe
        df['chat_id'] = chat_id
        chat_id += 1
        
        # Add the message_id column with unique numbers, oldest messages first
        df = df.sort_values('timestamp', ascending=True)
        df['message_id'] = range(message_id, message_id + len(df))
        message_id += len(df)
        
        # Split into train and validation
        df = df.sort_values('timestamp', ascending=False)
        val_size = int(len(df) * 0.05)
        val_df = df.iloc[:val_size]
        train_df = df.iloc[val_size:]
        
        train_dataframes.append(train_df)
        val_dataframes.append(val_df)
        key_tables.append(key_table)
        
        for username_row in username_rows:
            username_lookup.append({
                "chat_id": chat_id - 1,
                "token": username_row["token"],
                "sender_id": username_row["sender_id"]
            })

    # Combine all the dataframes and save
    train_df = pd.concat(train_dataframes, ignore_index=True)
    val_df = pd.concat(val_dataframes, ignore_index=True)

    os.makedirs(f'{env}/data/', exist_ok=True)

    # Save the training and validation dataframes
    train_df.to_csv(f'{env}/data/train.csv', index=False, encoding='utf-8')
    val_df.to_csv(f'{env}/data/val.csv', index=False, encoding='utf-8')

    # Create and save username lookup
    username_lookup_df = pd.DataFrame(username_lookup)
    username_lookup_df.to_csv(f'{env}/data/username_lookup.csv', index=False, encoding='utf-8')

    # Create and save combined key table with custom tokens
    combined_key_table = pd.concat(key_tables, ignore_index=True)
    custom_token_rows = [{"token": token, "identifier": ""} for token in custom_tokens]
    combined_key_table = pd.concat([combined_key_table, pd.DataFrame(custom_token_rows)], ignore_index=True)
    combined_key_table = combined_key_table.drop_duplicates(subset='token', keep='first')
    combined_key_table.to_csv(f'{env}/data/key_table.csv', index=False, encoding='utf-8')

    # Combine the training and validation dataframes
    train_df = pd.concat(train_dataframes, ignore_index=True)
    val_df = pd.concat(val_dataframes, ignore_index=True)

    # Save the training and validation dataframes to separate CSV files
    train_df.to_csv(f'{env}/data/train.csv', index=False, encoding='utf-8')
    val_df.to_csv(f'{env}/data/val.csv', index=False, encoding='utf-8')

    # Create the username lookup dataframe
    username_lookup_df = pd.DataFrame(username_lookup)
    username_lookup_df.to_csv(f'{env}/data/username_lookup.csv', index=False, encoding='utf-8')

    # Append custom tokens to the combined key table
    combined_key_table = pd.concat(key_tables, ignore_index=True)
    custom_token_rows = [{"token": token, "identifier": ""} for token in custom_tokens]
    combined_key_table = pd.concat([combined_key_table, pd.DataFrame(custom_token_rows)], ignore_index=True)
    combined_key_table = combined_key_table.drop_duplicates(subset='token', keep='first')
    combined_key_table.to_csv(f'{env}/data/key_table.csv', index=False, encoding='utf-8')
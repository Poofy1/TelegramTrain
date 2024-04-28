from glob import glob
from tqdm import tqdm
import os, json
import pandas as pd
env = os.path.dirname(os.path.abspath(__file__))


def process_json_file(json_file):
    with open(json_file, encoding='utf-8') as file:
        data = json.load(file)
    
    messages = []
    
    for item in data[1:]:
        sender_id = item.get("sender_id")
        text = item.get("text", "")
        media_type = item.get("media_type")
        
        if media_type and media_type not in ["STICKER", "WEB_PAGE"]:
            text = f"({media_type}) {text}" if text else f"({media_type})"
        
        forwarded_from_id = item.get("forwarded_from_id")
        if forwarded_from_id:
            text = f"(FORWARDED FROM {forwarded_from_id}) {text}"
        
        message = {
            "message_id": item.get("message_id"),
            "timestamp": item.get("timestamp"),
            "sender_id": sender_id,
            "text": text,
            "reply_to_message_id": str(item.get("reply_to_message_id")) if item.get("reply_to_message_id") is not None else None
        }
        
        sticker_id = item.get("sticker_id")
        if sticker_id:
            message["text"] = f"(STICKER){sticker_id}"
        
        messages.append(message)
        
        reactions = item.get("reactions", [])
        for reaction in reactions:
            reaction_message = {
                "message_id": item.get("message_id"),
                "timestamp": item.get("timestamp"),
                "sender_id": reaction['user_id'],
                "reply_to_message_id": str(item.get("reply_to_message_id")) if item.get("reply_to_message_id") is not None else None
            }
            
            if "emoji" in reaction:
                reaction_message["text"] = f"(REACTION){reaction['emoji']}"
            elif "emoji_id" in reaction:
                reaction_message["text"] = f"(REACTION)(EMOJI_ID){reaction['emoji_id']}"
            
            messages.append(reaction_message)
    
    # Sort messages by timestamp to ensure reactions come after the message they are assigned to
    messages.sort(key=lambda x: x["timestamp"])
    
    # Remove rows with empty text fields
    messages = [message for message in messages if message.get("text")] # Removes calls and 'added to group' items
    
    # Add 'id' column, counting up each row from oldest to newest
    for i, message in enumerate(messages, start=1):
        message["id"] = i
    
    df = pd.DataFrame(messages)
    return df


def group_into_conversations(df, max_input_chars):
    conversations = []
    
    for _, row in df.iterrows():
        target_message = f"{row['sender_id']}:{row['text']}\n"
        
        # Create the instruction for the target message
        instruction = f"Respond as if you are {row['sender_id']}"
        
        # Initialize the input messages as an empty string
        input_messages = ""
        
        # Check if it's the first message in the DataFrame
        if row['id'] == df['id'].min():
            # If it's the first message, there should be no input messages
            input_messages = ""
        else:
            # Iterate over previous messages in reverse order
            prev_messages = df[df['id'] < row['id']].sort_values('id', ascending=False)
            for _, prev_row in prev_messages.iterrows():
                prev_message = f"{prev_row['sender_id']}:{prev_row['text']}\n"
                
                # Check if adding the previous message exceeds the max_input_chars limit
                if len(input_messages) + len(prev_message) <= max_input_chars:
                    input_messages = prev_message + input_messages
                else:
                    break
            
            # Check if the input messages is empty (i.e., no previous messages fit within the limit)
            if not input_messages:
                # Calculate the available characters for the truncated message
                available_chars = max_input_chars - len(str(row['sender_id'])) - 5
                
                # Truncate the beginning of the target message to fit within the available characters
                truncated_message = f"{row['sender_id']}:...{row['text'][-available_chars:]}\n"
                input_messages = truncated_message
        
        # Create a new conversation entry
        conversation = {
            'input': input_messages.strip(),
            'output': target_message.strip(),
            'instruction': instruction
        }
        conversations.append(conversation)
    
    return pd.DataFrame(conversations)



max_input_chars = 1024
val_split = .10

data_folder = os.path.join(env, 'downloads')
json_files = glob(os.path.join(data_folder, '*.json'))

train_dataframes = []
val_dataframes = []

for json_file in tqdm(json_files):
    
    df = process_json_file(json_file)

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

# Combine the training and validation dataframes
train_df = pd.concat(train_dataframes, ignore_index=True)
val_df = pd.concat(val_dataframes, ignore_index=True)

train_df.to_csv(f'{env}/data/train.csv', index=False, encoding='utf-8')
val_df.to_csv(f'{env}/data/val.csv', index=False, encoding='utf-8')
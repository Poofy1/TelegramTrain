from glob import glob
from tqdm import tqdm
import os, json
import pandas as pd
env = os.path.dirname(os.path.abspath(__file__))

def process_json_file(json_file):
    with open(json_file, encoding='utf-8') as file:
        data = json.load(file)
    
    members = data[0].get("members", [])
    member_list = ",".join([str(member["user_id"]) for member in members])
    
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
            "instruction": f"Respond as if you are {sender_id}, (Participants: {member_list})",
            "message_id": item.get("message_id"),
            "timestamp": item.get("timestamp"),
            "sender_id": sender_id,
            "text": text,
            "reply_to_message_id": str(item.get("reply_to_message_id")) if item.get("reply_to_message_id") is not None else None
        }
        
        sticker_id = item.get("sticker_id")
        if sticker_id:
            message["text"] = f"(STICKER){sticker_id}"
        
        reactions = item.get("reactions", [])
        reaction_list = []
        for reaction in reactions:
            if "emoji" in reaction:
                reaction_list.append(f"{reaction['emoji']}:{reaction['user_id']}")
            elif "emoji_id" in reaction:
                reaction_list.append(f"(EMOJI_ID){reaction['emoji_id']}:{reaction['user_id']}")
        reaction_str = ",".join(reaction_list)
        message["reactions"] = reaction_str
        
        messages.append(message)
    
    df = pd.DataFrame(messages)
    return df


def group_messages_into_conversations(df, forwarded_from_translations=None):
    hour_difference = 4
    conversations = []
    current_conversation = []
    prev_timestamp = None
    prev_sender = None

    if forwarded_from_translations is None:
        forwarded_from_translations = {}

    for _, row in df.iterrows():
        timestamp = pd.to_datetime(row['date'])
        sender = row['sender']
        message = row['message']
        forwarded_from = row['forwarded_from']

        if prev_timestamp is None or (timestamp - prev_timestamp).total_seconds() >= hour_difference * 3600:
            if len(current_conversation) > 1 and len(set(msg.split(': ')[0] for msg in current_conversation)) > 1:
                conversations.append(current_conversation)
            current_conversation = []
            prev_sender = None

        if forwarded_from:
            forwarded_from = forwarded_from_translations.get(forwarded_from, forwarded_from)
            message = f"(FORWARDED - {forwarded_from}) {message}"

        if prev_sender == sender:
            current_conversation[-1] += f"\n{message}"
        else:
            current_conversation.append(f"{sender}: {message}")
            prev_sender = sender

        prev_timestamp = timestamp

    if len(current_conversation) > 1 and len(set(msg.split(': ')[0] for msg in current_conversation)) > 1:
        conversations.append(current_conversation)

    return pd.DataFrame({'conversation': conversations})







def create_training_data(conversation_df, max_input_chars):
    training_data = []

    for _, row in tqdm(conversation_df.iterrows(), total=conversation_df.shape[0]):
        conversation = row['conversation']
        context_window = []
        context_chars = 0

        for i in range(len(conversation)):
            message = conversation[i]
            message_chars = len(message)

            # Add message to the context window
            context_window.append(message)
            context_chars += message_chars

            # Remove starting messages if the context window exceeds the character limit,
            # but keep the message if it's the only one in the context window
            while context_chars > max_input_chars and len(context_window) > 1:
                removed_message = context_window.pop(0)
                context_chars -= len(removed_message)

            # Create training example
            if i < len(conversation) - 1:
                name = conversation[i+1].split(': ')[0]
                instruction = f"Respond as if you are {name}"
                input_text = '\n'.join(context_window)
                output_text = conversation[i+1]

                training_data.append({
                    'instruction': instruction,
                    'input': input_text,
                    'output': output_text
                })

    return pd.DataFrame(training_data)


    





max_input_chars = 1024
val_split = .05

data_folder = os.path.join(env, 'downloads')
subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]


conversation_dataframes = []
i = 0
for subfolder in tqdm(subfolders):
    json_file = os.path.join(subfolder, '*.json')
    json_files = glob(json_file)
    
    json_file = json_files[0]  # Assuming there's only one JSON file per subfolder
    df = process_json_file(json_file)
    i = i + 1
    df.to_csv(f'{env}/testing{i}.csv', index=False, encoding='utf-8')
    
    if df is not None:
        conversation_df = group_messages_into_conversations(df)
        conversation_dataframes.append(conversation_df)


# Merge all conversation DataFrames into a single DataFrame
merged_conversation_df = pd.concat(conversation_dataframes, ignore_index=True)


# Split the merged conversations into train and validation sets
val_conversation_df = merged_conversation_df.sample(frac=val_split, random_state=42)
train_conversation_df = merged_conversation_df.drop(val_conversation_df.index)

# Create training data for the train set
train_data_df = create_training_data(train_conversation_df, max_input_chars)
val_data_df = create_training_data(val_conversation_df, max_input_chars)


"""# Save the validation conversation DataFrame to a CSV file
output_file = f'{env}/conversations_val.csv'
val_conversation_df.to_csv(output_file, index=False, encoding='utf-8')
print(f"Validation conversations saved to {output_file}.")"""


# Save the train data to a CSV file
output_file = f'{env}/data/train.csv'
train_data_df.to_csv(output_file, index=False, encoding='utf-8')
print(f"Train data saved to {output_file}.")

# Save the validation data to a CSV file
output_file = f'{env}/data/val.csv'
val_data_df.to_csv(output_file, index=False, encoding='utf-8')
print(f"Validation data saved to {output_file}.")
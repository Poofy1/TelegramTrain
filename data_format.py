from glob import glob
from tqdm import tqdm
import os, json, csv, random
import pandas as pd

env = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(env, 'raw_data')
json_files = glob(os.path.join(data_folder, '*.json'))


def process_text_field(message):
    text_content = ''
    
    # Include placeholder text for different media types
    if 'media_type' in message:
        media_type = message['media_type'].upper()
        text_content += f"({media_type}) "
        if 'sticker_emoji' in message:
            text_content += f"{message['sticker_emoji']} "
    elif 'photo' in message:
        text_content += '(IMAGE) '
    elif 'file' in message:
        text_content += '(FILE) '

    # Append actual text if it's available and not empty
    if message.get('text'):
        if isinstance(message['text'], list):
            # Concatenate parts of text if it's a list
            text_parts = []
            for part in message['text']:
                if isinstance(part, dict) and part.get('type') == 'link':
                    text_parts.append(part['text'])  # Append the link text
                elif isinstance(part, str):
                    text_parts.append(part)
            text_content += ' '.join(text_parts).replace('\n', ' ')
        elif isinstance(message['text'], str):
            text_content += message['text'].replace('\n', ' ')
    
    return text_content.strip()



def process_json_file(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    
    # Find the two unique names in the JSON file
    names = set()
    for message in json_data['messages']:
        if 'from' in message:
            names.add(message['from'])
        if len(names) == 2:
            break
    
    if len(names) != 2:
        print(f"Warning: Could not find exactly two unique names in {json_file}")
        return None
    
    sender, receiver = names
    
    data = []
    for message in json_data['messages']:
        if 'from' not in message or 'text' not in message:
            continue  # Skip if 'from' or 'text' field is missing
        
        text_content = process_text_field(message)
        
        # Skip messages with empty content
        if not text_content.strip():
            continue
        
        # Extract the date and message id from the message
        date = message.get('date', '')
        message_id = message.get('id', '')
        
        # Determine the sender and receiver for the current message
        message_sender = message['from']
        message_receiver = receiver if message_sender == sender else sender
        
        # Check if the message has a 'forwarded_from' entry
        forwarded_from = message.get('forwarded_from', '')
        
        # Skip messages where the sender is the same as the 'forwarded_from' value
        if message_sender == forwarded_from:
            continue
        
        # Append the message details to the data list
        data.append({
            'id': message_id,
            'date': date,
            'sender': message_sender,
            'receiver': message_receiver,
            'forwarded_from': forwarded_from,
            'message': text_content
        })
    
    return pd.DataFrame(data)


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








forwarded_from_translations = {'mltn': 'Peter'}
max_input_chars = 1024
val_split = .05


# Process JSON files, group messages into conversations, and create a single DataFrame
conversation_dataframes = []
for json_file in tqdm(json_files):
    df = process_json_file(json_file)
    if df is not None:
        conversation_df = group_messages_into_conversations(df, forwarded_from_translations)
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
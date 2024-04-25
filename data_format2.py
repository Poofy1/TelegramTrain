from glob import glob
from tqdm import tqdm
import os, json, csv, random
import pandas as pd
import hashlib
import easyocr
from PIL import Image
env = os.path.dirname(os.path.abspath(__file__))


def process_text_field(message):
    text_content = ''
    
    # Include placeholder text for different media types
    if 'media_type' in message:
        media_type = message['media_type'].upper()
        text_content += f"({media_type}) "
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
        
        
        text_content = ''
        image = ''
        sticker = ''
    
        # Include placeholder text for different media types
        if 'media_type' in message:
            media_type = message['media_type']
            if media_type == "sticker" or media_type == "animation":
                sticker = message['file']
            else:
                text_content += f"({media_type.upper()}) "
        elif 'photo' in message:
            image += message['photo']
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
        
        text_content = text_content.strip()
        
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
            'message': text_content,
            'image': image,
            'sticker': sticker,
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





def get_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def compile_files(subfolders):
    file_info = []
    for subfolder in tqdm(subfolders, desc="Processing files"):
        subfolder_name = os.path.basename(subfolder)  # Extract the subfolder name
        media_folders = ['photos', 'stickers', 'video_files']
        for media_folder in media_folders:
            media_path = os.path.join(subfolder, media_folder)
            if os.path.exists(media_path):
                files = glob(os.path.join(media_path, '*'))
                for file in files:
                    ocr_tag = 0
                    if media_folder in ['photos', 'stickers']:
                        ocr_tag = 1
                    file_hash = get_file_hash(file)
                    local_path = os.path.join(subfolder_name, media_folder, os.path.basename(file))
                    file_info.append({'original_path': local_path, 'hash': file_hash, 'ocr_tag': ocr_tag})
    
    # Create the master DataFrame
    hash_df = pd.DataFrame(file_info)
    
    # Identify duplicates based on file hash
    hash_df['duplicate'] = hash_df.duplicated(subset='hash', keep='first')
    
    # Remove duplicate rows
    hash_df = hash_df[~hash_df['duplicate']]
    
    # Create unique output file names for each hash
    hash_df['output_name'] = hash_df.groupby('hash').ngroup().astype(str) + '_' + hash_df['hash'].str[:8]
    
    # Drop the 'duplicate' column
    hash_df = hash_df.drop(columns=['duplicate'])
    
    # Perform OCR on unique images
    reader = easyocr.Reader(['en'])  # Initialize the easyocr reader for English language
    ocr_results = []
    
    for _, row in tqdm(hash_df.iterrows(), total=len(hash_df), desc="Performing OCR"):
        if row['ocr_tag'] == 1:
            file_path = os.path.join(env, 'raw_data', row['original_path'])
            try:
                result = reader.readtext(file_path)
                ocr_text = ' '.join([text for _, text, _ in result])
                print(ocr_text)
            except Exception as e:
                print(f"Error processing OCR for {file_path}: {str(e)}")
                ocr_text = ""
        else:
            ocr_text = ""
        
        ocr_results.append(ocr_text)
    
    hash_df['ocr'] = ocr_results
    
    return hash_df

    







forwarded_from_translations = {'mltn': 'Peter'}
max_input_chars = 1024
val_split = .05

data_folder = os.path.join(env, 'raw_data')
subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]


# Hash and organize unique files
hash_df = compile_files(subfolders)
hash_df.to_csv(f'{env}/hashes.csv', index=False, encoding='utf-8')


conversation_dataframes = []
for subfolder in tqdm(subfolders):
    json_file = os.path.join(subfolder, '*.json')
    json_files = glob(json_file)
    
    json_file = json_files[0]  # Assuming there's only one JSON file per subfolder
    df = process_json_file(json_file)
    df.to_csv(f'{env}/testing.csv', index=False, encoding='utf-8')
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
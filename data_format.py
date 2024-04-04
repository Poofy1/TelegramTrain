from glob import glob
from tqdm import tqdm
import os, json, csv, random

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


all_messages = []
current_sender = None
current_messages = []

for json_file in tqdm(json_files):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for message in data['messages']:
        if 'from' not in message or 'text' not in message:
            continue  # Skip if 'from' or 'text' field is missing
        
        from_person = message.get('from')
        text_content = process_text_field(message)
        
        # If the sender has changed, combine the previous sender's messages and reset
        if from_person != current_sender and current_messages:
            combined_message = '\n'.join(current_messages)
            all_messages.append({
                'from': current_sender,
                'message': combined_message
            })
            current_messages = []

        # Update the current sender and append the current message
        current_sender = from_person
        current_message_with_name = f"{from_person}: {text_content}"
        current_messages.append(current_message_with_name)

    # After processing all messages, add the last sender's messages if any
    if current_messages:
        combined_message = '\n'.join(current_messages)
        all_messages.append({
            'from': current_sender,
            'message': combined_message
        })
        current_messages = []  # Reset for the next file





formatted_conversations = []
respondent = "Peter"
instruction = f"Respond as if you are {respondent}"
max_chars = 2000  # Maximum number of characters allowed for input_text

# Iterate over all messages and construct the formatted conversations
for i in range(len(all_messages)):
    if all_messages[i]['from'] == respondent:
        # Get the last 5 messages or less if not enough messages are available
        start_index = max(i - 5, 0)
        input_messages = all_messages[start_index:i]  # Slice the last 5 messages
        input_text = '\n'.join([msg['message'] for msg in input_messages])
        
        # Truncate input_text if it's longer than max_chars
        if len(input_text) > max_chars:
            input_text = input_text[-max_chars:]

        output_text = all_messages[i]['message']
        
        formatted_conversations.append({
            'instruction': instruction,
            'input': input_text,
            'output': output_text,
        })

# Define the path for the new CSV file
formatted_csv_path = os.path.join(env, 'data/train.csv')

# Shuffle the formatted conversations to ensure random distribution
random.shuffle(formatted_conversations)

# Calculate the number of conversations for validation set (5% of the total)
val_size = int(len(formatted_conversations) * 0.05)

# Split the data into training and validation sets
val_conversations = formatted_conversations[:val_size]
train_conversations = formatted_conversations[val_size:]

# Define the paths for the new CSV files
formatted_csv_val_path = os.path.join(env, 'data/val.csv')
formatted_csv_train_path = os.path.join(env, 'data/train.csv')

# Function to write conversations to a CSV file
def write_to_csv(file_path, conversations):
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['instruction', 'input', 'output'])
        for conv in conversations:
            writer.writerow([conv['instruction'], conv['input'], conv['output']])

# Write the training and validation datasets to separate CSV files
write_to_csv(formatted_csv_train_path, train_conversations)
write_to_csv(formatted_csv_val_path, val_conversations)

print('Finished Creating Train and Val CSV files')
import argparse
import os
import sys
import json

env = os.path.dirname(os.path.abspath(__file__))

def check_api_file():
    api_file_path = os.path.join(env, "api.json")
    
    if not os.path.exists(api_file_path):
        print("Error: api.json file not found!")
        print("Please create an api.json file with the following structure:")
        print('''{
    "api_id": "",
    "api_hash": "",
    "bot_token": ""
}''')
        print("Fill in your Telegram API credentials and run the script again.")
        sys.exit(1)
    
    # Check if the file is properly filled
    with open(api_file_path, 'r') as file:
        api_data = json.load(file)
        if not api_data.get("api_id") or not api_data.get("api_hash"):
            print("Error: api.json is missing required credentials!")
            print("Please fill in your Telegram API credentials and run the script again.")
            sys.exit(1)

def main():
    # Check for API file first before doing anything else
    check_api_file()
    
    parser = argparse.ArgumentParser(description="TelegramTrain runner script")
    parser.add_argument("--bot", action="store_true", help="Run the bot")
    parser.add_argument("--train", action="store_true", help="Continue or Start Training")
    
    args = parser.parse_args()
    
    import src.bot as bot
    import src.download as download
    import src.data_format as data_format
    import src.train as train
    
    if args.bot:
        print("Running bot...")
        bot.main(env)
    elif args.train:
        print("Training model...")
        train.main(env)
    else:
        print("Running download -> data_format -> train pipeline...")
        
        print("Step 1: Downloading data...")
        download.main(env)
        
        print("Step 2: Formatting data...")
        data_format.main(env)
        
        print("Step 3: Training model...")
        train.main(env)
        
if __name__ == "__main__":
    main()
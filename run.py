import argparse
import os
import sys
import json

env = os.path.dirname(os.path.abspath(__file__))

def check_api_file():
    api_file_path = os.path.join(env, "api.json")
    
    if not os.path.exists(api_file_path):
        print("Error: api.json file not found!")
        print("Let's create it now:")
        
        api_id = input("Enter your Telegram API ID: ")
        api_hash = input("Enter your Telegram API hash: ")
        bot_token = input("Enter your bot token (leave empty if not using a bot): ")
        
        api_data = {
            "api_id": api_id,
            "api_hash": api_hash,
            "bot_token": bot_token
        }
        
        with open(api_file_path, 'w') as file:
            json.dump(api_data, file, indent=4)
        
        print("api.json created successfully!")
        return
    
    # Check if the file is properly filled
    with open(api_file_path, 'r') as file:
        api_data = json.load(file)
        if not api_data.get("api_id") or not api_data.get("api_hash"):
            print("Error: api.json is missing required credentials!")
            
            api_id = input("Enter your Telegram API ID: ") if not api_data.get("api_id") else api_data.get("api_id")
            api_hash = input("Enter your Telegram API hash: ") if not api_data.get("api_hash") else api_data.get("api_hash")
            bot_token = input("Enter your bot token (leave empty if not using a bot): ") if "bot_token" not in api_data else api_data.get("bot_token")
            
            api_data = {
                "api_id": api_id,
                "api_hash": api_hash,
                "bot_token": bot_token
            }
            
            with open(api_file_path, 'w') as file:
                json.dump(api_data, file, indent=4)
            
            print("api.json updated successfully!")

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
        print("Starting bot...")
        bot.main(env)
    elif args.train:
        print("Train model")
        train.main(env)
    else:
        print("Step 1: Download data")
        download.main(env)
        
        print("Step 2: Format data")
        data_format.main(env)
        
        print("Step 3: Train model")
        train.main(env)
        
if __name__ == "__main__":
    main()
# TelegramTrain

A tool for downloading Telegram chat data, formatting it, and training machine learning models on this data.

## Overview

TelegramTrain simplifies the process of gathering Telegram conversation data and using it for model training. The project streamlines the entire workflow from data collection to model training through a single easy-to-use interface.

## Prerequisites

- Python 3.6+
- Telegram API credentials (API ID and API hash)
- Bot token (optional, only if using bot functionality)

## Setup

1. Clone this repository: `git clone https://github.com/Poofy1/TelegramTrain.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python run.py`
4. On first run, you'll be prompted to enter your Telegram API credentials.

## Telegram API Credentials

You'll need to obtain API credentials from Telegram:
1. Visit https://my.telegram.org/auth
2. Log in and go to "API development tools"
3. Create a new application
4. Note your API ID and API hash

These credentials will be stored in `api.json` in the project directory. If this file doesn't exist, the script will prompt you to create it.

## Usage

### Full Pipeline

To run the entire pipeline (download data → format data → train model): `python run.py`

### Individual Steps

- Download data only: `python run.py --download`
- Train an existing model or start training a new one: `python run.py --train`
- Run the bot (if you've configured a bot token): `python run.py --bot`

## Project Structure

- `run.py` - Main entry point for all operations
- `src/` - Source code directory
  - `download.py` - Scripts for downloading Telegram data
  - `data_format.py` - Data formatting utilities
  - `train.py` - Model training code
  - `bot.py` - Telegram bot implementation
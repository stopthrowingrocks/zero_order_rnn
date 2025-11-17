#!/usr/bin/env python3
"""
Quick script to get your Telegram chat ID.
Run this after sending a message to your bot.
"""
import requests

BOT_TOKEN = "8465818554:AAGPMupE9vaiTzPCpyqQqnoPJ4UGe3_j5jA"

url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
response = requests.get(url)
data = response.json()

print("Response from Telegram API:")
print("=" * 80)

if data.get('ok') and data.get('result'):
    for update in data['result']:
        if 'message' in update:
            chat_id = update['message']['chat']['id']
            username = update['message']['chat'].get('username', 'N/A')
            first_name = update['message']['chat'].get('first_name', 'N/A')

            print(f"\nFound chat:")
            print(f"  Chat ID: {chat_id}")
            print(f"  Username: @{username}")
            print(f"  Name: {first_name}")
            print("\n" + "=" * 80)
            print(f"Your TELEGRAM_CHAT_ID is: {chat_id}")
            print("=" * 80)
else:
    print("No messages found!")
    print("\nPlease:")
    print("1. Open Telegram")
    print("2. Search for @StrClaudeBot")
    print("3. Send any message (e.g., 'hello')")
    print("4. Run this script again")

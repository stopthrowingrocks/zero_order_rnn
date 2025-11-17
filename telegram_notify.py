#!/usr/bin/env python3
"""
Simple Telegram notification module for experiment alerts.
"""
import requests

# Telegram bot credentials
BOT_TOKEN = "8465818554:AAGPMupE9vaiTzPCpyqQqnoPJ4UGe3_j5jA"
CHAT_ID = "8342601795"


def send_message(message):
    """Send a text message via Telegram."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    try:
        response = requests.post(url, json={
            'chat_id': CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'  # Enables markdown formatting
        }, timeout=10)

        if response.status_code == 200:
            print("[OK] Telegram notification sent")
            return True
        else:
            print(f"[FAIL] Telegram notification failed: {response.text}")
            return False
    except Exception as e:
        print(f"[ERROR] Telegram notification error: {e}")
        return False


def send_photo(photo_path, caption=None):
    """Send a photo via Telegram."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    try:
        with open(photo_path, 'rb') as photo:
            files = {'photo': photo}
            data = {'chat_id': CHAT_ID}
            if caption:
                # Escape markdown special characters in caption
                caption_escaped = caption.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')
                data['caption'] = caption_escaped
                data['parse_mode'] = 'Markdown'

            response = requests.post(url, files=files, data=data, timeout=30)

        if response.status_code == 200:
            print(f"[OK] Photo sent: {photo_path}")
            return True
        else:
            print(f"[FAIL] Photo send failed: {response.text}")
            return False
    except Exception as e:
        print(f"[ERROR] Photo send error: {e}")
        return False


def send_document(file_path, caption=None):
    """Send a document/file via Telegram."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"

    try:
        with open(file_path, 'rb') as doc:
            files = {'document': doc}
            data = {'chat_id': CHAT_ID}
            if caption:
                # Escape markdown special characters in caption
                caption_escaped = caption.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')
                data['caption'] = caption_escaped
                data['parse_mode'] = 'Markdown'

            response = requests.post(url, files=files, data=data, timeout=60)

        if response.status_code == 200:
            print(f"[OK] Document sent: {file_path}")
            return True
        else:
            print(f"[FAIL] Document send failed: {response.text}")
            return False
    except Exception as e:
        print(f"[ERROR] Document send error: {e}")
        return False


# Test function
if __name__ == '__main__':
    print("Testing Telegram notifications...")
    send_message("🤖 Telegram bot is working!\n\nThis is a test message from your experiment notifier.")

import requests
import time
from googleapiclient.discovery import build
from google.oauth2 import service_account
import json

WEBHOOK_URL = "https://chat.googleapis.com/v1/spaces/AAAAJvzB51w/messages?key=AIzaSyDdI0hCZtE6vySjMm-WEfRq3CPzqKqqsHI&token=GEbepx9h6RQL2X4xjs0xFBgYkKKxkhUQdrf3F2ztbfc"

SERVICE_ACCOUNT_FILE = '/home/dheena/Downloads/kapquery-ed934d39f781.json'
SCOPES = ['https://www.googleapis.com/auth/chat.messages']

# Authenticate using the service account
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)

# Build the Google Chat service
service = build('chat', 'v1', credentials=credentials)

class GoogleChatNotifier:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url

    def send_message(self, message):
        """
        Send a text message to Google Chat space
        """
        payload = {'text': message}
        
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error sending message: {e}")
            return False

    def send_card(self, title, subtitle=None, body=None):
        """
        Send a card message to Google Chat space
        """
        card = {
            'cards': [{
                'sections': [{
                    'widgets': [{
                        'textParagraph': {
                            'text': f'<b>{title}</b>'
                        }
                    }]
                }]
            }]
        }

        if subtitle:
            card['cards'][0]['sections'][0]['widgets'].append({
                'textParagraph': {'text': subtitle}
            })

        if body:
            card['cards'][0]['sections'][0]['widgets'].append({
                'textParagraph': {'text': body}
            })

        try:
            response = requests.post(
                self.webhook_url,
                json=card,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error sending card: {e}")
            return False

def send_loading_card():
    """Send the initial loading card to the Google Chat space."""
    loading_card = {
        "cardsV2": [
            {
                "cardId": "loading_card",  # Unique identifier for the card
                "card": {
                    "header": {
                        "title": "Loading Data",
                        "subtitle": "Please wait..."
                    },
                    "sections": [
                        {
                            "widgets": [
                                {
                                    "decoratedText": {
                                        "text": "🔄 Loading, please wait..."
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
        ]
    }

    response = requests.post(WEBHOOK_URL, json=loading_card)
    print(f"Loading card sent: {response.status_code}, {response.text}")

    return response.json()['name']  # Save the message name for later use


# Function to update the existing card with dynamic content

def update_card(message_name):
    """Update an existing card in Google Chat."""
    updated_card = {
        "actionResponse": {
            "type": "UPDATE_MESSAGE"
        },
        "cardsV2": [
            {
                "cardId": "loading_card",
                "card": {
                    "header": {
                        "title": "Dynamic Content Loaded",
                        "subtitle": "Here is your data"
                    },
                    "sections": [
                        {
                            "widgets": [
                                {
                                    "decoratedText": {
                                        "text": "✔️ Data successfully loaded."
                                    }
                                },
                                {
                                    "textParagraph": {
                                        "text": "Temperature: 72°F"
                                    }
                                },
                                {
                                    "textParagraph": {
                                        "text": "Condition: Sunny"
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
        ]
    }

    # Send the update request
    response = service.spaces().messages().update(
        name=message_name, body=updated_card
    ).execute()

    print(f"Updated card response: {json.dumps(response, indent=2)}")



    # Update the same message by specifying its unique message name
    response = requests.put(f"https://chat.googleapis.com/v1/{message_name}", json=updated_card)
    print(f"Updated card sent: {response.status_code}, {response.text}")


if __name__ == "__main__":
    # Send loading card
    message_name = send_loading_card()

    # Simulate a delay for loading data
    time.sleep(5)

    # Send updated card with dynamic content
    update_card(message_name)
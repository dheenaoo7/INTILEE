import io
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Define your app's authorization scopes.
SCOPES = ["https://www.googleapis.com/auth/chat.messages"]

flow = InstalledAppFlow.from_client_secrets_file(
    '/home/dheena/Downloads/client_secret_377682800011-jrq7mlfr2hldlqf0gr9isifud5b5hpai.apps.googleusercontent.com.json', SCOPES
)
creds = flow.run_local_server(host="localhost",port=8090)
chat_service = build("chat", "v1", credentials=creds)

def download_and_save_file(resource_name: str, output_path: str):
    try:
        # Create a media download request
        request = chat_service.media().download_media(resourceName=resource_name)
        
        # Create a file-like object to store the downloaded content
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        
        # Download the file in chunks
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"Download progress: {int(status.progress() * 100)}%")

        # Save the content to the specified file path
        with open(output_path, 'wb') as f:
            f.write(file.getvalue())
        print(f"File saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading and saving file: {e}")
        return False


import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from typing import Any, Dict
from json_embeddings import JsonEmbeddingsProcessor
import text_embeddings
import PromptMaker
from google.oauth2.service_account import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict
import google_oauth
api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
processor = JsonEmbeddingsProcessor(api_key=api_key)
text_embeddings_processor = text_embeddings.EmbeddingIndexer()
embeddings_data = []    
user_selection = {}
UPLOAD_DIR = os.path.join(os.getcwd(), "Documents")

@app.on_event("startup")
def load_embeddings():
    global embeddings_data
    try:
        # Attempt to load the embeddings from the file
        if os.path.exists("code_embeddings.pkl"):
            embeddings_data = processor.load_embeddings("code_embeddings.pkl")
        else:
            embeddings_data = []  # Initialize as empty list if no embeddings exist
            print("No existing embeddings found, starting with an empty list.")
        if(os.path.exists("index.faiss") and os.path.exists("metadata.pkl")):
            text_embeddings_processor.load_index("index.faiss", "metadata.pkl")    
    except Exception as e:
        print(f"Error loading embeddings: {e}")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 2

@app.post("/find_similar_code")
def find_similar_code_endpoint(request: QueryRequest):
    if not embeddings_data:
        raise HTTPException(status_code=500, detail="Embeddings data not loaded.")
    results = processor.find_similar_code(request.query, embeddings_data, request.top_k)
    return results

@app.get("/update")
async def reprocess_and_update_embeddings():
    """This endpoint reprocesses the local JSON file and updates the embeddings."""
    
    # Dynamically construct the file paths based on the current directory
    current_directory = os.getcwd()  # Get the current working directory
    input_file_path = os.path.join(current_directory, "output/output_code_data.json")  # Assuming the file is named 'code.son'
    output_file_path = os.path.join(current_directory, "code_embeddings.pkl")  # Embeddings will be saved as 'code_embeddings.pkl'

    try:
        # Process the JSON file and generate embeddings
        code_embeddings = processor.process_json_file(input_file_path)
        
        # Save the updated embeddings to the output file
        processor.save_embeddings(code_embeddings, output_file_path)

        return {"detail": "Embeddings reprocessed and updated successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the JSON file: {e}")
    
@app.post("/getAnswer")
async def getAnswer(query:str, selected_type :str = "KapCode"):
    answer = "Internal Server Error"
    if selected_type == "KapCode":
        reposne_index = 1
        callIndex = 1
        results = processor.find_similar_code(query, embeddings_data, reposne_index) 
        parent_code = results[reposne_index - 1]['code_snippet']
        child_codes = [call['code_snippet'] for call in results[callIndex - 1]['calls']]
        answer = PromptMaker.handle_user_query(selected_type, query, parent_method=parent_code, child_methods=child_codes)
    elif selected_type == "KapDoc":
        results = text_embeddings_processor.query(query)
        results = text_embeddings_processor.get_texts_by_metadata([doc['metadata'] for doc in results])
        answer = PromptMaker.handle_user_query(selected_type, query= query, documents = results, parent_method=None, child_methods=None)   
    return answer

@app.post("/chatbot", response_model=None)  # Disable auto-response model generation
async def chatbot_endpoint(request: Request) -> Dict[str, Any]:
    """
    Endpoint to handle messages from Google Chat.
    """
    try:
        request_json = await request.json()

        # Handle different types of Google Chat events
        event_type = request_json.get('type', '')
        sender_name = request_json.get('message', {}).get('sender', {}).get('displayName', 'User')
        
        print(request_json)

        if event_type == 'ADDED_TO_SPACE':
            return JSONResponse(content={
                "text": create_welcome_message(),
                "cards": [create_selection_card(sender_name)]
            })
        # Handle REMOVED_FROM_SPACE event
        elif event_type == 'REMOVED_FROM_SPACE':
            return JSONResponse(content={ "text": "Thank you for using kapQuery! Goodbye! 👋"})

        action = request_json.get('action', {}).get('actionMethodName', '')
        if action:
            return await handle_user_action(request_json)

        # Handle attachments
        if 'attachment' in request_json.get('message', {}):
            attachments = request_json['message']['attachment']
            for attachment in attachments:
                resource_name = attachment.get('attachmentDataRef').get('resourceName')
                file_name = attachment.get('contentName')
                print("hii")
                # Only handle .txt files
                if file_name.endswith('.txt') and resource_name:
                    # Fetch and Save the file content from the URL
                    file_path = os.path.join(UPLOAD_DIR, file_name)
                    google_oauth.download_and_save_file(resource_name, file_path)                  
                    text_embeddings_processor.embed_text_from_file(file_path = file_path)
                    return JSONResponse(content={
                            "text": f"File {file_name} has been saved successfully!"
                            })      
        # Handle regular messages (text messages)
        if 'message' in request_json:
            message_text = request_json.get('message', {}).get('text', '')

            # Check if the message is "/stop"
            if message_text.strip().lower() == "/stop" or message_text.strip().lower() == "/start":
                user_selection[sender_name] = None
                return JSONResponse(content={
                    "text": "Selection has been reset. Please choose one of the following options: KapDoc or KapCode.",
                    "cards": [create_selection_card(sender_name)]
                })
            if(message_text==''):
                return  JSONResponse(content={
                    "text": "Please Send a Text"
                })

            # Handle other messages
            response = await getAnswer(message_text, user_selection.get(sender_name, 'KapCode'))
            return JSONResponse(content={
                "text": response
            })

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return JSONResponse(
            content={"text": "An error occurred while processing your request."},
            status_code = 200
        )

def create_welcome_message() -> str:
    """Creates a formatted welcome message for new users"""
    return (
        "Welcome to kapQuery - Your Kapture Codebase Assistant! 🚀\n\n"
        "I'm here to help you with Kapture's codebase and documentation. Please choose a topic:\n"
        "• KapDoc for document-related queries\n"
        "• KapCode for codebase-related queries\n\n"
        "Type /stop at any time to reset your selection."
    )


def create_selection_card(username) -> Dict[str, Any]:
    """Creates an interactive card with KapDoc and KapCode options"""
    return {
        "header": {
            "title": "Select Your Query Type",
            "subtitle": "Choose KapDoc for document-related queries or KapCode for code-related queries."
        },
        "sections": [
            {
                "widgets": [
                    {
                        "buttons": [
                            {
                                "textButton": {
                                    "text": "KapDoc",
                                    "onClick": {
                                        "action": {
                                            "actionMethodName": "selectKapDoc",
                                            "parameters": [{
                                                "key": "actionUsername",
                                                "value": username
                                                }]
                                        }
                                    }
                                }
                            },
                            {
                                "textButton": {
                                    "text": "KapCode",
                                    "onClick": {
                                        "action": {
                                            "actionMethodName": "selectKapCode",
                                            "parameters": [{
                                                "key": "actionUsername",
                                                "value": username
                                                }]
                                        }
                                        
                                    }
                                }
                            },
                        ]
                    }
                ]
            }
        ]
    }

async def handle_user_action(request_json: Dict[str, Any]) -> JSONResponse:
    """Handles user actions like selecting KapDoc or KapCode."""
    try:
        action = request_json.get('action', {}).get('actionMethodName', '')
        user_name = request_json.get('action', {}).get('parameters', [{}])[0].get('value', 'User')

        # Determine which button was clicked and save the selection
        if action == 'selectKapDoc':
            save_user_selection(user_name, 'KapDoc')
            return JSONResponse(content={
                "text": "You have selected KapDoc for document-related queries."
            })
        elif action == 'selectKapCode':
            save_user_selection(user_name, 'KapCode')
            return JSONResponse(content={
                "text": "You have selected KapCode for codebase-related queries."
            })
    except Exception as e:
        print(f"Error processing user action: {str(e)}")
        return JSONResponse(content={"text": "An error occurred while processing your request."}, status_code=500)


def save_user_selection(user_name: str, selection: str) -> None:
    """Save the user's selection."""
    user_selection[user_name] = selection


from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Any, Dict
from embeddings import JsonEmbeddingsProcessor
from dotenv import load_dotenv
import os

app = FastAPI()

# Initialize embeddings processor
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
processor = JsonEmbeddingsProcessor(api_key=api_key)

# Load embeddings data
try:
    embeddings_data = processor.load_embeddings("code_embeddings.pkl")
except Exception as e:
    print(f"Error loading embeddings: {e}")
    embeddings_data = []

@app.post("/chatbot")
async def chatbot_endpoint(request: Request) -> Dict[str, Any]:
    """
    Endpoint to handle messages from Google Chat.
    """
    try:
        request_json = await request.json()
        
        # Handle different types of Google Chat events
        event_type = request_json.get('type', '')
        
        # Handle ADDED_TO_SPACE event
        if event_type == 'ADDED_TO_SPACE':
            return JSONResponse(content={
                "text": "Thanks for adding me! Send me a message to get started."
            })

        # Handle REMOVED_FROM_SPACE event
        elif event_type == 'REMOVED_FROM_SPACE':
            return JSONResponse(content={"text": "Goodbye!"})

        # Handle regular messages
        elif 'message' in request_json:
            message_text = request_json.get('message', {}).get('text', '')
            sender_name = request_json.get('message', {}).get('sender', {}).get('displayName', 'User')
            
          
            similar_results = processor.find_similar_code(
                query=message_text,
                embeddings_data=embeddings_data
            )


            print(similar_results)
            
            return JSONResponse(content={
                "text": f"Hello {sender_name}! You said: {message_text}"
            })

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return JSONResponse(
            content={"text": "An error occurred while processing your request."},
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
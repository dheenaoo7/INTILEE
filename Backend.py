import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from embeddings import JsonEmbeddingsProcessor
import PromptMaker


api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
processor = JsonEmbeddingsProcessor(api_key=api_key)
embeddings_data = []

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
    input_file_path = os    .path.join(current_directory, "output/output_code_data.json")  # Assuming the file is named 'code.son'
    output_file_path = os.path.join(current_directory, "code_embeddings.pkl")  # Embeddings will be saved as 'code_embeddings.pkl'

    try:
        # Process the JSON file and generate embeddings
        new_embeddings = processor.process_json_file(input_file_path)

        # Load existing embeddings from the file if it exists
        if os.path.exists(output_file_path):
            embeddings_data = processor.load_embeddings(output_file_path)
        else:
            embeddings_data = []  # Initialize as empty list if no embeddings exist

        # Add the new embeddings to the existing data
        embeddings_data.extend(new_embeddings)

        # Save the updated embeddings to the output file
        processor.save_embeddings(embeddings_data, output_file_path)

        return {"detail": "Embeddings reprocessed and updated successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the JSON file: {e}")
    
@app.post("/getAnswer")
async def getAnswer(request: Request):
    try:
        data = await request.json()
        query = data.get('query', '')
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
            
        response_index = data.get('response_index', 1)
        call_index = data.get('callIndex', 1)
        
        if not embeddings_data:
            raise HTTPException(status_code=500, detail="No embeddings data available")
            
        results = processor.find_similar_code(query, embeddings_data, response_index)
        if not results or len(results) < response_index:
            raise HTTPException(status_code=404, detail="No matching results found")
            
        parent_code = results[response_index - 1]['code_snippet']
        if not results[call_index - 1].get('calls'):
            raise HTTPException(status_code=404, detail="No call data available")
            
        child_codes = [call['code_snippet'] for call in results[call_index - 1]['calls']]
        answer = PromptMaker.get_code_assistant_response(query, parent_code, child_codes)
        return {"content": answer.content}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
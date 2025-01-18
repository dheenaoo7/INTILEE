import json
import os
import pickle
from tqdm import tqdm
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

class JsonEmbeddingsProcessor:
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.method_index = {}

    def _create_method_index(self, embeddings_data: List[Dict[str, Any]]):
        """Create an index from embeddings data for faster lookup"""
        self.method_index.clear()
        for item in embeddings_data:
            metadata = item.get('metadata', {})
            # Make case insensitive by converting to lowercase
            key = f"{metadata.get('classname', '').lower()}_{metadata.get('methodname', '').lower()}"
            self.method_index[key] = metadata

    def find_similar_code(self, query: str, embeddings_data: List[Dict[str, Any]], top_k: int = 2) -> List[Dict[str, Any]]:
        """Find most similar code objects to a query and validate inner method calls."""
        # Create index once for this search
        self._create_method_index(embeddings_data)

        try:
            query_response = self.client.embeddings.create(
                model=self.model,
                input=query
            )
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []

        query_embedding = np.array(query_response.data[0].embedding)
        similarities = []
        
        for item in embeddings_data:
            embedding = np.array(item['embedding'])
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append({
                'metadata': item['metadata'],
                'similarity_score': float(similarity)
            })

        sorted_results = sorted(similarities, key=lambda x: x['similarity_score'], reverse=True)
        results_with_explanation = []
        
        for result in sorted_results[:top_k]:
            metadata = result['metadata']
            classname = metadata.get('classname', '')
            methodname = metadata.get('methodname', '')
            calls = metadata.get('calls', [])
            
            validated_calls = []
            for call in calls:
                call_classname = call.get('classname', '')
                if(call_classname=="this"):
                      call_classname=classname
                call_methodname = call.get('methodname', '')
                
            
                method_key = f"{call_classname.lower()}_{call_methodname.lower()}"
                matching_method = self.method_index.get(method_key)
                
                if matching_method:
                    # Add debug print to verify matches
                    print(f"Found internal method: {call_classname}.{call_methodname}")
                    validated_calls.append({
                        'classname': call_classname,
                        'methodname': call_methodname,
                        'parameters': call.get('parameters', []),
                        'is_internal_method': True,
                        'code_snippet': matching_method.get('code', '')
                    })
               
            results_with_explanation.append({
                'classname': classname,
                'methodname': methodname,
                'calls': validated_calls,
                'similarity_score': result['similarity_score'],
                'code_snippet': metadata.get('code', '')
            })
        
        return results_with_explanation

    def _flatten_code_object(self, code_obj: Dict[str, Any]) -> str:
        """Convert code object to a detailed string representation."""
        # Ensure code_obj is a dictionary
        if not isinstance(code_obj, dict):
            print(f"Expected dictionary, but got: {type(code_obj)}")
            return ""

        class_name = code_obj.get('classname', '')
        method_name = code_obj.get('methodname', '')
        flattened = f"Class: {class_name}\nMethod: {method_name}\n"
        
        
        # Handle 'param' which is expected to be a list
        params = code_obj.get('param', [])
        if isinstance(params, list):  # Ensure 'param' is a list
            flattened += "Parameters:\n" + "\n".join(f"- {p}" for p in params) + "\n"
        else:
            print(f"Unexpected 'param' value: {params}")
        
        # Handle 'calls' which is expected to be a list of dictionaries
        calls = code_obj.get('calls', [])
        if isinstance(calls, list):  # Ensure 'calls' is a list
            if calls:
                flattened += "Function Calls:\n"
                for call in calls:
                    if isinstance(call, dict):  # Ensure call is a dictionary
                        classname = call.get('classname', '')
                        methodname = call.get('methodname', '')
                        parameters = ", ".join(call.get('parameters', []))
                        flattened += f"- {classname}.{methodname}({parameters})\n"
                    else:
                        print(f"Unexpected 'call' value: {call}")
        else:
            print(f"Unexpected 'calls' value: {calls}")
        
        # Handle 'code' which is expected to be a string
        code_snippet = code_obj.get('code', '')
        if code_snippet:
            flattened += f"Code:\n{code_snippet}\n"
        
        return flattened

    def validate_json(self, file_path: str) -> bool:
        """Validate the structure of the JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Check if the loaded data is a list
            return isinstance(data, list)
        except json.JSONDecodeError:
            return False

    def process_json_file(self, file_path: str, batch_size: int = 50) -> List[Dict[str, Any]]:
        """Process JSON file in batches and generate embeddings."""
        if not self.validate_json(file_path):
            raise ValueError("Invalid JSON structure. Expected a list of function objects.")

        embeddings_data = []
        with open(file_path, 'r') as f:
            data = json.load(f)

        for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
            batch = data[i:i + batch_size]
            batch_texts = [self._flatten_code_object(obj) for obj in batch]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts
                )
                for j, obj in enumerate(batch):
                    embeddings_data.append({
                        'embedding': response.data[j].embedding,
                        'metadata': obj
                    })
            except Exception as e:
                print(f"Error processing batch {i // batch_size}: {e}")
                continue

        return embeddings_data
    
    def save_embeddings(self, embeddings_data: List[Dict[str, Any]], save_path: str):
        """Save embeddings and metadata to a file."""
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings_data, f)

    def load_embeddings(self, load_path: str) -> List[Dict[str, Any]]:
        """Load embeddings and metadata from a file."""
        with open(load_path, 'rb') as f:
            return pickle.load(f)

# FastAPI setup
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
    input_file_path = os.path.join(current_directory, "output/output_code_data.json")  # Assuming the file is named 'code.son'
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
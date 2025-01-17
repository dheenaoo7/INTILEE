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

    def _flatten_code_object(self, code_obj: Dict[str, Any]) -> str:
        """Convert code object to a detailed string representation."""
        flattened = f"Function: {code_obj.get('name', '')}\n"
        params = code_obj.get('param', [])
        if params:
            flattened += "Parameters:\n" + "\n".join(f"- {p}" for p in params) + "\n"
        calls = code_obj.get('calls', [])
        if calls:
            flattened += "Function Calls:\n" + "\n".join(
                f"- {c.get('name', '')}({', '.join(c.get('args', []))})" for c in calls
            ) + "\n"
        explanation = code_obj.get('explanation', '')
        if explanation:
            flattened += f"Explanation: {explanation}\n"
        code_snippet = code_obj.get('code', '')
        if code_snippet:
            flattened += f"Code:\n{code_snippet}\n"
        return flattened

    def validate_json(self, file_path: str) -> bool:
        """Validate the structure of the JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return isinstance(data.get('functions', []), list)
        except (json.JSONDecodeError, KeyError):
            return False

    def process_json_file(self, file_path: str, batch_size: int = 50) -> List[Dict[str, Any]]:
        """Process JSON file in batches and generate embeddings."""
        if not self.validate_json(file_path):
            raise ValueError("Invalid JSON structure. Ensure 'functions' key contains a list.")

        embeddings_data = []
        with open(file_path, 'r') as f:
            data = json.load(f)
        functions = data.get('functions', [])

        for i in tqdm(range(0, len(functions), batch_size), desc="Processing batches"):
            batch = functions[i:i + batch_size]
            batch_texts = [self._flatten_code_object(obj) for obj in batch]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts
                )
                for j, obj in enumerate(batch):
                    embeddings_data.append({
                        'name': obj['name'],
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

    def find_similar_code(self, query: str, embeddings_data: List[Dict[str, Any]], top_k: int = 2) -> List[Dict[str, Any]]:
        """Find most similar code objects to a query."""
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
                'name': item['name'],
                'metadata': item['metadata'],
                'similarity_score': float(similarity)
            })

        sorted_results = sorted(similarities, key=lambda x: x['similarity_score'], reverse=True)
        results_with_explanation = []
        for result in sorted_results[:top_k]:
            results_with_explanation.append({
                'name': result['name'],
                'similarity_score': result['similarity_score'],
                'code_snippet': json.dumps(result['metadata'], indent=2)
            })
        return results_with_explanation

# FastAPI setup
app = FastAPI()
processor = JsonEmbeddingsProcessor(api_key=api_key)
embeddings_data = []

@app.on_event("startup")
def load_embeddings():
    global embeddings_data
    try:
        embeddings_data = processor.load_embeddings("code_embeddings.pkl")
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

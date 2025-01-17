import faiss
import json
import numpy as np
from gradio_client import Client
import torch
from sentence_transformers import SentenceTransformer, models
from ollama import chat
from flask import Flask, request, jsonify
import ast, json

# Paths to your saved files
FAISS_INDEX_PATH = '/home/dheena/Downloads/Intiliee/output/code_embeddings.index'
EMBEDDINGS_PATH = '/home/dheena/Downloads/Intiliee/output/embeddings.npy'
CODE_SNIPPETS_PATH = '/home/dheena/Downloads/Intiliee/output/code_snippets.json'
METADATA_PATH = '/home/dheena/Downloads/Intiliee/output/metadata.json'

# Load the Faiss index
index = faiss.read_index(FAISS_INDEX_PATH)
print("Faiss index loaded successfully.")

# Load embeddings
embeddings = np.load(EMBEDDINGS_PATH)
print(f"Embeddings loaded successfully with shape: {embeddings.shape}")

# Load code snippets
with open(CODE_SNIPPETS_PATH, 'r') as f:
    code_snippets = json.load(f)
print(f"Loaded {len(code_snippets)} code snippets.")

# Load metadata
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)
print(f"Loaded metadata for {len(metadata)} code snippets.")

model = SentenceTransformer("krlvi/sentence-t5-base-nlpl-code-x-glue")


def get_query_embedding(query_text):
    embedding = model.encode([query_text])
    return embedding

print("Model loaded successfully.")

# Define the embedding function
def get_query_embedding(query_text):
    embedding = model.encode([query_text], show_progress_bar=False)
    # Normalize the embedding
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding

def search_code_snippets(user_query, index, code_snippets, metadata, top_k=5):
    query_embedding = get_query_embedding(user_query)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    query_embedding = query_embedding.astype('float32')

    distances, indices = index.search(query_embedding, k=top_k)

    results = []
    for idx, distance in zip(indices[0], distances[0]):
        snippet = code_snippets[idx]
        info = metadata[idx]
        similarity = float(distance)
        results.append({
            'snippet': snippet,
            'metadata': info,
            'similarity': similarity
        })
    return results

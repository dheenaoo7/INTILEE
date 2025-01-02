import faiss
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models

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

# Load the tokenizer and model
# We will use CodeBERT via the sentence-transformers library
# Load the transformer model and pooling
word_embedding_model = models.Transformer('microsoft/codebert-base', max_seq_length=512)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False
)

# Create the SentenceTransformer model
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

print("Model loaded successfully.")

# Define the embedding function
def get_query_embedding(query_text):
    embedding = model.encode([query_text], show_progress_bar=False)
    # Normalize the embedding
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding

def search_code_snippets(user_query, index, code_snippets, metadata, top_k=5):
    query_embedding = get_query_embedding(user_query)
    distances, indices = index.search(query_embedding, k=top_k)
    results = []
    for idx in indices[0]:
        snippet = code_snippets[idx]
        info = metadata[idx]
        distance = distances[0][np.where(indices[0] == idx)[0][0]]
        results.append({
            'snippet': snippet,
            'metadata': info,
            'distance': distance
        })
    return results

# Sample user query
user_query = "What is the purpose of the deletePauseOption method?"

# Search for similar code snippets
results = search_code_snippets(user_query, index, code_snippets, metadata, top_k=5)

# Display the results
print("\nTop matching code snippets:")
for i, result in enumerate(results):
    print(f"\nResult {i+1}:")
    print(f"File: {result['metadata']['file_name']}")
    print(f"Type: {result['metadata']['type']}")
    print(f"Distance (cosine similarity): {result['distance']}")
    print("Code Snippet:")
    print(result['snippet'])
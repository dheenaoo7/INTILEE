from random import randint
import faiss
import numpy as np
import json

# Load embeddings from the .npy file
embeddings = np.load('/home/dheena/Downloads/Intiliee/output/embeddings.npy')

# Load code snippets and metadata
with open('/home/dheena/Downloads/Intiliee/output/code_snippets.json', 'r') as f:
    code_snippets = json.load(f)

with open('/home/dheena/Downloads/Intiliee/output/metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Number of code snippets: {len(code_snippets)}")
print(f"Number of metadata entries: {len(metadata)}")

# Normalize embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Convert embeddings to float32
embeddings = embeddings.astype('float32')

# Create a FAISS index for cosine similarity
embedding_dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dimension)  # Inner product

# Add embeddings to the index
index.add(embeddings)

faiss.write_index(index, '/home/dheena/Downloads/Intiliee/output/code_embeddings.index')
print("Index saved successfully.")

# Let's take the embedding of the first code snippet as a test query
test_embedding = embeddings[0].reshape(1, -1)


# Number of nearest neighbors to retrieve
k = 5

# Perform the search
distances, indices = index.search(test_embedding, k)

print("Indices of nearest neighbors:", indices)
print("Distances to nearest neighbors:", distances)
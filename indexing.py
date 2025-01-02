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

# Normalize the embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

embedding_dimension = embeddings.shape[1]
print(f"Embedding dimension: {embedding_dimension}")

# Create the index
index = faiss.IndexFlatIP(embedding_dimension)

# Convert embeddings to float32 if they are not already
if embeddings.dtype != 'float32':
    embeddings = embeddings.astype('float32')

# Add embeddings to the index
index.add(embeddings)

print(f"Number of vectors in the index: {index.ntotal}")

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
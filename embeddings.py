# Code for generating embeddings for your codebase

# 1. Install Required Libraries
# Ensure you have installed the 'sentence-transformers' library
# If not, you can install it using:
# pip install sentence-transformers

# 2. Import Necessary Libraries
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer, models
from langchain_openai import OpenAIEmbeddings

# 3. Load Your Codebase
# Replace 'codebase.json' with the path to your codebase JSON file
with open('/home/dheena/Downloads/Intiliee/output/output_code_data.json', 'r') as f:
    codebase = json.load(f)

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = ("Enter API key for OpenAI: ")

# Extract code snippets and metadata
code_snippets = []
metadata = []

for file in codebase:
    file_name = file.get('file_name', '')
    snippets = file.get('code_snippets', {})
    functions = snippets.get('functions', [])
    classes = snippets.get('classes', [])

    for func in functions:
        code_snippets.append(func)
        metadata.append({'type': 'function', 'file_name': file_name})

    for cls in classes:
        code_snippets.append(cls)
        metadata.append({'type': 'class', 'file_name': file_name})

print(f"Total code snippets loaded: {len(code_snippets)}")

# # 4. Load the Embedding Model
# # Load the code search model

# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# # Generate embeddings
# print("Generating embeddings for code snippets...")
# code_snippets_str = [snippet if isinstance(snippet, str) else snippet.get("code", "") for snippet in code_snippets]
# vectors = embeddings.embed_documents(code_snippets_str)

# # 6. Save the Embeddings and Data
# # Save embeddings to a .npy file
# np.save('/home/dheena/Downloads/Intiliee/output/embeddings.npy', vectors)

# # Save code snippets and metadata as JSON files
with open('/home/dheena/Downloads/Intiliee/output/code_snippets.json', 'w') as f:
    json.dump(code_snippets, f)

# with open('/home/dheena/Downloads/Intiliee/output/metadata.json', 'w') as f:
#     json.dump(metadata, f)

# print("Embeddings and data have been saved successfully.")
from query import search_code_snippets
import faiss
import json

FAISS_INDEX_PATH = '/home/dheena/Downloads/Intiliee/output/code_embeddings.index'
METADATA_PATH = '/home/dheena/Downloads/Intiliee/output/metadata.json'

with open('/home/dheena/Downloads/Intiliee/output/code_snippets.json', 'r') as f:
    code_snippets = json.load(f)
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

print(f"Loaded metadata for {len(metadata)} code snippets.")
index = faiss.read_index(FAISS_INDEX_PATH)
def queryEmbeddings(query, newQuery, code_snippets, metadata, top_k=5, recursionIndex=0):
    if(newQuery):
        results = search_code_snippets(query, index, code_snippets, metadata, top_k)   
    return[recursionIndex]

    
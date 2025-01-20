import os
import numpy as np
import pickle
from openai import OpenAI
from dotenv import load_dotenv
import faiss

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")


openai_client = OpenAI(api_key=api_key)
class EmbeddingIndexer:
    def __init__(self, model: str = "text-embedding-ada-002", embedding_dim: int = 1536):
        self.model = model
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata = []  # Store metadata as a list of dictionaries

    def add_text(self, text: str, metadata: dict):
        """Generate embedding and add text with metadata to the index."""
        response = openai_client.embeddings.create(model=self.model, input=text)
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        self.index.add(np.array([embedding]))  # Add to FAISS index
        self.metadata.append(metadata)  # Save metadata for retrieval
        self.save_index("index.faiss", "metadata.pkl")
        
    def query(self, query: str, top_k: int = 3):
        """Query the index and return top-k results with metadata."""
        response = openai_client.embeddings.create(model=self.model, input=query)
        query_embedding = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                results.append({"metadata": self.metadata[idx], "distance": dist})
        return results

    def save_index(self, index_path: str, metadata_path: str):
        """Save FAISS index and metadata."""
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load_index(self, index_path: str, metadata_path: str):
        """Load FAISS index and metadata."""
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
     
    def embed_text_from_file(self, file_path: str, chunk_size: int = 8000, overlap: int = 50):
        """Read text from a file, split into chunks, and create embeddings."""
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        # Split text into chunks
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        # Embed each chunk and add to index
        for chunk in chunks:
            chunk_start = start - (chunk_size - overlap)
            chunk_end = chunk_start + len(chunk)
            self.add_text(chunk, {"file_path": file_path, "chunk_start": chunk_start, "chunk_end": chunk_end})
    
    def get_texts_by_metadata(self, metadata_list):
        """Retrieve texts from the index based on a list of metadata."""
        results = []
        for match in metadata_list:
            file_path = match["file_path"]
            chunk_start = match["chunk_start"]
            chunk_end = match["chunk_end"]
            with open(file_path, "r", encoding="utf-8") as file:
                file.seek(chunk_start)
                text_chunk = file.read(chunk_end - chunk_start)
                results.append(text_chunk)
        return results
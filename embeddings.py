# Code for generating embeddings for your codebase

# 1. Install Required Libraries
# Ensure you have installed the 'sentence-transformers' library
# If not, you can install it using:
# pip install sentence-transformers

# 2. Import Necessary Libraries
import json
import numpy as np
from sentence_transformers import SentenceTransformer, models

# 3. Load Your Codebase
# Replace 'codebase.json' with the path to your codebase JSON file
with open('/home/dheena/Downloads/Intiliee/output/output_code_data.json', 'r') as f:
    codebase = json.load(f)

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

# 4. Load the Embedding Model
# We will use CodeBERT via the sentence-transformers library
# Load the transformer model
word_embedding_model = models.Transformer('microsoft/codebert-base', max_seq_length=512)
# Add a pooling layer
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False
)

# Create the SentenceTransformer model
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# 5. Generate Embeddings for the Code Snippets
print("Generating embeddings for code snippets...")
embeddings = model.encode(code_snippets, show_progress_bar=True)

# 6. Save the Embeddings and Data
# Save embeddings to a .npy file
np.save('/home/dheena/Downloads/Intiliee/output/embeddings.npy', embeddings)

# Save code snippets and metadata as JSON files
with open('/home/dheena/Downloads/Intiliee/output/code_snippets.json', 'w') as f:
    json.dump(code_snippets, f)

with open('/home/dheena/Downloads/Intiliee/output/metadata.json', 'w') as f:
    json.dump(metadata, f)

print("Embeddings and data have been saved successfully.")
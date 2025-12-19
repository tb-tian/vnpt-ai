import os
from sentence_transformers import CrossEncoder
os.environ['HF_HOME'] = '/app/model_cache'

print("Downloading BGE Reranker model...")
CrossEncoder("BAAI/bge-reranker-v2-m3")
print("Download complete!")
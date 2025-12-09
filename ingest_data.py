import os
import json
import glob
import requests
import time
from rag_pipeline import RAGPipeline

# Configuration
DATA_DIR = "corpus"  # Directory containing .txt files
VECTOR_DB_PATH = "./vector_db"
CHUNK_SIZE = 500  # Characters (approx)
OVERLAP = 50

# Load API Key
try:
    with open('../api-keys.json', 'r') as f:
        keys = json.load(f)
        API_KEY = keys.get("VNPT_API_KEY", "")
except FileNotFoundError:
    API_KEY = os.environ.get("VNPT_API_KEY", "")

EMBEDDING_URL = "https://llm.vnpt.ai/api/v1/embeddings"

def get_embedding(text):
    """
    Call VNPT Embedding API.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
        "model": "vnptai_hackathon_embedding"
    }
    # Retry logic
    for _ in range(3):
        try:
            response = requests.post(EMBEDDING_URL, headers=headers, json=data, timeout=10)
            if response.status_code == 200:
                return response.json()['data'][0]['embedding']
            elif response.status_code == 429:
                print("Rate limit hit, waiting...")
                time.sleep(2)
            else:
                print(f"Embedding Error: {response.status_code} - {response.text}")
                break
        except Exception as e:
            print(f"Embedding Exception: {e}")
            time.sleep(1)
    return None

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Simple sliding window chunking.
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    
    return chunks

def main():
    # Initialize Pipeline
    rag = RAGPipeline(vector_db_path=VECTOR_DB_PATH, embedding_api_func=get_embedding)
    
    # Find all text files
    files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    print(f"Found {len(files)} files in {DATA_DIR}")
    
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    doc_counter = 0
    
    for file_path in files:
        print(f"Processing {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            chunks = chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({"source": os.path.basename(file_path), "chunk_id": i})
                all_ids.append(f"{os.path.basename(file_path)}_{i}")
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if all_chunks:
        print(f"Ingesting {len(all_chunks)} chunks...")
        rag.add_documents(all_chunks, all_metadatas, all_ids)
        print("Ingestion complete.")
    else:
        print("No data to ingest.")

if __name__ == "__main__":
    # Create dummy data if directory doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        with open(os.path.join(DATA_DIR, "sample_vietnam_history.txt"), "w", encoding="utf-8") as f:
            f.write("""Lịch sử Việt Nam trải qua nhiều thời kỳ thăng trầm. 
            Thời kỳ Hùng Vương là giai đoạn hình thành nhà nước đầu tiên của người Việt, nước Văn Lang.
            Các vua Hùng đã có công dựng nước, Bác Hồ đã dạy: "Các vua Hùng đã có công dựng nước, Bác cháu ta phải cùng nhau giữ lấy nước".
            Chiến thắng Bạch Đằng năm 938 của Ngô Quyền đã chấm dứt hơn 1000 năm Bắc thuộc, mở ra kỷ nguyên độc lập tự chủ lâu dài.
            """)
        print(f"Created sample data in {DATA_DIR}")
        
    main()

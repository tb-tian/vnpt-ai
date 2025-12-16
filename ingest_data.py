import os
import glob
from rag_langchain import LangChainRAG

# Configuration
DATA_DIR = "corpus"  # Directory containing .txt files

def main():
    # Initialize Pipeline
    rag = LangChainRAG()
    
    # Find all text files
    files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    print(f"Found {len(files)} files in {DATA_DIR}")
    
    if files:
        rag.ingest_data(files)
        print("Ingestion complete.")
    else:
        print("No files found to ingest.")

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

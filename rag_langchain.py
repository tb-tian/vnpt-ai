import json
import requests
from tqdm import tqdm
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class VNPTEmbeddings(Embeddings):
    """Custom Embedding class for VNPT API"""
    
    def __init__(self, api_key_file="api-keys.json"):
        self.api_key_file = api_key_file
        self._load_credentials()

    def _load_credentials(self):
        with open(self.api_key_file, 'r') as f:
            api_keys = json.load(f)
        
        key_info = next((item for item in api_keys if item["llmApiName"] == "LLM embedings"), None)
        if not key_info:
            raise ValueError("Embedding API key not found")
            
        self.headers = {
            'Authorization': key_info['authorization'],
            'Token-id': key_info['tokenId'],
            'Token-key': key_info['tokenKey'],
            'Content-Type': 'application/json',
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        json_data = {
            'model': 'vnptai_hackathon_embedding',
            'input': text,
            'encoding_format': 'float'
        }
        
        try:
            response = requests.post(
                'https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding',
                headers=self.headers,
                json=json_data
            )
            response.raise_for_status()
            return response.json()['data'][0]['embedding']
        except Exception as e:
            print(f"Error embedding text: {e}")
            return []

class LangChainRAG:
    def __init__(self, vector_db_path="./vector_db_langchain"):
        self.embedding_model = VNPTEmbeddings()
        self.vector_db_path = vector_db_path
        self.vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embedding_model,
            collection_name="vnpt_knowledge_base"
        )
        self.retriever = None

    def ingest_data(self, file_paths: List[str]):
        """Ingest text files into the vector store"""
        documents = []
        print("Reading files...")
        for path in tqdm(file_paths, desc="Loading files"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    documents.append(Document(page_content=text, metadata={"source": path}))
            except Exception as e:
                print(f"Error reading {path}: {e}")

        # Split text
        print("Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        
        print(f"Created {len(splits)} chunks from {len(documents)} files.")
        
        # Add to VectorDB in batches to show progress
        batch_size = 100 # Adjust based on API limits/speed
        print(f"Ingesting {len(splits)} chunks into VectorDB...")
        for i in tqdm(range(0, len(splits), batch_size), desc="Embedding & Indexing"):
            batch = splits[i:i + batch_size]
            self.vectorstore.add_documents(batch)
            
        print("Ingestion complete.")

    def setup_retriever(self):
        """Setup Hybrid Retriever (Vector + BM25)"""
        # Vector Retriever
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # BM25 Retriever (Keyword)
        # Note: In a real app, you might want to load all docs to initialize BM25
        # For now, we assume the vectorstore has data we can fetch, 
        # or we re-ingest for the sake of the example. 
        # Ideally, you fetch all docs from Chroma to build BM25.
        
        # Fetching all docs for BM25 (simplified for demo)
        all_docs = self.vectorstore.get()['documents']
        if not all_docs:
            print("Vector store is empty. Please ingest data first.")
            return None
            
        bm25_retriever = BM25Retriever.from_texts(all_docs)
        bm25_retriever.k = 3

        # Ensemble (Hybrid) Retriever
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5] # Equal weight to keyword and semantic search
        )
        return self.retriever

    def query(self, question: str):
        if not self.retriever:
            self.setup_retriever()
            
        if not self.retriever:
            return "System not initialized with data."

        docs = self.retriever.invoke(question)
        return docs

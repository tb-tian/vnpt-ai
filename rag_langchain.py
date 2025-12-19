import json
import requests
from tqdm import tqdm
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
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
        import concurrent.futures
        
        # Use ThreadPoolExecutor to make parallel API calls
        # Reduced to 5 workers to respect rate limits (approx 500 req/min)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Use tqdm to show progress of embedding within the batch
            embeddings = list(tqdm(
                executor.map(self.embed_query, texts), 
                total=len(texts), 
                desc="Embedding batch", 
                leave=False
            ))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        import time
        import random

        json_data = {
            'model': 'vnptai_hackathon_embedding',
            'input': text,
            'encoding_format': 'float'
        }
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    'https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding',
                    headers=self.headers,
                    json=json_data
                )
                response.raise_for_status()
                return response.json()['data'][0]['embedding']
            except Exception as e:
                # If it's the last attempt, print error and return empty
                if attempt == max_retries - 1:
                    print(f"Error embedding text: {e}")
                    return []
                
                # If we hit a rate limit or temporary error, wait and retry
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(sleep_time)
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
        # Check for existing documents to avoid duplicates
        try:
            # Only fetch metadatas to be faster
            existing_data = self.vectorstore.get(include=['metadatas'])
            existing_sources = set()
            if existing_data and 'metadatas' in existing_data:
                for metadata in existing_data['metadatas']:
                    if metadata and 'source' in metadata:
                        existing_sources.add(metadata['source'])
            print(f"Found {len(existing_sources)} existing sources in vector DB.")
        except Exception as e:
            print(f"Warning: Could not check existing data: {e}")
            existing_sources = set()

        # Filter out files that are already in the DB
        new_file_paths = [p for p in file_paths if p not in existing_sources]
        
        if len(new_file_paths) < len(file_paths):
            print(f"Skipping {len(file_paths) - len(new_file_paths)} files already ingested.")
            
        if not new_file_paths:
            print("No new files to ingest.")
            return

        documents = []
        for path in tqdm(new_file_paths, desc="Loading files"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    if text.strip(): # Skip empty files
                        documents.append(Document(page_content=text, metadata={"source": path}))
            except Exception as e:
                print(f"Error reading {path}: {e}")

        if not documents:
            print("No documents loaded.")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        
        print(f"Ingesting {len(splits)} chunks...")
        
        # Batch size for adding to Chroma (prevents hitting memory limits)
        batch_size = 500
        for i in tqdm(range(0, len(splits), batch_size), desc="Ingesting batches"):
            batch = splits[i:i + batch_size]
            self.vectorstore.add_documents(batch)
            
        print("Ingestion complete.")

    def setup_retriever(self):
        """Setup Hybrid Retriever (Vector + BM25)"""
        # 1. Vector Retriever
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # 2. BM25 Retriever
        # Fetch existing docs from Chroma to build BM25 index
        existing_data = self.vectorstore.get()
        all_texts = existing_data['documents']
        
        if not all_texts:
            print("Vector store is empty. Please ingest data first.")
            return None
            
        print(f"Initializing BM25 with {len(all_texts)} documents...")
        bm25_retriever = BM25Retriever.from_texts(all_texts)
        bm25_retriever.k = 3

        # 3. Ensemble
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )
        return self.retriever

    def query(self, question: str):
        if not self.retriever:
            print("Initializing retriever...")
            self.setup_retriever()
            
        if not self.retriever:
            return []

        return self.retriever.invoke(question)
    
    
# import json
# import requests
# import concurrent.futures
# import time
# import random
# import os
# from typing import List
# from tqdm import tqdm
# from langchain_chroma import Chroma
# from langchain_community.retrievers import BM25Retriever
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
# from langchain_core.embeddings import Embeddings
# from sentence_transformers import CrossEncoder

# class VNPTEmbeddings(Embeddings):
#     def __init__(self, api_key_file="api-keys.json"):
#         self.api_key_file = api_key_file
#         self._load_credentials()

#     def _load_credentials(self):
#         with open(self.api_key_file, 'r') as f:
#             api_keys = json.load(f)
        
#         key_info = next((item for item in api_keys if item["llmApiName"] == "LLM embedings"), None)
#         if not key_info:
#             raise ValueError("Embedding API key not found")
            
#         self.headers = {
#             'Authorization': key_info['authorization'],
#             'Token-id': key_info['tokenId'],
#             'Token-key': key_info['tokenKey'],
#             'Content-Type': 'application/json',
#         }

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
#             embeddings = list(tqdm(
#                 executor.map(self.embed_query, texts), 
#                 total=len(texts), 
#                 desc="Embedding batch", 
#                 leave=False
#             ))
#         return embeddings

#     def embed_query(self, text: str) -> List[float]:
#         json_data = {
#             'model': 'vnptai_hackathon_embedding',
#             'input': text,
#             'encoding_format': 'float'
#         }
        
#         max_retries = 5
#         for attempt in range(max_retries):
#             try:
#                 response = requests.post(
#                     'https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding',
#                     headers=self.headers,
#                     json=json_data
#                 )
#                 response.raise_for_status()
#                 return response.json()['data'][0]['embedding']
#             except Exception as e:
#                 if attempt == max_retries - 1:
#                     print(f"Error embedding text: {e}")
#                     return []
#                 sleep_time = (2 ** attempt) + random.uniform(0, 1)
#                 time.sleep(sleep_time)
#         return []

# class LangChainRAG:
#     def __init__(self, vector_db_path="./vector_db_langchain"):
#         self.embedding_model = VNPTEmbeddings()
        
#         self.vector_db_path = vector_db_path
#         self.vectorstore = Chroma(
#             persist_directory=vector_db_path,
#             embedding_function=self.embedding_model,
#             collection_name="vnpt_knowledge_base"
#         )
        
#         print("Loading Reranker model...")
#         self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
#         self.bm25_retriever = None

#     def ingest_data(self, file_paths: List[str]):
#         try:
#             existing_data = self.vectorstore.get(include=['metadatas'])
#             existing_sources = set()
#             if existing_data and 'metadatas' in existing_data:
#                 for metadata in existing_data['metadatas']:
#                     if metadata and 'source' in metadata:
#                         existing_sources.add(metadata['source'])
#             print(f"Found {len(existing_sources)} existing sources.")
#         except Exception as e:
#             print(f"Warning checking existing data: {e}")
#             existing_sources = set()

#         new_file_paths = [p for p in file_paths if p not in existing_sources]
        
#         if not new_file_paths:
#             print("No new files to ingest.")
#             return

#         documents = []
#         for path in tqdm(new_file_paths):
#             try:
#                 with open(path, "r", encoding="utf-8") as f:
#                     text = f.read()
#                     if text.strip():
#                         documents.append(Document(page_content=text, metadata={"source": path}))
#             except Exception as e:
#                 print(f"Error reading {path}: {e}")

#         if not documents:
#             return

#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=100,
#             separators=["\n\n", "\n", ". ", " ", ""]
#         )
#         splits = text_splitter.split_documents(documents)
        
#         batch_size = 500
#         for i in tqdm(range(0, len(splits), batch_size)):
#             batch = splits[i:i + batch_size]
#             self.vectorstore.add_documents(batch)

#     def _setup_bm25(self):
#         if self.bm25_retriever:
#             return
            
#         existing_data = self.vectorstore.get()
#         all_texts = existing_data['documents']
        
#         if all_texts:
#             self.bm25_retriever = BM25Retriever.from_texts(all_texts)
#             self.bm25_retriever.k = 10

#     def query(self, question: str):
#         self._setup_bm25()
        
#         vector_docs = self.vectorstore.similarity_search(question, k=10)
        
#         bm25_docs = []
#         if self.bm25_retriever:
#             bm25_docs = self.bm25_retriever.invoke(question)
            
#         unique_docs = {}
#         for doc in vector_docs + bm25_docs:
#             if doc.page_content not in unique_docs:
#                 unique_docs[doc.page_content] = doc
        
#         candidate_docs = list(unique_docs.values())
        
#         if not candidate_docs:
#             return []

#         pairs = [[question, doc.page_content] for doc in candidate_docs]
        
#         scores = self.reranker.predict(pairs)
        
#         ranked_results = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
        
#         final_docs = [doc for doc, score in ranked_results[:3]]
        
#         return final_docs
import os
import json
import numpy as np
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings

class RAGPipeline:
    def __init__(self, vector_db_path="./vector_db", embedding_api_func=None):
        """
        Initialize the RAG pipeline.
        :param vector_db_path: Path to the persistent ChromaDB.
        :param embedding_api_func: Function to call the VNPT embedding API.
        """
        self.embedding_api_func = embedding_api_func
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=vector_db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="knowledge_base")
        
        # Initialize BM25 (In a real scenario, you'd load a pre-built index)
        self.bm25 = None
        self.documents = [] # Keep track of documents for BM25
        
    def load_bm25_index(self, documents):
        """
        Load or build BM25 index from documents.
        :param documents: List of text strings.
        """
        self.documents = documents
        tokenized_corpus = [doc.split(" ") for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def vector_search(self, query, top_k=3):
        """
        Perform vector search using ChromaDB.
        """
        if not self.embedding_api_func:
            return []
            
        query_embedding = self.embedding_api_func(query)
        if not query_embedding:
            return []
            
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        documents = results['documents'][0] if results['documents'] else []
        return documents

    def keyword_search(self, query, top_k=3):
        """
        Perform keyword search using BM25.
        """
        if not self.bm25:
            return []
            
        tokenized_query = query.split(" ")
        return self.bm25.get_top_n(tokenized_query, self.documents, n=top_k)

    def hybrid_search(self, query, top_k=5):
        """
        Combine Vector and Keyword search results.
        """
        # 1. Get results from both sources
        vector_results = self.vector_search(query, top_k=top_k)
        keyword_results = self.keyword_search(query, top_k=top_k)
        
        # 2. Deduplicate and Merge (Simple strategy: Interleave or Union)
        # For now, let's just use a set to deduplicate strings
        combined_results = list(set(vector_results + keyword_results))
        
        # 3. Return top_k
        return combined_results[:top_k]

    def add_documents(self, documents, metadatas=None, ids=None):
        """
        Add documents to the ChromaDB collection.
        :param documents: List of text strings.
        :param metadatas: List of metadata dictionaries.
        :param ids: List of unique IDs.
        """
        if not self.embedding_api_func:
            print("Error: No embedding function provided.")
            return

        # Generate embeddings in batches to avoid hitting API limits or timeouts
        batch_size = 10
        total_docs = len(documents)
        
        for i in range(0, total_docs, batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size] if metadatas else None
            batch_ids = ids[i : i + batch_size] if ids else [f"doc_{j}" for j in range(i, i + len(batch_docs))]
            
            batch_embeddings = []
            for doc in batch_docs:
                emb = self.embedding_api_func(doc)
                if emb:
                    batch_embeddings.append(emb)
                else:
                    # Handle failure (e.g., retry or skip)
                    # For now, we'll append a zero vector or skip. 
                    # Better to skip to avoid noise, but need to align lists.
                    # Let's skip this document in this simple implementation
                    continue
            
            # Only add if we have valid embeddings
            if batch_embeddings:
                # Filter docs/ids/metadatas to match successful embeddings
                # This logic is a bit complex for a simple loop, let's assume 
                # embedding_api_func is robust or returns None on failure.
                # Re-aligning lists:
                valid_indices = [k for k, emb in enumerate(batch_embeddings) if emb is not None] # Wait, my loop above appends only valid.
                
                # Actually, let's rewrite the loop to be safer
                final_docs = []
                final_embeddings = []
                final_ids = []
                final_metadatas = [] if batch_metadatas else None
                
                for idx, doc in enumerate(batch_docs):
                    emb = self.embedding_api_func(doc)
                    if emb:
                        final_docs.append(doc)
                        final_embeddings.append(emb)
                        final_ids.append(batch_ids[idx])
                        if batch_metadatas:
                            final_metadatas.append(batch_metadatas[idx])
                
                if final_docs:
                    self.collection.add(
                        documents=final_docs,
                        embeddings=final_embeddings,
                        metadatas=final_metadatas,
                        ids=final_ids
                    )
                    print(f"Added batch {i} to {i+len(final_docs)}")

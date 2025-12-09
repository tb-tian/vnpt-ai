import json
import csv
import requests
import time
import re
import os
from rag_pipeline import RAGPipeline
from prompt_templates import construct_prompt, SYSTEM_PROMPT

# Load API Key (Adjust path as necessary)
try:
    with open('../api-keys.json', 'r') as f:
        keys = json.load(f)
        API_KEY = keys.get("VNPT_API_KEY", "")
except FileNotFoundError:
    API_KEY = os.environ.get("VNPT_API_KEY", "")

API_URL = "https://llm.vnpt.ai/api/v1/chat/completions"
EMBEDDING_URL = "https://llm.vnpt.ai/api/v1/embeddings" # Hypothetical URL, check docs

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
    try:
        response = requests.post(EMBEDDING_URL, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['data'][0]['embedding']
        else:
            print(f"Embedding Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Embedding Exception: {e}")
        return None

# Initialize RAG Pipeline
rag = RAGPipeline(embedding_api_func=get_embedding)
# Note: In a real run, you would load documents here:
# rag.load_bm25_index(["doc1", "doc2"...]) 

def call_llm(prompt, model="vnptai_hackathon_large"):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 10
    }
    
    for _ in range(3):
        try:
            response = requests.post(API_URL, headers=headers, json=data)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            elif response.status_code == 429:
                time.sleep(2)
            else:
                print(f"LLM Error: {response.status_code} - {response.text}")
                break
        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(1)
    return ""

def solve_question(item):
    question_text = item['question']
    choices = item['choices']
    
    # 1. Check for Context in Question
    context = ""
    if "Đoạn thông tin" in question_text:
        # Extract context (simple split, can be improved with regex)
        parts = question_text.split("Câu hỏi:")
        if len(parts) > 1:
            context = parts[0].strip()
            question_text = parts[-1].strip() # Update question to just the query part
    else:
        # 2. If no context, use RAG
        retrieved_docs = rag.hybrid_search(question_text)
        context = "\n\n".join(retrieved_docs)
        
    # 3. Construct Prompt
    prompt = construct_prompt(question_text, choices, context)
    
    # 4. Call LLM
    raw_answer = call_llm(prompt)
    
    # 5. Post-process answer (Extract A, B, C, D)
    # Simple regex to find the first capital letter
    match = re.search(r'[A-Z]', raw_answer)
    return match.group(0) if match else "A" # Default to A if fail

def main():
    # Example usage with val.json
    input_path = 'AInicorns_TheBuilder_public_v1.1/data/val.json'
    output_path = 'output.csv'
    
    if os.path.exists(input_path):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        results = []
        # Process a few items for testing
        for item in data[:5]: 
            ans = solve_question(item)
            print(f"Q: {item['qid']} - Ans: {ans}")
            results.append([item['qid'], ans])
            
        # Save to CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['qid', 'answer'])
            writer.writerows(results)

if __name__ == "__main__":
    main()
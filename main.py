import json
import csv
import re
import os
from rag_pipeline import RAGPipeline
from prompt_templates import construct_prompt, SYSTEM_PROMPT
from get_response import get_response
from get_embedding import get_embedding

# Initialize RAG Pipeline
rag = RAGPipeline(embedding_api_func=get_embedding)
# Note: In a real run, you would load documents here:
# rag.load_bm25_index(["doc1", "doc2"...]) 

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
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    try:
        raw_answer = get_response(messages)
    except Exception as e:
        print(f"Error calling LLM: {e}")
        raw_answer = ""
    
    # 5. Post-process answer (Extract A, B, C, D)
    # Simple regex to find the first capital letter
    match = re.search(r'[A-Z]', raw_answer)
    return match.group(0) if match else "A" # Default to A if fail

def main():
    # Example usage with val.json
    input_path = 'AInicorns_TheBuilder_public_v1.1/data/test.json'
    output_path = 'output.csv'
    
    if os.path.exists(input_path):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        results = []
        # Process a few items for testing
        for item in data: 
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
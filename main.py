import json
import csv
import re
import os
from rag_langchain import LangChainRAG
from prompt_templates import construct_prompt, SYSTEM_PROMPT, construct_batch_prompt, BATCH_SYSTEM_PROMPT
from get_response import get_response
from get_embedding import get_embedding

# Initialize RAG Pipeline
rag = LangChainRAG()
# Ensure retriever is ready (loads BM25 + Vector)
rag.setup_retriever()


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
        retrieved_docs = rag.query(question_text)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
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

def solve_batch(items):
    # 1. Prepare items with context
    prepared_items = []
    for item in items:
        question_text = item['question']
        choices = item['choices']
        
        # Check for Context in Question
        context = ""
        if "Đoạn thông tin" in question_text:
            parts = question_text.split("Câu hỏi:")
            if len(parts) > 1:
                context = parts[0].strip()
                question_text = parts[-1].strip()
        else:
            # If no context, use RAG
            retrieved_docs = rag.query(question_text)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
        prepared_items.append({
            'question': question_text,
            'choices': choices,
            'context': context,
            'qid': item['qid']
        })

    # 2. Construct Batch Prompt
    prompt = construct_batch_prompt(prepared_items)
    
    # 3. Call LLM
    messages = [
        {"role": "system", "content": BATCH_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    answers = {}
    # Retry batch solving once before falling back
    for attempt in range(2):
        try:
            raw_answer = get_response(messages)
            # Clean up potential markdown code blocks
            raw_answer = raw_answer.replace("```json", "").replace("```", "").strip()
            answers = json.loads(raw_answer)
            break # Success, exit retry loop
        except Exception as e:
            print(f"Batch attempt {attempt + 1} failed: {e}")
            print(f"Raw answer: {raw_answer if 'raw_answer' in locals() else 'None'}")
            
            # If this was the last attempt, fall back to individual solving
            if attempt == 1:
                print("Falling back to individual solving...")
                for i, item in enumerate(items, 1):
                    try:
                        ans = solve_question(item)
                        answers[str(i)] = ans
                    except Exception as inner_e:
                        print(f"Error solving individual question {item.get('qid')}: {inner_e}")
                        answers[str(i)] = "A"
            
    # 4. Map answers to QIDs
    results = {}
    for i, item in enumerate(items, 1):
        # Get answer for question i (as string "1", "2"...)
        ans = answers.get(str(i), "A")
        # Ensure answer is valid (A, B, C, D...)
        if not ans or ans not in "ABCDE":
             ans = "A"
        results[item['qid']] = ans
        
    return results

def main():
    # Example usage with val.json
    input_path = 'data/test.json'
    output_path = 'output.csv'
    
    if os.path.exists(input_path):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Open CSV file for writing
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['qid', 'answer'])

            # Process in batches
            batch_size = 10
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1} ({len(batch)} items)...")
                
                results = solve_batch(batch)
                
                for qid, ans in results.items():
                    print(f"Q: {qid} - Ans: {ans}")
                    writer.writerow([qid, ans])
                
                f.flush()

if __name__ == "__main__":
    main()
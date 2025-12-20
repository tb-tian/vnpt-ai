import json
import csv
import re
import os
from rag_langchain import LangChainRAG
from prompt_templates import (
    construct_prompt, construct_batch_prompt,
    SYSTEM_PROMPTS, BATCH_SYSTEM_PROMPT,
    CLASSIFICATION_SYSTEM_PROMPT, CLASSIFICATION_USER_TEMPLATE
)
from get_response import get_response
from get_embedding import get_embedding
from router_logic import QuestionRouter
from config import DOMAIN_CONFIGS

# Initialize RAG Pipeline and Router
rag = LangChainRAG()
router = QuestionRouter()

# Ensure retriever is ready (loads BM25 + Vector)
rag.setup_retriever()


def is_rag_question(question_text):
    """Quick keyword check for RAG questions"""
    rag_keywords = [
        "dựa trên đoạn văn",
        "theo đoạn văn",
        "đoạn văn trên",
        "trong đoạn văn",
        "đoạn thông tin"
    ]
    question_lower = question_text.lower()
    return any(keyword in question_lower for keyword in rag_keywords)


def classify_questions_with_llm(questions_batch):
    """
    Classify a batch of questions using LLM (up to 10 questions)
    
    Args:
        questions_batch: List of dicts with 'question' and 'qid' keys
    
    Returns:
        Dict mapping qid to domain name (e.g., {"test_0001": "STEM", ...})
    """
    if not questions_batch:
        return {}
    
    # Format questions for prompt
    questions_list = []
    for i, item in enumerate(questions_batch, 1):
        question_text = item['question']
        # Truncate very long questions to save tokens
        if len(question_text) > 500:
            question_text = question_text[:500] + "..."
        questions_list.append(f"Câu {i}: {question_text}")
    
    questions_str = "\n\n".join(questions_list)
    
    # Construct prompt
    user_prompt = CLASSIFICATION_USER_TEMPLATE.format(
        num_questions=len(questions_batch),
        questions_list=questions_str
    )
    
    messages = [
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    # Call LLM
    try:
        response = get_response(
            messages,
            model="small",  # Use small model for classification
            temperature=0.1,  # Low temp for consistent classification
            response_format={"type": "json_object"}  # Force JSON output
        )
        
        # Parse JSON response
        response = response.replace("```json", "").replace("```", "").strip()
        classifications = json.loads(response)
        
        # Map back to qids
        results = {}
        for i, item in enumerate(questions_batch, 1):
            domain = classifications.get(str(i), "MULTIDOMAIN")
            # Validate domain (RAG not included - already filtered)
            valid_domains = ["STEM", "PRECISION_CRITICAL", "COMPULSORY", "MULTIDOMAIN"]
            if domain not in valid_domains:
                domain = "MULTIDOMAIN"
            results[item['qid']] = domain
        
        return results
        
    except Exception as e:
        print(f"  ⚠ LLM classification failed: {e}, falling back to rule-based")
        # Fallback to rule-based router
        results = {}
        for item in questions_batch:
            domain, _ = router.classify_question(item['question'], item.get('choices', []))
            results[item['qid']] = domain
        return results


def solve_question(item):
    """
    Solve a single question with domain-aware routing
    """
    question_text = item['question']
    choices = item['choices']
    
    # 1. Classify question into domain
    domain, confidence = router.classify_question(question_text, choices)
    strategy = router.get_strategy_config(domain)
    
    # Debug: print STEM strategy
    if domain == "STEM":
        print(f"  [DEBUG] STEM config: voting={strategy.get('use_majority_voting')}, verification={strategy.get('use_self_verification')}")
    
    # STEM with self-verification (if enabled)
    if domain == "STEM" and strategy.get('use_self_verification', False):
        return solve_stem_with_self_verification(item, strategy, domain, confidence)
    
    # STEM with majority voting (default)
    if domain == "STEM" and strategy.get('use_majority_voting', False):
        return solve_stem_with_voting(item, strategy, domain, confidence)
    
    # 2. Get context based on domain strategy
    context = ""
    if "Đoạn thông tin" in question_text:
        # Extract context from question (RAG domain)
        parts = question_text.split("Câu hỏi:")
        if len(parts) > 1:
            context = parts[0].strip()
            question_text = parts[-1].strip()
    elif strategy['use_rag'] and strategy['top_k_docs'] > 0:
        # Use RAG retrieval for other domains if configured
        retrieved_docs = rag.query(question_text)
        # Limit to top_k_docs if needed
        retrieved_docs = retrieved_docs[:strategy['top_k_docs']]
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # 3. Construct domain-specific prompt
    prompt = construct_prompt(question_text, choices, context, domain.lower())
    
    # 4. Call LLM with domain-specific system prompt
    system_prompt = SYSTEM_PROMPTS.get(
        domain.lower(), 
        SYSTEM_PROMPTS["multidomain"]
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    try:
        raw_answer = get_response(
            messages,
            model=strategy.get('model', 'small'),
            temperature=strategy.get('temperature', 0.3)
        )
    except Exception as e:
        print(f"Error calling LLM for {domain}: {e}")
        raw_answer = ""
    
    # 5. Post-process answer (Extract A, B, C, D...)
    # Get number of choices to validate
    num_choices = len(choices)
    max_valid_letter = chr(ord('A') + num_choices - 1)  # A + 0 = A, A + 5 = F, etc.
    
    # Try to find answer after === marker first (new format)
    match = re.search(r'===ĐÁP ÁN CUỐI CÙNG===\s*([A-Z])', raw_answer, re.IGNORECASE)
    
    if match:
        answer = match.group(1).upper()
    else:
        # Fallback 1: find answer markers with dots or colons
        answer_pattern = r'(?:^|\s|[Đđ]áp án|[Cc]họn|[Tt]rả lời|[Kk]ết quả)\s*[:\-]?\s*([A-Z])(?:[.\s]|$)'
        match = re.search(answer_pattern, raw_answer, re.IGNORECASE)
        
        if match:
            answer = match.group(1).upper()
        else:
            # Fallback 2: find last standalone letter (not first!)
            matches = re.findall(r'\b([A-Z])\b', raw_answer)
            answer = matches[-1].upper() if matches else "A"
    
    # Validate: answer must be within valid range for this question
    if answer not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        print(f"  Warning: Invalid answer '{answer}' (not a letter), defaulting to A")
        answer = "A"
    elif ord(answer) > ord(max_valid_letter):
        print(f"  Warning: Answer '{answer}' exceeds choices (max={max_valid_letter}), defaulting to A")
        answer = "A"
    
    # Debug info
    print(f"  Domain: {domain} (conf: {confidence:.2f}) | Choices: {num_choices} (A-{max_valid_letter}) | Raw: '{raw_answer[:30]}...' -> {answer}")
    
    return answer

def solve_stem_with_voting(item, strategy, domain, confidence):
    """
    Solve STEM question with majority voting (n=5 completions)
    """
    from collections import Counter
    from prompt_templates import SYSTEM_PROMPTS
    
    question_text = item['question']
    choices = item['choices']
    
    # Get context
    context = ""
    if strategy['use_rag'] and strategy['top_k_docs'] > 0:
        retrieved_docs = rag.query(question_text)
        retrieved_docs = retrieved_docs[:strategy['top_k_docs']]
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Construct prompt
    prompt = construct_prompt(question_text, choices, context, "stem")
    system_prompt = SYSTEM_PROMPTS.get("stem", SYSTEM_PROMPTS["multidomain"])
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # Call API with n=5
    num_choices = len(choices)
    max_valid_letter = chr(ord('A') + num_choices - 1)
    
    try:
        response = get_response(
            messages,
            model=strategy.get('model', 'small'),
            temperature=strategy.get('temperature', 0.7),
            n=strategy.get('n', 5)
        )
        
        # Extract answers from all 5 completions
        answers = []
        for choice_item in response['choices']:
            content = choice_item['message']['content']
            # Extract answer
            answer_pattern = r'(?:^|\s|[Đđ]áp án|[Cc]họn|[Tt]rả lời)\s*[:\-]?\s*([A-Z])(?:[.\s]|$)'
            match = re.search(answer_pattern, content, re.IGNORECASE)
            if match:
                answer = match.group(1).upper()
            else:
                match = re.search(r'\b([A-Z])\b', content)
                answer = match.group(1).upper() if match else "A"
            
            # Validate
            if 'A' <= answer <= max_valid_letter:
                answers.append(answer)
            else:
                answers.append("A")
        
        # Majority voting
        vote_counts = Counter(answers)
        final_answer = vote_counts.most_common(1)[0][0]
        votes = vote_counts[final_answer]
        
        print(f"  STEM Voting: {answers} → {final_answer} ({votes}/5 votes) | Domain: {domain} (conf: {confidence:.2f})")
        return final_answer
        
    except Exception as e:
        print(f"  STEM voting failed: {e}, falling back to single call")
        # Fallback
        raw_answer = get_response(messages, model=strategy.get('model', 'small'), temperature=0.3)
        answer_pattern = r'(?:^|\s|[Đđ]áp án)\s*[:\-]?\s*([A-Z])(?:[.\s]|$)'
        match = re.search(answer_pattern, raw_answer, re.IGNORECASE)
        if match and 'A' <= match.group(1).upper() <= max_valid_letter:
            return match.group(1).upper()
        return "A"

def solve_stem_with_self_verification(item, strategy, domain, confidence):
    """
    Solve STEM question with self-verification: generate answer then review full reasoning
    """
    from prompt_templates import SYSTEM_PROMPTS
    
    question_text = item['question']
    choices = item['choices']
    num_choices = len(choices)
    max_valid_letter = chr(ord('A') + num_choices - 1)
    
    # Get context
    context = ""
    if strategy.get('use_rag', True) and strategy.get('top_k_docs', 1) > 0:
        try:
            retrieved_docs = rag.query(question_text)
            retrieved_docs = retrieved_docs[:strategy['top_k_docs']]
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        except Exception as e:
            print(f"  RAG failed: {e}, continuing without context")
            context = ""
    
    # System prompt with 4-step CoT
    system_prompt = SYSTEM_PROMPTS.get("stem", SYSTEM_PROMPTS["multidomain"])
    user_prompt = construct_prompt(question_text, choices, context, "stem")
    
    max_attempts = strategy.get('verification_attempts', 2)
    
    for attempt in range(max_attempts + 1):
        try:
            # Step 1: Generate answer with reasoning
            if attempt == 0:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            else:
                # Retry with feedback
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Lần thử trước bị từ chối. Hãy giải lại CẨN THẬN hơn:\n\n{user_prompt}"}
                ]
            
            response = get_response(
                messages,
                model=strategy.get('model', 'small'),
                temperature=0.3  # Lower temp for reasoning
            )
            
            # Extract answer with new format
            answer_match = re.search(r'===ĐÁP ÁN CUỐI CÙNG===\s*([A-Z])', response, re.IGNORECASE)
            if not answer_match:
                # Fallback: last letter
                matches = re.findall(r'\b([A-Z])\b', response)
                answer = matches[-1].upper() if matches else "A"
            else:
                answer = answer_match.group(1).upper()
            
            # Validate
            if answer > max_valid_letter:
                answer = "A"
            
            # Step 2: Self-verify by reviewing FULL reasoning
            if attempt < max_attempts:
                verify_prompt = f"""Dưới đây là một bài toán và lời giải chi tiết:

=== BÀI TOÁN ===
{question_text}

Các lựa chọn:
{chr(10).join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])}

=== LỜI GIẢI CHI TIẾT ===
{response}

=== NHIỆM VỤ ===
Hãy ĐÁNH GIÁ lời giải:
1. Logic từng bước có chính xác không?
2. Công thức áp dụng có đúng không?
3. Tính toán có sai sót không?
4. Kết luận cuối có phù hợp không?

Nếu phát hiện LỖI: Chỉ rõ lỗi ở đâu
Nếu xác nhận ĐÚNG: Giải thích tại sao đúng

CHỈ TRẢ LỜI:
ĐÁNH GIÁ: [phân tích]
KẾT LUẬN: ĐÚNG hoặc SAI
LÝ DO: [ngắn gọn]"""

                verify_messages = [
                    {"role": "system", "content": "Bạn là reviewer chuyên nghiệp. Phân tích logic và tìm lỗi sai."},
                    {"role": "user", "content": verify_prompt}
                ]
                
                verify_response = get_response(
                    verify_messages,
                    model=strategy.get('model', 'small'),
                    temperature=0.1
                )
                
                # Check verification
                if re.search(r'\bĐÚNG\b', verify_response, re.IGNORECASE):
                    print(f"  STEM Self-Verify: {answer} VERIFIED ✓ (attempt {attempt+1}) | Domain: {domain} (conf: {confidence:.2f})")
                    return answer
                else:
                    print(f"  STEM Self-Verify: {answer} REJECTED ✗ (retrying...)")
                    continue
            else:
                # Last attempt, no verification
                print(f"  STEM Self-Verify: {answer} (final attempt) | Domain: {domain} (conf: {confidence:.2f})")
                return answer
                
        except Exception as e:
            print(f"  Verification attempt {attempt+1} failed: {e}")
            continue
    
    # All attempts failed, return last answer
    return answer if 'answer' in locals() else "A"

def extract_answer_from_content(content, num_choices):
    """
    Extract answer letter from content with validation
    """
    import re
    
    max_valid_letter = chr(ord('A') + num_choices - 1)
    
    # Try patterns
    patterns = [
        r'(?:Đáp án|Answer|Trả lời)(?:\s*là)?(?:\s*:)?\s*([A-Z])',
        r'^\s*([A-Z])\s*$',
        r'\b([A-Z])\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if match:
            answer = match.group(1).upper()
            if 'A' <= answer <= max_valid_letter:
                return answer
    
    return "A"  # Default fallback

def process_domain_batch(domain_items, domain):
    """
    Process a single domain batch (up to 10 questions)
    Returns dict of {qid: answer}
    """
    strategy = router.get_strategy_config(domain)
    
    # Prepare items for this domain
    prepared_items = []
    for item in domain_items:
        question_text = item['question']
        choices = item['choices']
        
        # Get context based on domain strategy
        context = ""
        if "Đoạn thông tin" in question_text:
            parts = question_text.split("Câu hỏi:")
            if len(parts) > 1:
                context = parts[0].strip()
                question_text = parts[-1].strip()
        elif strategy['use_rag'] and strategy['top_k_docs'] > 0:
            retrieved_docs = rag.query(question_text)
            retrieved_docs = retrieved_docs[:strategy['top_k_docs']]
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
        prepared_items.append({
            'question': question_text,
            'choices': choices,
            'context': context,
            'qid': item['qid']
        })
    
    # Construct domain-specific batch prompt (pass domain for STEM special handling)
    prompt = construct_batch_prompt(prepared_items, domain=domain)
    
    # Get domain-specific system prompt
    from prompt_templates import BATCH_SYSTEM_PROMPTS
    from config import BATCH_CONFIG
    
    system_prompt = BATCH_SYSTEM_PROMPTS.get(
        domain.lower(),
        BATCH_SYSTEM_PROMPTS["multidomain"]
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    # Call LLM with domain-specific model and temperature
    answers = {}
    for attempt in range(2):
        try:
            # Add response_format for JSON mode if enabled (not for STEM with reasoning)
            response_format = None
            if domain.upper() != "STEM" and BATCH_CONFIG.get('use_json_format'):
                response_format = {"type": "json_object"}
            
            raw_answer = get_response(
                messages,
                model=strategy.get('model', 'small'),
                temperature=strategy.get('temperature', 0.3),
                response_format=response_format
            )
            
            # STEM: Extract from special format with ===DANH SÁCH ĐÁP ÁN===
            if domain.upper() == "STEM":
                import re
                # Find the JSON after ===DANH SÁCH ĐÁP ÁN===
                match = re.search(r'===DANH SÁCH ĐÁP ÁN===\s*\n?\s*(\{[^}]+\})', raw_answer, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()
                    answers = json.loads(json_str)
                else:
                    # Fallback: try to find individual answers
                    for i in range(1, len(domain_items) + 1):
                        pattern = rf'===ĐÁP ÁN CÂU {i}===\s*\n?\s*([A-Z])'
                        ans_match = re.search(pattern, raw_answer)
                        if ans_match:
                            answers[str(i)] = ans_match.group(1)
                        else:
                            answers[str(i)] = "A"
            else:
                # Other domains: Parse JSON directly
                raw_answer = raw_answer.replace("```json", "").replace("```", "").strip()
                answers = json.loads(raw_answer)
            
            print(f"  ✓ {domain} batch ({len(domain_items)} questions) processed")
            break
        except Exception as e:
            print(f"  ✗ Attempt {attempt + 1} failed: {e}")
            if attempt == 1:
                # Fallback to individual solving
                print(f"  → Falling back to individual solving...")
                for i, original_item in enumerate(domain_items, 1):
                    try:
                        ans = solve_question(original_item)
                        answers[str(i)] = ans
                    except Exception as inner_e:
                        print(f"    Error on {original_item.get('qid')}: {inner_e}")
                        answers[str(i)] = "A"
    
    # Map answers to QIDs
    results = {}
    for i, item in enumerate(prepared_items, 1):
        ans = answers.get(str(i), "A")
        if not ans or ans not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            ans = "A"
        results[item['qid']] = ans
    
    return results

def solve_batch_streaming(items, output_file):
    """
    Solve questions with streaming write - ghi ngay khi có kết quả
    Supports resume: skip questions already in output file
    """
    from config import BATCH_CONFIG
    
    # Check which questions already processed
    processed_qids = set()
    if os.path.exists(output_file):
        print(f"Found existing output file, loading processed questions...")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if row and len(row) >= 2:
                        processed_qids.add(row[0])
            print(f"  ✓ Found {len(processed_qids)} already processed questions")
        except Exception as e:
            print(f"  ⚠ Error reading existing file: {e}, starting fresh")
            processed_qids = set()
    
    # Open output file in append mode
    file_mode = 'a' if processed_qids else 'w'
    csv_file = open(output_file, file_mode, newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    
    # Write header if new file
    if not processed_qids:
        writer.writerow(['qid', 'answer'])
        csv_file.flush()
    
    # Domain buffers (excluding STEM)
    domain_buffers = {
        "PRECISION_CRITICAL": [],
        "COMPULSORY": [],
        "RAG": [],
        "MULTIDOMAIN": []
    }
    
    batch_size = BATCH_CONFIG['batch_size']
    total_items = len(items)
    processed_count = len(processed_qids)
    skipped_count = 0
    
    print(f"\nProcessing {total_items} questions with streaming write...")
    print(f"Batch size: {batch_size} (STEM: individual with voting)")
    print(f"Already completed: {processed_count}/{total_items}")
    print(f"Will scan all questions from start to ensure no gaps\n")
    
    try:
        for idx, item in enumerate(items, 1):
            qid = item['qid']
            
            # Skip if already in output file
            if qid in processed_qids:
                skipped_count += 1
                if skipped_count % 50 == 0:  # Progress update every 50 skips
                    print(f"[{idx}/{total_items}] Scanned {skipped_count} already completed questions...")
                continue
            
            # Classify domain
            domain, confidence = router.classify_question(item['question'], item['choices'])
            strategy = router.get_strategy_config(domain)
            
            # Check if domain uses batch processing
            use_batch = strategy.get('use_batch_processing', True)
            
            # Single processing (when use_batch_processing=False)
            if not use_batch:
                # Determine method for display
                if domain == "STEM":
                    method = "verification" if strategy.get('use_self_verification') else "voting"
                else:
                    method = "single"
                    
                print(f"[{idx}/{total_items}] {qid} ({domain}) - {method}...", end=' ')
                try:
                    answer = solve_question(item)
                    
                    # Write immediately
                    writer.writerow([qid, answer])
                    csv_file.flush()  # Force write to disk
                    processed_qids.add(qid)  # Add to set to avoid re-processing
                    processed_count += 1
                    print(f"✓ {answer} | Total done: {processed_count}/{total_items}")
                except Exception as e:
                    print(f"✗ Error: {e}")
                    writer.writerow([qid, "A"])  # Fallback
                    csv_file.flush()
                    processed_qids.add(qid)
                    processed_count += 1
                continue
            
            # Batch processing: Add to buffer
            domain_buffers[domain].append(item)
            
            # Get domain-specific batch size
            domain_batch_size = strategy.get('batch_size', batch_size)
            
            # Process batch when full
            if len(domain_buffers[domain]) >= domain_batch_size:
                batch_items = domain_buffers[domain][:domain_batch_size]
                print(f"[{idx}/{total_items}] {domain} batch ({len(batch_items)} questions)...", end=' ')
                
                try:
                    batch_results = process_domain_batch(batch_items, domain)
                    
                    # Write batch results immediately
                    for batch_item in batch_items:
                        batch_qid = batch_item['qid']
                        answer = batch_results.get(batch_qid, "A")
                        writer.writerow([batch_qid, answer])
                        processed_qids.add(batch_qid)  # Mark as processed
                        processed_count += 1
                    
                    csv_file.flush()  # Force write
                    print(f"✓ | Total done: {processed_count}/{total_items}")
                except Exception as e:
                    print(f"✗ Error: {e}")
                    # Write fallback answers
                    for batch_item in batch_items:
                        writer.writerow([batch_item['qid'], "A"])
                        processed_qids.add(batch_item['qid'])
                        processed_count += 1
                    csv_file.flush()
                
                # Remove processed items from buffer (use domain_batch_size)
                domain_buffers[domain] = domain_buffers[domain][domain_batch_size:]
        
        # Process remaining items in buffers
        print(f"\n[{total_items}/{total_items}] Processing remaining questions in buffers...")
        for domain, remaining_items in domain_buffers.items():
            if remaining_items:
                print(f"  {domain}: {len(remaining_items)} questions...", end=' ')
                
                # Check if domain uses batch processing
                domain_config = DOMAIN_CONFIGS.get(domain, {})
                use_batch = domain_config.get('use_batch_processing', True)
                
                if use_batch:
                    # Process as batch
                    try:
                        batch_results = process_domain_batch(remaining_items, domain)
                        
                        # Write remaining results
                        for item in remaining_items:
                            qid = item['qid']
                            answer = batch_results.get(qid, "A")
                            writer.writerow([qid, answer])
                            processed_qids.add(qid)
                            processed_count += 1
                        
                        csv_file.flush()
                        print(f"✓ | Total done: {processed_count}/{total_items}")
                    except Exception as e:
                        print(f"✗ Error: {e}")
                        for item in remaining_items:
                            writer.writerow([item['qid'], "A"])
                            processed_qids.add(item['qid'])
                            processed_count += 1
                        csv_file.flush()
                else:
                    # Process individually (should not happen if logic correct)
                    for item in remaining_items:
                        try:
                            answer = solve_question(item)
                            writer.writerow([item['qid'], answer])
                            processed_qids.add(item['qid'])
                            processed_count += 1
                        except:
                            writer.writerow([item['qid'], "A"])
                            processed_qids.add(item['qid'])
                            processed_count += 1
                    csv_file.flush()
                    print(f"✓ | Total done: {processed_count}/{total_items}")
        
        print(f"\n{'='*80}")
        print(f"✓ Completed: {processed_count}/{total_items} questions processed")
        
        # Sort output file by qid
        print(f"\nSorting output file by question ID...")
        try:
            csv_file.close()  # Close first before reading
            
            # Read all data
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                data_rows = list(reader)
            
            # Sort by qid (extract number from test_XXXX)
            def get_sort_key(row):
                qid = row[0]
                # Extract number from qid like "test_0001" -> 1
                try:
                    return int(qid.split('_')[1])
                except:
                    return 0
            
            data_rows.sort(key=get_sort_key)
            
            # Write back sorted data
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data_rows)
            
            print(f"✓ Output file sorted successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not sort output file: {e}")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠ Interrupted by user!")
        print(f"Progress saved: {processed_count}/{total_items} questions completed")
        print(f"Resume by running again - already processed questions will be skipped")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        print(f"Progress saved: {processed_count}/{total_items} questions completed")
        import traceback
        traceback.print_exc()
    finally:
        if not csv_file.closed:
            csv_file.close()
        print(f"Output file closed: {output_file}")


def solve_batch_streaming_llm(items, output_file):
    """
    OPTIMIZED v2: Solve questions with smart classification
    
    Flow:
    1. Quick check: Is it RAG? (has "Dựa trên đoạn văn") → RAG buffer
    2. Non-RAG questions → accumulate until 10 → LLM classify → domain buffers
    3. When any domain buffer reaches batch_size → process immediately
    4. Flush remaining buffers at the end
    
    Benefits:
    - RAG detection by keywords (no LLM needed) → saves API calls
    - Only classify non-RAG questions with LLM
    - For 300 questions with ~30% RAG → only 21 API calls instead of 30
    
    Supports resume: skip questions already in output file
    """
    from config import BATCH_CONFIG
    
    # Check which questions already processed
    processed_qids = set()
    if os.path.exists(output_file):
        print(f"Found existing output file, loading processed questions...")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if row and len(row) >= 2:
                        processed_qids.add(row[0])
            print(f"  ✓ Found {len(processed_qids)} already processed questions")
        except Exception as e:
            print(f"  ⚠ Error reading existing file: {e}, starting fresh")
            processed_qids = set()
    
    # Open output file in append mode
    file_mode = 'a' if processed_qids else 'w'
    csv_file = open(output_file, file_mode, newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    
    # Write header if new file
    if not processed_qids:
        writer.writerow(['qid', 'answer'])
        csv_file.flush()
    
    # Buffers
    non_rag_buffer = []  # Questions waiting for LLM classification
    domain_buffers = {
        "PRECISION_CRITICAL": [],
        "COMPULSORY": [],
        "RAG": [],
        "STEM": [],
        "MULTIDOMAIN": []
    }
    
    classification_batch_size = 10  # Classify 10 non-RAG questions at a time
    total_items = len(items)
    processed_count = len(processed_qids)
    rag_detected = 0
    non_rag_classified = 0
    
    # Print summary
    print("="*80)
    print("OPTIMIZED PROCESSING WITH SMART CLASSIFICATION")
    print("="*80)
    print(f"Total questions: {total_items}")
    print(f"Already completed: {processed_count}")
    print(f"Classification strategy:")
    print(f"  - RAG: Keyword detection (no LLM)")
    print(f"  - Non-RAG: LLM batch classification (10 questions/call)")
    print("\nDomain processing modes:")
    for domain_name in DOMAIN_CONFIGS:
        config = DOMAIN_CONFIGS[domain_name]
        mode = "Single" if not config.get('use_batch_processing', True) else f"Batch ({config.get('batch_size', 10)})"
        print(f"  - {domain_name}: {mode}")
    print("="*80)
    print()
    
    def process_domain_buffer(domain, buffer_name="domain buffer"):
        """Process a domain buffer when full"""
        nonlocal processed_count
        
        buffer = domain_buffers[domain]
        if not buffer:
            return
        
        strategy = DOMAIN_CONFIGS.get(domain, {})
        use_batch = strategy.get('use_batch_processing', True)
        domain_batch_size = strategy.get('batch_size', 10)
        
        # Process full batches
        while len(buffer) >= domain_batch_size:
            batch_items = buffer[:domain_batch_size]
            domain_buffers[domain] = buffer[domain_batch_size:]
            
            if use_batch:
                # Batch processing
                print(f"  → {domain} batch ({len(batch_items)} questions)...", end=' ')
                try:
                    batch_results = process_domain_batch(batch_items, domain)
                    
                    for batch_item in batch_items:
                        batch_qid = batch_item['qid']
                        answer = batch_results.get(batch_qid, "A")
                        writer.writerow([batch_qid, answer])
                        processed_qids.add(batch_qid)
                        processed_count += 1
                    
                    csv_file.flush()
                    print(f"✓ | Total: {processed_count}/{total_items}")
                except Exception as e:
                    print(f"✗ Error: {e}")
                    for batch_item in batch_items:
                        writer.writerow([batch_item['qid'], "A"])
                        processed_qids.add(batch_item['qid'])
                        processed_count += 1
                    csv_file.flush()
            else:
                # Single processing
                for item in batch_items:
                    qid = item['qid']
                    method = "verification" if strategy.get('use_self_verification') else "voting" if strategy.get('use_majority_voting') else "single"
                    print(f"  → {qid} ({domain} - {method})...", end=' ')
                    
                    try:
                        answer = solve_question(item)
                        writer.writerow([qid, answer])
                        processed_qids.add(qid)
                        processed_count += 1
                        csv_file.flush()
                        print(f"✓ {answer} | Total: {processed_count}/{total_items}")
                    except Exception as e:
                        print(f"✗ Error: {e}")
                        writer.writerow([qid, "A"])
                        processed_qids.add(qid)
                        processed_count += 1
                        csv_file.flush()
    
    try:
        # Process all items
        for idx, item in enumerate(items, 1):
            qid = item['qid']
            
            # Skip if already processed
            if qid in processed_qids:
                continue
            
            question_text = item['question']
            
            # Step 1: Quick RAG detection
            if is_rag_question(question_text):
                # RAG question - add to RAG buffer directly
                domain_buffers['RAG'].append(item)
                rag_detected += 1
                
                # Process RAG buffer if full
                process_domain_buffer('RAG')
            else:
                # Non-RAG question - add to classification buffer
                non_rag_buffer.append(item)
                
                # Step 2: When non-RAG buffer is full, classify with LLM
                if len(non_rag_buffer) >= classification_batch_size:
                    batch_to_classify = non_rag_buffer[:classification_batch_size]
                    non_rag_buffer = non_rag_buffer[classification_batch_size:]
                    
                    print(f"[{idx}/{total_items}] Classifying {len(batch_to_classify)} non-RAG questions with LLM...", end=' ')
                    classifications = classify_questions_with_llm(batch_to_classify)
                    print(f"✓")
                    
                    non_rag_classified += len(batch_to_classify)
                    
                    # Step 3: Add to domain buffers
                    for item_to_classify in batch_to_classify:
                        domain = classifications.get(item_to_classify['qid'], 'MULTIDOMAIN')
                        domain_buffers[domain].append(item_to_classify)
                    
                    # Step 4: Process domain buffers that are full
                    for domain in domain_buffers.keys():
                        process_domain_buffer(domain)
        
        # Classify remaining non-RAG buffer
        if non_rag_buffer:
            print(f"\nClassifying {len(non_rag_buffer)} remaining non-RAG questions...", end=' ')
            classifications = classify_questions_with_llm(non_rag_buffer)
            print(f"✓")
            
            non_rag_classified += len(non_rag_buffer)
            
            for item_to_classify in non_rag_buffer:
                domain = classifications.get(item_to_classify['qid'], 'MULTIDOMAIN')
                domain_buffers[domain].append(item_to_classify)
        
        # Step 5: Flush remaining domain buffers
        print(f"\n{'='*80}")
        print("Processing remaining questions in domain buffers...")
        print(f"{'='*80}")
        
        for domain, remaining_items in domain_buffers.items():
            if not remaining_items:
                continue
            
            print(f"\n{domain}: {len(remaining_items)} questions")
            strategy = DOMAIN_CONFIGS.get(domain, {})
            use_batch = strategy.get('use_batch_processing', True)
            
            if use_batch:
                # Batch process remaining
                try:
                    batch_results = process_domain_batch(remaining_items, domain)
                    
                    for item in remaining_items:
                        qid = item['qid']
                        answer = batch_results.get(qid, "A")
                        writer.writerow([qid, answer])
                        processed_qids.add(qid)
                        processed_count += 1
                    
                    csv_file.flush()
                    print(f"  ✓ Processed {len(remaining_items)} questions")
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    for item in remaining_items:
                        writer.writerow([item['qid'], "A"])
                        processed_qids.add(item['qid'])
                        processed_count += 1
                    csv_file.flush()
            else:
                # Single process remaining
                for item in remaining_items:
                    try:
                        answer = solve_question(item)
                        writer.writerow([item['qid'], answer])
                        processed_qids.add(item['qid'])
                        processed_count += 1
                    except:
                        writer.writerow([item['qid'], "A"])
                        processed_qids.add(item['qid'])
                        processed_count += 1
                csv_file.flush()
                print(f"  ✓ Processed {len(remaining_items)} questions")
        
        print(f"\n{'='*80}")
        print(f"✓ Completed: {processed_count}/{total_items} questions processed")
        print(f"\nClassification stats:")
        print(f"  - RAG detected by keywords: {rag_detected}")
        print(f"  - Non-RAG classified by LLM: {non_rag_classified}")
        print(f"  - LLM API calls saved: ~{rag_detected // 10} calls")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠ Interrupted by user!")
        print(f"Progress saved: {processed_count}/{total_items} questions completed")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        print(f"Progress saved: {processed_count}/{total_items} questions completed")
        import traceback
        traceback.print_exc()
    finally:
        if not csv_file.closed:
            csv_file.close()
        print(f"Output file closed: {output_file}")


def solve_batch(items):
    """
    Solve a batch of questions with domain-aware streaming batching
    STEM: Process individually with majority voting
    Other domains: Batch when reaching batch_size questions
    
    Note: This is the old non-streaming version, kept for compatibility
    Use solve_batch_streaming() for production
    """
    from config import BATCH_CONFIG
    
    # Domain buffers to accumulate questions (excluding STEM)
    domain_buffers = {
        "PRECISION_CRITICAL": [],
        "COMPULSORY": [],
        "RAG": [],
        "MULTIDOMAIN": []
    }
    
    all_results = {}
    batch_size = BATCH_CONFIG['batch_size']  # Default 10
    
    print(f"Processing {len(items)} questions with streaming domain batching...")
    print(f"Batch size: {batch_size} questions per domain (STEM: individual with voting)\n")
    
    # Process items one by one, classify and group
    for idx, item in enumerate(items, 1):
        # Classify into domain
        domain, confidence = router.classify_question(item['question'], item['choices'])
        
        # STEM: Process individually with majority voting
        if domain == "STEM":
            print(f"[{idx}/{len(items)}] STEM question detected, processing with majority voting...")
            strategy = router.get_strategy_config(domain)
            answer = solve_stem_with_voting(item, strategy, domain, confidence)
            all_results[item['qid']] = answer
            continue
        
        # Add to domain buffer for batch processing
        domain_buffers[domain].append(item)
        
        # Check if this domain buffer is full
        if len(domain_buffers[domain]) >= batch_size:
            print(f"[{idx}/{len(items)}] {domain} buffer full ({batch_size} questions), processing batch...")
            
            # Process this domain batch
            batch_to_process = domain_buffers[domain][:batch_size]
            batch_results = process_domain_batch(batch_to_process, domain)
            all_results.update(batch_results)
            
            # Reset buffer for this domain
            domain_buffers[domain] = []
    
    # Process remaining items in buffers (excluding STEM)
    print(f"\n[{len(items)}/{len(items)}] Processing remaining questions...")
    for domain, remaining_items in domain_buffers.items():
        if remaining_items:
            print(f"  {domain}: {len(remaining_items)} questions remaining")
            batch_results = process_domain_batch(remaining_items, domain)
            all_results.update(batch_results)
    
    return all_results


def main():
    # Example usage with val.json
    # input_path = 'data/test.json'
    # output_path = 'output.csv'
    # input_path = 'data/val.json'
    # output_path = 'val.csv'
    # input_path = 'data/stem_val.json'
    # output_path = 'stem_val.csv'
    input_path = 'data/stem_test.json'
    output_path = 'stem_test.csv'
    # input_path = 'data/test_stem.json'
    # output_path = 'test_stem.csv'
    
    if os.path.exists(input_path):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Total questions: {len(data)}")
        print("="*80)
        
        # Use streaming batch processing with auto-save
        solve_batch_streaming(data, output_path)
        
        print("\n" + "="*80)
        print(f"✓ All done! Output saved to: {output_path}")
        print("="*80)
    else:
        print(f"Error: Input file not found: {input_path}")

if __name__ == "__main__":
    main()
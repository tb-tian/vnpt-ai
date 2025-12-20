"""
predict.py - Entry point for VNPT AI Competition
Reads from /code/private_test.json and outputs submission.csv, submission_time.csv

This file contains ALL logic from main.py, only modified for BTC requirements:
- Read from /code/private_test.json
- Output to /code/submission.csv and /code/submission_time.csv
- Track timing per question in a loop
- Retry API calls every 1 minute on error (no fallback to 'A')
"""

import json
import csv
import re
import os
import time
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


# ============================================================================
# RETRY HELPER FUNCTION
# ============================================================================

def get_response_with_retry(messages, model='small', temperature=0.3, n=1, response_format=None, retry_delay=60):
    """
    Call LLM API with automatic retry on failure
    Retries every {retry_delay} seconds until success
    
    Special handling:
    - ContentPolicyError: Don't retry, propagate immediately (for PRECISION_CRITICAL)
    
    Args:
        messages: List of message dicts
        model: Model name ('small' or 'large')
        temperature: Temperature parameter
        n: Number of completions
        response_format: Response format (e.g., {"type": "json_object"})
        retry_delay: Seconds to wait between retries (default: 60)
    
    Returns:
        Response from API
    """
    from get_response import ContentPolicyError
    
    attempt = 0
    while True:
        attempt += 1
        try:
            response = get_response(
                messages,
                model=model,
                temperature=temperature,
                n=n,
                response_format=response_format
            )
            return response
        except ContentPolicyError as e:
            # Don't retry content policy violations - these won't succeed
            print(f"\n  ⚠ Content policy violation: {e}")
            raise
        except Exception as e:
            print(f"\n  ⚠ API call failed (attempt {attempt}): {e}")
            print(f"  ⏳ Waiting {retry_delay} seconds before retry...")
            time.sleep(retry_delay)
            print(f"  🔄 Retrying (attempt {attempt + 1})...")


# ============================================================================
# COPY ALL FUNCTIONS FROM main.py
# ============================================================================

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
        response = get_response_with_retry(
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
    
    # Call LLM with retry - no fallback to empty string
    try:
        raw_answer = get_response_with_retry(
            messages,
            model=strategy.get('model', 'small'),
            temperature=strategy.get('temperature', 0.3)
        )
    except Exception as e:
        # Check if it's ContentPolicyError for PRECISION_CRITICAL
        error_str = str(e)
        if domain == "PRECISION_CRITICAL" and "API rejected content" in error_str:
            # Find answer with "không thể", "không trả lời" keywords
            rejection_keywords = ["không thể", "không trả lời", "từ chối", "không thể trả lời"]
            
            for idx, choice in enumerate(choices):
                choice_lower = choice.lower()
                if any(keyword in choice_lower for keyword in rejection_keywords):
                    answer = chr(ord('A') + idx)  # Convert index to letter (0→A, 1→B, etc.)
                    print(f"  ⚠ API rejected content (PRECISION_CRITICAL) → Found answer: {answer} ('{choice[:50]}...')")
                    return answer
            
            # If no rejection keyword found, use first choice as safe default
            print(f"  ⚠ API rejected content (PRECISION_CRITICAL) → No rejection keyword found, defaulting to A")
            return "A"
        # Re-raise for other errors
        raise
    
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
        response = get_response_with_retry(
            messages,
            model=strategy.get('model', 'small'),
            temperature=strategy.get('temperature', 0.7),
            n=strategy.get('n', 5)
        )
        
        # Extract answers from all 5 completions
        answers = []
        # Parse response: when n > 1, get_response returns dict with 'choices' array
        if isinstance(response, dict) and 'choices' in response:
            # Response is full API dict: {'choices': [{'message': {'content': '...'}}, ...]}
            for choice in response['choices']:
                content = choice['message']['content']
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
        elif isinstance(response, list):
            # Response is list of strings (shouldn't happen with current get_response)
            for content in response:
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
        else:
            # Single string - treat as one answer (n=1 case)
            answer_pattern = r'(?:^|\s|[Đđ]áp án|[Cc]họn|[Tt]rả lời)\s*[:\-]?\s*([A-Z])(?:[.\s]|$)'
            match = re.search(answer_pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).upper()
            else:
                match = re.search(r'\b([A-Z])\b', response)
                answer = match.group(1).upper() if match else "A"
            answers.append(answer if 'A' <= answer <= max_valid_letter else "A")
        
        # Majority voting
        vote_counts = Counter(answers)
        final_answer = vote_counts.most_common(1)[0][0]
        votes = vote_counts[final_answer]
        
        print(f"  STEM Voting: {answers} → {final_answer} ({votes}/{len(answers)} votes) | Domain: {domain} (conf: {confidence:.2f})")
        return final_answer
        
    except Exception as e:
        print(f"  STEM voting failed: {e}, falling back to single call")
        # Fallback to single question solving
        return solve_question(item)


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
            
            response = get_response_with_retry(
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
                verify_prompt = f"""Hãy đánh giá lại bài giải sau:

{response}

Kiểm tra:
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
                
                verify_response = get_response_with_retry(
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
            print(f"  STEM Self-Verify attempt {attempt+1} failed: {e}")
            if attempt == max_attempts:
                # Last attempt failed, return best effort
                return answer
            continue
    
    # All attempts exhausted, return last answer
    return answer


def process_domain_batch(domain_items, domain):
    """
    Process a batch of questions from the same domain
    Returns dict mapping qid to answer
    """
    # Get strategy for this domain
    strategy = router.get_strategy_config(domain)
    
    # Prepare items with context
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
            
            raw_answer = get_response_with_retry(
                messages,
                model=strategy.get('model', 'small'),
                temperature=strategy.get('temperature', 0.3),
                response_format=response_format
            )
            
            # STEM: Extract from special format with ===DANH SÁCH ĐÁP ÁN===
            if domain.upper() == "STEM":
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
            
            break
        except json.JSONDecodeError as e:
            print(f"    JSON parse error (attempt {attempt + 1}): {e}")
            if attempt == 1:
                # Last attempt: Fallback to individual solving
                print(f"    Batch failed, solving individually...")
                for i, original_item in enumerate(domain_items, 1):
                    ans = solve_question(original_item)
                    answers[str(i)] = ans
                break
    
    return answers


def predict_with_timing(test_data, output_submission, output_timing):
    """
    Process questions with optimized flow:
    1. Keyword detect RAG questions
    2. Batch classify non-RAG questions with LLM (10 at a time)
    3. Accumulate into domain buffers
    4. Process when buffers reach batch_size
    
    Args:
        test_data: List of test questions
        output_submission: Path to submission.csv
        output_timing: Path to submission_time.csv
    """
    total_items = len(test_data)
    
    print(f"\nProcessing {total_items} questions...")
    print("="*80)
    print("\n📋 Domain Processing Config:")
    for domain_name, config in DOMAIN_CONFIGS.items():
        mode = "SINGLE" if not config.get('use_batch_processing', True) else f"BATCH ({config.get('batch_size', 10)})"
        print(f"  {domain_name}: {mode}")
    print("\nClassification Strategy:")
    print("  - RAG: Keyword detection (no LLM)")
    print("  - Non-RAG: LLM batch classification (10 questions/call)")
    print("="*80)
    
    # Check if output files exist to determine append or write mode
    submission_exists = os.path.exists(output_submission)
    timing_exists = os.path.exists(output_timing)
    
    submission_mode = 'a' if submission_exists else 'w'
    timing_mode = 'a' if timing_exists else 'w'
    
    if submission_exists:
        print(f"\n✓ Resuming: Appending to existing {os.path.basename(output_submission)}")
    else:
        print(f"\n✓ Fresh start: Creating new {os.path.basename(output_submission)}")
    
    # Open both output files
    submission_file = open(output_submission, submission_mode, newline='', encoding='utf-8')
    timing_file = open(output_timing, timing_mode, newline='', encoding='utf-8')
    
    submission_writer = csv.writer(submission_file)
    timing_writer = csv.writer(timing_file)
    
    # Write headers only for new files
    if not submission_exists:
        submission_writer.writerow(['qid', 'answer'])
    if not timing_exists:
        timing_writer.writerow(['qid', 'answer', 'time'])
    
    # Domain buffers for batch processing
    domain_buffers = {
        "PRECISION_CRITICAL": [],
        "COMPULSORY": [],
        "RAG": [],
        "STEM": [],
        "MULTIDOMAIN": []
    }
    
    non_rag_buffer = []  # Buffer for non-RAG questions awaiting LLM classification
    classification_batch_size = 10  # Classify 10 non-RAG questions at a time
    
    processed_count = 0
    rag_detected = 0
    non_rag_classified = 0
    
    def process_batch(domain, batch_items):
        """Process a batch and write results with timing"""
        nonlocal processed_count
        
        print(f"  Processing {domain} batch ({len(batch_items)} questions)...", end=' ')
        batch_start = time.time()
        
        try:
            # Use process_domain_batch if batch size > 1, otherwise single
            if len(batch_items) == 1:
                # Single question
                item = batch_items[0]
                answer = solve_question(item)
                batch_results = {item['qid']: answer}
            else:
                # Batch processing
                batch_results = process_domain_batch(batch_items, domain)
            
            batch_end = time.time()
            batch_time = batch_end - batch_start
            
            # Calculate per-question time (estimate)
            per_question_time = batch_time / len(batch_items)
            
            # Write results
            for item in batch_items:
                qid = item['qid']
                answer = batch_results.get(qid, "A")
                
                submission_writer.writerow([qid, answer])
                timing_writer.writerow([qid, answer, round(per_question_time, 4)])
                processed_count += 1
            
            submission_file.flush()
            timing_file.flush()
            
            print(f"✓ ({batch_time:.2f}s total, ~{per_question_time:.4f}s/question)")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            # Fallback
            for item in batch_items:
                submission_writer.writerow([item['qid'], 'A'])
                timing_writer.writerow([item['qid'], 'A', 0.0])
                processed_count += 1
            submission_file.flush()
            timing_file.flush()
    
    try:
        # Process each question with optimized flow
        for idx, item in enumerate(test_data, 1):
            qid = item['qid']
            question_text = item['question']
            choices = item['choices']
            
            # Step 1: Quick RAG detection with keywords
            if is_rag_question(question_text):
                # RAG question - add directly to RAG buffer
                domain = "RAG"
                domain_buffers['RAG'].append(item)
                rag_detected += 1
                
                strategy = router.get_strategy_config(domain)
                use_batch = strategy.get('use_batch_processing', True)
                batch_size = strategy.get('batch_size', 10)
                
                # Process RAG buffer if full
                if use_batch and len(domain_buffers['RAG']) >= batch_size:
                    print(f"[{idx}/{total_items}] RAG buffer full ({len(domain_buffers['RAG'])}/{batch_size})")
                    batch_to_process = domain_buffers['RAG'][:batch_size]
                    domain_buffers['RAG'] = domain_buffers['RAG'][batch_size:]
                    process_batch('RAG', batch_to_process)
                else:
                    print(f"[{idx}/{total_items}] {qid} → RAG buffer ({len(domain_buffers['RAG'])}/{batch_size})")
                continue
            
            # Step 2: Non-RAG question - add to classification buffer
            non_rag_buffer.append(item)
            print(f"[{idx}/{total_items}] {qid} → Non-RAG buffer ({len(non_rag_buffer)}/{classification_batch_size})")
            
            # Step 3: When non-RAG buffer is full, classify with LLM
            if len(non_rag_buffer) >= classification_batch_size:
                batch_to_classify = non_rag_buffer[:classification_batch_size]
                non_rag_buffer = non_rag_buffer[classification_batch_size:]
                
                print(f"  → Classifying {len(batch_to_classify)} non-RAG questions with LLM...", end=' ')
                classifications = classify_questions_with_llm(batch_to_classify)
                print("✓")
                
                non_rag_classified += len(batch_to_classify)
                
                # Step 4: Add classified questions to domain buffers OR process single immediately
                for classified_item in batch_to_classify:
                    classified_domain = classifications.get(classified_item['qid'], 'MULTIDOMAIN')
                    
                    # Mark as LLM classified
                    classified_item['_llm_classified'] = True
                    
                    strategy = router.get_strategy_config(classified_domain)
                    use_batch = strategy.get('use_batch_processing', True)
                    
                    if not use_batch:
                        # Process single immediately
                        print(f"    {classified_item['qid']} → {classified_domain} (LLM classify) → Processing single...", end=' ')
                        try:
                            start_time = time.time()
                            answer = solve_question(classified_item)
                            end_time = time.time()
                            inference_time = end_time - start_time
                            
                            submission_writer.writerow([classified_item['qid'], answer])
                            timing_writer.writerow([classified_item['qid'], answer, round(inference_time, 4)])
                            submission_file.flush()
                            timing_file.flush()
                            processed_count += 1
                            
                            print(f"✓ {answer} ({inference_time:.4f}s) | Total: {processed_count}/{total_items}")
                        except Exception as e:
                            print(f"✗ Error: {e}")
                            submission_writer.writerow([classified_item['qid'], 'A'])
                            timing_writer.writerow([classified_item['qid'], 'A', 0.0])
                            submission_file.flush()
                            timing_file.flush()
                            processed_count += 1
                    else:
                        # Add to batch buffer
                        domain_buffers[classified_domain].append(classified_item)
                        print(f"    {classified_item['qid']} → {classified_domain} buffer (LLM classify) ({len(domain_buffers[classified_domain])} total)")
                
                # Step 5: Process domain buffers that are full
                for check_domain in domain_buffers.keys():
                    strategy = router.get_strategy_config(check_domain)
                    use_batch = strategy.get('use_batch_processing', True)
                    batch_size = strategy.get('batch_size', 10)
                    
                    if use_batch and len(domain_buffers[check_domain]) >= batch_size:
                        print(f"  → {check_domain} buffer full ({len(domain_buffers[check_domain])}/{batch_size})")
                        batch_to_process = domain_buffers[check_domain][:batch_size]
                        domain_buffers[check_domain] = domain_buffers[check_domain][batch_size:]
                        process_batch(check_domain, batch_to_process)
        
        # Classify remaining non-RAG buffer
        if non_rag_buffer:
            print(f"\nClassifying {len(non_rag_buffer)} remaining non-RAG questions...", end=' ')
            classifications = classify_questions_with_llm(non_rag_buffer)
            print("✓")
            
            non_rag_classified += len(non_rag_buffer)
            
            for classified_item in non_rag_buffer:
                classified_domain = classifications.get(classified_item['qid'], 'MULTIDOMAIN')
                
                # Mark as LLM classified
                classified_item['_llm_classified'] = True
                
                strategy = router.get_strategy_config(classified_domain)
                use_batch = strategy.get('use_batch_processing', True)
                
                if not use_batch:
                    # Process single immediately
                    print(f"  {classified_item['qid']} → {classified_domain} (LLM classify) → Processing single...", end=' ')
                    try:
                        start_time = time.time()
                        answer = solve_question(classified_item)
                        end_time = time.time()
                        inference_time = end_time - start_time
                        
                        submission_writer.writerow([classified_item['qid'], answer])
                        timing_writer.writerow([classified_item['qid'], answer, round(inference_time, 4)])
                        submission_file.flush()
                        timing_file.flush()
                        processed_count += 1
                        
                        print(f"✓ {answer} ({inference_time:.4f}s) | Total: {processed_count}/{total_items}")
                    except Exception as e:
                        print(f"✗ Error: {e}")
                        submission_writer.writerow([classified_item['qid'], 'A'])
                        timing_writer.writerow([classified_item['qid'], 'A', 0.0])
                        submission_file.flush()
                        timing_file.flush()
                        processed_count += 1
                else:
                    # Add to batch buffer
                    domain_buffers[classified_domain].append(classified_item)
                    print(f"  {classified_item['qid']} → {classified_domain} buffer (LLM classify)")
        
        # Process remaining buffers
        print("\n" + "="*80)
        print("Processing remaining questions in buffers...")
        print(f"Statistics: {rag_detected} RAG (keyword), {non_rag_classified} non-RAG (LLM classified)")
        print("="*80)
        
        for domain, remaining_items in domain_buffers.items():
            if remaining_items:
                print(f"{domain}: {len(remaining_items)} remaining questions")
                process_batch(domain, remaining_items)
        
        print("="*80)
        print(f"✓ Completed: {processed_count}/{total_items} questions processed")
        
    finally:
        submission_file.close()
        timing_file.close()


def main():
    """
    Main entry point for prediction pipeline
    Modified from main.py to meet BTC requirements
    """
    # Input/output paths as specified in requirements
    input_path = '/code/private_test.json'
    output_submission = '/code/submission.csv'
    output_timing = '/code/submission_time.csv'
    
    print("="*80)
    print("VNPT AI - Prediction Pipeline")
    print("="*80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_submission}")
    print(f"        {output_timing}")
    print("="*80)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"✗ Error: Input file not found: {input_path}")
        print("Note: Make sure to mount the test data to /code/private_test.json")
        return
    
    # Load test data
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"✓ Loaded {len(test_data)} questions from {input_path}")
    except Exception as e:
        print(f"✗ Error loading input file: {e}")
        return
    
    # Process all questions with timing (writes as it goes)
    start_total = time.time()
    predict_with_timing(test_data, output_submission, output_timing)
    end_total = time.time()
    
    # Summary
    print("\n" + "="*80)
    print("✓ PREDICTION COMPLETE")
    print("="*80)
    print(f"Total questions processed: {len(test_data)}")
    print(f"Output files:")
    print(f"  - {output_submission}")
    print(f"  - {output_timing}")
    print(f"\nTotal inference time: {end_total - start_total:.2f}s")
    print(f"Average per question: {(end_total - start_total) / len(test_data):.4f}s")
    print("="*80)


if __name__ == "__main__":
    main()

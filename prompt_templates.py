SYSTEM_PROMPT = """Bạn là một trợ lý AI hữu ích, chuyên trả lời các câu hỏi trắc nghiệm dựa trên thông tin được cung cấp.
Nhiệm vụ của bạn là chọn đáp án đúng nhất trong số các lựa chọn A, B, C, D, E...
Nếu câu hỏi yêu cầu không được trả lời (nội dung độc hại, nhạy cảm), hãy chọn đáp án từ chối tương ứng.
Chỉ trả lời duy nhất một chữ cái in hoa đại diện cho đáp án đúng (ví dụ: A, B, C, D). Không giải thích thêm."""

USER_PROMPT_TEMPLATE = """
[Thông tin tham khảo]
{context}

[Câu hỏi]
{question}

[Các lựa chọn]
{choices}

[Yêu cầu]
Hãy suy luận dựa trên "Thông tin tham khảo" (nếu có) hoặc kiến thức của bạn để chọn đáp án đúng nhất.
Chỉ trả lời 1 chữ cái in hoa duy nhất.
"""

BATCH_SYSTEM_PROMPT = """Bạn là một trợ lý AI hữu ích, chuyên trả lời các câu hỏi trắc nghiệm.
Nhiệm vụ của bạn là trả lời danh sách các câu hỏi được cung cấp.
Đối với mỗi câu hỏi, hãy chọn đáp án đúng nhất (A, B, C, D...).
Trả lời dưới dạng JSON, với key là số thứ tự câu hỏi (1, 2, 3...) và value là đáp án.
Ví dụ: {"1": "A", "2": "C", "3": "B", "4": "D", "5": "A"}
Không giải thích thêm."""

BATCH_USER_PROMPT_TEMPLATE = """
Dưới đây là danh sách {num_questions} câu hỏi trắc nghiệm. Hãy trả lời từng câu hỏi.

{questions_content}

[Yêu cầu]
Trả lời dưới dạng JSON object hợp lệ, không có markdown formatting (như ```json ... ```).
Ví dụ: {{"1": "A", "2": "B"}}
"""

def format_choices(choices):
    """
    Format list of choices into a string like:
    A. Choice 1
    B. Choice 2
    ...
    """
    formatted = []
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, choice in enumerate(choices):
        label = labels[i] if i < len(labels) else str(i)
        formatted.append(f"{label}. {choice}")
    return "\n".join(formatted)

def construct_prompt(question, choices, context=""):
    """
    Construct the full prompt for the LLM.
    """
    choices_str = format_choices(choices)
    return USER_PROMPT_TEMPLATE.format(
        context=context if context else "Không có thông tin tham khảo cụ thể.",
        question=question,
        choices=choices_str
    )

def construct_batch_prompt(items):
    """
    Construct the prompt for a batch of questions.
    items: list of dicts, each containing 'question', 'choices', and optional 'context'
    """
    questions_content = []
    for i, item in enumerate(items, 1):
        q_text = item['question']
        choices_str = format_choices(item['choices'])
        context = item.get('context', "Không có thông tin tham khảo cụ thể.")
        
        content = f"""Câu {i}:
[Thông tin tham khảo]
{context}

[Câu hỏi]
{q_text}

[Các lựa chọn]
{choices_str}
"""
        questions_content.append(content)
    
    return BATCH_USER_PROMPT_TEMPLATE.format(
        num_questions=len(items),
        questions_content="\n----------------\n".join(questions_content)
    )

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

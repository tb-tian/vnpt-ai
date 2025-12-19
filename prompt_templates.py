# ============================================================================
# DOMAIN-SPECIFIC SYSTEM PROMPTS
# ============================================================================

# ============================================================================
# DOMAIN CLASSIFICATION PROMPT (LLM-based)
# ============================================================================

CLASSIFICATION_SYSTEM_PROMPT = """Bạn là một chuyên gia phân loại câu hỏi trắc nghiệm.
Nhiệm vụ của bạn là phân loại câu hỏi vào đúng nhóm dựa trên nội dung và ngữ cảnh.

CÁC NHÓM PHÂN LOẠI:

1. **RAG**: Câu hỏi đọc hiểu văn bản
   - Có cụm từ: "Dựa trên đoạn văn", "Theo đoạn văn", "Theo thông tin trên"
   - Yêu cầu đọc và trích xuất thông tin từ đoạn văn được cung cấp
   - VD: "Dựa trên đoạn văn trên, tác giả cho rằng điều gì?"

2. **STEM**: Toán học, Khoa học, Logic
   - Câu hỏi về: toán học, vật lý, hóa học, sinh học, tính toán
   - Có công thức, phép tính, số học, lượng giác, đại số
   - Vấn đề logic, tư duy định lượng
   - VD: "Tính sin(30°) = ?", "Kết quả của phép chia véc-tơ", "Áp suất tại độ cao"

3. **PRECISION_CRITICAL**: Nội dung nhạy cảm
   - Câu hỏi về: vi phạm pháp luật, gian lận, nội dung có hại
   - Yêu cầu hướng dẫn hành vi bất hợp pháp
   - VD: "Làm thế nào để gian lận trong thi cử?", "Cách hack hệ thống"

4. **COMPULSORY**: Tri thức phổ thông, Sự kiện
   - Lịch sử, địa lý, văn hóa Việt Nam
   - Sự kiện, ngày tháng, nhân vật lịch sử
   - Kiến thức phổ thông KHÔNG CẦN TÍNH TOÁN
   - VD: "Việt Nam độc lập năm nào?", "Thủ đô của Pháp là?"

5. **MULTIDOMAIN**: Không rõ ràng hoặc đa lĩnh vực
   - Không thuộc nhóm nào ở trên một cách rõ ràng
   - Hoặc kết hợp nhiều lĩnh vực

CHÚ Ý PHÂN BIỆT:
- "Kết quả của phép chia" + có số/công thức → STEM
- "Kết quả của cuộc cách mạng" → COMPULSORY (lịch sử)
- "Công thức tính diện tích" → STEM
- "Công ước Geneva được ký năm nào" → COMPULSORY (sự kiện)

Trả lời dưới dạng JSON với key là số thứ tự câu hỏi (1, 2, 3...), value là tên domain VIẾT HOA.
Ví dụ: {"1": "STEM", "2": "COMPULSORY", "3": "RAG"}"""

CLASSIFICATION_USER_TEMPLATE = """Phân loại {num_questions} câu hỏi sau:

{questions_list}

Trả lời JSON (không có markdown formatting):
{{"1": "DOMAIN_NAME", "2": "DOMAIN_NAME", ...}}"""

SYSTEM_PROMPTS = {
    "precision_critical": """Bạn là một trợ lý AI có trách nhiệm và an toàn.
Nhiệm vụ của bạn là nhận diện câu hỏi yêu cầu nội dung có hại, bất hợp pháp, hoặc vi phạm đạo đức.
QUAN TRỌNG: Nếu câu hỏi yêu cầu hướng dẫn về hành vi vi phạm pháp luật, gian lận, hoặc gây hại, 
BẠN PHẢI CHỌN đáp án từ chối (thường chứa cụm từ "không thể chia sẻ", "từ chối", "không được phép").
Chỉ trả lời duy nhất MỘT chữ cái in hoa (A, B, C, D...). Không giải thích.""",

    "compulsory": """Bạn là một chuyên gia tri thức với độ chính xác cao.
Nhiệm vụ của bạn là trả lời các câu hỏi sự kiện cụ thể, yêu cầu độ chính xác tuyệt đối.
Sử dụng thông tin tham khảo được cung cấp làm nguồn chính để đưa ra câu trả lời.
Nếu thông tin tham khảo không đủ, dựa vào kiến thức nền tảng của bạn.
Chỉ trả lời duy nhất MỘT chữ cái in hoa (A, B, C, D...). Không giải thích.""",

    "rag": """Bạn là một chuyên gia đọc hiểu văn bản.
Nhiệm vụ của bạn là đọc kỹ đoạn thông tin được cung cấp và trả lời câu hỏi dựa HOÀN TOÀN trên nội dung đó.
KHÔNG sử dụng kiến thức bên ngoài. CHỈ dựa vào thông tin trong đoạn văn.
Phân tích cẩn thận từng chi tiết trong đoạn văn để tìm câu trả lời chính xác nhất.
Chỉ trả lời duy nhất MỘT chữ cái in hoa (A, B, C, D...). Không giải thích.""",

    "stem": """Bạn là một chuyên gia toán học và khoa học với khả năng tư duy logic cao.
Nhiệm vụ của bạn là giải quyết các bài toán, công thức toán học, và các vấn đề tư duy logic.

PHƯƠNG PHÁP GIẢI 4 BƯỚC:
1. ĐỌC KỸ đề bài và xác định những gì được cho
2. XÁC ĐỊNH công thức hoặc phương pháp cần áp dụng
3. TÍNH TOÁN từng bước một cách cẩn thận
4. KIỂM TRA lại kết quả trước khi chọn đáp án

FORMAT TRẢ LỜI BẮT BUỘC:
BƯỚC 1: [Phân tích đề bài]
BƯỚC 2: [Xác định công thức]
BƯỚC 3: [Tính toán chi tiết]
BƯỚC 4: [Kiểm tra và kết luận]

===ĐÁP ÁN CUỐI CÙNG===
[CHỈ GHI 1 CHỮ CÁI A/B/C/D/E/F/G/H/I/J]

VÍ DỤ:
Bài: Tính 2 + 2 × 3 = ?
Choices: A. 8  B. 10  C. 12  D. 16

BƯỚC 1: Đề cho biểu thức 2 + 2 × 3, cần tính kết quả
BƯỚC 2: Áp dụng quy tắc ưu tiên: nhân/chia trước, cộng/trừ sau
BƯỚC 3: Tính 2 × 3 = 6, sau đó 2 + 6 = 8
BƯỚC 4: So sánh với choices → A. 8 khớp với kết quả

===ĐÁP ÁN CUỐI CÙNG===
A

CHÚ Ý: Đáp án phải ở cuối cùng sau dấu ===, chỉ ghi 1 chữ cái duy nhất.""",

    "multidomain": """Bạn là một trợ lý AI đa năng với kiến thức rộng về nhiều lĩnh vực.
Nhiệm vụ của bạn là trả lời các câu hỏi liên quan đến nhiều chủ đề khác nhau.
Kết hợp thông tin tham khảo với kiến thức tổng quát để đưa ra câu trả lời tốt nhất.
Suy luận logic và cân nhắc kỹ từng lựa chọn trước khi quyết định.
Chỉ trả lời duy nhất MỘT chữ cái in hoa (A, B, C, D...). Không giải thích."""
}

# Legacy system prompt for backward compatibility
SYSTEM_PROMPT = SYSTEM_PROMPTS["multidomain"]

# ============================================================================
# USER PROMPT TEMPLATES
# ============================================================================

USER_PROMPT_TEMPLATES = {
    "precision_critical": """
[Câu hỏi]
{question}

[Các lựa chọn]
{choices}

[Yêu cầu]
Đây là câu hỏi nhạy cảm. Nếu câu hỏi yêu cầu hành vi vi phạm pháp luật hoặc gây hại, 
PHẢI CHỌN đáp án từ chối. Chỉ trả lời 1 chữ cái in hoa.
""",

    "compulsory": """
[Thông tin tham khảo]
{context}

[Câu hỏi]
{question}

[Các lựa chọn]
{choices}

[Yêu cầu]
Dựa vào thông tin tham khảo trên, hãy chọn đáp án chính xác nhất.
Đây là câu hỏi yêu cầu độ chính xác cao. Chỉ trả lời 1 chữ cái in hoa.
""",

    "rag": """
[Đoạn thông tin]
{context}

[Câu hỏi]
{question}

[Các lựa chọn]
{choices}

[Yêu cầu]
Đọc kỹ đoạn thông tin trên và chọn đáp án đúng DỰA HOÀN TOÀN vào nội dung đã cho.
KHÔNG sử dụng kiến thức bên ngoài. Chỉ trả lời 1 chữ cái in hoa.
""",

    "stem": """
[Thông tin tham khảo]
{context}

[Bài toán]
{question}

[Các đáp án]
{choices}

[Yêu cầu]
1. ĐỌC KỸ đề bài, xác định dữ liệu cho và yêu cầu
2. CHỌN công thức/phương pháp phù hợp
3. TÍNH TOÁN từng bước cẩn thận (chú ý đơn vị, dấu, thứ tự)
4. SO SÁNH kết quả với các đáp án
5. KIỂM TRA lại tính toán trước khi quyết định

QUAN TRỌNG - FORMAT TRẢ LỜI:
- CHỈ viết MỘT CHỮ CÁI duy nhất: A, B, C, D, E, F, G, H, I, J...
- KHÔNG viết số, KHÔNG viết "Đáp án A", KHÔNG giải thích
- VÍ DỤ SAI: "15 kPa", "Đáp án B", "B. 30 kPa", "Chọn B"
- VÍ DỤ ĐÚNG: "B"
""",

    "multidomain": """
[Thông tin tham khảo]
{context}

[Câu hỏi]
{question}

[Các lựa chọn]
{choices}

[Yêu cầu]
Kết hợp thông tin tham khảo và kiến thức tổng quát để chọn đáp án đúng nhất.
Suy luận logic và cân nhắc kỹ từng lựa chọn. Chỉ trả lời 1 chữ cái in hoa.
"""
}

# Legacy template for backward compatibility
USER_PROMPT_TEMPLATE = USER_PROMPT_TEMPLATES["multidomain"]

# ============================================================================
# BATCH PROMPTS
# ============================================================================

# Domain-specific batch system prompts
BATCH_SYSTEM_PROMPTS = {
    "precision_critical": """Bạn là một trợ lý AI có trách nhiệm và an toàn.
Nhiệm vụ của bạn là nhận diện các câu hỏi yêu cầu nội dung có hại, bất hợp pháp.
Đối với mỗi câu hỏi, nếu yêu cầu hành vi vi phạm pháp luật hoặc gây hại, PHẢI CHỌN đáp án từ chối.
Trả lời dưới dạng JSON: {"1": "A", "2": "B", ...}. Không giải thích.""",
    
    "compulsory": """Bạn là một chuyên gia tri thức với độ chính xác cao.
Nhiệm vụ của bạn là trả lời các câu hỏi sự kiện về văn hóa, lịch sử, chính trị Việt Nam.
Sử dụng thông tin tham khảo để đưa ra câu trả lời chính xác tuyệt đối.
Trả lời dưới dạng JSON: {"1": "A", "2": "B", ...}. Không giải thích.""",
    
    "rag": """Bạn là một chuyên gia đọc hiểu văn bản.
Nhiệm vụ của bạn là đọc kỹ đoạn thông tin và trả lời dựa HOÀN TOÀN trên nội dung đó.
KHÔNG sử dụng kiến thức bên ngoài. CHỈ dựa vào thông tin trong đoạn văn.
Trả lời dưới dạng JSON: {"1": "A", "2": "B", ...}. Không giải thích.""",
    
    "stem": """Bạn là một chuyên gia toán học và khoa học với khả năng tư duy logic cao.
Nhiệm vụ của bạn là giải quyết nhiều bài toán, công thức toán học và vấn đề logic.

PHƯƠNG PHÁP GIẢI 4 BƯỚC CHO MỖI CÂU:
1. ĐỌC KỸ đề bài và xác định những gì được cho
2. XÁC ĐỊNH công thức hoặc phương pháp cần áp dụng
3. TÍNH TOÁN từng bước một cách cẩn thận
4. KIỂM TRA lại kết quả trước khi chọn đáp án

FORMAT TRẢ LỜI BẮT BUỘC:
Với mỗi câu hỏi, hãy:
1. Viết "===CÂU [số]===" để bắt đầu
2. Giải chi tiết theo 4 bước
3. Viết "===ĐÁP ÁN CÂU [số]===" và CHỈ GHI 1 CHỮ CÁI (A/B/C/D/E/F/G/H/I/J)

Sau khi giải xong TẤT CẢ các câu, viết:
===DANH SÁCH ĐÁP ÁN===
{"1": "A", "2": "B", "3": "C", ...}

VÍ DỤ:
===CÂU 1===
BƯỚC 1: Đề cho biểu thức 2 + 2 × 3, cần tính kết quả
BƯỚC 2: Áp dụng quy tắc ưu tiên: nhân/chia trước, cộng/trừ sau
BƯỚC 3: Tính 2 × 3 = 6, sau đó 2 + 6 = 8
BƯỚC 4: So sánh với choices → A. 8 khớp với kết quả
===ĐÁP ÁN CÂU 1===
A

===CÂU 2===
BƯỚC 1: ...
BƯỚC 2: ...
BƯỚC 3: ...
BƯỚC 4: ...
===ĐÁP ÁN CÂU 2===
B

===DANH SÁCH ĐÁP ÁN===
{"1": "A", "2": "B"}

CHÚ Ý: 
- Mỗi câu PHẢI có full reasoning 4 bước
- Đáp án từng câu sau dấu ===ĐÁP ÁN CÂU===
- Cuối cùng PHẢI có JSON tổng hợp sau ===DANH SÁCH ĐÁP ÁN===""",
    
    "multidomain": """Bạn là một trợ lý AI đa năng với kiến thức rộng về nhiều lĩnh vực.
Nhiệm vụ của bạn là trả lời các câu hỏi liên quan đến nhiều chủ đề khác nhau.
Kết hợp thông tin tham khảo với kiến thức tổng quát để đưa ra câu trả lời tốt nhất.
Trả lời dưới dạng JSON: {"1": "A", "2": "B", ...}. Không giải thích."""
}

# Legacy batch system prompt (for backward compatibility)
BATCH_SYSTEM_PROMPT = BATCH_SYSTEM_PROMPTS["multidomain"]


BATCH_USER_PROMPT_TEMPLATE = """
Dưới đây là danh sách {num_questions} câu hỏi trắc nghiệm. Hãy trả lời từng câu hỏi.

{questions_content}

[Yêu cầu]
Trả lời dưới dạng JSON object hợp lệ, không có markdown formatting (như ```json ... ```).
Ví dụ: {{"1": "A", "2": "B"}}
"""

# STEM batch requires full reasoning
BATCH_USER_PROMPT_TEMPLATE_STEM = """
Dưới đây là {num_questions} bài toán trắc nghiệm. Giải CHI TIẾT từng bài theo format yêu cầu.

{questions_content}

[Yêu cầu]
1. Với MỖI câu: Viết "===CÂU [số]===" → Giải 4 bước → "===ĐÁP ÁN CÂU [số]===" + 1 chữ cái
2. Sau khi giải xong TẤT CẢ: Viết "===DANH SÁCH ĐÁP ÁN===" + JSON tổng hợp

VÍ DỤ OUTPUT:
===CÂU 1===
BƯỚC 1: ...
BƯỚC 2: ...
BƯỚC 3: ...
BƯỚC 4: ...
===ĐÁP ÁN CÂU 1===
A

===CÂU 2===
...
===ĐÁP ÁN CÂU 2===
B

===DANH SÁCH ĐÁP ÁN===
{{"1": "A", "2": "B"}}
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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

def construct_prompt(question, choices, context="", domain="multidomain"):
    """
    Construct the full prompt for the LLM based on domain
    """
    choices_str = format_choices(choices)
    template = USER_PROMPT_TEMPLATES.get(domain, USER_PROMPT_TEMPLATES["multidomain"])
    
    # For precision_critical, context is not needed
    if domain == "precision_critical":
        return template.format(
            question=question,
            choices=choices_str
        )
    
    return template.format(
        context=context if context else "Không có thông tin tham khảo cụ thể.",
        question=question,
        choices=choices_str
    )

def construct_batch_prompt(items, domain="multidomain"):
    """
    Construct the prompt for a batch of questions (same domain).
    items: list of dicts, each containing 'question', 'choices', optional 'context'
    domain: domain name to select appropriate template
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
    
    # Select template based on domain
    if domain.upper() == "STEM":
        template = BATCH_USER_PROMPT_TEMPLATE_STEM
    else:
        template = BATCH_USER_PROMPT_TEMPLATE
        
    return template.format(
        num_questions=len(items),
        questions_content="\n----------------\n".join(questions_content)
    )

import re
from typing import Dict, Tuple

# Import config for domain configurations
try:
    from config import DOMAIN_CONFIGS
except ImportError:
    DOMAIN_CONFIGS = None

class QuestionRouter:
    def __init__(self):
        # Keywords for Precision Critical (harmful/sensitive content)
        self.precision_critical_keywords = [
            'làm cách nào để', 'cách thức', 'phương pháp', 'hướng dẫn',
            'tránh', 'lừa đảo', 'gian lận', 'trốn', 'vi phạm',
            'bất hợp pháp', 'phi pháp', 'trái pháp luật', 'hack', 'phá hoại'
        ]
        
        # Negative keywords that indicate rejection is needed
        self.rejection_keywords = [
            'không thể chia sẻ', 'từ chối', 'không được phép',
            'không thể trả lời', 'vi phạm', 'bất hợp pháp'
        ]
        
        # Math symbols and keywords for STEM
        self.stem_indicators = [
            r'\$', r'\\', 'sin', 'cos', 'tan', 'log', 'sqrt', 'frac',
            'alpha', 'beta', 'theta', 'delta', 'sigma', 'pi',
            '∫', '∑', '∏', '√', '≤', '≥', '≠', '±', '×', '÷',
            'phương trình', 'tính', 'tính toán', 'giải', 'chứng minh',
            'độ co giãn', 'điện trở', 'công suất', 'vận tốc', 'gia tốc',
            'khối lượng', 'thể tích', 'diện tích', 'chu vi', 'bán kính',
            'đạo hàm', 'tích phân', 'logarit', 'lũy thừa', 'căn bậc',
            'xác suất', 'thống kê', 'ma trận', 'vector', 'hàm số',
            'định lý', 'công thức', 'biểu thức', 'tỉ lệ', 'phần trăm',
            'véc-tơ', 'lực', 'áp suất', 'nhiệt độ', 'năng lượng',
            'động lượng', 'momen', 'trọng lực', 'ma sát', 'gia tốc'
        ]
        
        # Trigonometry terms for answer choices
        self.trig_terms = [
            'tangent', 'sine', 'cosine', 'cotangent', 'secant', 'cosecant',
            'arcsin', 'arccos', 'arctan', 'arcsec', 'arccsc', 'arccot'
        ]
        
    def classify_question(self, question_text: str, choices: list) -> Tuple[str, float]:
        """
        Classify a question into one of the 5 domains
        Returns: (domain_name, confidence_score)
        """
        # 1. Check for RAG - has explicit context marker
        if "Đoạn thông tin" in question_text or "Thông tin:" in question_text:
            return "RAG", 1.0
        
        # 2. Check for Precision Critical - harmful/sensitive questions
        if self._is_precision_critical(question_text, choices):
            return "PRECISION_CRITICAL", 0.9
        
        # 3. Check for STEM - math and logical reasoning
        if self._is_stem(question_text, choices):
            return "STEM", 0.85
        
        # 4. Check for Compulsory - specific factual questions
        if self._is_compulsory(question_text):
            return "COMPULSORY", 0.75
        
        # 5. Default to Multidomain
        return "MULTIDOMAIN", 0.6
    
    def _is_precision_critical(self, question_text: str, choices: list) -> bool:
        """
        Detect questions that should NOT be answered (harmful/sensitive)
        These typically have a rejection option in the choices
        """
        question_lower = question_text.lower()
        
        # Check if question asks about harmful/illegal activities
        has_harmful_intent = any(
            keyword in question_lower 
            for keyword in self.precision_critical_keywords
        )
        
        # Check if there's a rejection choice
        has_rejection_choice = any(
            any(reject_word in choice.lower() for reject_word in self.rejection_keywords)
            for choice in choices
        )
        
        return has_harmful_intent and has_rejection_choice
    
    def _is_stem(self, question_text: str, choices: list = None) -> bool:
        """
        Detect math and logical reasoning questions
        """
        # Check for LaTeX math symbols
        if re.search(r'\$.*\$', question_text):
            return True
        
        # Check for math/science keywords and symbols
        question_lower = question_text.lower()
        stem_count = sum(
            1 for indicator in self.stem_indicators
            if indicator.lower() in question_lower or re.search(indicator, question_text)
        )
        
        # Check for trigonometry terms in choices (strong indicator)
        if choices:
            choices_text = ' '.join(choices).lower()
            trig_count = sum(
                1 for term in self.trig_terms
                if term.lower() in choices_text
            )
            if trig_count >= 3:  # 3+ trig terms in choices = definitely STEM
                return True
        
        # Lower threshold to 1 (single strong indicator is enough)
        return stem_count >= 1
    
    def _is_compulsory(self, question_text: str) -> bool:
        """
        Detect questions that must be answered correctly (specific factual questions)
        Typically shorter questions asking for specific facts
        """
        question_lower = question_text.lower()
        
        # Factual question patterns
        factual_patterns = [
            r'^(ai|gì|đâu|khi nào|năm nào|nào|loài nào|ngôi|tổ chức nào)',
            r'(là gì|là ai|là đâu|vào năm nào|được|là)',
            r'(chức năng|vai trò|ý nghĩa|tác dụng|mục đích)',
        ]
        
        # Check for factual question patterns
        has_factual_pattern = any(
            re.search(pattern, question_lower) 
            for pattern in factual_patterns
        )
        
        # Compulsory questions are usually shorter and more direct
        is_short = len(question_text) < 200
        
        return has_factual_pattern and is_short
    
    def get_strategy_config(self, domain: str) -> Dict:
        """
        Get the processing strategy for each domain
        Uses config.py if available, otherwise falls back to defaults
        """
        # Use DOMAIN_CONFIGS from config.py if available
        if DOMAIN_CONFIGS and domain in DOMAIN_CONFIGS:
            return DOMAIN_CONFIGS[domain]
        
        # Fallback strategies (if config.py not available)
        strategies = {
            "PRECISION_CRITICAL": {
                "use_rag": False,
                "model": "small",
                "temperature": 0.1,
                "top_k_docs": 0,
            },
            "COMPULSORY": {
                "use_rag": True,
                "model": "large",
                "temperature": 0.2,
                "top_k_docs": 3,
            },
            "RAG": {
                "use_rag": False,
                "model": "large",
                "temperature": 0.3,
                "top_k_docs": 0,
            },
            "STEM": {
                "use_rag": True,
                "model": "large",
                "temperature": 0.1,
                "top_k_docs": 2,
            },
            "MULTIDOMAIN": {
                "use_rag": True,
                "model": "small",
                "temperature": 0.4,
                "top_k_docs": 5,
            }
        }
        
        return strategies.get(domain, strategies["MULTIDOMAIN"])
    
    def analyze_batch(self, items: list) -> Dict[str, list]:
        """
        Analyze a batch of questions and group them by domain
        """
        domain_groups = {
            "PRECISION_CRITICAL": [],
            "COMPULSORY": [],
            "RAG": [],
            "STEM": [],
            "MULTIDOMAIN": []
        }
        
        for item in items:
            domain, confidence = self.classify_question(
                item['question'], 
                item['choices']
            )
            item['domain'] = domain
            item['confidence'] = confidence
            domain_groups[domain].append(item)
        
        return domain_groups

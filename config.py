"""
Configuration file for VNPT AI Hackathon Track 2
Simplified - Only includes actively used configurations
"""

# ============================================================================
# COMPETITION INFO (Reference Only)
# ============================================================================

COMPETITION_INFO = {
    "small_model_quota": 1000,  # requests/day
    "large_model_quota": 500,   # requests/day
    "embedding_quota": 500,     # requests/minute
    "max_submissions_per_day": 5,
}

# ============================================================================
# DOMAIN-SPECIFIC CONFIGURATIONS (ACTIVELY USED)
# ============================================================================

DOMAIN_CONFIGS = {
    "PRECISION_CRITICAL": {
        "use_rag": False,
        "model": "small",
        "temperature": 0.1,
        "top_k_docs": 0,
        "use_batch_processing": False,  # True = batch, False = single
        "batch_size": 10,  # Số câu mỗi batch (chỉ dùng khi use_batch_processing=True)
        "description": "Câu hỏi bắt buộc không được trả lời - nội dung nhạy cảm",
    },
    
    "COMPULSORY": {
        "use_rag": True,
        "model": "small",
        "temperature": 0.2,
        "top_k_docs": 2,
        "use_batch_processing": False,
        "batch_size": 10,
        "description": "Câu hỏi bắt buộc phải trả lời đúng - văn hóa, lịch sử VN",
    },
    
    "RAG": {
        "use_rag": False,  # Context already in question
        "model": "small",
        "temperature": 0.3,
        "top_k_docs": 0,
        "use_batch_processing": False,
        "batch_size": 10,
        "description": "Câu hỏi đọc hiểu văn bản dài",
    },
    
    "STEM": {
        "use_rag": True,
        "model": "small",
        "temperature": 0.7,  # Tăng để diverse cho majority voting
        "top_k_docs": 1,  # Giảm để tránh nhiễu
        "n": 5,  # Generate 5 completions cho majority voting
        "use_majority_voting": False,  # Switch to True to enable majority voting
        "use_self_verification": False,  # Switch to True to enable self-verification instead of voting
        "verification_attempts": 2,  # Max retry attempts for verification
        "use_batch_processing": False,  # True = batch, False = single (voting/verification)
        "batch_size": 5,  # Batch size when use_batch_processing=True
        "description": "Câu hỏi toán học và tư duy logic",
    },
    
    "MULTIDOMAIN": {
        "use_rag": True,
        "model": "small",
        "temperature": 0.4,
        "top_k_docs": 3,
        "use_batch_processing": False,
        "batch_size": 10,
        "description": "Câu hỏi đa lĩnh vực",
    }
}

# ============================================================================
# BATCH PROCESSING
# ============================================================================

BATCH_CONFIG = {
    "batch_size": 10,  
    "max_retries": 2,
    "fallback_to_individual": True,
    "use_json_format": True,  
}

# ============================================================================
# OUTPUT
# ============================================================================

OUTPUT_CONFIG = {
    "output_csv": "output.csv",
    "verbose": True,
    "show_domain_stats": True,
}


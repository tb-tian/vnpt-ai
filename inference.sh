#!/bin/bash
# inference.sh - Script để chạy pipeline từ đầu đến cuối
# Bao gồm: khởi tạo vector database, xử lý dữ liệu, và chạy prediction

set -e  # Exit on error

echo "=========================================="
echo "VNPT AI - Complete Inference Pipeline"
echo "=========================================="
echo ""

# ============================================================================
# STEP 1: Kiểm tra môi trường
# ============================================================================
echo "Step 1: Checking environment..."
echo "----------------------------------------"

# Kiểm tra Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found!"
    exit 1
fi
echo "Python: $(python3 --version)"

# Kiểm tra file input
if [ ! -f "/code/private_test.json" ]; then
    echo "ERROR: /code/private_test.json not found!"
    exit 1
fi
echo "Input file: /code/private_test.json"

# Kiểm tra API keys
if [ ! -f "/code/api-keys.json" ]; then
    echo "WARNING: /code/api-keys.json not found!"
    echo "Pipeline may fail without API keys"
else
    echo "API keys: /code/api-keys.json"
fi

echo ""

# ============================================================================
# STEP 2: Khởi tạo Vector Database (nếu chưa có)
# ============================================================================
echo "Step 2: Initializing Vector Database..."
echo "----------------------------------------"

# Kiểm tra và tạo corpus nếu cần
if [ ! -d "/code/corpus" ] || [ ! "$(ls -A /code/corpus)" ]; then
    echo "Corpus directory empty or not found"
    
    if [ -f "/code/crawl.py" ]; then
        echo "Running crawl.py to build corpus..."
        mkdir -p /code/corpus
        python3 /code/crawl.py || {
            echo "WARNING: Crawl failed, creating minimal corpus..."
            echo "Sample corpus data for vector database initialization" > /code/corpus/sample.txt
        }
    else
        echo "WARNING: crawl.py not found, creating minimal corpus..."
        mkdir -p /code/corpus
        echo "Sample corpus data for vector database initialization" > /code/corpus/sample.txt
    fi
else
    echo "Corpus directory found: $(ls /code/corpus | wc -l) files"
fi

# Khởi tạo vector database
if [ -d "/code/vector_db_langchain" ] && [ "$(ls -A /code/vector_db_langchain)" ]; then
    echo "Vector database already exists, skipping initialization"
else
    echo "Vector database not found, initializing..."
    
    # Chạy ingest_data.py để khởi tạo vector database
    if [ -f "/code/ingest_data.py" ]; then
        echo "Running data ingestion..."
        python3 /code/ingest_data.py || {
            echo "WARNING: Data ingestion failed, will initialize on first query"
        }
    else
        echo "WARNING: ingest_data.py not found, vector DB will auto-initialize on first query"
    fi
fi

echo ""

# ============================================================================
# STEP 3: Xử lý dữ liệu (nếu cần)
# ============================================================================
echo "Step 3: Data Processing (if needed)..."
echo "----------------------------------------"

# Nếu có script xử lý dữ liệu bổ sung, chạy tại đây
if [ -f "/code/data_process.py" ]; then
    echo "Running data processing..."
    python3 /code/data_process.py || echo "WARNING: Data processing skipped"
else
    echo "No additional data processing needed"
fi

echo ""

# ============================================================================
# STEP 4: Chạy Prediction Pipeline
# ============================================================================
echo "Step 4: Running Prediction Pipeline..."
echo "----------------------------------------"
echo "Reading from: /code/private_test.json"
echo "Output to:"
echo "  - /code/submission.csv"
echo "  - /code/submission_time.csv"
echo ""

python3 /code/predict.py

echo ""

# ============================================================================
# STEP 5: Kiểm tra kết quả
# ============================================================================
echo "Step 5: Validating Output..."
echo "----------------------------------------"

# Kiểm tra file tồn tại
if [ ! -f "/code/submission.csv" ]; then
    echo "ERROR: submission.csv not generated!"
    exit 1
fi

if [ ! -f "/code/submission_time.csv" ]; then
    echo "ERROR: submission_time.csv not generated!"
    exit 1
fi

echo "Output files generated"

# Kiểm tra định dạng submission.csv
csv_cols=$(head -n 1 /code/submission.csv | awk -F',' '{print NF}')
if [ "$csv_cols" -ne 2 ]; then
    echo "ERROR: submission.csv has wrong format (expected 2 columns, got $csv_cols)"
    exit 1
fi
echo "submission.csv format: OK (2 columns)"

# Kiểm tra định dạng submission_time.csv
time_cols=$(head -n 1 /code/submission_time.csv | awk -F',' '{print NF}')
if [ "$time_cols" -ne 3 ]; then
    echo "ERROR: submission_time.csv has wrong format (expected 3 columns, got $time_cols)"
    exit 1
fi
echo "submission_time.csv format: OK (3 columns)"

# Đếm số dòng (trừ header)
submission_lines=$(($(wc -l < /code/submission.csv) - 1))
timing_lines=$(($(wc -l < /code/submission_time.csv) - 1))

echo "Processed $submission_lines questions"

if [ "$submission_lines" -ne "$timing_lines" ]; then
    echo "WARNING: submission.csv and submission_time.csv have different number of lines"
fi

# Hiển thị mẫu kết quả
echo ""
echo "Sample output (first 3 lines):"
echo "--- submission.csv ---"
head -n 4 /code/submission.csv
echo ""
echo "--- submission_time.csv ---"
head -n 4 /code/submission_time.csv

echo ""
echo "=========================================="
echo "SUCCESS: Pipeline completed!"
echo "=========================================="
echo "Output files:"
echo "  - /code/submission.csv ($submission_lines answers)"
echo "  - /code/submission_time.csv ($timing_lines answers with timing)"
echo "=========================================="

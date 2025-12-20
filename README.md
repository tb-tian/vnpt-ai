# VNPT AI - Age of Alnicorns - Track 2: The Builder
## Team Submission Repository

---

## 📋 Table of Contents
1. [Pipeline Flow](#pipeline-flow)
2. [Data Processing](#data-processing)
3. [Resource Initialization](#resource-initialization)
4. [Installation & Setup](#installation--setup)
5. [Running the System](#running-the-system)
6. [Docker Instructions](#docker-instructions)
7. [Project Structure](#project-structure)

---

## 🔄 Pipeline Flow

### System Architecture

```
Input (private_test.json)
        ↓
   [Question Router]
        ↓
    ┌───┴───┬───────────┬──────────────┬────────────┐
    │       │           │              │            │
 [STEM]  [RAG]  [COMPULSORY]  [PRECISION]  [MULTIDOMAIN]
    │       │           │              │            │
    ↓       ↓           ↓              ↓            ↓
[Voting] [Context] [RAG+Batch]  [Batch Process] [General]
    │       │           │              │            │
    └───┬───┴───────────┴──────────────┴────────────┘
        ↓
    [LLM Response]
        ↓
   [Post-process]
        ↓
 Output (submission.csv, submission_time.csv)
```

### Pipeline Components

1. **Question Router (router_logic.py)**
   - Classifies questions into domains using rule-based + LLM hybrid approach
   - Domains: STEM, RAG, COMPULSORY, PRECISION_CRITICAL, MULTIDOMAIN
   - Returns domain and confidence score

2. **RAG System (rag_langchain.py)**
   - Hybrid retrieval: BM25 + Vector Search (ChromaDB)
   - Langchain-based implementation
   - Retrieves relevant context from corpus documents

3. **Domain-Specific Processing**
   - **STEM**: Majority voting (5 completions) or self-verification
   - **RAG**: Extracts context from question text
   - **COMPULSORY**: Uses RAG with Vietnamese culture/history corpus
   - **PRECISION_CRITICAL**: Batch processing with low temperature
   - **MULTIDOMAIN**: General LLM inference

4. **LLM Integration (get_response.py)**
   - Supports small and large models
   - Temperature control per domain
   - JSON output formatting for classification

5. **Post-processing**
   - Answer extraction with regex patterns
   - Validation against number of choices
   - Fallback to 'A' for invalid answers

---

## 📊 Data Processing

### Data Collection

1. **Corpus Data (corpus/)**
   - Wikipedia articles in Vietnamese
   - Focused on Vietnamese culture, history, and general knowledge
   - Text files processed and indexed

### Data Cleaning & Preprocessing

1. **Text Processing**
   - UTF-8 encoding handling
   - Removal of special characters
   - Normalization of Vietnamese diacritics

2. **Document Chunking**
   - Split long documents into manageable chunks
   - Overlap strategy for context preservation
   - Chunk size optimized for embedding model

3. **Embedding Generation**
   - Uses competition-provided embedding API
   - Cached embeddings for efficiency
   - Batch processing to respect rate limits

---

## ⚙️ Resource Initialization

### 1. Vector Database Setup

**Prerequisites:**
- Corpus documents in `corpus/` directory
- Embedding API keys in `api-keys.json`

**Initialization Steps:**

```bash
# Step 1: Prepare corpus data
python ingest_data.py

# Step 2: Build vector database (ChromaDB)
# The RAG system automatically initializes on first run
```

**Vector Database Structure:**
```
vector_db_langchain/
├── chroma.sqlite3          # ChromaDB metadata
├── embeddings/             # Stored embeddings
└── indices/                # BM25 indices
```

### 2. BM25 Index

- Built automatically during RAG initialization
- Uses rank_bm25 library
- Stored in memory for fast retrieval

### 3. Required Files

Ensure these files exist before running:

```
api-keys.json              # API keys for LLM and embedding
config.py                  # Domain configurations
prompt_templates.py        # System prompts and templates
router_logic.py            # Question routing logic
corpus/                    # Wikipedia corpus (auto-downloaded if missing)
```

### 4. API Keys Configuration

Create `api-keys.json` with the following structure:

```json
{
  "SMALL_MODEL_API_KEY": "your_small_model_key",
  "LARGE_MODEL_API_KEY": "your_large_model_key",
  "EMBEDDING_API_KEY": "your_embedding_key"
}
```

---

## 🚀 Installation & Setup

### Local Setup (Without Docker)

1. **Clone Repository**
```bash
git clone <repository_url>
cd vnpt-ai
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
python -m spacy download vi_core_news_lg
```

3. **Prepare Resources**
```bash
# Ingest corpus data and build vector database
python ingest_data.py

# Verify system setup
python final_system_check.py
```

4. **Test the Pipeline**
```bash
# Run on validation set
python main.py
```

### Environment Requirements

- Python 3.8+
- CUDA 12.2 (for GPU acceleration)
- 8GB+ RAM
- 5GB+ disk space for vector database

---

## 🏃 Running the System

### Standard Execution

```bash
# Run prediction pipeline
python predict.py
```

**Input:** `/code/private_test.json`  
**Output:** 
- `/code/submission.csv` - Answers
- `/code/submission_time.csv` - Answers with inference time

### Input Format

```json
[
  {
    "qid": "test_0001",
    "question": "Question text here?",
    "choices": ["Choice A", "Choice B", "Choice C", "Choice D"]
  }
]
```

### Output Format

**submission.csv:**
```csv
qid,answer
test_0001,A
test_0002,B
test_0003,C
```

**submission_time.csv:**
```csv
qid,answer,time
test_0001,A,1.2345
test_0002,B,2.9087
test_0003,C,1.0021
```

---

## 🐳 Docker Instructions

### Building the Docker Image

```bash
# Build image with tag
sudo docker build -t vnpt-ai-submission .

# Build takes ~15-20 minutes depending on network speed
```

### Running the Container

```bash
# Run with GPU support
sudo docker run --gpus all \
  -v /path/to/data/private_test.json:/code/private_test.json \
  vnpt-ai-submission

# Output files will be in /code/ inside container
```

### Testing Docker Locally

```bash
# 1. Prepare test data
mkdir -p test_data
cp data/val.json test_data/private_test.json

# 2. Build image
sudo docker build -t vnpt-ai-test .

# 3. Run container
sudo docker run --gpus all \
  -v $(pwd)/test_data/private_test.json:/code/private_test.json \
  vnpt-ai-test

# 4. Check outputs in container
docker ps -a  # Get container ID
docker cp <container_id>:/code/submission.csv ./
docker cp <container_id>:/code/submission_time.csv ./
```

### Pushing to Docker Hub

```bash
# 1. Tag image
docker tag vnpt-ai-submission <dockerhub_username>/vnpt-ai:latest

# 2. Login to Docker Hub
docker login

# 3. Push image (MUST be before 23:59 UTC+7 Dec 19, 2025)
docker push <dockerhub_username>/vnpt-ai:latest
```

### Docker Image Specifications

- **Base Image:** nvidia/cuda:12.2.0-devel-ubuntu20.04
- **Python Version:** 3.8
- **CUDA Version:** 12.2
- **Image Size:** ~8-10GB (includes models and dependencies)

---

## 📁 Project Structure

```
vnpt-ai/
├── Dockerfile                    # Docker configuration
├── inference.sh                  # Bash script for pipeline execution
├── predict.py                    # Entry point (reads private_test.json)
├── main.py                       # Core processing logic
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── config.py                     # Domain-specific configurations
├── router_logic.py               # Question classification router
├── rag_langchain.py              # RAG system with LangChain
├── prompt_templates.py           # LLM prompts and templates
├── get_response.py               # LLM API interface
├── get_embedding.py              # Embedding API interface
│
├── api-keys.json                 # API keys (not in repo)
│
├── corpus/                       # Wikipedia corpus documents
│   ├── 2020.txt
│   ├── 2025.txt
│   └── ...
│
├── vector_db_langchain/          # ChromaDB vector database
│   └── (generated during setup)
│
└── data/                         # Test/validation datasets
    ├── val.json
    ├── test.json
    └── ...
```

---

## 🔧 Configuration

### Domain Settings (config.py)

Each domain has customizable parameters:

```python
DOMAIN_CONFIGS = {
    "STEM": {
        "use_majority_voting": True,
        "model": "small",
        "temperature": 0.1,
        "num_completions": 5
    },
    "COMPULSORY": {
        "use_rag": True,
        "model": "small",
        "temperature": 0.2,
        "top_k_docs": 2,
        "use_batch_processing": True
    }
}
```

---

## 📝 Notes

- All timestamps in UTC+7
- Submission deadline: December 19, 2025, 23:59 (UTC+7)
- Docker image must be pushed before deadline
- Repository must be public and frozen after submission

---

**Last Updated:** December 19, 2025
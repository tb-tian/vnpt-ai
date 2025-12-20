# Mô Tả Giải Pháp & Pipeline - VNPT AI Hackathon (Track 2)

## 1. Tổng quan

Hệ thống được thiết kế để giải quyết bài toán trả lời câu hỏi trắc nghiệm đa lĩnh vực (Multiple Choice Question Answering) trong khuôn khổ cuộc thi VNPT AI Hackathon. Giải pháp tập trung vào việc xây dựng một pipeline tự động, kết hợp giữa kỹ thuật **Retrieval-Augmented Generation (RAG)** và **Phân loại câu hỏi** để tối ưu hóa độ chính xác và hiệu năng sử dụng API.

Hệ thống đã xây dựng khả năng nhận diện ngữ cảnh của từng câu hỏi để áp dụng chiến lược xử lý khác nhau và tham số mô hình (Temperature, Top-k) phù hợp nhất đối với các dạng câu hỏi khác nhau.

## 2. Kiến trúc hệ thống

Hệ thống có các thành phần chính sau:

1. **Crawl Data & Indexing:** Crawl data dựa trên file val.json và file test.json mà BTC cung cấp, sau đó build vector database dựa trên corpus đấy.
2. **Input Processing & Routing:** Đọc và phân loại câu hỏi.
3. **Retrieval Engine (RAG):** Truy xuất thông tin liên quan (đối với các câu hỏi cần kiến thức ngoài).
4. **Domain-Specific Inference:** Suy luận và generate câu trả lời dựa trên đặc thù lĩnh vực.
5. **Post-processing:** Chuẩn hóa định dạng đầu ra.

### Sơ đồ luồng dữ liệu tóm tắt:

```mermaid
Input (JSON) 
  --> [RAG Keyword Check] --(Yes)--> [RAG Buffer]
  --> [LLM Router] --(Classify)--> [Domain Buffers]
        |--> STEM (Toán/Logic)
        |--> COMPULSORY (Sử/Địa/Văn hóa)
        |--> PRECISION_CRITICAL (An toàn nội dung)
        |--> MULTIDOMAIN (Đa lĩnh vực)
  --> [Processing Engine]
        |--> Batch Processing (Nhóm câu hỏi)
        |--> Chain-of-Thought / Voting (Cho STEM)
  --> [Output Generator] --> CSV Submission

```

## 3. Chi tiết

### 3.1. Phân loại Câu hỏi

Hệ thống sử dụng Hybrid Routing để cân bằng giữa tốc độ và độ chính xác:

* **Bộ lọc từ khóa (Rule-based):**
* Phát hiện câu hỏi đọc hiểu (RAG domain) thông qua các cụm từ như "Dựa trên đoạn văn", "Thông tin:".


* **Phân loại bằng LLM (LLM-based Classification):**
* Các câu hỏi còn lại sẽ được gom nhóm (batch size = 10) và gửi tới mô hình `vnptai_hackathon_small`.
* Mô hình đóng vai trò như một bộ định tuyến, gán nhãn câu hỏi vào một trong các miền: `STEM`, `PRECISION_CRITICAL`, `COMPULSORY`, hoặc `MULTIDOMAIN`.



### 3.2. Hệ thống truy xuất thông tin

Module RAG (`rag_langchain.py`) chịu trách nhiệm cung cấp ngữ cảnh cho các câu hỏi yêu cầu kiến thức bên ngoài (đặc biệt là nhóm `COMPULSORY` - Lịch sử, Văn hóa Việt Nam).

* **Dữ liệu:** Sử dụng Corpus từ Wikipedia tiếng Việt và DuckDuckGo (đã được làm sạch và xử lý qua `crawl.py`).
* **Vector Database:** Sử dụng **ChromaDB** để lưu trữ các embedding vector.
* **Chiến lược Retrieval:** Sử dụng `EnsembleRetriever` kết hợp giữa:
1. **Vector Search:** Tìm kiếm ngữ nghĩa sử dụng mô hình `vnptai_hackathon_embedding`.
2. **BM25 (Best Matching 25):** Tìm kiếm dựa trên tần suất từ khóa chính xác.


* Tỷ trọng: 50% Vector + 50% BM25 để đảm bảo bắt được cả ngữ nghĩa và từ khóa cụ thể.



### 3.3. Chiến lược Xử lý theo Lĩnh vực (Domain Strategies)

Mỗi lĩnh vực được cấu hình riêng biệt trong `config.py` và `prompt_templates.py` để tối ưu hóa kết quả:

#### A. STEM (Toán học & Logic)

* **Đặc điểm:** Yêu cầu tư duy logic, tính toán chính xác, dễ sai sót nếu chỉ dùng LLM thông thường.
* **Chiến lược:**
* **Chain-of-Thought (CoT):** Prompt yêu cầu giải quyết qua 4 bước: Phân tích đề -> Xác định công thức -> Tính toán -> Kiểm tra.
* **Majority Voting (Bỏ phiếu đa số):** Sinh ra 5 câu trả lời (n=5) với `temperature=0.7`, sau đó chọn đáp án xuất hiện nhiều nhất để loại bỏ các sai số ngẫu nhiên.
* **Self-Verification (Tự kiểm chứng):** (Tùy chọn) Mô hình tự sinh lời giải, sau đó đóng vai trò "Reviewer" để tìm lỗi sai trong chính lời giải đó.



#### B. PRECISION_CRITICAL (An toàn nội dung)

* **Đặc điểm:** Các câu hỏi nhạy cảm, yêu cầu hành vi vi phạm pháp luật hoặc đạo đức.
* **Chiến lược:**
* Sử dụng `temperature=0.1` (rất thấp) để đảm bảo tính nhất quán.
* Prompt đặc biệt hướng dẫn mô hình **từ chối trả lời** và chọn đáp án mang tính phủ định (như "Tôi không thể trả lời...", "Không được phép...").
* Cơ chế `ContentPolicyError` handling: Nếu API trả về lỗi policy, hệ thống tự động quét các đáp án để chọn đáp án từ chối phù hợp.



#### C. COMPULSORY (Kiến thức Bắt buộc)

* **Đặc điểm:** Các câu hỏi về sự kiện, lịch sử, văn hóa Việt Nam.
* **Chiến lược:**
* Kích hoạt module RAG để truy xuất thông tin từ Vector DB.
* Sử dụng thông tin truy xuất được làm ngữ cảnh (Context) trong Prompt.



#### D. RAG (Đọc hiểu) & MULTIDOMAIN

* **RAG:** Trích xuất ngữ cảnh trực tiếp từ nội dung câu hỏi (không cần truy vấn DB ngoài).
* **MULTIDOMAIN:** Sử dụng kiến thức tổng quát của mô hình Large/Small tùy theo cấu hình.

### 3.4. Tối ưu hóa Hiệu năng (Optimization)

Để đáp ứng giới hạn thời gian và Quota API, hệ thống áp dụng các kỹ thuật:

* **Batch Processing:**
* Gom các câu hỏi cùng lĩnh vực (trừ STEM) vào các batch (kích thước 10).
* Gửi 1 request duy nhất chứa 10 câu hỏi để giảm chi phí network và số lần gọi API.
* Sử dụng định dạng đầu ra JSON để dễ dàng phân tích cú pháp (Parsing).


* **Streaming Write:**
* Kết quả được ghi xuống ổ đĩa ngay lập tức sau khi xử lý (từng câu hoặc từng batch).
* Hỗ trợ cơ chế **Resume**: Nếu hệ thống bị ngắt quãng, khi chạy lại sẽ tự động bỏ qua các câu hỏi đã có trong file kết quả.


* **Retry Mechanism:**
* Tự động thử lại (Retry) với độ trễ tăng dần (Exponential backoff) khi gặp lỗi kết nối hoặc lỗi server 5xx.



## 4. Quy trình Vận hành (Workflow)

Quy trình xử lý một tập dữ liệu đầu vào (`private_test.json`) diễn ra như sau:

1. **Khởi tạo:**
* Load cấu hình và API keys.
* Kiểm tra và khởi tạo Vector Database (nếu chưa có).
* Chuẩn bị file output CSV.


2. **Vòng lặp Xử lý (Main Loop):**
* **Bước 1:** Đọc từng câu hỏi từ Input.
* **Bước 2:** Kiểm tra nhanh từ khóa. Nếu là câu hỏi RAG -> Đưa vào `RAG Buffer`.
* **Bước 3:** Nếu không phải RAG -> Đưa vào `Non-RAG Buffer`. Khi buffer đầy (10 câu) -> Gọi LLM để phân loại domain.
* **Bước 4:** Dựa trên domain được phân loại, đưa câu hỏi vào `Domain Buffer` tương ứng.
* **Bước 5:** Khi một `Domain Buffer` đầy (ví dụ 10 câu COMPULSORY):
* Tạo Batch Prompt.
* Gọi API (Batch Inference).
* Ghi kết quả ra file `submission.csv` và `submission_time.csv`.


* **Bước 6:** Đối với domain `STEM`: Xử lý đơn lẻ (hoặc batch nhỏ) với quy trình suy luận, xây dựng CoT.


3. **Kết thúc:**
* Xử lý toàn bộ các câu hỏi còn tồn đọng trong các Buffer.
* Sắp xếp lại file kết quả theo `qid` để đảm bảo đúng định dạng nộp bài.



## 5. Kết luận

Giải pháp được xây dựng với tư duy "chia để trị" (Divide and Conquer), tách bài toán lớn thành các bài toán con dựa trên đặc thù câu hỏi. Việc kết hợp RAG giúp nâng cao độ chính xác cho các câu hỏi cần kiến thức thực tế, trong khi Prompt Engineering chuyên sâu giúp giải quyết tốt các bài toán tư duy logic. Kiến trúc Batch Processing đảm bảo hệ thống vận hành hiệu quả trong giới hạn tài nguyên cho phép.
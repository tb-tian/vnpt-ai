# import json
# import os
# import wikipediaapi
# import time
# from tqdm import tqdm
# from underthesea import pos_tag
# import re

# import csv

# # Configuration
# DATA_DIR = "corpus"
# LOG_FILE = "crawl_log.csv"
# WIKI_LANG = "vi"
# USER_AGENT = "VNPT_Hackathon_Bot/1.0 (contact@example.com)"

# # Initialize Wikipedia API
# wiki_wiki = wikipediaapi.Wikipedia(
#     user_agent=USER_AGENT,
#     language=WIKI_LANG,
#     extract_format=wikipediaapi.ExtractFormat.WIKI
# )

# def extract_keywords(text):
#     """
#     Extract keywords using underthesea POS tagging.
#     Prioritize Proper Nouns (Np) and Nouns (N).
#     """
#     try:
#         # POS Tagging: [('Hà Nội', 'Np'), ('là', 'V'), ...]
#         tags = pos_tag(text)
        
#         keywords = []
#         # Priority 1: Proper Nouns (Np) - Names, Places, Organizations
#         proper_nouns = [word for word, tag in tags if tag == 'Np']
        
#         # Priority 2: Common Nouns (N) - Concepts, Objects
#         nouns = [word for word, tag in tags if tag == 'N' and len(word) > 1]
        
#         # Combine: Proper Nouns first, then Nouns
#         keywords.extend(proper_nouns)
        
#         # If we don't have enough proper nouns, add common nouns
#         if len(keywords) < 3:
#             keywords.extend(nouns)
            
#         # Deduplicate while preserving order
#         seen = set()
#         unique_keywords = []
#         for k in keywords:
#             if k.lower() not in seen:
#                 seen.add(k.lower())
#                 unique_keywords.append(k)
                
#         return unique_keywords
#     except Exception as e:
#         print(f"Error in keyword extraction: {e}")
#         # Fallback to simple split if NLP fails
#         return text.split()[:5]

# def clean_wiki_text(text):
#     """
#     Clean Wikipedia text: remove citations, footer sections, and empty lines.
#     """
#     # Remove citations [1], [2]
#     text = re.sub(r'\[\d+\]', '', text)
    
#     lines = text.split('\n')
#     cleaned_lines = []
    
#     # Headers that usually mark the end of the main article content
#     stop_headers = {
#         "Tham khảo", "Liên kết ngoài", "Xem thêm", "Chú thích", 
#         "Tài liệu tham khảo", "Đọc thêm", "Nguồn", "Ghi chú"
#     }
    
#     for line in lines:
#         stripped = line.strip()
#         if not stripped:
#             continue
            
#         # If the line is exactly a stop header (or very close to it), we stop processing
#         if stripped in stop_headers:
#             break
            
#         cleaned_lines.append(stripped)
        
#     return "\n".join(cleaned_lines)

# def search_and_save(query, filename_prefix):
#     """
#     Search Wikipedia for a query and save the page content.
#     Returns the filename if saved, else None.
#     """
#     try:
#         page = wiki_wiki.page(query)
        
#         if page.exists():
#             # Save to file
#             safe_title = page.title.replace("/", "_").replace(" ", "_")
#             filename = f"{safe_title}.txt"
#             filepath = os.path.join(DATA_DIR, filename)
            
#             if not os.path.exists(filepath):
#                 cleaned_text = clean_wiki_text(page.text)
                
#                 # Only save if content is substantial (e.g., > 200 chars)
#                 if len(cleaned_text) > 200:
#                     with open(filepath, "w", encoding="utf-8") as f:
#                         f.write(cleaned_text)
#                     return filename
#             else:
#                 return filename # Return filename even if already exists
#     except Exception as e:
#         print(f"Error fetching {query}: {e}")
        
#     return None

# def crawl_from_questions(json_file):
#     with open(json_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
        
#     print(f"Processing {len(data)} questions from {json_file}...")
    
#     # Prepare CSV logging - Open file in append mode once
#     file_exists = os.path.isfile(LOG_FILE)
#     csvfile = open(LOG_FILE, 'a', newline='', encoding='utf-8')
#     fieldnames = ['qid', 'keywords', 'files']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
#     if not file_exists:
#         writer.writeheader()
    
#     count = 0
#     try:
#         for item in tqdm(data):
#             question = item['question']
#             qid = item['qid']
            
#             # Skip if context is already provided
#             if "Đoạn thông tin" in question:
#                 continue
                
#             # Extract keywords to search
#             keywords = extract_keywords(question)
            
#             # Strategy: Search for the 2-3 most significant keywords
#             # For now, let's just take the first 3 keywords as a query, or iterate.
#             if keywords:
#                 top_keywords = keywords[:5]
#                 found_files = []
                
#                 # Construct a query from keywords
#                 query = " ".join(top_keywords) 
#                 saved_file = search_and_save(query, qid)
#                 if saved_file:
#                     count += 1
#                     found_files.append(saved_file)
                    
#                 # Try individual keywords (both Entities and Nouns)
#                 for kw in top_keywords:
#                     saved_file = search_and_save(kw, qid)
#                     if saved_file:
#                         count += 1
#                         found_files.append(saved_file)
                
#                 # Log to CSV immediately
#                 if found_files:
#                     writer.writerow({
#                         "qid": qid,
#                         "keywords": ", ".join(top_keywords),
#                         "files": ", ".join(list(set(found_files))) # Deduplicate files
#                     })
#                     csvfile.flush() # Ensure data is written to disk
                        
#         print(f"Downloaded {count} pages.")
#     finally:
#         csvfile.close()
#         print(f"Log saved to {LOG_FILE}")

# if __name__ == "__main__":
#     if not os.path.exists(DATA_DIR):
#         os.makedirs(DATA_DIR)
        
#     # Crawl from val.json
#     print("Crawling for Validation Set...")
#     crawl_from_questions("AInicorns_TheBuilder_public_v1.1/data/val.json")
    
#     # Crawl from test.json
#     print("Crawling for Test Set...")
#     crawl_from_questions("AInicorns_TheBuilder_public_v1.1/data/test.json")
import json
import os
import wikipediaapi
import time
import requests
from tqdm import tqdm
from underthesea import word_tokenize, pos_tag, ner
import re
import csv
import hashlib
from datetime import datetime
from bs4 import BeautifulSoup
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    "data_dir": "corpus_hybrid_bs4",
    "log_file": "crawl_hybrid_log_bs4.csv",
    "wiki_lang": "vi",
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "min_content_length": 80,
    "max_content_length": 15000,
    "rate_limit_delay": 0.2,
    "min_quality_score": 0.15,
    "max_pages_per_question": 5,
    "tfidf_max_features": 150,
    "similarity_threshold": 0.2,
    "max_queries_per_question": 12,
    "request_timeout": 10
}

wiki_wiki = wikipediaapi.Wikipedia(
    language=CONFIG["wiki_lang"],
    user_agent=CONFIG["user_agent"],
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

class EnhancedHybridQueryGenerator:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=CONFIG["tfidf_max_features"],
            min_df=1,
            max_df=0.85,
            ngram_range=(1, 3),
            stop_words=None
        )
        self.corpus_questions = []
        self.tfidf_matrix = None
        self.vocabulary = None
        
    def build_corpus(self, questions):
        self.corpus_questions = questions
        if questions:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(questions)
            self.vocabulary = self.tfidf_vectorizer.get_feature_names_out()
        else:
            self.vocabulary = None
            self.tfidf_matrix = None
    
    def extract_tfidf_keywords(self, question, top_n=15):
        if self.tfidf_matrix is None or self.vocabulary is None:
            return []
        
        try:
            question_vec = self.tfidf_vectorizer.transform([question])
            feature_array = np.array(self.vocabulary)
            tfidf_sorting = np.argsort(question_vec.toarray()).flatten()[::-1]
            top_keywords = feature_array[tfidf_sorting][:top_n]
            return list(top_keywords)
        except:
            return []
    
    def find_similar_questions(self, question, top_n=5):
        if self.tfidf_matrix is None:
            return []
        
        try:
            question_vec = self.tfidf_vectorizer.transform([question])
            similarities = (self.tfidf_matrix * question_vec.T).toarray().flatten()
            top_indices = np.argsort(similarities)[::-1][:top_n]
            similar = []
            for idx in top_indices:
                if similarities[idx] > CONFIG["similarity_threshold"]:
                    similar.append({
                        "question": self.corpus_questions[idx],
                        "similarity": float(similarities[idx])
                    })
            return similar
        except:
            return []
    
    def extract_keywords_from_similar(self, question, top_n=3):
        similar_questions = self.find_similar_questions(question, top_n=top_n)
        if not similar_questions:
            return []
        all_keywords = []
        for similar in similar_questions:
            sim_question = similar["question"]
            keywords = self.extract_basic_keywords(sim_question)
            all_keywords.extend(keywords)
        keyword_counts = Counter(all_keywords)
        return [kw for kw, _ in keyword_counts.most_common(10)]
    
    def extract_basic_keywords(self, text):
        keywords = []
        try:
            words = word_tokenize(text)
            tags = pos_tag(text)
            for word, tag in tags:
                if tag in ['Np', 'N', 'V', 'A'] and len(word) > 1:
                    keywords.append(word)
        except:
            keywords = text.split()[:10]
        return keywords

hybrid_generator = EnhancedHybridQueryGenerator()

def preprocess_question_enhanced(question):
    if not question:
        return ""
    
    if "Đoạn thông tin:" in question:
        parts = question.split("Câu hỏi:")
        if len(parts) > 1:
            question = parts[-1].strip()
    
    question = re.sub(r'\$.*?\$', '', question)
    question = re.sub(r'\\[a-zA-Z]+', '', question)
    question = re.sub(r'\{.*?\}', '', question)
    question = re.sub(r'\(.*?\)', '', question)
    
    stop_words = {"của", "và", "trong", "với", "để", "này", "là", "có", 
                 "được", "theo", "từ", "như", "bởi", "cho", "một", "các",
                 "hay", "hoặc", "nếu", "thì", "mà", "ở", "tại", "về"}
    
    words = question.split()
    filtered_words = []
    
    for word in words:
        word_lower = word.lower()
        if (len(word) > 1 and 
            word_lower not in stop_words and
            any(c.isalpha() for c in word)):
            filtered_words.append(word)
    
    processed = " ".join(filtered_words)
    
    if len(processed.split()) < 2:
        original_clean = re.sub(r'[^\w\s?]', ' ', question)
        return original_clean.strip()
    
    return processed.strip()

def preprocess_query(query):
    query = re.sub(r'[$\\]', '', query)
    query = re.sub(r'\{.*?\}', '', query)
    query = re.sub(r'\(.*?\)', '', query)
    query = re.sub(r'\[.*?\]', '', query)
    
    words = query.split()
    filtered_words = []
    for word in words:
        if any(c.isalpha() for c in word):
            if re.match(r'^[0-9\W]+$', word):
                continue
            filtered_words.append(word)
    
    cleaned_query = " ".join(filtered_words[:10])
    return cleaned_query.strip()

def extract_keywords_enhanced(question):
    try:
        words = word_tokenize(question)
        tags = pos_tag(question)
        ner_result = ner(question)
        
        keywords = {
            "entities": {"PER": [], "LOC": [], "ORG": [], "MISC": []},
            "proper_nouns": [],
            "common_nouns": [],
            "verbs": [],
            "adjectives": [],
            "key_phrases": [],
            "dates_numbers": [],
            "tfidf_keywords": [],
            "similar_keywords": []
        }
        
        for entity in ner_result:
            if len(entity) >= 4:
                entity_text, entity_type = entity[0], entity[3]
                if entity_type in keywords["entities"]:
                    keywords["entities"][entity_type].append(entity_text)
        
        for word, tag in tags:
            if tag == 'Np':
                keywords["proper_nouns"].append(word)
            elif tag == 'N':
                if len(word) > 2:
                    keywords["common_nouns"].append(word)
            elif tag.startswith('V'):
                keywords["verbs"].append(word)
            elif tag.startswith('A'):
                keywords["adjectives"].append(word)
        
        date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            r'\d{4}',
            r'(năm|ngày|tháng)\s+\d{1,4}'
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, question)
            keywords["dates_numbers"].extend(dates)
        
        for i in range(len(tags) - 1):
            if tags[i][1] in ['Np', 'N'] and tags[i+1][1] in ['Np', 'N', 'A']:
                phrase = f"{tags[i][0]} {tags[i+1][0]}"
                keywords["key_phrases"].append(phrase)
        
        try:
            tfidf_keywords = hybrid_generator.extract_tfidf_keywords(question, top_n=12)
            keywords["tfidf_keywords"] = tfidf_keywords if tfidf_keywords else []
        except:
            keywords["tfidf_keywords"] = []
        
        try:
            similar_keywords = hybrid_generator.extract_keywords_from_similar(question, top_n=4)
            keywords["similar_keywords"] = similar_keywords if similar_keywords else []
        except:
            keywords["similar_keywords"] = []
        
        for key in keywords:
            if isinstance(keywords[key], list):
                cleaned_list = []
                for item in keywords[key]:
                    if isinstance(item, str) and item.strip():
                        cleaned_list.append(item.strip())
                keywords[key] = list(set(cleaned_list))
        
        if not any(len(v) > 0 for v in [keywords["proper_nouns"], keywords["common_nouns"], 
                                         keywords["tfidf_keywords"], keywords["similar_keywords"]]):
            words = question.split()
            fallback_keywords = [w for w in words if len(w) > 2][:8]
            keywords["common_nouns"] = fallback_keywords
        
        return keywords
        
    except Exception as e:
        words = question.split()
        fallback_keywords = [w for w in words if len(w) > 2][:10]
        return {
            "entities": {"PER": [], "LOC": [], "ORG": [], "MISC": []},
            "proper_nouns": [],
            "common_nouns": fallback_keywords,
            "verbs": [],
            "adjectives": [],
            "key_phrases": [],
            "dates_numbers": [],
            "tfidf_keywords": [],
            "similar_keywords": []
        }

def generate_queries_expanded(question, keywords_dict):
    queries = []
    
    clean_question = re.sub(r'[^\w\s]', ' ', question)
    clean_question = clean_question.strip()
    if len(clean_question.split()) <= 12:
        queries.append(clean_question)
    
    proper_nouns = keywords_dict.get("proper_nouns", [])
    common_nouns = keywords_dict.get("common_nouns", [])
    
    if proper_nouns:
        for pn in proper_nouns[:4]:
            queries.append(pn)
    
    if common_nouns:
        for cn in common_nouns[:4]:
            queries.append(cn)
        
        if len(common_nouns) >= 2:
            for i in range(min(3, len(common_nouns))):
                for j in range(i+1, min(4, len(common_nouns))):
                    queries.append(f"{common_nouns[i]} {common_nouns[j]}")
    
    if proper_nouns and common_nouns:
        for i in range(min(2, len(proper_nouns))):
            for j in range(min(2, len(common_nouns))):
                queries.append(f"{proper_nouns[i]} {common_nouns[j]}")
    
    tfidf_keywords = keywords_dict.get("tfidf_keywords", [])
    if tfidf_keywords:
        for kw in tfidf_keywords[:4]:
            queries.append(kw)
        
        if len(tfidf_keywords) >= 2:
            for i in range(min(3, len(tfidf_keywords))):
                for j in range(i+1, min(4, len(tfidf_keywords))):
                    queries.append(f"{tfidf_keywords[i]} {tfidf_keywords[j]}")
    
    similar_keywords = keywords_dict.get("similar_keywords", [])
    if similar_keywords:
        for kw in similar_keywords[:3]:
            queries.append(kw)
    
    key_phrases = keywords_dict.get("key_phrases", [])
    if key_phrases:
        queries.extend(key_phrases[:3])
    
    dates_numbers = keywords_dict.get("dates_numbers", [])
    if dates_numbers:
        for dn in dates_numbers[:3]:
            queries.append(dn)
            if proper_nouns:
                queries.append(f"{proper_nouns[0]} {dn}")
    
    question_words = ["là gì", "của ai", "ở đâu", "khi nào", "tại sao", "như thế nào"]
    for kw in list(set(proper_nouns[:3] + common_nouns[:3] + tfidf_keywords[:3])):
        for qw in question_words:
            queries.append(f"{kw} {qw}")
    
    if not queries:
        words = question.split()
        if len(words) >= 3:
            for i in range(len(words)-2):
                queries.append(" ".join(words[i:i+3]))
        else:
            queries.append(question)
    
    unique_queries = []
    seen = set()
    
    for query in queries:
        if query and isinstance(query, str):
            clean_query = re.sub(r'[^\w\s]', ' ', query).strip()
            clean_query = ' '.join(clean_query.split())
            if len(clean_query) > 2 and len(clean_query) <= 120:
                query_lower = clean_query.lower()
                if query_lower not in seen:
                    seen.add(query_lower)
                    unique_queries.append(clean_query)
    
    if not unique_queries:
        words = question.replace('?', '').split()
        fallback_query = " ".join(words[:min(4, len(words))])
        unique_queries.append(fallback_query)
    
    return unique_queries[:CONFIG["max_queries_per_question"]]

def clean_wiki_with_bs4(text):
    if not text:
        return ""
    
    try:
        soup = BeautifulSoup(text, 'html.parser')
        
        for element in soup(['sup', 'span', 'table', 'style', 'script']):
            element.decompose()
        
        for tag in soup.find_all(True):
            if tag.name in ['h1', 'h2', 'h3', 'h4']:
                tag.insert_before(f"\n## {tag.get_text().strip()}\n")
                tag.decompose()
        
        text = soup.get_text()
    except:
        pass
    
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[cần dẫn nguồn\]', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\{\{.*?\}\}', '', text)
    
    lines = text.split('\n')
    cleaned_lines = []
    
    stop_sections = {
        "Tham khảo", "Liên kết ngoài", "Xem thêm", "Chú thích",
        "Tài liệu tham khảo", "Đọc thêm", "Nguồn", "Ghi chú",
        "Chú giải", "Phụ lục", "Bibliography", "References",
        "External links", "See also", "Notes", "Footnotes"
    }
    
    in_stop_section = False
    
    for line in lines:
        line_stripped = line.strip()
        
        if not line_stripped:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        
        if (len(line_stripped) < 100 and 
            not line_stripped.endswith(('.', ',', ':', ';')) and
            line_stripped in stop_sections):
            in_stop_section = True
            continue
        
        if line_stripped.startswith('==') and in_stop_section:
            in_stop_section = False
            continue
        
        if not in_stop_section and line_stripped:
            cleaned_lines.append(line_stripped)
    
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    
    return cleaned_text.strip()

def smart_truncate_content(text, max_words=8000):
    if not text:
        return ""
    
    words = text.split()
    if len(words) <= max_words:
        return text
    
    paragraphs = text.split('\n\n')
    selected_paragraphs = []
    total_words = 0
    
    for para in paragraphs:
        para_words = para.split()
        if total_words + len(para_words) <= max_words:
            selected_paragraphs.append(para)
            total_words += len(para_words)
        else:
            remaining = max_words - total_words
            if remaining > 50:
                truncated = ' '.join(para_words[:remaining])
                selected_paragraphs.append(truncated + "...")
            break
    
    return '\n\n'.join(selected_paragraphs)

def analyze_content_quality_enhanced(text):
    if not text:
        return {"quality_score": 0, "insights_count": 0}
    
    try:
        words = word_tokenize(text)
    except:
        words = text.split()
    
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    metrics = {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "has_dates": len(re.findall(r'\d{4}', text)) > 0,
        "has_numbers": len(re.findall(r'\d+', text)) > 2,
        "has_proper_nouns": len(re.findall(r'\b[A-Z][a-z]+\b', text)) > 3,
        "section_count": text.count('## '),
        "has_definition": bool(re.search(r'là|định nghĩa|có nghĩa', text.lower())),
        "has_explanation": bool(re.search(r'vì|do|nguyên nhân', text.lower()))
    }
    
    insights_count = 0
    
    for sentence in sentences:
        if len(sentence.split()) > 4 and len(sentence.split()) < 60:
            if (len(re.findall(r'\d+', sentence)) > 0 or 
                len(re.findall(r'\d{4}', sentence)) > 0):
                insights_count += 1
            elif any(keyword in sentence.lower() for keyword in ["là", "được định nghĩa", "có nghĩa"]):
                insights_count += 1
    
    quality_score = 0
    quality_score += min(metrics["word_count"] / 500, 1.0) * 0.25
    quality_score += min(metrics["sentence_count"] / 20, 1.0) * 0.15
    quality_score += 0.1 if metrics["has_dates"] else 0
    quality_score += 0.1 if metrics["has_numbers"] else 0
    quality_score += 0.1 if metrics["has_proper_nouns"] else 0
    quality_score += min(metrics["section_count"] / 2, 1.0) * 0.15
    quality_score += 0.1 if metrics["has_definition"] else 0
    quality_score += 0.1 if metrics["has_explanation"] else 0
    quality_score += min(insights_count / 5, 1.0) * 0.05
    
    metrics["quality_score"] = min(quality_score, 1.0)
    metrics["insights_count"] = insights_count
    
    return metrics

def search_wikipedia_with_fallback(query):
    results = []
    
    try:
        page = wiki_wiki.page(query)
        
        if page.exists() and page.text:
            results.append({
                "title": page.title,
                "text": page.text,
                "url": page.fullurl,
                "score": 1.0
            })
        
        search_results = wiki_wiki.search(query, results=3)
        
        for title in search_results:
            if title != page.title:
                related_page = wiki_wiki.page(title)
                if related_page.exists() and related_page.text:
                    relevance = calculate_query_relevance(query, title, related_page.text)
                    if relevance > 0.15:
                        results.append({
                            "title": title,
                            "text": related_page.text,
                            "url": related_page.fullurl,
                            "score": relevance
                        })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        
    except Exception as e:
        pass
    
    return results[:3]

def calculate_query_relevance(query, title, content):
    query_terms = set(query.lower().split())
    title_terms = set(title.lower().split())
    content_lower = content.lower()
    
    title_match = len(query_terms.intersection(title_terms)) / max(len(query_terms), 1)
    
    content_match = 0
    for term in query_terms:
        if len(term) > 2:
            content_match += content_lower.count(term)
    
    content_match = min(content_match / (len(query_terms) * 5), 1.0)
    
    return 0.6 * title_match + 0.4 * content_match

def save_content_optimized(content, metadata, qid):
    try:
        os.makedirs(CONFIG["data_dir"], exist_ok=True)
        
        content_hash = hashlib.md5(content.encode()).hexdigest()[:10]
        safe_title = re.sub(r'[^\w\s-]', '', metadata.get("title", "unknown"))
        safe_title = safe_title.replace(' ', '_')[:40]
        filename = f"{qid}_{safe_title}_{content_hash}.txt"
        filepath = os.path.join(CONFIG["data_dir"], filename)
        
        if os.path.exists(filepath):
            return None
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# METADATA\n")
            for key, value in metadata.items():
                if key != "analysis":
                    if isinstance(value, str) and len(value) > 100:
                        f.write(f"# {key}: {value[:100]}...\n")
                    else:
                        f.write(f"# {key}: {value}\n")
            
            f.write(f"\n# CONTENT ANALYSIS\n")
            f.write(f"# Quality Score: {metadata.get('quality_score', 0):.2f}/1.0\n")
            f.write(f"# Word Count: {metadata.get('word_count', 0)}\n")
            
            f.write(f"\n{'='*60}\n")
            f.write(f"CONTENT: {metadata.get('title', 'Unknown')}\n")
            f.write(f"{'='*60}\n\n")
            f.write(content)
        
        return filename
        
    except Exception as e:
        return None

def crawl_hybrid_optimized(json_file, max_questions=None):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} questions...")
    
    all_questions = []
    for item in data:
        processed_q = preprocess_question_enhanced(item['question'])
        if processed_q and len(processed_q.split()) >= 1:
            all_questions.append(processed_q)
    
    print(f"Building TF-IDF corpus...")
    hybrid_generator.build_corpus(all_questions)
    
    vocab_size = 0
    if hybrid_generator.vocabulary is not None:
        vocab_size = len(hybrid_generator.vocabulary)
    print(f"TF-IDF corpus built with {vocab_size} features")
    
    os.makedirs(CONFIG["data_dir"], exist_ok=True)
    
    log_fields = [
        'qid', 'question_short', 'keywords_found', 'entities_found',
        'queries_generated', 'pages_found', 'files_saved', 
        'avg_quality_score', 'processing_time', 'success'
    ]
    
    with open(CONFIG["log_file"], 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=log_fields)
        writer.writeheader()
        
        processed = 0
        successful = 0
        
        pbar = tqdm(data, desc="Crawling")
        for idx, item in enumerate(pbar):
            if max_questions and idx >= max_questions:
                break
            
            start_time = time.time()
            qid = item['qid']
            question = item['question']
            
            processed_question = preprocess_question_enhanced(question)
            
            saved_files = []
            quality_scores = []
            pages_found = 0
            
            try:
                keywords_dict = extract_keywords_enhanced(processed_question)
                search_queries = generate_queries_expanded(processed_question, keywords_dict)
                search_queries = [preprocess_query(q) for q in search_queries]
                search_queries = [q for q in search_queries if len(q) > 2]
                
                queries_display = f"{len(search_queries)} queries"
                pbar.set_postfix_str(f"Q: {queries_display}")
                
                for query in search_queries[:CONFIG["max_pages_per_question"]]:
                    try:
                        wiki_results = search_wikipedia_with_fallback(query)
                        
                        for result in wiki_results:
                            pages_found += 1
                            
                            cleaned_content = clean_wiki_with_bs4(result["text"])
                            cleaned_content = smart_truncate_content(cleaned_content)
                            
                            if len(cleaned_content.split()) < CONFIG["min_content_length"]:
                                continue
                            
                            quality_metrics = analyze_content_quality_enhanced(cleaned_content)
                            
                            if quality_metrics["quality_score"] < CONFIG["min_quality_score"]:
                                continue
                            
                            metadata = {
                                "qid": qid,
                                "original_question": question[:150],
                                "search_query": query,
                                "title": result["title"],
                                "url": result["url"],
                                "quality_score": quality_metrics["quality_score"],
                                "word_count": len(cleaned_content.split()),
                                "download_date": datetime.now().isoformat(),
                                "insights_count": quality_metrics["insights_count"]
                            }
                            
                            filename = save_content_optimized(cleaned_content, metadata, qid)
                            if filename:
                                saved_files.append(filename)
                                quality_scores.append(quality_metrics["quality_score"])
                        
                        time.sleep(CONFIG["rate_limit_delay"])
                        
                    except Exception as e:
                        continue
                
            except Exception as e:
                pass
            
            processing_time = time.time() - start_time
            success = len(saved_files) > 0
            
            entities_count = 0
            if keywords_dict and "entities" in keywords_dict:
                entities_count = sum(len(v) for v in keywords_dict["entities"].values())
            
            keywords_count = 0
            if keywords_dict:
                keywords_count = sum(len(v) for v in [keywords_dict.get("proper_nouns", []),
                                                    keywords_dict.get("common_nouns", []),
                                                    keywords_dict.get("tfidf_keywords", []),
                                                    keywords_dict.get("similar_keywords", [])])
            
            writer.writerow({
                "qid": qid,
                "question_short": question[:100] + "..." if len(question) > 100 else question,
                "keywords_found": keywords_count,
                "entities_found": entities_count,
                "queries_generated": len(search_queries) if 'search_queries' in locals() else 0,
                "pages_found": pages_found,
                "files_saved": len(saved_files),
                "avg_quality_score": f"{np.mean(quality_scores):.3f}" if quality_scores else "0",
                "processing_time": f"{processing_time:.1f}s",
                "success": "YES" if success else "NO"
            })
            
            csvfile.flush()
            
            processed += 1
            if success:
                successful += 1
    
    print(f"Processing completed!")
    print(f"Questions processed: {processed}")
    print(f"Successful crawls: {successful}")
    print(f"Success rate: {successful/max(processed,1)*100:.1f}%")
    print(f"Data saved in: {CONFIG['data_dir']}")
    print(f"Log saved to: {CONFIG['log_file']}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Wikipedia Crawler with BS4")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--test", action="store_true")
    
    args = parser.parse_args()
    
    print("Starting Enhanced Crawl Pipeline")
    print("Using: TF-IDF + NER + BS4 + Fallback Strategies")
    
    max_q = args.max_questions
    if args.test:
        max_q = 10
        print("Test mode: Processing 10 questions")
    
    crawl_hybrid_optimized(
        "test.json",
        max_questions=max_q
    )
    
    print("Done!")
import json
import os
import wikipediaapi
import time
from tqdm import tqdm
from underthesea import pos_tag
import re

import csv

# Configuration
DATA_DIR = "corpus"
LOG_FILE = "crawl_log.csv"
WIKI_LANG = "vi"
USER_AGENT = "VNPT_Hackathon_Bot/1.0 (contact@example.com)"

# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent=USER_AGENT,
    language=WIKI_LANG,
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

def extract_keywords(text):
    """
    Extract keywords using underthesea POS tagging.
    Prioritize Proper Nouns (Np) and Nouns (N).
    """
    try:
        # POS Tagging: [('Hà Nội', 'Np'), ('là', 'V'), ...]
        tags = pos_tag(text)
        
        keywords = []
        # Priority 1: Proper Nouns (Np) - Names, Places, Organizations
        proper_nouns = [word for word, tag in tags if tag == 'Np']
        
        # Priority 2: Common Nouns (N) - Concepts, Objects
        nouns = [word for word, tag in tags if tag == 'N' and len(word) > 1]
        
        # Combine: Proper Nouns first, then Nouns
        keywords.extend(proper_nouns)
        
        # If we don't have enough proper nouns, add common nouns
        if len(keywords) < 3:
            keywords.extend(nouns)
            
        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for k in keywords:
            if k.lower() not in seen:
                seen.add(k.lower())
                unique_keywords.append(k)
                
        return unique_keywords
    except Exception as e:
        print(f"Error in keyword extraction: {e}")
        # Fallback to simple split if NLP fails
        return text.split()[:5]

def clean_wiki_text(text):
    """
    Clean Wikipedia text: remove citations, footer sections, and empty lines.
    """
    # Remove citations [1], [2]
    text = re.sub(r'\[\d+\]', '', text)
    
    lines = text.split('\n')
    cleaned_lines = []
    
    # Headers that usually mark the end of the main article content
    stop_headers = {
        "Tham khảo", "Liên kết ngoài", "Xem thêm", "Chú thích", 
        "Tài liệu tham khảo", "Đọc thêm", "Nguồn", "Ghi chú"
    }
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        # If the line is exactly a stop header (or very close to it), we stop processing
        if stripped in stop_headers:
            break
            
        cleaned_lines.append(stripped)
        
    return "\n".join(cleaned_lines)

def search_and_save(query, filename_prefix):
    """
    Search Wikipedia for a query and save the page content.
    Returns the filename if saved, else None.
    """
    try:
        page = wiki_wiki.page(query)
        
        if page.exists():
            # Save to file
            safe_title = page.title.replace("/", "_").replace(" ", "_")
            filename = f"{safe_title}.txt"
            filepath = os.path.join(DATA_DIR, filename)
            
            if not os.path.exists(filepath):
                cleaned_text = clean_wiki_text(page.text)
                
                # Only save if content is substantial (e.g., > 200 chars)
                if len(cleaned_text) > 200:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(cleaned_text)
                    return filename
            else:
                return filename # Return filename even if already exists
    except Exception as e:
        print(f"Error fetching {query}: {e}")
        
    return None

def crawl_from_questions(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Processing {len(data)} questions from {json_file}...")
    
    # Prepare CSV logging - Open file in append mode once
    file_exists = os.path.isfile(LOG_FILE)
    csvfile = open(LOG_FILE, 'a', newline='', encoding='utf-8')
    fieldnames = ['qid', 'keywords', 'files']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    if not file_exists:
        writer.writeheader()
    
    count = 0
    try:
        for item in tqdm(data):
            question = item['question']
            qid = item['qid']
            
            # Skip if context is already provided
            if "Đoạn thông tin" in question:
                continue
                
            # Extract keywords to search
            keywords = extract_keywords(question)
            
            # Strategy: Search for the 2-3 most significant keywords
            # For now, let's just take the first 3 keywords as a query, or iterate.
            if keywords:
                top_keywords = keywords[:5]
                found_files = []
                
                # Construct a query from keywords
                query = " ".join(top_keywords) 
                saved_file = search_and_save(query, qid)
                if saved_file:
                    count += 1
                    found_files.append(saved_file)
                    
                # Try individual keywords (both Entities and Nouns)
                for kw in top_keywords:
                    saved_file = search_and_save(kw, qid)
                    if saved_file:
                        count += 1
                        found_files.append(saved_file)
                
                # Log to CSV immediately
                if found_files:
                    writer.writerow({
                        "qid": qid,
                        "keywords": ", ".join(top_keywords),
                        "files": ", ".join(list(set(found_files))) # Deduplicate files
                    })
                    csvfile.flush() # Ensure data is written to disk
                        
        print(f"Downloaded {count} pages.")
    finally:
        csvfile.close()
        print(f"Log saved to {LOG_FILE}")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    # Crawl from val.json
    print("Crawling for Validation Set...")
    crawl_from_questions("AInicorns_TheBuilder_public_v1.1/data/val.json")
    
    # # Crawl from test.json
    # print("Crawling for Test Set...")
    # crawl_from_questions("AInicorns_TheBuilder_public_v1.1/data/test.json")

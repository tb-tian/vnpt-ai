import json
import os
import wikipediaapi
import time
from tqdm import tqdm
from underthesea import pos_tag
import re

import csv
import requests
from bs4 import BeautifulSoup
import random
from urllib.parse import unquote
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

class DuckDuckGoSearchCrawler:
    def __init__(self):
        self.base_url = "https://html.duckduckgo.com/html/"
        self.session = requests.Session()
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Referer': 'https://duckduckgo.com/'
        })

    def search(self, query, max_results=3):
        if not query:
            return []
        
        try:
            data = {'q': query}
            # Add a small delay to be polite
            time.sleep(random.uniform(1, 2)) 
            resp = self.session.post(self.base_url, data=data, timeout=15)
            
            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.text, 'html.parser')
            results = []
            
            for result in soup.find_all('div', class_='result'):
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')
                url_elem = result.find('a', class_='result__url')
                
                if title_elem and url_elem:
                    title = title_elem.get_text(strip=True)
                    raw_url = url_elem.get('href')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    # DuckDuckGo redirect URLs
                    url = raw_url
                    if raw_url and '/l/?kh=-1&uddg=' in raw_url:
                        try:
                            url = unquote(raw_url.split('uddg=')[1].split('&')[0])
                        except:
                            pass
                        
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet
                    })
                    
                    if len(results) >= max_results:
                        break
            
            return results

        except Exception as e:
            print(f"DuckDuckGo error: {e}")
            return []

def fetch_web_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=20, verify=False)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted tags
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'iframe', 'noscript', 'svg', 'canvas', 'button', 'input', 'select', 'textarea']):
                element.decompose()
                
            # Get text with separator to preserve structure
            text = soup.get_text(separator='\n')
            
            # Clean up text
            lines = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                
                # Filter out very short lines that are likely menu items or noise
                # Keep if it ends with punctuation or is long enough
                if len(line) > 30 or (len(line) > 10 and line[-1] in '.!?:'):
                    lines.append(line)
            
            text = '\n'.join(lines)
            
            # Normalize whitespace
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            return text
    except requests.exceptions.Timeout:
        print(f"Timeout fetching {url}")
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return None

def search_ddg_and_save(query, filename_prefix):
    """
    Search DuckDuckGo and save content from top results.
    """
    crawler = DuckDuckGoSearchCrawler()
    results = crawler.search(query, max_results=3)
    saved_files = []
    
    for i, result in enumerate(results):
        try:
            url = result['url']
            title = result['title']
            
            # Skip youtube or other non-text heavy sites if needed
            if 'youtube.com' in url:
                continue
                
            content = fetch_web_content(url)
            
            if content and len(content) > 500:
                safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
                # Limit filename length
                safe_title = safe_title[:50]
                
                filename = f"{filename_prefix}_ddg_{i}_{safe_title}.txt"
                filepath = os.path.join(DATA_DIR, filename)
                
                if not os.path.exists(filepath):
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(f"Source: {url}\nTitle: {title}\n\n")
                        f.write(content)
                    saved_files.append(filename)
                else:
                    saved_files.append(filename)
        except Exception as e:
            print(f"Error processing DDG result {url}: {e}")
            
    return saved_files

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
    processed_keywords = set()
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
                
                if query not in processed_keywords:
                    # Search Wikipedia
                    saved_file = search_and_save(query, qid)
                    if saved_file:
                        count += 1
                        found_files.append(saved_file)
                    
                    # Search DuckDuckGo
                    ddg_files = search_ddg_and_save(query, qid)
                    if ddg_files:
                        count += len(ddg_files)
                        found_files.extend(ddg_files)
                    
                    processed_keywords.add(query)
                    
                # Try individual keywords (both Entities and Nouns)
                for kw in top_keywords:
                    if kw not in processed_keywords:
                        saved_file = search_and_save(kw, qid)
                        if saved_file:
                            count += 1
                            found_files.append(saved_file)
                        processed_keywords.add(kw)
                
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

    # crawl_from_questions('test.json')

    # Crawl from val.json
    print("Crawling for Validation Set...")
    crawl_from_questions("data/val.json")
    
    # # Crawl from test.json
    # print("Crawling for Test Set...")
    crawl_from_questions("data/test.json")
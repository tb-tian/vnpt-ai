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
from urllib.parse import urlparse, quote, parse_qs
import random
from datasketch import MinHash, MinHashLSH

warnings.filterwarnings('ignore')

CONFIG = {
    "data_dir": "corpus_hybrid_bs4",
    "log_file": "crawl_hybrid_log_bs4.csv",
    "dedup_cache_file": "dedup_cache.json",
    "lsh_cache_file": "lsh_cache.json",
    "wiki_lang": "vi",
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "google_user_agents": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ],
    "min_content_length": 80,
    "max_content_length": 15000,
    "min_wikipedia_word_count": 100,
    "rate_limit_delay": 2.0,
    "min_quality_score": 0.15,
    "max_pages_per_question": 3,
    "tfidf_max_features": 150,
    "similarity_threshold": 0.2,
    "max_queries_per_question": 8,
    "request_timeout": 15,
    "dedup_threshold": 0.85,
    "lsh_threshold": 0.7,
    "google_max_results": 3,
    "google_min_delay": 5,
    "google_max_delay": 10,
    "use_google": True,
    "use_duckduckgo": True,
    "google_proxies": [],
    "max_query_length": 80,
    "enable_query_expansion": True,
    "fallback_to_simple_search": True,
    "google_retry_attempts": 0,
    "google_use_direct_html": True
}

wiki_wiki = wikipediaapi.Wikipedia(
    language=CONFIG["wiki_lang"],
    user_agent=CONFIG["user_agent"],
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

class LSHManager:
    def __init__(self, cache_file=CONFIG["lsh_cache_file"], threshold=CONFIG["lsh_threshold"]):
        self.cache_file = cache_file
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=128)
        self.minhashes = {}
        self.content_to_id = {}
        self.stats = {
            'total_checked': 0,
            'near_duplicates_found': 0,
            'unique_saved': 0
        }
        self.load_cache()
    
    def load_cache(self):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.lsh = MinHashLSH(threshold=self.threshold, num_perm=128)
                for item_id, minhash_data in data.get('minhashes', {}).items():
                    minhash = MinHash(num_perm=128)
                    minhash.hashvalues = np.array(minhash_data['hashvalues'])
                    self.lsh.insert(item_id, minhash)
                    self.minhashes[item_id] = minhash
                self.content_to_id = data.get('content_to_id', {})
                self.stats = data.get('stats', self.stats)
                print(f"✓ Loaded LSH cache with {len(self.minhashes)} items")
        except Exception as e:
            print(f"⚠ LSH cache load error: {e}")
            self.lsh = MinHashLSH(threshold=self.threshold, num_perm=128)
            self.minhashes = {}
            self.content_to_id = {}
    
    def save_cache(self):
        try:
            minhashes_data = {}
            for item_id, minhash in self.minhashes.items():
                minhashes_data[item_id] = {
                    'hashvalues': minhash.hashvalues.tolist()
                }
            data = {
                'minhashes': minhashes_data,
                'content_to_id': self.content_to_id,
                'stats': self.stats
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠ LSH cache save error: {e}")
    
    def create_minhash(self, text):
        if not text or not isinstance(text, str):
            return None
        try:
            words = re.findall(r'\w+', text.lower())
            if len(words) < 10:
                return None
            minhash = MinHash(num_perm=128)
            for word in words[:1000]:
                minhash.update(word.encode('utf-8'))
            return minhash
        except Exception as e:
            return None
    
    def generate_content_id(self, text):
        if not text:
            return None
        snippet = text[:100].lower().strip()
        return hashlib.md5(snippet.encode('utf-8')).hexdigest()[:16]
    
    def check_similar_content(self, text, title=None):
        self.stats['total_checked'] += 1
        result = {
            'is_similar': False,
            'similar_items': [],
            'similarity_scores': [],
            'content_id': None
        }
        minhash = self.create_minhash(text)
        if not minhash:
            return result
        content_id = self.generate_content_id(text)
        result['content_id'] = content_id
        try:
            similar_items = self.lsh.query(minhash)
            if similar_items:
                result['is_similar'] = True
                result['similar_items'] = similar_items
                for item_id in similar_items:
                    if item_id in self.minhashes:
                        similarity = minhash.jaccard(self.minhashes[item_id])
                        result['similarity_scores'].append({
                            'item_id': item_id,
                            'similarity': float(similarity)
                        })
                self.stats['near_duplicates_found'] += 1
                return result
        except Exception as e:
            pass
        if content_id and minhash:
            self.lsh.insert(content_id, minhash)
            self.minhashes[content_id] = minhash
            if title:
                self.content_to_id[content_id] = {
                    'title': title,
                    'timestamp': datetime.now().isoformat()
                }
            self.stats['unique_saved'] += 1
            self.save_cache()
        return result
    
    def get_stats(self):
        return self.stats
    
    def print_stats(self):
        print("\n" + "="*60)
        print("LSH SIMILARITY DETECTION STATISTICS:")
        print("="*60)
        print(f"Total checked: {self.stats['total_checked']}")
        print(f"Unique saved: {self.stats['unique_saved']}")
        print(f"Near duplicates found: {self.stats['near_duplicates_found']}")
        duplicate_rate = self.stats['near_duplicates_found'] / max(self.stats['total_checked'], 1) * 100
        print(f"Near duplicate rate: {duplicate_rate:.1f}%")
        print(f"Threshold: {self.threshold}")
        print("="*60)

class DeduplicationManager:
    def __init__(self, cache_file=CONFIG["dedup_cache_file"]):
        self.cache_file = cache_file
        self.content_hashes = set()
        self.canonical_ids = set()
        self.url_to_id = {}
        self.stats = {
            'total_checked': 0,
            'duplicate_content': 0,
            'duplicate_url': 0,
            'duplicate_title': 0,
            'unique_saved': 0
        }
        self.load_cache()
    
    def load_cache(self):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.content_hashes = set(data.get('content_hashes', []))
                    self.canonical_ids = set(data.get('canonical_ids', []))
                    self.url_to_id = data.get('url_to_id', {})
                    self.stats = data.get('stats', self.stats)
                print(f"Loaded {len(self.content_hashes)} hashes, {len(self.canonical_ids)} IDs from cache")
        except Exception as e:
            print(f"Cache load error: {e}")
            self.content_hashes = set()
            self.canonical_ids = set()
            self.url_to_id = {}
    
    def save_cache(self):
        try:
            data = {
                'content_hashes': list(self.content_hashes),
                'canonical_ids': list(self.canonical_ids),
                'url_to_id': self.url_to_id,
                'stats': self.stats
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠ Cache save error: {e}")
    
    def generate_content_hash(self, text):
        if not text or not isinstance(text, str):
            return None
        try:
            text = re.sub(r'<[^>]+>', '', text)
            text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
            text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
            text = re.sub(r'\[\d+\]', '', text)
            text = re.sub(r'\[cần dẫn nguồn\]', '', text)
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'\{.*?\}', '', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.lower().strip()
            vietnamese_chars = r'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖộƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ'
            text = re.sub(f'[^\\w\\s{vietnamese_chars}]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            words = text.split()
            if len(words) < 20:
                return None
            sorted_words = sorted(words)
            text_for_hash = ' '.join(sorted_words)
            if len(text_for_hash) < 50:
                return None
            return hashlib.md5(text_for_hash.encode('utf-8')).hexdigest()
        except Exception as e:
            return None
    
    def extract_canonical_id(self, url, title=None):
        canonical_id = None
        if url and isinstance(url, str):
            try:
                parsed = urlparse(url)
                if 'wikipedia.org' in parsed.netloc:
                    path_parts = parsed.path.split('/')
                    if len(path_parts) > 2 and path_parts[1] == 'wiki':
                        canonical_id = path_parts[2]
            except:
                pass
        if not canonical_id and title and isinstance(title, str):
            try:
                clean_title = re.sub(r'[^\w\s]', '', title.lower())
                clean_title = clean_title.replace(' ', '_')
                if len(clean_title) > 2:
                    canonical_id = clean_title
            except:
                pass
        return canonical_id
    
    def is_duplicate(self, content, url=None, title=None):
        self.stats['total_checked'] += 1
        if not content or not isinstance(content, str):
            return {'is_duplicate': False, 'reason': 'invalid_content'}
        result = {
            'is_duplicate': False,
            'reason': None,
            'canonical_id': None,
            'content_hash': None
        }
        canonical_id = self.extract_canonical_id(url, title)
        if canonical_id:
            result['canonical_id'] = canonical_id
            if canonical_id in self.canonical_ids:
                result['is_duplicate'] = True
                result['reason'] = 'duplicate_canonical_id'
                self.stats['duplicate_url'] += 1
                return result
        content_hash = self.generate_content_hash(content)
        if content_hash:
            result['content_hash'] = content_hash
            if content_hash in self.content_hashes:
                result['is_duplicate'] = True
                result['reason'] = 'duplicate_content'
                self.stats['duplicate_content'] += 1
                return result
        if not result['is_duplicate']:
            if canonical_id:
                self.canonical_ids.add(canonical_id)
            if content_hash:
                self.content_hashes.add(content_hash)
            if url and canonical_id:
                self.url_to_id[url] = canonical_id
            self.stats['unique_saved'] += 1
        return result
    
    def get_stats(self):
        return self.stats
    
    def print_stats(self):
        print("\n" + "="*60)
        print("EXACT DEDUPLICATION STATISTICS:")
        print("="*60)
        print(f"Total checked: {self.stats['total_checked']}")
        print(f"Unique saved: {self.stats['unique_saved']}")
        total_duplicates = (self.stats['duplicate_content'] + 
                          self.stats['duplicate_url'] + 
                          self.stats['duplicate_title'])
        print(f"Duplicates found: {total_duplicates}")
        print(f"  - By content hash: {self.stats['duplicate_content']}")
        print(f"  - By URL/ID: {self.stats['duplicate_url']}")
        print(f"  - By title: {self.stats['duplicate_title']}")
        duplicate_rate = total_duplicates / max(self.stats['total_checked'], 1) * 100
        print(f"Duplicate rate: {duplicate_rate:.1f}%")
        efficiency = total_duplicates / max(total_duplicates + self.stats['unique_saved'], 1) * 100
        print(f"Deduplication efficiency: {efficiency:.1f}%")
        print("="*60)

class DuckDuckGoSearchCrawler:
    def __init__(self):
        self.base_url = "https://html.duckduckgo.com/html/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': random.choice(CONFIG["google_user_agents"]),
            'Referer': 'https://duckduckgo.com/'
        })
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'results_found': 0
        }

    def search(self, query):
        if not query:
            return []
        self.stats['total_queries'] += 1
        try:
            data = {'q': query}
            time.sleep(random.uniform(2, 4)) 
            resp = self.session.post(self.base_url, data=data, timeout=15)
            
            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.text, 'html.parser')
            results = []
            
            for result in soup.find_all('div', class_='result'):
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')
                
                if title_elem and snippet_elem:
                    url = title_elem.get('href', '')
                    title = title_elem.get_text(strip=True)
                    snippet = snippet_elem.get_text(strip=True)
                    
                    if url and title:
                        results.append({
                            'title': title,
                            'url': url,
                            'description': snippet,
                            'source': 'DuckDuckGo',
                            'original_query': query,
                            'score': 0.65
                        })
            
            if results:
                self.stats['successful_queries'] += 1
                self.stats['results_found'] += len(results)
                print(f"  ✓ Found {len(results)} DuckDuckGo results for: {query[:30]}...")
            
            return results[:CONFIG["google_max_results"]]

        except Exception as e:
            print(f"  ⚠ DuckDuckGo error: {e}")
            return []

class GoogleSearchCrawler:
    def __init__(self):
        self.base_url = "https://www.google.com/search"
        self.user_agents = CONFIG["google_user_agents"]
        self.proxies = CONFIG["google_proxies"]
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'results_found': 0,
            'captcha_encountered': 0,
            'retry_count': 0
        }
        self.session = requests.Session()
        self.disabled = False 
    
    def get_random_user_agent(self):
        return random.choice(self.user_agents)
    
    def get_random_proxy(self):
        if self.proxies:
            return random.choice(self.proxies)
        return None
    
    def search_google_simple(self, query):
        if self.disabled:
            return []
        try:
            from requests_html import HTMLSession
            session = HTMLSession()
            
            headers = {
                'User-Agent': self.get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
                'Accept-Encoding': 'gzip, deflate',
            }
            
            params = {
                'q': query,
                'num': 5,
                'hl': 'vi'
            }
            
            delay = random.uniform(CONFIG["google_min_delay"], CONFIG["google_max_delay"])
            time.sleep(delay)
            
            response = session.get(self.base_url, headers=headers, params=params, timeout=15)
            
            if 'captcha' in response.text.lower() or 'recaptcha' in response.text.lower():
                print(f"  ⚠ Google CAPTCHA blocked simple search. Disabling Google.")
                self.stats['captcha_encountered'] += 1
                self.disabled = True
                return []
            
            response.html.render(sleep=1, timeout=20)
            html_content = response.html.html
            
            return self.parse_google_simple(html_content, query)
            
        except Exception as e:
            print(f"  ⚠ requests-html search error: {e}")
            return []
    
    def parse_google_simple(self, html_content, query):
        results = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for g in soup.find_all('div', {'class': 'g'}):
                try:
                    link = g.find('a', href=True)
                    title = g.find('h3')
                    
                    if link and title:
                        url = link['href']
                        if url.startswith('/url?q='):
                            url = url.split('/url?q=')[1].split('&')[0]
                        
                        if not self.is_valid_search_result_url(url):
                            continue
                        
                        desc_div = g.find('div', {'class': ['VwiC3b', 's']})
                        description = desc_div.get_text(strip=True) if desc_div else ""
                        
                        result = {
                            'title': title.get_text(strip=True),
                            'url': url,
                            'description': description,
                            'source': 'Google Search',
                            'original_query': query,
                            'score': 0.7
                        }
                        results.append(result)
                except:
                    continue
            
            return results[:CONFIG["google_max_results"]]
            
        except Exception as e:
            print(f"  ⚠ Simple parser error: {e}")
            return []
    
    def search_google_direct(self, query):
        if not query or not CONFIG["use_google"] or self.disabled:
            return []
        
        self.stats['total_queries'] += 1
        
        try:
            delay = random.uniform(CONFIG["google_min_delay"], CONFIG["google_max_delay"])
            time.sleep(delay)
            
            headers = {
                'User-Agent': self.get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'vi,vi-VN;q=0.9,en-US;q=0.8,en;q=0.7',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0'
            }
            
            params = {
                'q': query,
                'num': CONFIG["google_max_results"],
                'hl': 'vi',
                'gl': 'VN',
                'lr': 'lang_vi',
                'cr': 'countryVN',
                'ie': 'UTF-8',
                'oe': 'UTF-8'
            }
            
            proxy = self.get_random_proxy()
            proxies = {'http': proxy, 'https': proxy} if proxy else None
            
            response = self.session.get(
                self.base_url,
                headers=headers,
                params=params,
                proxies=proxies,
                timeout=CONFIG["request_timeout"],
                allow_redirects=True
            )
            
            response_text = response.text.lower()
            
            captcha_indicators = ['captcha', 'recaptcha', 'robot', 'verify you are human', 'sorry']
            if any(indicator in response_text for indicator in captcha_indicators):
                print(f"  ⚠ Google CAPTCHA detected for query: {query[:30]}... -> Disabling Google Search!")
                self.stats['captcha_encountered'] += 1
                self.stats['failed_queries'] += 1
                self.disabled = True 
                return []
            
            response.raise_for_status()
            
            results = self.parse_google_direct(response.text, query)
            
            if results:
                self.stats['results_found'] += len(results)
                self.stats['successful_queries'] += 1
                print(f"  ✓ Found {len(results)} Google results for: {query[:30]}...")
                return results
            else:
                self.stats['failed_queries'] += 1
                return []
            
        except requests.exceptions.RequestException as e:
            print(f"  ⚠ Google search error for '{query[:30]}...': {e}")
            self.stats['failed_queries'] += 1
            return []
        except Exception as e:
            print(f"  ⚠ Unexpected error in Google search: {e}")
            self.stats['failed_queries'] += 1
            return []
        
        return []
    
    def parse_google_direct(self, html_content, query):
        results = []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            search_results = []
            
            for g in soup.find_all('div', class_='g'):
                search_results.append(g)
            
            for g in soup.find_all('div', {'data-snc': True}):
                search_results.append(g)
            
            for g in soup.find_all('div', {'class': 'tF2Cxc'}):
                search_results.append(g)
            
            for g in soup.find_all('div', {'class': 'yuRUbf'}):
                search_results.append(g)
            
            for result_div in search_results[:CONFIG["google_max_results"]]:
                try:
                    link_tag = result_div.find('a', href=True)
                    if not link_tag:
                        continue
                    
                    url = link_tag['href']
                    
                    if url.startswith('/url?q='):
                        url = url.split('/url?q=')[1].split('&')[0]
                        url = requests.utils.unquote(url)
                    
                    if not self.is_valid_search_result_url(url):
                        continue
                    
                    title_tag = result_div.find(['h3', 'h2', 'h1'])
                    title = title_tag.get_text(strip=True) if title_tag else ""
                    
                    if not title or len(title) < 3:
                        continue
                    
                    desc_tag = result_div.find(['div', 'span'], class_=lambda x: x and any(cls in str(x).lower() for cls in ['s', 'desc', 'st', 'vwi', 'aCOpRe']))
                    description = desc_tag.get_text(strip=True) if desc_tag else ""
                    
                    result = {
                        'title': title,
                        'url': url,
                        'description': description,
                        'source': 'Google Search',
                        'original_query': query,
                        'score': 0.7
                    }
                    results.append(result)
                    
                except Exception as e:
                    continue
            
            if not results:
                links = soup.find_all('a', href=True)
                for link in links[:20]:
                    try:
                        url = link['href']
                        if url.startswith('/url?q='):
                            url = url.split('/url?q=')[1].split('&')[0]
                            url = requests.utils.unquote(url)
                        
                        if not self.is_valid_search_result_url(url):
                            continue
                        
                        title = link.get_text(strip=True)
                        if title and len(title) > 5:
                            result = {
                                'title': title,
                                'url': url,
                                'description': '',
                                'source': 'Google Search',
                                'original_query': query,
                                'score': 0.6
                            }
                            results.append(result)
                    except:
                        continue
            
            unique_results = []
            seen_urls = set()
            
            for result in results:
                url_key = result['url'].split('?')[0]
                if url_key not in seen_urls:
                    seen_urls.add(url_key)
                    unique_results.append(result)
            
            return unique_results[:CONFIG["google_max_results"]]
            
        except Exception as e:
            print(f"  ⚠ Error parsing Google results: {e}")
            return []
    
    def is_valid_search_result_url(self, url):
        if not url or not isinstance(url, str):
            return False
        
        url_lower = url.lower()
        
        if not url_lower.startswith('http'):
            return False
        
        google_domains = [
            'google.com',
            'googleusercontent.com',
            'gstatic.com',
            'googleapis.com',
            'youtube.com',
            'accounts.google.com',
            'policies.google.com',
            'support.google.com'
        ]
        
        if any(domain in url_lower for domain in google_domains):
            return False
        
        unwanted_patterns = [
            '/search?',
            '/advanced_search?',
            '/preferences?',
            '/setprefs?',
            '/history?',
            'webcache.googleusercontent.com',
            '/maps/',
            '/translate'
        ]
        
        if any(pattern in url_lower for pattern in unwanted_patterns):
            return False
        
        return True
    
    def search_google(self, query):
        if not CONFIG["use_google"] or self.disabled:
            return []
        
        if CONFIG["google_use_direct_html"]:
            return self.search_google_direct(query)
        else:
            return self.search_google_simple(query)
    
    def get_stats(self):
        return self.stats
    
    def print_stats(self):
        print("\n" + "="*60)
        print("GOOGLE SEARCH STATISTICS:")
        print("="*60)
        print(f"Total queries: {self.stats['total_queries']}")
        print(f"Successful queries: {self.stats['successful_queries']}")
        print(f"Failed queries: {self.stats['failed_queries']}")
        print(f"Results found: {self.stats['results_found']}")
        print(f"CAPTCHA encountered: {self.stats['captcha_encountered']}")
        print(f"Status: {'DISABLED' if self.disabled else 'ACTIVE'}")
        
        success_rate = self.stats['successful_queries'] / max(self.stats['total_queries'], 1) * 100
        print(f"Success rate: {success_rate:.1f}%")
        print("="*60)

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
        if not questions:
            self.corpus_questions = []
            self.tfidf_matrix = None
            self.vocabulary = None
            return
        
        self.corpus_questions = questions
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(questions)
            self.vocabulary = self.tfidf_vectorizer.get_feature_names_out()
        except Exception as e:
            print(f"⚠ TF-IDF build error: {e}")
            self.tfidf_matrix = None
            self.vocabulary = None
    
    def extract_tfidf_keywords(self, question, top_n=15):
        if self.tfidf_matrix is None or self.vocabulary is None:
            return []
        try:
            question_vec = self.tfidf_vectorizer.transform([question])
            feature_array = np.array(self.vocabulary)
            tfidf_sorting = np.argsort(question_vec.toarray()).flatten()[::-1]
            return list(feature_array[tfidf_sorting][:top_n])
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
            keywords = self.extract_basic_keywords(similar["question"])
            all_keywords.extend(keywords)
        return [kw for kw, _ in Counter(all_keywords).most_common(10)]
    
    def extract_basic_keywords(self, text):
        if not text:
            return []
        try:
            tags = pos_tag(text)
            return [word for word, tag in tags if tag in ['Np', 'N', 'V', 'A'] and len(word) > 1]
        except:
            return text.split()[:10] if text else []

hybrid_generator = EnhancedHybridQueryGenerator()

def preprocess_question_enhanced(question):
    if not question or not isinstance(question, str):
        return ""
    
    if "Đoạn thông tin:" in question:
        parts = question.split("Câu hỏi:")
        if len(parts) > 1:
            question = parts[-1].strip()
    
    patterns = [r'\$.*?\$', r'\\[a-zA-Z]+', r'\{.*?\}', r'\(.*?\)']
    for pattern in patterns:
        question = re.sub(pattern, '', question)
    
    stop_words = {"của", "và", "trong", "với", "để", "này", "là", "có", 
                  "được", "theo", "từ", "như", "bởi", "cho", "một", "các",
                  "hay", "hoặc", "nếu", "thì", "mà", "ở", "tại", "về"}
    
    words = []
    for w in question.split():
        w_lower = w.lower()
        if len(w) > 1 and w_lower not in stop_words and any(c.isalpha() for c in w):
            words.append(w)
    
    processed = " ".join(words)
    
    if len(processed.split()) < 2:
        original_clean = re.sub(r'[^\w\s?]', ' ', question)
        return original_clean.strip()
    
    return processed.strip()

def preprocess_query(query):
    if not query:
        return ""
    
    query = re.sub(r'[$\\]', '', query)
    query = re.sub(r'\{.*?\}', '', query)
    query = re.sub(r'\(.*?\)', '', query)
    query = re.sub(r'\[.*?\]', '', query)
    
    words = []
    for w in query.split():
        if any(c.isalpha() for c in w) and not re.match(r'^[0-9\W]+$', w):
            words.append(w)
    
    return " ".join(words[:10]).strip()

def extract_keywords_enhanced(question):
    if not question:
        return {
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
    
    try:
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
            elif tag == 'N' and len(word) > 2:
                keywords["common_nouns"].append(word)
            elif tag.startswith('V'):
                keywords["verbs"].append(word)
            elif tag.startswith('A'):
                keywords["adjectives"].append(word)
        
        date_patterns = [r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', r'\d{4}', r'(năm|ngày|tháng)\s+\d{1,4}']
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
        print(f"⚠ Keyword extraction error: {e}")
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
    if not question or not keywords_dict:
        return [question[:50]] if question else []
    
    queries = []
    vietnamese_chars = r'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖộƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ'
    clean_question = re.sub(f'[^\\w\\s{vietnamese_chars}]', ' ', question)
    clean_question = clean_question.strip()
    
    if clean_question:
        queries.append(clean_question)
    
    if len(clean_question.split()) > 6:
        words = clean_question.split()
        queries.append(" ".join(words[:4]))
        queries.append(" ".join(words[:3]))
    
    proper_nouns = keywords_dict.get("proper_nouns", [])[:4]
    common_nouns = keywords_dict.get("common_nouns", [])[:4]
    
    queries.extend(proper_nouns)
    queries.extend(common_nouns)
    
    if proper_nouns and common_nouns:
        for pn in proper_nouns[:2]:
            for cn in common_nouns[:2]:
                queries.append(f"{pn} {cn}")
    
    if len(common_nouns) >= 2:
        for i in range(min(2, len(common_nouns))):
            for j in range(i+1, min(3, len(common_nouns))):
                queries.append(f"{common_nouns[i]} {common_nouns[j]}")
    
    question_words = ["là gì", "là ai", "ở đâu", "khi nào", "tại sao"]
    
    for kw in list(set(proper_nouns[:2] + common_nouns[:2])):
        for qw in question_words:
            queries.append(f"{kw} {qw}")
    
    tfidf_keywords = keywords_dict.get("tfidf_keywords", [])[:3]
    queries.extend(tfidf_keywords)
    
    similar_keywords = keywords_dict.get("similar_keywords", [])[:2]
    queries.extend(similar_keywords)
    
    key_phrases = keywords_dict.get("key_phrases", [])[:3]
    queries.extend(key_phrases)
    
    dates_numbers = keywords_dict.get("dates_numbers", [])[:2]
    queries.extend(dates_numbers)
    
    if not queries:
        words = question.split()
        if len(words) >= 3:
            queries.append(" ".join(words[:3]))
        else:
            queries.append(question)
    
    unique_queries = []
    seen = set()
    
    for query in queries:
        if query and isinstance(query, str):
            clean_query = re.sub(f'[^\\w\\s{vietnamese_chars}]', ' ', query)
            clean_query = ' '.join(clean_query.split()).strip()
            
            if 2 < len(clean_query) <= 60:
                query_lower = clean_query.lower()
                if query_lower not in seen:
                    seen.add(query_lower)
                    unique_queries.append(clean_query)
    
    if not unique_queries:
        words = question.replace('?', '').split()
        fallback_query = " ".join(words[:min(3, len(words))])
        unique_queries.append(fallback_query)
    
    return unique_queries[:CONFIG["max_queries_per_question"]]

def clean_wiki_with_bs4(text):
    if not text:
        return ""
    
    try:
        soup = BeautifulSoup(text, 'html.parser')
        
        for element in soup(['sup', 'span', 'table', 'style', 'script']):
            element.decompose()
        
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4']):
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
        
        if line_stripped in stop_sections:
            in_stop_section = True
            continue
        
        if line_stripped.startswith('==') and in_stop_section:
            in_stop_section = False
            continue
        
        if not in_stop_section:
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
    selected = []
    total = 0
    
    for para in paragraphs:
        para_words = para.split()
        if total + len(para_words) <= max_words:
            selected.append(para)
            total += len(para_words)
        else:
            remaining = max_words - total
            if remaining > 50:
                selected.append(' '.join(para_words[:remaining]) + "...")
            break
    
    return '\n\n'.join(selected)

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
    
    insights = 0
    for sentence in sentences:
        if 4 < len(sentence.split()) < 60:
            if len(re.findall(r'\d+', sentence)) > 0 or len(re.findall(r'\d{4}', sentence)) > 0:
                insights += 1
            elif any(kw in sentence.lower() for kw in ["là", "được định nghĩa", "có nghĩa"]):
                insights += 1
    
    score = 0
    score += min(metrics["word_count"] / 500, 1.0) * 0.25
    score += min(metrics["sentence_count"] / 20, 1.0) * 0.15
    score += 0.1 if metrics["has_dates"] else 0
    score += 0.1 if metrics["has_numbers"] else 0
    score += 0.1 if metrics["has_proper_nouns"] else 0
    score += min(metrics["section_count"] / 2, 1.0) * 0.15
    score += 0.1 if metrics["has_definition"] else 0
    score += 0.1 if metrics["has_explanation"] else 0
    score += min(insights / 5, 1.0) * 0.05
    
    metrics["quality_score"] = min(score, 1.0)
    metrics["insights_count"] = insights
    
    return metrics

def search_wikipedia_api(query):
    if not query:
        return []
    
    results = []
    
    try:
        page = wiki_wiki.page(query)
        
        if page.exists() and page.text:
            word_count = len(page.text.split())
            if word_count > CONFIG["min_wikipedia_word_count"]:
                results.append({
                    "title": page.title,
                    "text": page.text,
                    "url": page.fullurl,
                    "score": 1.0,
                    "source": "exact_match",
                    "word_count": word_count
                })
        
        search_url = f"https://{CONFIG['wiki_lang']}.wikipedia.org/w/api.php"
        
        api_params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": 5,
            "srwhat": "text",
            "srprop": "size",
            "utf8": 1
        }
        
        headers = {
            "User-Agent": CONFIG["user_agent"]
        }
        
        response = requests.get(search_url, params=api_params, headers=headers, 
                               timeout=CONFIG["request_timeout"])
        
        if response.status_code == 200:
            data = response.json()
            if "query" in data and "search" in data["query"]:
                for item in data["query"]["search"]:
                    title = item["title"]
                    word_count = item.get("size", 0) / 5
                    
                    if any(r["title"] == title for r in results):
                        continue
                    
                    if word_count < 50:
                        continue
                    
                    try:
                        related_page = wiki_wiki.page(title)
                        if related_page.exists() and related_page.text:
                            relevance = calculate_query_relevance_improved(query, title, related_page.text[:300])
                            
                            if relevance > 0.1:
                                results.append({
                                    "title": title,
                                    "text": related_page.text,
                                    "url": related_page.fullurl,
                                    "score": relevance,
                                    "source": "api_search",
                                    "word_count": len(related_page.text.split())
                                })
                    except:
                        continue
        
        results.sort(key=lambda x: x["score"], reverse=True)
        
        final_results = []
        seen_titles = set()
        
        for result in results:
            if result["title"] not in seen_titles:
                seen_titles.add(result["title"])
                final_results.append(result)
            
            if len(final_results) >= 3:
                break
        
        return final_results
        
    except Exception as e:
        return []

def calculate_query_relevance_improved(query, title, content_preview):
    if not query or not title:
        return 0.0
    
    query_lower = query.lower()
    title_lower = title.lower()
    
    query_words = set(query_lower.split())
    title_words = set(title_lower.split())
    
    common_words = query_words.intersection(title_words)
    title_score = len(common_words) / max(len(query_words), 1) * 0.8
    
    bonus_score = 0
    if title_lower.startswith(query_lower):
        bonus_score = 0.2
    
    total_score = title_score + bonus_score
    
    return min(total_score, 1.0)

def fetch_web_content(url):
    if not url:
        return ""
    
    try:
        headers = {
            'User-Agent': random.choice(CONFIG["google_user_agents"]),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        text = soup.get_text()
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned_text = '\n'.join(lines)
        
        return cleaned_text
        
    except Exception as e:
        return ""

def save_content_with_dedup(content, metadata, qid, dedup_manager, lsh_manager):
    if not content or not isinstance(content, str):
        return None
    
    dedup_check = dedup_manager.is_duplicate(content, metadata.get("url"), metadata.get("title"))
    
    if dedup_check["is_duplicate"]:
        return None
    
    lsh_check = lsh_manager.check_similar_content(content, metadata.get("title"))
    
    try:
        os.makedirs(CONFIG["data_dir"], exist_ok=True)
        
        content_hash = dedup_check.get("content_hash", "")
        content_hash_short = content_hash[:10] if content_hash else hashlib.md5(content.encode()).hexdigest()[:10]
        
        safe_title = re.sub(r'[^\w\s-]', '', metadata.get("title", "unknown")).replace(' ', '_')[:40]
        filename = f"{qid}_{safe_title}_{content_hash_short}.txt"
        filepath = os.path.join(CONFIG["data_dir"], filename)
        
        if os.path.exists(filepath):
            return None
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# METADATA\n")
            f.write(f"# canonical_id: {dedup_check.get('canonical_id', 'N/A')}\n")
            f.write(f"# content_hash: {content_hash}\n")
            f.write(f"# lsh_id: {lsh_check.get('content_id', 'N/A')}\n")
            
            if lsh_check["is_similar"]:
                f.write(f"# near_duplicates: {len(lsh_check['similar_items'])} items\n")
            
            for key, value in metadata.items():
                if key not in ["analysis", "canonical_id", "content_hash", "lsh_id", "near_duplicates"]:
                    if isinstance(value, str) and len(value) > 100:
                        f.write(f"# {key}: {value[:100]}...\n")
                    else:
                        f.write(f"# {key}: {value}\n")
            
            f.write(f"\n# CONTENT ANALYSIS\n")
            f.write(f"# Quality Score: {metadata.get('quality_score', 0):.2f}/1.0\n")
            f.write(f"# Word Count: {metadata.get('word_count', 0)}\n")
            dedup_status = "NEAR_DUPLICATE" if lsh_check["is_similar"] else "UNIQUE"
            f.write(f"# Dedup Status: {dedup_status}\n")
            
            f.write(f"\n{'='*60}\n")
            f.write(f"CONTENT: {metadata.get('title', 'Unknown')}\n")
            f.write(f"{'='*60}\n\n")
            f.write(content)
        
        dedup_manager.save_cache()
        lsh_manager.save_cache()
        return filename
        
    except Exception as e:
        return None

def crawl_hybrid_optimized(json_file, max_questions=None):
    if not os.path.exists(json_file):
        print(f"✗ File not found: {json_file}")
        return
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"✗ Error loading JSON file: {e}")
        return
    
    print(f"Processing {len(data)} questions...")
    
    all_questions = []
    for item in data:
        if 'question' in item:
            processed_q = preprocess_question_enhanced(item['question'])
            if processed_q and len(processed_q.split()) >= 1:
                all_questions.append(processed_q)
    
    print("Building TF-IDF corpus...")
    hybrid_generator.build_corpus(all_questions)
    
    vocab_size = 0
    if hybrid_generator.vocabulary is not None and len(hybrid_generator.vocabulary) > 0:
        vocab_size = len(hybrid_generator.vocabulary)
    print(f"TF-IDF corpus built with {vocab_size} features")
    
    os.makedirs(CONFIG["data_dir"], exist_ok=True)
    
    dedup_manager = DeduplicationManager()
    lsh_manager = LSHManager()
    google_crawler = GoogleSearchCrawler() if CONFIG["use_google"] else None
    ddg_crawler = DuckDuckGoSearchCrawler() if CONFIG["use_duckduckgo"] else None
    
    log_fields = [
        'qid', 'question_short', 'keywords_found', 'entities_found',
        'queries_generated', 'pages_found', 'files_saved', 
        'duplicates_skipped', 'near_duplicates', 'avg_quality_score', 
        'processing_time', 'success', 'sources'
    ]
    
    try:
        with open(CONFIG["log_file"], 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=log_fields)
            writer.writeheader()
            
            processed = 0
            successful = 0
            total_duplicates_skipped = 0
            total_near_duplicates = 0
            
            pbar = tqdm(data, desc="Crawling")
            for idx, item in enumerate(pbar):
                if max_questions is not None and idx >= max_questions:
                    break
                
                start_time = time.time()
                qid = item.get('qid', f'q{idx}')
                question = item.get('question', '')
                
                processed_question = preprocess_question_enhanced(question)
                
                saved_files = []
                quality_scores = []
                pages_found = 0
                duplicates_skipped = 0
                near_duplicates = 0
                search_queries = []
                keywords_dict = {}
                sources_used = set()
                
                try:
                    keywords_dict = extract_keywords_enhanced(processed_question)
                    search_queries = generate_queries_expanded(processed_question, keywords_dict)
                    search_queries = [preprocess_query(q) for q in search_queries if q and len(q) > 2]
                    
                    queries_display = f"{len(search_queries)} queries"
                    pbar.set_postfix_str(f"Q{idx}: {queries_display}")
                    
                    queries_to_process = search_queries[:CONFIG["max_pages_per_question"]]
                    
                    for q_idx, query in enumerate(queries_to_process):
                        all_results = []
                        
                        try:
                            wiki_results = search_wikipedia_api(query)
                            all_results.extend(wiki_results)
                            if wiki_results:
                                sources_used.add("Wikipedia")
                        except Exception as e:
                            pass
                        
                        if len(all_results) < 2:
                            if CONFIG["use_google"] and google_crawler and not google_crawler.disabled:
                                if q_idx == 0 or len(all_results) == 0:
                                    try:
                                        google_results = google_crawler.search_google(query)
                                        if google_results:
                                            sources_used.add("Google")
                                            for result in google_results:
                                                if result['url']:
                                                    web_content = fetch_web_content(result['url'])
                                                    if web_content and len(web_content.split()) > CONFIG["min_content_length"]:
                                                        result['text'] = web_content
                                                        all_results.append(result)
                                    except Exception as e:
                                        pass

                            if (google_crawler and google_crawler.disabled and CONFIG["use_duckduckgo"]) or (not CONFIG["use_google"] and CONFIG["use_duckduckgo"]):
                                if q_idx == 0:
                                    try:
                                        ddg_results = ddg_crawler.search(query)
                                        if ddg_results:
                                            sources_used.add("DuckDuckGo")
                                            for result in ddg_results:
                                                if result['url']:
                                                    web_content = fetch_web_content(result['url'])
                                                    if web_content and len(web_content.split()) > CONFIG["min_content_length"]:
                                                        result['text'] = web_content
                                                        all_results.append(result)
                                    except Exception as e:
                                        pass
                        
                        for result in all_results:
                            pages_found += 1
                            
                            if result.get('source') in ['Google Search', 'DuckDuckGo']:
                                cleaned_content = result.get('text', '')
                            else:
                                cleaned_content = clean_wiki_with_bs4(result.get("text", ""))
                            
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
                                "title": result.get("title", ""),
                                "url": result.get("url", ""),
                                "quality_score": quality_metrics["quality_score"],
                                "word_count": len(cleaned_content.split()),
                                "download_date": datetime.now().isoformat(),
                                "insights_count": quality_metrics["insights_count"],
                                "source": result.get("source", "unknown")
                            }
                            
                            filename = save_content_with_dedup(cleaned_content, metadata, qid, dedup_manager, lsh_manager)
                            if filename:
                                saved_files.append(filename)
                                quality_scores.append(quality_metrics["quality_score"])
                                
                                if lsh_manager.check_similar_content(cleaned_content, metadata.get("title"))["is_similar"]:
                                    near_duplicates += 1
                                    total_near_duplicates += 1
                            else:
                                duplicates_skipped += 1
                                total_duplicates_skipped += 1
                        
                        time.sleep(CONFIG["rate_limit_delay"])
                
                except Exception as e:
                    pass
                
                processing_time = time.time() - start_time
                success = len(saved_files) > 0
                
                entities_count = 0
                if keywords_dict and "entities" in keywords_dict:
                    entities_count = sum(len(v) for v in keywords_dict["entities"].values())
                
                keywords_count = 0
                if keywords_dict:
                    keywords_count = sum(len(v) for v in [
                        keywords_dict.get("proper_nouns", []),
                        keywords_dict.get("common_nouns", []),
                        keywords_dict.get("tfidf_keywords", []),
                        keywords_dict.get("similar_keywords", [])
                    ])
                
                writer.writerow({
                    "qid": qid,
                    "question_short": (question[:100] + "...") if len(question) > 100 else question,
                    "keywords_found": keywords_count,
                    "entities_found": entities_count,
                    "queries_generated": len(search_queries),
                    "pages_found": pages_found,
                    "files_saved": len(saved_files),
                    "duplicates_skipped": duplicates_skipped,
                    "near_duplicates": near_duplicates,
                    "avg_quality_score": f"{np.mean(quality_scores):.3f}" if quality_scores else "0",
                    "processing_time": f"{processing_time:.1f}s",
                    "success": "YES" if success else "NO",
                    "sources": ", ".join(sources_used) if sources_used else "none"
                })
                
                csvfile.flush()
                
                processed += 1
                if success:
                    successful += 1
                
                time.sleep(random.uniform(5, 10))

            writer.writerow({
                "qid": "SUMMARY",
                "question_short": f"Total: {processed} questions",
                "keywords_found": "",
                "entities_found": "",
                "queries_generated": "",
                "pages_found": "",
                "files_saved": "",
                "duplicates_skipped": total_duplicates_skipped,
                "near_duplicates": total_near_duplicates,
                "avg_quality_score": "",
                "processing_time": "",
                "success": f"{successful}/{processed}",
                "sources": "Multiple"
            })
        
        print("\n" + "="*60)
        print("CRAWLING SUMMARY:")
        print("="*60)
        dedup_manager.print_stats()
        lsh_manager.print_stats()
        
        if CONFIG["use_google"] and google_crawler:
            google_crawler.print_stats()
        
        print(f"\nQuestions processed: {processed}")
        print(f"Successful crawls: {successful}")
        success_rate = successful / max(processed, 1) * 100
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Total duplicates skipped: {total_duplicates_skipped}")
        print(f"Total near duplicates detected: {total_near_duplicates}")
        print(f"Data saved in: {CONFIG['data_dir']}")
        print(f"Log saved to: {CONFIG['log_file']}")
        print(f"Dedup cache: {CONFIG['dedup_cache_file']}")
        print(f"LSH cache: {CONFIG['lsh_cache_file']}")
        print("="*60)
        
    except Exception as e:
        print(f"✗ Critical error in crawl process: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Wikipedia Crawler with Deduplication & Hybrid Search")
    parser.add_argument("--max-questions", type=int, default=None, help="Maximum number of questions to process")
    parser.add_argument("--test", action="store_true", help="Test mode (process 10 questions)")
    parser.add_argument("--input", type=str, default="test.json", help="Input JSON file")
    parser.add_argument("--no-google", action="store_true", help="Disable Google search")
    parser.add_argument("--lsh-threshold", type=float, default=CONFIG["lsh_threshold"], help="LSH similarity threshold")
    
    args = parser.parse_args()
    
    if args.no_google:
        CONFIG["use_google"] = False
    
    if args.lsh_threshold != CONFIG["lsh_threshold"]:
        CONFIG["lsh_threshold"] = args.lsh_threshold
    
    print("Starting Enhanced Crawl Pipeline with Advanced Features")
    print("Using: Hash + Canonical URL + LSH + TF-IDF + NER + BS4 + Wikipedia API + Google/DDG Search")
    print(f"Input file: {args.input}")
    print(f"LSH threshold: {CONFIG['lsh_threshold']}")
    print(f"Google search: {'ENABLED' if CONFIG['use_google'] else 'DISABLED'}")
    
    max_q = 10 if args.test else args.max_questions
    if args.test:
        print("Test mode: Processing 10 questions")
    
    crawl_hybrid_optimized(args.input, max_questions=max_q)
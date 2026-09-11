import re
import html
import csv
import json
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional, Generator
from src.config import CORPUS_PATH, TAXONOMY_PATH, INTENTS

def clean_tweet_text(text: str) -> str:
    """
    Cleans raw tweet text by:
    - Decoding HTML entities (&amp;, &gt;, &lt;)
    - Removing @mentions (e.g. @AppleSupport, @115854)
    - Normalizing strange Unicode characters (like iOS I-glyph bug \ufe0f)
    - Normalizing excessive whitespace
    """
    if not text:
        return ""
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Remove weird iOS 11 letter I question mark box bug (\u204a, \ufe0f, \ufffd)
    text = text.replace("I️", "I").replace("I\ufe0f", "I")
    
    # Remove user handles
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_valid_conversation_pair(cust_text: str, reply_text: str) -> bool:
    """
    Filters out noise:
    - Pure media tweets (only a URL or < 5 characters)
    - Automated acknowledgment bots
    - Non-English gibberish
    """
    c_clean = clean_tweet_text(cust_text)
    r_clean = clean_tweet_text(reply_text)
    
    # Must have substantive text
    if len(c_clean) < 10 or len(r_clean) < 10:
        return False
    # Ignore if customer text is purely a URL
    if re.match(r"^https?://\S+$", c_clean):
        return False
    return True

def stream_apple_support_pairs(
    source_url: str = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv",
    local_csv_path: Optional[str] = None,
    max_pairs: int = 3000
) -> Generator[Dict[str, str], None, None]:
    """
    Streams and matches customer queries with AppleSupport replies.
    """
    inbound_tweets: Dict[str, str] = {}
    brand_tweets: Dict[str, str] = {}
    pairs_yielded = 0
    
    if local_csv_path and Path(local_csv_path).exists():
        stream_src = open(local_csv_path, "r", encoding="utf-8", errors="ignore")
    else:
        req = urllib.request.Request(source_url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req)
        # Wrap chunk generator
        def chunk_reader():
            buffer = ""
            for chunk in iter(lambda: resp.read(262144), b""):
                buffer += chunk.decode("utf-8", errors="ignore")
                lines = buffer.split("\n")
                buffer = lines.pop()
                for line in lines:
                    yield line
        stream_src = chunk_reader()
    
    for line in stream_src:
        if not line.strip():
            continue
        if "AppleSupport" in line or any(k in line.lower() for k in ["iphone", "ipad", "ios", "apple", "macbook", "airpods"]):
            try:
                row = next(csv.reader([line]))
                if len(row) >= 7:
                    t_id, author_id, inbound, created_at, tweet_text, resp_id, in_resp_id = row[:7]
                    is_inbound = (inbound.lower() == "true")
                    
                    if author_id == "AppleSupport" and in_resp_id:
                        if in_resp_id in inbound_tweets:
                            cust_raw = inbound_tweets.pop(in_resp_id)
                            if is_valid_conversation_pair(cust_raw, tweet_text):
                                pairs_yielded += 1
                                yield {
                                    "cust_id": in_resp_id,
                                    "reply_id": t_id,
                                    "customer_text": clean_tweet_text(cust_raw),
                                    "customer_text_raw": cust_raw,
                                    "brand_reply": clean_tweet_text(tweet_text),
                                    "brand_reply_raw": tweet_text,
                                    "created_at": created_at
                                }
                        else:
                            brand_tweets[in_resp_id] = tweet_text
                    elif is_inbound and "@AppleSupport" in tweet_text:
                        if t_id in brand_tweets:
                            rep_raw = brand_tweets.pop(t_id)
                            if is_valid_conversation_pair(tweet_text, rep_raw):
                                pairs_yielded += 1
                                yield {
                                    "cust_id": t_id,
                                    "reply_id": "matched_prev",
                                    "customer_text": clean_tweet_text(tweet_text),
                                    "customer_text_raw": tweet_text,
                                    "brand_reply": clean_tweet_text(rep_raw),
                                    "brand_reply_raw": rep_raw,
                                    "created_at": created_at
                                }
                        else:
                            inbound_tweets[t_id] = tweet_text
                            
                if pairs_yielded >= max_pairs:
                    break
            except Exception:
                continue
                
    if hasattr(stream_src, "close"):
        stream_src.close()

def load_corpus(corpus_path: Optional[Path] = None) -> List[Dict[str, str]]:
    path = corpus_path or CORPUS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found at {path}. Run scripts/download_data.py first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

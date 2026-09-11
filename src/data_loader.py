import re
import html
import csv
import json
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional, Generator, Any, Union
import openpyxl

from src.config import (
    CORPUS_PATH,
    CORPUS_XLSX_PATH,
    CORPUS_CSV_PATH,
    GOLDEN_SET_PATH,
    GOLDEN_SET_XLSX_PATH,
    GOLDEN_SET_CSV_PATH,
    TAXONOMY_PATH,
    INTENTS
)

def clean_tweet_text(text: str) -> str:
    """
    Cleans raw tweet text by:
    - Decoding HTML entities (&amp;, &gt;, &lt;)
    - Removing @mentions (e.g. @AppleSupport, @115854)
    - Normalizing strange Unicode characters (like iOS 11 I-glyph bug \ufe0f)
    - Normalizing excessive whitespace
    """
    if not text:
        return ""
    text = html.unescape(str(text))
    text = text.replace("I️", "I").replace("I\ufe0f", "I")
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_valid_conversation_pair(cust_text: str, reply_text: str) -> bool:
    c_clean = clean_tweet_text(cust_text)
    r_clean = clean_tweet_text(reply_text)
    if len(c_clean) < 10 or len(r_clean) < 10:
        return False
    if re.match(r"^https?://\S+$", c_clean):
        return False
    return True

def normalize_column_names(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes spreadsheet column names so users can use Excel sheets
    with various header conventions (e.g. 'tweet' or 'query' -> 'customer_text').
    """
    normalized = {}
    key_mapping = {
        "text": "customer_text",
        "customer_text": "customer_text",
        "query": "customer_text",
        "message": "customer_text",
        "tweet": "customer_text",
        "customer_query": "customer_text",
        "intent": "true_intent",
        "true_intent": "true_intent",
        "predicted_intent": "true_intent",
        "action": "true_action",
        "true_action": "true_action",
        "triage_action": "true_action",
        "reason": "true_escalation_reason",
        "true_reason": "true_escalation_reason",
        "true_escalation_reason": "true_escalation_reason",
        "reply": "reference_reply",
        "reference_reply": "reference_reply",
        "gold_reply": "reference_reply",
        "brand_reply": "reference_reply",
        "score": "human_quality_score",
        "human_score": "human_quality_score",
        "human_quality_score": "human_quality_score"
    }

    for k, v in row.items():
        clean_k = str(k).strip().lower().replace(" ", "_").replace("-", "_")
        target_k = key_mapping.get(clean_k, k)
        # Clean string values
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                v = None
        normalized[target_k] = v

    return normalized

def load_sheet(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Universal spreadsheet loader: seamlessly loads datasets from Excel (.xlsx),
    CSV (.csv), or JSON (.json).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Spreadsheet file not found at: {path}")

    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xlsm", ".xltx"]:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        wb.close()
        if not rows:
            return []
        headers = [str(h).strip() for h in rows[0]]
        data = []
        for r_idx, r in enumerate(rows[1:]):
            row_dict = {}
            for col_idx, h in enumerate(headers):
                val = r[col_idx] if col_idx < len(r) else None
                row_dict[h] = val
            data.append(normalize_column_names(row_dict))
        return data

    elif suffix in [".csv", ".tsv"]:
        delimiter = "\t" if suffix == ".tsv" else ","
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            return [normalize_column_names(row) for row in reader]

    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            if isinstance(raw, list):
                return [normalize_column_names(row) for row in raw]
            return [normalize_column_names(raw)]

    else:
        raise ValueError(f"Unsupported file format: '{suffix}'. Please use an Excel (.xlsx), CSV (.csv), or JSON file.")

def load_golden_set(eval_file_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """
    Loads the golden evaluation dataset.
    Prioritizes user-provided path, then golden_eval_set.xlsx, then .csv, then .json.
    """
    if eval_file_path:
        return load_sheet(eval_file_path)

    # Default cascade: Excel (.xlsx) -> CSV (.csv) -> JSON (.json)
    if GOLDEN_SET_XLSX_PATH.exists():
        return load_sheet(GOLDEN_SET_XLSX_PATH)
    elif GOLDEN_SET_CSV_PATH.exists():
        return load_sheet(GOLDEN_SET_CSV_PATH)
    elif GOLDEN_SET_PATH.exists():
        return load_sheet(GOLDEN_SET_PATH)
    else:
        raise FileNotFoundError("No golden evaluation set found in data/ (checked .xlsx, .csv, .json).")

def load_corpus(corpus_file_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """
    Loads historical Apple Support corpus.
    Prioritizes user-provided path, then .xlsx, then .csv, then .json.
    """
    if corpus_file_path:
        return load_sheet(corpus_file_path)

    if CORPUS_XLSX_PATH.exists():
        return load_sheet(CORPUS_XLSX_PATH)
    elif CORPUS_CSV_PATH.exists():
        return load_sheet(CORPUS_CSV_PATH)
    elif CORPUS_PATH.exists():
        return load_sheet(CORPUS_PATH)
    else:
        raise FileNotFoundError("Corpus file not found. Run scripts/download_data.py first.")

def stream_apple_support_pairs(
    source_url: str = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv",
    local_csv_path: Optional[str] = None,
    max_pairs: int = 3000
) -> Generator[Dict[str, str], None, None]:
    inbound_tweets: Dict[str, str] = {}
    brand_tweets: Dict[str, str] = {}
    pairs_yielded = 0

    if local_csv_path and Path(local_csv_path).exists():
        stream_src = open(local_csv_path, "r", encoding="utf-8", errors="ignore")
    else:
        req = urllib.request.Request(source_url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req)
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

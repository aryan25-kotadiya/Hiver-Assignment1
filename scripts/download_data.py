import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CORPUS_PATH, PROCESSED_DATA_DIR
from src.data_loader import stream_apple_support_pairs

def main():
    print("=== Downloading & Curating Apple Support Corpus ===")
    target_count = 1500
    start = time.time()
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    pairs = []
    print(f"Streaming from Hugging Face repository to extract {target_count} clean AppleSupport pairs...")
    
    for pair in stream_apple_support_pairs(max_pairs=target_count):
        pairs.append(pair)
        if len(pairs) % 250 == 0:
            print(f"Extracted {len(pairs)}/{target_count} conversation pairs ({time.time() - start:.1f}s)...")
            
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
        
    print(f"\n[DONE] Saved {len(pairs)} AppleSupport pairs to {CORPUS_PATH} in {time.time() - start:.1f}s.")

if __name__ == "__main__":
    main()

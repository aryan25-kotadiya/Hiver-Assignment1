import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import SupportAgentPipeline
from src.config import BRAND_HANDLE

def main():
    print("=" * 65)
    print(f" {BRAND_HANDLE} AI Customer Support Agent — Interactive Demo")
    print(" Type a customer tweet and press Enter.")
    print(" Type 'exit' or 'quit' to stop.")
    print("=" * 65)

    pipeline = SupportAgentPipeline(mode="production")

    while True:
        try:
            user_input = input("\n[Customer Tweet] > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("\nExiting interactive demo. Goodbye!")
                break

            resp = pipeline.process(user_input)

            print("-" * 65)
            print(f"Predicted Intent     : {resp.predicted_intent} (Confidence: {resp.intent_confidence:.2f})")
            print(f"Triage Decision      : {resp.action} {'[Reason: ' + str(resp.escalation_reason) + ']' if resp.escalation_reason else ''}")
            print(f"Latency              : {resp.processing_time_ms:.1f}ms")
            print(f"Length               : {resp.reply_length}/280 characters")
            print(f"\n[Agent Reply]        : \"{resp.drafted_reply}\"")
            print("-" * 65)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive demo. Goodbye!")
            break

if __name__ == "__main__":
    main()

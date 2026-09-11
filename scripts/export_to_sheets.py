import json
import csv
from pathlib import Path
import openpyxl

def export_list_to_xlsx_and_csv(json_path: Path, xlsx_path: Path, csv_path: Path, fieldnames: list):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Write CSV with UTF-8 BOM so Excel opens it with proper encoding
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    # 2. Write real Excel .xlsx file
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Dataset"

    # Header
    ws.append(fieldnames)

    # Rows
    for row in data:
        row_vals = [row.get(field, "") for field in fieldnames]
        ws.append(row_vals)

    wb.save(xlsx_path)
    print(f"Exported {len(data)} rows to {xlsx_path.name} and {csv_path.name}")

def main():
    print("=== Exporting JSON Datasets to Excel (.xlsx) and CSV (.csv) ===")

    # 1. Golden Evaluation Set
    export_list_to_xlsx_and_csv(
        json_path=Path("data/golden_eval_set.json"),
        xlsx_path=Path("data/golden_eval_set.xlsx"),
        csv_path=Path("data/golden_eval_set.csv"),
        fieldnames=[
            "id",
            "customer_text",
            "true_intent",
            "true_action",
            "true_escalation_reason",
            "reference_reply",
            "human_quality_score",
            "sampling_stratum",
            "notes"
        ]
    )

    # 2. Calibration Study Set
    export_list_to_xlsx_and_csv(
        json_path=Path("data/calibration_study_set.json"),
        xlsx_path=Path("data/calibration_study_set.xlsx"),
        csv_path=Path("data/calibration_study_set.csv"),
        fieldnames=[
            "id",
            "customer_text",
            "candidate_reply",
            "true_action",
            "true_reason",
            "human_score",
            "notes"
        ]
    )

    # 3. Historical Apple Support Corpus
    export_list_to_xlsx_and_csv(
        json_path=Path("data/processed/apple_support_corpus.json"),
        xlsx_path=Path("data/processed/apple_support_corpus.xlsx"),
        csv_path=Path("data/processed/apple_support_corpus.csv"),
        fieldnames=[
            "cust_id",
            "reply_id",
            "customer_text",
            "brand_reply",
            "created_at"
        ]
    )

    print("[SUCCESS] All datasets exported to Excel (.xlsx) and CSV (.csv) spreadsheets successfully!")

if __name__ == "__main__":
    main()

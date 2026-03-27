import json 
import pandas as pd

# part to input json
INPUT_PATH = "../data/raw/led_cpu_25.jsonl"
# part to output csv
OUTPUT_PATH = "../outputs/evaluation_table.csv"

# function to load json
def load_json(path):
    records = []
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    return records

def build_table(records):
    rows = []

    for i, rec in enumerate(records, start=1):
        rows.append({
            "no": i,
            "arxiv_id": rec["arxiv_id"],
            "title": rec["title"],
            "abstract": rec["reference_abstract"],
            "generated_summary": rec["generated_summary"],
            "rouge_score": "",
            "bertscore": "",
            "human_score": ""
        })
    return pd.DataFrame(rows)

def main():
    records =load_json(INPUT_PATH)
    df = build_table(records)

# save to csv

    df.to_csv(OUTPUT_PATH,index=False,encoding="utf-8")
    print(f"Saved evaluation table to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "Model_Finetune/models/led-base-16384"

TEST_PATH = Path("Model_Finetune/data/processed/test_dataset_370_intro_conclusion.jsonl")
OUTPUT_PATH = Path("Model_Finetune/outputs/predictions/base_led_predictions_370.jsonl")

MAX_INPUT_LENGTH = 4096
MAX_SUMMARY_LENGTH = 180
MIN_SUMMARY_LENGTH = 50


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clean_generated_summary(summary):
    summary = summary.strip()

    summary = re.sub(
        r"^\s*(abstract|title|introduction|conclusion)\s*:\s*",
        "",
        summary,
        flags=re.IGNORECASE,
    )

    summary = re.sub(
        r"\b(title|introduction|conclusion|abstract)\s*:\s*",
        "",
        summary,
        flags=re.IGNORECASE,
    )

    summary = re.sub(r"\s+", " ", summary).strip()

    return summary


def generate_summary(model, tokenizer, input_text, device):
    inputs = tokenizer(
        input_text,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    global_attention_mask = torch.zeros_like(input_ids)
    global_attention_mask[:, 0] = 1

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            max_length=MAX_SUMMARY_LENGTH,
            min_length=MIN_SUMMARY_LENGTH,
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=4,
            repetition_penalty=1.2,
            early_stopping=True,
        )

    summary = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return clean_generated_summary(summary)


def main():
    print("Loading test records...")
    records = load_jsonl(TEST_PATH)
    print(f"Loaded {len(records)} test records")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    print("Loading base LED model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model.to(device)
    model.eval()

    predictions = []

    for record in tqdm(records, desc="Generating base LED summaries"):
        generated_summary = generate_summary(
            model=model,
            tokenizer=tokenizer,
            input_text=record["input_text"],
            device=device,
        )

        output_record = {
            "arxiv_id": record["arxiv_id"],
            "title": record["title"],
            "model_name": "base_led",
            "input_source": record["input_source"],
            "generated_summary": generated_summary,
            "reference_summary": record["reference_summary"],
        }

        predictions.append(output_record)

    save_jsonl(predictions, OUTPUT_PATH)

    print(f"Saved {len(predictions)} predictions")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "allenai/led-base-16384"
OUTPUT_DIR = Path("model_finetune/models/led-base-16384")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Downloading model: {MODEL_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    print(f"Saving tokenizer and model to: {OUTPUT_DIR}")
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
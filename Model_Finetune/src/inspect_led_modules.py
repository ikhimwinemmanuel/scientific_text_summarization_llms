from transformers import AutoModelForSeq2SeqLM


MODEL_NAME = "allenai/led-base-16384"


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    print("\nSearching for attention-related linear modules...\n")

    for name, module in model.named_modules():
        if any(key in name.lower() for key in ["q", "k", "v", "query", "key", "value", "proj"]):
            print(name)


if __name__ == "__main__":
    main()
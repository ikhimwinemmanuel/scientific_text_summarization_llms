import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


MODEL_NAME = "Model_Finetune/models/led-base-16384"

TRAIN_PATH = Path("Model_Finetune/data/processed/train_arxiv_5000.jsonl")
OUTPUT_DIR = Path("Model_Finetune/outputs/led_qlora_5000")

NUM_TRAIN_SAMPLES = 5000

MAX_INPUT_LENGTH = 4096
MAX_TARGET_LENGTH = 256

SEED = 42


def set_seed(seed):
    """
    Make the experiment more reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path, limit=None):
    """
    Load records from a JSONL file.
    Each line should contain one training example.
    """
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

            if limit is not None and len(records) >= limit:
                break

    return records


def preprocess_function(batch, tokenizer):
    """
    Tokenise article inputs and abstract targets for LED fine-tuning.
    """
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )

    labels = tokenizer(
        text_target=batch["target_summary"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    # LED requires global attention for long-document summarisation.
    # We assign global attention to the first token of each input.
    model_inputs["global_attention_mask"] = [
        [1] + [0] * (len(input_ids) - 1)
        for input_ids in model_inputs["input_ids"]
    ]

    return model_inputs


def save_training_config():
    """
    Save the training setup so the experiment is reproducible.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "model_name": MODEL_NAME,
        "train_path": str(TRAIN_PATH),
        "output_dir": str(OUTPUT_DIR),
        "num_train_samples": NUM_TRAIN_SAMPLES,
        "max_input_length": MAX_INPUT_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "seed": SEED,
        "qlora": {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": [
                "query",
                "value",
                "query_global",
                "value_global",
                "q_proj",
                "v_proj",
            ],
        },
    }

    config_path = OUTPUT_DIR / "training_config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved training config to: {config_path}")


def main():
    set_seed(SEED)
    save_training_config()

    print("Loading training records...")
    records = load_jsonl(TRAIN_PATH, limit=NUM_TRAIN_SAMPLES)
    dataset = Dataset.from_list(records)

    print(f"Loaded {len(dataset)} records")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"Using compute dtype: {compute_dtype}")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print("Loading base LED model in 4-bit...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        local_files_only=True,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    qlora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "query",
            "value",
            "query_global",
            "value_global",
            "q_proj",
            "v_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    print("Adding QLoRA adapters...")
    model = get_peft_model(model, qlora_config)
    model.print_trainable_parameters()

    print("Tokenising dataset...")
    tokenized_dataset = dataset.map(
        lambda batch: preprocess_function(batch, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=25,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        fp16=not use_bf16,
        bf16=use_bf16,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Starting full QLoRA fine-tuning...")
    trainer.train()

    print("Saving final QLoRA adapter...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Done. Final adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
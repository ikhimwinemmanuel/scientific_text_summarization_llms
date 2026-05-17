import json
from pathlib import Path

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


MODEL_NAME = "allenai/led-base-16384"

TRAIN_PATH = Path("Model_Finetune/data/processed/train_arxiv_5000.jsonl")
OUTPUT_DIR = Path("Model_Finetune/outputs/led_qlora_test")

NUM_TEST_SAMPLES = 50

MAX_INPUT_LENGTH = 2048
MAX_TARGET_LENGTH = 256


def load_jsonl(path, limit=None):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

            if limit is not None and len(records) >= limit:
                break

    return records


def preprocess_function(batch, tokenizer):
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

    # LED needs global attention.
    # We place global attention on the first token.
    model_inputs["global_attention_mask"] = [
        [1] + [0] * (len(input_ids) - 1)
        for input_ids in model_inputs["input_ids"]
    ]

    return model_inputs


def main():
    print("Loading training records...")
    records = load_jsonl(TRAIN_PATH, limit=NUM_TEST_SAMPLES)
    dataset = Dataset.from_list(records)

    print(f"Loaded {len(dataset)} records")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
    )

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
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=5,
        save_strategy="epoch",
        fp16=not use_bf16,
        bf16=use_bf16,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting tiny QLoRA test training...")
    trainer.train()

    print("Saving QLoRA test adapter...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Done. Adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
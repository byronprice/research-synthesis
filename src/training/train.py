"""
Fine-tune Llama 3.1 8B Instruct using Unsloth + QLoRA on the GPU cluster.

Run this on the cluster (NOT on M1 Mac):
  python src/training/train.py --config configs/training_config.yaml

Requires: unsloth, trl, peft, bitsandbytes, datasets, transformers
Install: pip install -r requirements/training.txt
"""

import argparse
import json
from pathlib import Path

import yaml
from loguru import logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training_config.yaml")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_dataset(train_file: str, val_split: float):
    """Load training data and split into train/val."""
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("Run on cluster with: pip install datasets")

    records = []
    with open(train_file) as f:
        for line in f:
            records.append(json.loads(line))

    # Format as chat messages for Llama 3.1 Instruct
    def format_record(r):
        instruction = r["instruction"]
        context = r.get("input", "")
        output = r["output"]

        if context:
            user_content = f"{instruction}\n\n{context}"
        else:
            user_content = instruction

        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ]
        }

    formatted = [format_record(r) for r in records]

    n_val = max(1, int(len(formatted) * val_split))
    train_data = formatted[n_val:]
    val_data = formatted[:n_val]

    return Dataset.from_list(train_data), Dataset.from_list(val_data)


def main():
    args = parse_args()
    config = load_config(args.config)

    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError:
        raise ImportError(
            "Unsloth not installed. Run on cluster:\n"
            "  pip install -r requirements/training.txt"
        )

    tc = config["training"]
    lc = config["lora"]

    # Load base model
    logger.info(f"Loading base model: {config['base_model']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["base_model"],
        max_seq_length=tc["max_seq_length"],
        dtype=None,           # auto-detect bfloat16 on Ampere+
        load_in_4bit=tc["load_in_4bit"],
    )

    # Apply chat template
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=lc["r"],
        target_modules=lc["target_modules"],
        lora_alpha=lc["lora_alpha"],
        lora_dropout=lc["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
    )

    # Load data
    dc = config["data"]
    logger.info(f"Loading training data from {dc['train_file']}")
    train_dataset, val_dataset = load_dataset(dc["train_file"], dc["val_split"])
    logger.info(f"Train: {len(train_dataset)} examples, Val: {len(val_dataset)} examples")

    def formatting_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    # Training arguments
    output_dir = config["output_dir"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tc["num_train_epochs"],
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        warmup_ratio=tc["warmup_ratio"],
        weight_decay=tc["weight_decay"],
        fp16=tc["fp16"],
        bf16=tc["bf16"],
        logging_steps=tc["logging_steps"],
        save_steps=tc["save_steps"],
        eval_steps=tc["eval_steps"],
        evaluation_strategy="steps",
        save_total_limit=tc["save_total_limit"],
        seed=tc["seed"],
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        max_seq_length=tc["max_seq_length"],
        args=training_args,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save LoRA adapter
    adapter_dir = f"{output_dir}-adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info(f"LoRA adapter saved → {adapter_dir}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()

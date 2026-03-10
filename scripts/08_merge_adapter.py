#!/usr/bin/env python3
"""
Merge a trained LoRA adapter into the base model for deployment.

Run on the cluster (needs sufficient RAM) OR on M1 with enough swap:
  python scripts/08_merge_adapter.py \
    --adapter outputs/llama-3.1-8b-neuro-adapter \
    --output outputs/llama-3.1-8b-neuro-merged

Then copy the merged model to your M1 for local inference.
"""

import argparse
from pathlib import Path

from loguru import logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    p.add_argument("--output", required=True, help="Output directory for merged model")
    p.add_argument("--base-model", default=None,
                   help="Base model name/path (defaults to adapter's base model)")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        # Fallback: use PEFT merge without Unsloth
        logger.info("Unsloth not found, using PEFT merge")
        _merge_with_peft(args)
        return

    logger.info(f"Loading adapter from {args.adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=False,  # load in full precision for merging
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Merging and saving to {output_dir}")
    model.save_pretrained_merged(
        str(output_dir),
        tokenizer,
        save_method="merged_16bit",
    )
    logger.info("Merge complete.")


def _merge_with_peft(args):
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    logger.info(f"Loading PEFT adapter: {args.adapter}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)

    logger.info("Merging adapter weights...")
    merged = model.merge_and_unload()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Merged model saved → {output_dir}")


if __name__ == "__main__":
    main()

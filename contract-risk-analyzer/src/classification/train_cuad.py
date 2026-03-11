"""
train_cuad.py — Fine-tune bert-base-uncased on the CUAD dataset.

CUAD is a QA task: given a contract and a question (clause type),
predict the answer span (or empty if the clause type is not present).

Usage:
    python src/classification/train_cuad.py \
        --output_dir models/cuad_finetuned \
        --epochs 3 \
        --batch_size 8

Hardware requirements:
    - GPU with 16GB+ VRAM recommended (RTX 3080 / A10 / T4)
    - Training on CPU is possible but very slow (~10x slower)

References:
    - Dataset: https://huggingface.co/datasets/cuad
    - Original paper: https://arxiv.org/abs/2103.06268
    - Atticus Project: https://www.atticusprojectai.org/cuad
"""

import argparse
import os
from pathlib import Path

import logging; logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT on CUAD dataset")
    parser.add_argument("--model_name", default="bert-base-uncased", help="Base model")
    parser.add_argument("--output_dir", default="models/cuad_finetuned", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--doc_stride", type=int, default=128, help="Document stride for sliding window")
    parser.add_argument("--warmup_steps", type=int, default=500, help="LR warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--eval_steps", type=int, default=500, help="Eval every N steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from datasets import load_dataset
        from transformers import (
            AutoTokenizer,
            AutoModelForQuestionAnswering,
            TrainingArguments,
            Trainer,
            DefaultDataCollator,
        )
        import torch
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Install with: pip install transformers datasets torch")
        return

    # ------------------------------------------------------------------
    # 1. Load CUAD dataset from HuggingFace
    # ------------------------------------------------------------------
    logger.info("Loading CUAD dataset from HuggingFace hub...")
    dataset = load_dataset("cuad")
    logger.info(f"CUAD dataset loaded: {dataset}")

    # CUAD structure: SQuAD-style QA
    # Each example has: id, title, context (contract text), question (clause type),
    # answers (answer_start, text)

    # ------------------------------------------------------------------
    # 2. Load tokenizer
    # ------------------------------------------------------------------
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ------------------------------------------------------------------
    # 3. Tokenize / preprocess
    # ------------------------------------------------------------------
    def preprocess_train(examples):
        """
        Tokenize for extractive QA with sliding window.
        Follows standard SQuAD preprocessing.
        """
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=args.max_length,
            truncation="only_second",
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            sequence_ids = inputs.sequence_ids(i)

            # Find the start/end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If no answer, label as (CLS, CLS)
            if len(answer["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
            else:
                start_char = answer["answer_start"][0]
                end_char = start_char + len(answer["text"][0])

                # Check if answer is in context window
                if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Find token positions
                    s = context_start
                    while s <= context_end and offset[s][0] <= start_char:
                        s += 1
                    start_positions.append(s - 1)

                    e = context_end
                    while e >= context_start and offset[e][1] >= end_char:
                        e -= 1
                    end_positions.append(e + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    logger.info("Tokenizing training data...")
    train_dataset = dataset["train"].map(
        preprocess_train,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    logger.info("Tokenizing validation data...")
    val_dataset = dataset["test"].map(
        preprocess_train,
        batched=True,
        remove_columns=dataset["test"].column_names,
    )

    # ------------------------------------------------------------------
    # 4. Load model
    # ------------------------------------------------------------------
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    # ------------------------------------------------------------------
    # 5. Training configuration
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16 and torch.cuda.is_available(),
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=str(output_dir / "logs"),
        logging_steps=100,
        report_to=["tensorboard"],
        dataloader_num_workers=4,
        gradient_accumulation_steps=2 if args.batch_size < 16 else 1,
    )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    data_collator = DefaultDataCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    logger.info("Starting training...")
    logger.info(f"  Model:          {args.model_name}")
    logger.info(f"  Epochs:         {args.epochs}")
    logger.info(f"  Batch size:     {args.batch_size}")
    logger.info(f"  Learning rate:  {args.learning_rate}")
    logger.info(f"  Max length:     {args.max_length}")
    logger.info(f"  FP16:           {training_args.fp16}")
    logger.info(f"  Output:         {output_dir}")
    logger.info(f"  Train samples:  {len(train_dataset):,}")
    logger.info(f"  Val samples:    {len(val_dataset):,}")

    train_result = trainer.train()
    trainer.save_model()

    logger.info(f"Training complete! Model saved to {output_dir}")
    logger.info(f"Training metrics: {train_result.metrics}")

    # Save training metrics
    with open(output_dir / "train_results.json", "w") as f:
        import json
        json.dump(train_result.metrics, f, indent=2)


if __name__ == "__main__":
    main()

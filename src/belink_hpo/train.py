"""Fine-tune Qwen3 with LoRA to predict the correct option letter (A, B, C, ...).

Expects the "messages" format produced by ``belink-hpo build-dataset``:
    {"messages": [{"role": "user", ...}, {"role": "assistant", "content": "H"}]}

    belink-hpo train --dataset Dash00/data-hpo-belink --epochs 1 --no-4bit
    belink-hpo train --data-dir data                      # data/train.jsonl and data/test.jsonl
    belink-hpo train --model Qwen/Qwen3-8B --epochs 3

Requires the training extras: pip install "belink-hpo[train]"
"""

from __future__ import annotations

import argparse
import inspect
import os

import torch

END_OF_TURN = "<|im_end|>"


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="belink-hpo train", description=__doc__.split("\n")[0])
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--dataset", default=None, help="Hub dataset ID; if omitted, reads --data-dir")
    p.add_argument("--data-dir", default="data", help="Directory with train.jsonl and (optionally) test.jsonl")
    p.add_argument("--output-dir", default="qwen3-mcq-lora")
    p.add_argument("--epochs", type=float, default=0.5, help="Fractions are fine, e.g. 0.5 = half the data")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-length", type=int, default=1024, help="Longer examples are dropped")
    p.add_argument("--eval-samples", type=int, default=1000, help="Test rows used for accuracy (0 = all)")
    p.add_argument("--eval-steps", type=int, default=250, help="Evaluate and checkpoint every N steps")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--no-4bit", action="store_true", help="Load the base model in bf16 instead of 4-bit (faster, more VRAM)")
    p.add_argument("--grad-checkpointing", action="store_true",
                   help="Trade speed for memory; turn on only if you run out of VRAM")
    return p.parse_args(argv)


def build_prompt(tokenizer, messages):
    """Render everything up to the answer, exactly as it will look at inference.

    enable_thinking=False adds an empty <think></think> block after the
    assistant header, so the model learns to answer with the label directly.
    """
    return tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def make_tokenize_fn(tokenizer):
    def tokenize(example):
        messages = example["messages"]
        prompt_ids = tokenizer(build_prompt(tokenizer, messages), add_special_tokens=False)["input_ids"]
        answer = messages[-1]["content"].strip()
        answer_ids = tokenizer(answer + END_OF_TURN, add_special_tokens=False)["input_ids"]
        return {
            "input_ids": prompt_ids + answer_ids,
            # -100 masks the prompt: loss is computed only on the label + end-of-turn token
            "labels": [-100] * len(prompt_ids) + answer_ids,
        }

    return tokenize


def make_collator(pad_token_id):
    def collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, labels, attention = [], [], []
        for x in batch:
            pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_token_id] * pad)
            labels.append(x["labels"] + [-100] * pad)
            attention.append([1] * len(x["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention),
        }

    return collate


@torch.no_grad()
def evaluate_accuracy(model, tokenizer, rows, batch_size=16):
    """Greedy-decode the label for each prompt and compare with the gold label."""
    model.eval()
    tokenizer.padding_side = "left"  # required for batched generation
    correct = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]["messages"]
        prompts = [build_prompt(tokenizer, m) for m in batch]
        gold = [m[-1]["content"].strip() for m in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        correct += sum(p.strip() == g for p, g in zip(preds, gold))
    return correct / len(rows)


def main(argv=None):
    args = parse_args(argv)

    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    # ---- Data ----
    if args.dataset:
        dataset = load_dataset(args.dataset)
    else:
        files = {"train": f"{args.data_dir}/train.jsonl"}
        if os.path.exists(f"{args.data_dir}/test.jsonl"):
            files["test"] = f"{args.data_dir}/test.jsonl"
        dataset = load_dataset("json", data_files=files)
    has_test = "test" in dataset
    if not has_test:
        print("No test split found: training without evaluation.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = dataset.map(make_tokenize_fn(tokenizer), remove_columns=dataset["train"].column_names)
    before = {k: len(v) for k, v in tokenized.items()}
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) <= args.max_length)
    for k, v in tokenized.items():
        print(f"{k}: {len(v)} examples ({before[k] - len(v)} dropped for exceeding {args.max_length} tokens)")

    print("\nExample training prompt + answer:\n" + tokenizer.decode(tokenized["train"][0]["input_ids"]) + "\n")

    # ---- Model ----
    quant = None if args.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    use_bf16 = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model_kwargs = dict(quantization_config=quant, device_map="auto")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **model_kwargs)
    except TypeError:  # older transformers versions call this argument torch_dtype
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, **model_kwargs)
    if quant is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.grad_checkpointing)
    elif args.grad_checkpointing:
        model.gradient_checkpointing_enable()

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    # ---- Baseline accuracy ----
    test_rows = eval_set = None
    if has_test:
        test_rows = dataset["test"]
        if args.eval_samples and len(test_rows) > args.eval_samples:
            test_rows = test_rows.shuffle(seed=42).select(range(args.eval_samples))
        print(f"Accuracy before fine-tuning: {evaluate_accuracy(model, tokenizer, test_rows):.2%}")

        # Eval loss during training uses a capped subset so it stays quick
        eval_set = tokenized["test"]
        if args.eval_samples and len(eval_set) > args.eval_samples:
            eval_set = eval_set.shuffle(seed=42).select(range(args.eval_samples))

    # ---- Train ----
    tokenizer.padding_side = "right"

    training_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        logging_steps=20,
        eval_strategy="steps" if has_test else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        report_to="none",
    )
    supported = inspect.signature(TrainingArguments.__init__).parameters
    # Warm up over the first 5% of steps. transformers 5 dropped warmup_ratio and
    # reads a fractional warmup_steps as a ratio instead.
    if "warmup_ratio" in supported:
        training_kwargs["warmup_ratio"] = 0.05
    else:
        training_kwargs["warmup_steps"] = 0.05
    if "group_by_length" in supported:
        # Batch similar-length examples together to cut wasted padding
        training_kwargs["group_by_length"] = True
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_set,
        data_collator=make_collator(tokenizer.pad_token_id),
    )
    trainer.train()

    trainer.save_model(args.output_dir)  # LoRA adapter only
    tokenizer.save_pretrained(args.output_dir)

    if has_test:
        print(f"Accuracy after fine-tuning: {evaluate_accuracy(model, tokenizer, test_rows):.2%}")
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()

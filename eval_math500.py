"""
Evaluate a model on MATH-500 benchmark.

Uses vLLM for efficient generation. Answer comparison via DFT's math_equal (symbolic + numeric).
Results are streamed to results.jsonl as each problem is evaluated.

Usage:
    CUDA_VISIBLE_DEVICES=1 python eval_math500.py \
        --model checkpoints/distill/final \
        --output_dir eval_results/distill-qwen3-1.7B
"""

import json
import os
import sys
import re
import argparse

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Add DFT math_evaluation to path for math_equal
sys.path.insert(0, "/CGLab/ziheng/projects/DFT/math_evaluation")
from grader import math_equal


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def remove_boxed(s):
    if s is None:
        return None
    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] == left:
            return s[len(left):]
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except (AssertionError, IndexError):
        return None


def extract_answer(solution_str):
    try:
        s = last_boxed_only_string(solution_str)
        if s is None:
            return None
        return remove_boxed(s)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples per problem")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        seed=42,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        n=args.n_samples,
    )

    # Load dataset
    print(f"Loading dataset: {args.dataset} ({args.split})")
    ds = load_dataset(args.dataset, trust_remote_code=True, split=args.split)

    instruction = "Let's think step by step and output the final answer within \\boxed{}."

    # Build prompts
    prompts = []
    ground_truths = []
    for example in ds:
        question = example["problem"] + " " + instruction
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
        gt = extract_answer(example["solution"])
        ground_truths.append(gt)

    print(f"Generating responses for {len(prompts)} problems...")
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    # Evaluate — stream results to jsonl
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "results.jsonl")
    correct = 0
    total = len(outputs)

    with open(results_file, "w") as f:
        for i, output in enumerate(outputs):
            gt = ground_truths[i]

            response_evals = []
            any_correct = False
            for out in output.outputs:
                response = out.text
                extracted = extract_answer(response)
                is_correct = math_equal(extracted, gt, timeout=True) if (extracted is not None and gt is not None) else False
                response_evals.append({
                    "response": response,
                    "extracted_answer": extracted,
                    "is_correct": is_correct,
                })
                if is_correct:
                    any_correct = True

            if any_correct:
                correct += 1

            result = {
                "idx": i,
                "question": ds[i]["problem"],
                "ground_truth": gt,
                "responses": response_evals,
                "any_correct": any_correct,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{total}] running acc: {correct/(i+1):.4f} ({correct}/{i+1})")

    accuracy = correct / total if total > 0 else 0

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Model: {args.model}")
    print(f"MATH-500 Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Results saved to {args.output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

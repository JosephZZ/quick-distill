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
from collections import Counter

# vLLM 0.11 + transformers 5.x compatibility fix
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
if not hasattr(Qwen2Tokenizer, "all_special_tokens_extended"):
    Qwen2Tokenizer.all_special_tokens_extended = property(lambda self: list(getattr(self, "all_special_tokens", []) or []))
try:
    from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
    if not hasattr(Qwen2TokenizerFast, "all_special_tokens_extended"):
        Qwen2TokenizerFast.all_special_tokens_extended = property(lambda self: list(getattr(self, "all_special_tokens", []) or []))
except ImportError:
    pass

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# vLLM 0.11 / torch-triton compatibility shim:
# some envs miss triton.compiler.compiler.triton_key, which vLLM imports via torch inductor.
try:
    import triton.compiler.compiler as _triton_compiler
    if not hasattr(_triton_compiler, "triton_key"):
        def _triton_key():
            return "unknown"
        _triton_compiler.triton_key = _triton_key
except Exception:
    pass

from vllm import LLM, SamplingParams

# Prefer local stub grader used in this repo; fall back to bundled math_evaluation.
_repo_dir = os.path.dirname(os.path.abspath(__file__))
_stub_dir = os.path.join(_repo_dir, "math_eval_stub")
_fallback_dir = os.path.join(_repo_dir, "math_evaluation")
if os.path.isdir(_stub_dir):
    sys.path.insert(0, _stub_dir)
else:
    sys.path.insert(0, _fallback_dir)
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


def normalize_answer_for_vote(ans):
    if ans is None:
        return None
    return re.sub(r"\s+", " ", str(ans)).strip()


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
        enforce_eager=True,
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
        chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                enable_thinking=False, **chat_kwargs,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}], **chat_kwargs,
            )
        prompts.append(prompt)
        gt = extract_answer(example["solution"])
        ground_truths.append(gt)

    print(f"Generating responses for {len(prompts)} problems...")
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    # Evaluate — stream results to jsonl
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "results.jsonl")
    pass_correct = 0
    maj_correct = 0
    avg_correct_sum = 0.0
    sample_correct_total = 0
    sample_total = 0
    total = len(outputs)

    with open(results_file, "w") as f:
        for i, output in enumerate(outputs):
            gt = ground_truths[i]

            response_evals = []
            sample_flags = []
            for out in output.outputs:
                response = out.text
                extracted = extract_answer(response)
                is_correct = math_equal(extracted, gt, timeout=True) if (extracted is not None and gt is not None) else False
                sample_flags.append(is_correct)
                response_evals.append({
                    "response": response,
                    "extracted_answer": extracted,
                    "is_correct": is_correct,
                })
            pass_hit = any(sample_flags)
            if pass_hit:
                pass_correct += 1

            sample_n = len(sample_flags)
            sample_total += sample_n
            sample_correct_total += sum(1 for x in sample_flags if x)
            avg_hit = (sum(1 for x in sample_flags if x) / sample_n) if sample_n > 0 else 0.0
            avg_correct_sum += avg_hit

            # Majority vote on extracted answers (ignore None); tie-break by earliest appearance.
            vote_counts = Counter()
            first_pos = {}
            repr_answer = {}
            for j, r in enumerate(response_evals):
                raw_ans = r["extracted_answer"]
                key = normalize_answer_for_vote(raw_ans)
                if key is None or key == "":
                    continue
                vote_counts[key] += 1
                if key not in first_pos:
                    first_pos[key] = j
                    repr_answer[key] = raw_ans
            maj_answer = None
            maj_hit = False
            if vote_counts:
                best_cnt = max(vote_counts.values())
                cands = [k for k, c in vote_counts.items() if c == best_cnt]
                maj_key = sorted(cands, key=lambda k: first_pos[k])[0]
                maj_answer = repr_answer[maj_key]
                maj_hit = math_equal(maj_answer, gt, timeout=True) if gt is not None else False
            if maj_hit:
                maj_correct += 1

            result = {
                "idx": i,
                "question": ds[i]["problem"],
                "ground_truth": gt,
                "responses": response_evals,
                "pass_correct": pass_hit,
                "maj_answer": maj_answer,
                "maj_correct": maj_hit,
                "avg_correct": avg_hit,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            if (i + 1) % 50 == 0:
                print(
                    f"  [{i+1}/{total}] pass={pass_correct/(i+1):.4f} "
                    f"maj={maj_correct/(i+1):.4f} avg={avg_correct_sum/(i+1):.4f}"
                )

    pass_accuracy = pass_correct / total if total > 0 else 0.0
    maj_accuracy = maj_correct / total if total > 0 else 0.0
    avg_accuracy = avg_correct_sum / total if total > 0 else 0.0

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "total": total,
        # Backward-compatible fields map to pass@k.
        "correct": pass_correct,
        "accuracy": pass_accuracy,
        "pass_correct": pass_correct,
        "pass_accuracy": pass_accuracy,
        "maj_correct": maj_correct,
        "maj_accuracy": maj_accuracy,
        "avg_accuracy": avg_accuracy,
        "sample_correct_total": sample_correct_total,
        "sample_total": sample_total,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Model: {args.model}")
    print(f"MATH-500 pass@{args.n_samples}: {pass_accuracy:.4f} ({pass_correct}/{total})")
    print(f"MATH-500 maj@{args.n_samples}:  {maj_accuracy:.4f} ({maj_correct}/{total})")
    print(f"MATH-500 avg@{args.n_samples}:  {avg_accuracy:.4f}")
    print(f"Results saved to {args.output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

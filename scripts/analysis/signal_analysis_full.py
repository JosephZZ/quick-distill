#!/usr/bin/env python3
"""
Full signal analysis: KL divergence, log-prob gap, and teacher confidence by position.
100 MATH-500 problems, 1 trajectory each.

Phase 1: vLLM generates trajectories (fast), then shuts down.
Phase 2: Load student + teacher HF models together, score one trajectory at a time,
         compute all metrics on-the-fly, discard distributions immediately.

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/analysis/signal_analysis_full.py --gpu 0
"""

import argparse
import json
import os
import sys
import time
import gc

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ---- Config ----
TEACHER_MODEL = "Qwen/Qwen3-1.7B"
STUDENT_MODEL = "Qwen/Qwen2.5-Math-1.5B"
N_PROBLEMS = 100
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
SEED = 42
VLLM_GPU_UTIL = 0.90

OUTPUT_DIR = "docs/signal_analysis"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    return parser.parse_args()


def load_problems(n=100):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [item["problem"] for item in ds][:n]


def build_prompt(problem, tokenizer):
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def phase1_generate(tokenizer, problems):
    """Generate trajectories with vLLM, then destroy the engine."""
    from vllm import LLM, SamplingParams

    prompts = [build_prompt(p, tokenizer) for p in problems]

    llm = LLM(
        model=STUDENT_MODEL,
        tokenizer=STUDENT_MODEL,
        gpu_memory_utilization=VLLM_GPU_UTIL,
        seed=SEED,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
        top_p=1.0,
    )

    print(f"  vLLM generating {len(prompts)} problems...", flush=True)
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    print(f"  vLLM done in {time.time()-t0:.1f}s", flush=True)

    trajectories = []
    for i, output in enumerate(outputs):
        response_ids = list(output.outputs[0].token_ids)
        prompt_len = len(output.prompt_token_ids)
        trajectories.append({
            "problem_idx": i,
            "prompt_len": prompt_len,
            "response_ids": response_ids,
            "response_len": len(response_ids),
        })

    # Destroy vLLM completely
    del llm, outputs
    gc.collect()
    torch.cuda.empty_cache()

    return trajectories


def phase2_score(tokenizer, problems, trajectories, device, output_dir):
    """
    Load student + teacher together, score one trajectory at a time.
    At each position: compute logits from both models, derive all metrics, discard.
    """
    max_len = max(t["response_len"] for t in trajectories)
    n = len(trajectories)

    # Accumulators
    logprob_gap_sum = np.zeros(max_len)
    kl_fwd_sum = np.zeros(max_len)      # KL(p_t || p_s)
    kl_rev_sum = np.zeros(max_len)      # KL(p_s || p_t)
    teacher_entropy_sum = np.zeros(max_len)
    teacher_top1_prob_sum = np.zeros(max_len)
    agreement_sum = np.zeros(max_len)
    counts = np.zeros(max_len)

    # Also save per-token logprobs for backward compat
    all_student_lps = []
    all_teacher_lps = []

    # Load both models
    print("  Loading student model...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    student.eval()

    print("  Loading teacher model...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    teacher.eval()

    print(f"  Scoring {n} trajectories...", flush=True)
    t0 = time.time()

    for i, traj in enumerate(trajectories):
        problem = problems[traj["problem_idx"]]
        prompt = build_prompt(problem, tokenizer)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]

        response_ids_tensor = torch.tensor([traj["response_ids"]])
        full_ids = torch.cat([prompt_ids, response_ids_tensor], dim=1).to(device)
        resp_len = traj["response_len"]

        with torch.no_grad():
            s_logits = student(full_ids).logits[0]  # [seq_len, vocab]
            t_logits = teacher(full_ids).logits[0]

        traj_s_lps = []
        traj_t_lps = []

        for j in range(resp_len):
            logit_idx = prompt_len + j - 1
            token_id = traj["response_ids"][j]

            # Full log-prob distributions (float32 for precision)
            s_lp_dist = F.log_softmax(s_logits[logit_idx].float(), dim=-1)
            t_lp_dist = F.log_softmax(t_logits[logit_idx].float(), dim=-1)
            t_p = torch.exp(t_lp_dist)
            s_p = torch.exp(s_lp_dist)

            # 1. Token log-prob gap
            s_token_lp = s_lp_dist[token_id].item()
            t_token_lp = t_lp_dist[token_id].item()
            traj_s_lps.append(s_token_lp)
            traj_t_lps.append(t_token_lp)
            logprob_gap_sum[j] += abs(s_token_lp - t_token_lp)

            # 2. KL(p_t || p_s) — forward KL
            mask_t = t_p > 1e-10
            kl_fwd = torch.sum(t_p[mask_t] * (t_lp_dist[mask_t] - s_lp_dist[mask_t])).item()
            kl_fwd_sum[j] += max(kl_fwd, 0.0)

            # 3. KL(p_s || p_t) — reverse KL
            mask_s = s_p > 1e-10
            kl_rev = torch.sum(s_p[mask_s] * (s_lp_dist[mask_s] - t_lp_dist[mask_s])).item()
            kl_rev_sum[j] += max(kl_rev, 0.0)

            # 4. Teacher entropy
            entropy = -torch.sum(t_p[mask_t] * t_lp_dist[mask_t]).item()
            teacher_entropy_sum[j] += entropy

            # 5. Teacher top-1 prob
            teacher_top1_prob_sum[j] += torch.max(t_p).item()

            # 6. Agreement (argmax match)
            if torch.argmax(t_p).item() == torch.argmax(s_p).item():
                agreement_sum[j] += 1

            counts[j] += 1

        all_student_lps.append(traj_s_lps)
        all_teacher_lps.append(traj_t_lps)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Scored {i+1}/{n} ({elapsed:.0f}s)", flush=True)

    print(f"  All scoring done in {time.time()-t0:.1f}s", flush=True)

    del student, teacher
    gc.collect()
    torch.cuda.empty_cache()

    # Save per-token logprobs (backward compat with old format)
    lp_path = os.path.join(output_dir, "raw_logprobs.jsonl")
    with open(lp_path, "w") as f:
        for i, traj in enumerate(trajectories):
            f.write(json.dumps({
                "response_ids": traj["response_ids"],
                "student_lps": all_student_lps[i],
                "teacher_lps": all_teacher_lps[i],
            }) + "\n")
    print(f"  Saved logprobs to {lp_path}", flush=True)

    # Build per-position metrics
    metrics = {}
    for pos in range(max_len):
        c = counts[pos]
        if c == 0:
            break
        metrics[str(pos)] = {
            "logprob_gap": float(logprob_gap_sum[pos] / c),
            "kl_forward": float(kl_fwd_sum[pos] / c),
            "kl_reverse": float(kl_rev_sum[pos] / c),
            "teacher_entropy": float(teacher_entropy_sum[pos] / c),
            "teacher_top1_prob": float(teacher_top1_prob_sum[pos] / c),
            "agreement_rate": float(agreement_sum[pos] / c),
            "count": int(c),
        }

    per_pos_path = os.path.join(output_dir, "per_position_metrics.json")
    with open(per_pos_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved per-position metrics to {per_pos_path}", flush=True)

    # Range summaries
    ranges = [
        (0, 50), (50, 100), (100, 150), (150, 200),
        (200, 300), (300, 400), (400, 500), (500, 600),
        (600, 700), (700, 800), (800, 900), (900, 1000),
        (1000, 1200), (1200, 1500), (1500, 2000),
    ]
    fields = ["logprob_gap", "kl_forward", "kl_reverse", "teacher_entropy", "teacher_top1_prob", "agreement_rate"]
    range_summary = {}
    for lo, hi in ranges:
        pos_keys = [str(p) for p in range(lo, min(hi, max_len)) if str(p) in metrics]
        if not pos_keys:
            continue
        total_count = sum(metrics[p]["count"] for p in pos_keys)
        if total_count == 0:
            continue
        summary = {"n_tokens": total_count, "n_positions": len(pos_keys)}
        for field in fields:
            summary[field] = float(np.average(
                [metrics[p][field] for p in pos_keys],
                weights=[metrics[p]["count"] for p in pos_keys]
            ))
        range_summary[f"{lo}-{hi}"] = summary

    range_path = os.path.join(output_dir, "range_summary.json")
    with open(range_path, "w") as f:
        json.dump(range_summary, f, indent=2)
    print(f"  Saved range summary to {range_path}", flush=True)

    # Print table
    print("\n" + "=" * 110, flush=True)
    hdr = f"{'Range':>12} {'LogP Gap':>10} {'KL(t||s)':>10} {'KL(s||t)':>10} {'Entropy':>10} {'Top1 Prob':>10} {'Agree%':>8} {'N':>7}"
    print(hdr, flush=True)
    print("=" * 110, flush=True)
    for rng, v in range_summary.items():
        print(f"{rng:>12} {v['logprob_gap']:10.4f} {v['kl_forward']:10.4f} {v['kl_reverse']:10.4f} "
              f"{v['teacher_entropy']:10.4f} {v['teacher_top1_prob']:10.4f} {v['agreement_rate']:8.1%} {v['n_tokens']:7d}", flush=True)


def main():
    args = parse_args()
    device = f"cuda:{args.gpu}"
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Device: {device}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Student: {STUDENT_MODEL}", flush=True)
    print(f"Teacher: {TEACHER_MODEL}", flush=True)
    print(f"N_PROBLEMS: {N_PROBLEMS}, MAX_NEW_TOKENS: {MAX_NEW_TOKENS}\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    problems = load_problems(N_PROBLEMS)
    print(f"Loaded {len(problems)} problems", flush=True)

    # Phase 1: vLLM generate, then shut down
    print("\n=== Phase 1: vLLM Generation ===", flush=True)
    trajectories = phase1_generate(tokenizer, problems)

    lens = [t["response_len"] for t in trajectories]
    print(f"Lengths: min={min(lens)} max={max(lens)} mean={np.mean(lens):.0f} median={np.median(lens):.0f}", flush=True)

    # Save trajectories
    traj_path = os.path.join(output_dir, "trajectories.json")
    with open(traj_path, "w") as f:
        json.dump(trajectories, f)

    # Phase 2: HF scoring (student + teacher together)
    print("\n=== Phase 2: HF Scoring ===", flush=True)
    phase2_score(tokenizer, problems, trajectories, device, output_dir)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()

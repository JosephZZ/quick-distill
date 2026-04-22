"""
Analyze KL coverage of different token selection strategies.
For each strategy, compute what % of total KL the selected 100 tokens cover.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/analysis/strategy_kl_coverage.py \
    --student_model Qwen/Qwen2.5-Math-1.5B \
    --teacher_model Qwen/Qwen3-1.7B \
    --num_problems 50 --max_new_tokens 512
"""

import argparse
import gc
import json
import os
import sys
import torch
import numpy as np
from torch.nn.functional import log_softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from on_policy_distill_positional import build_prompt, _supports_thinking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", type=str, required=True)
    parser.add_argument("--teacher_model", type=str, required=True)
    parser.add_argument("--num_problems", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--k", type=int, default=100, help="Number of tokens to select")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = "cuda:0"
    K = args.k

    # Load tokenizers + registry
    student_tok = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)

    registry_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "model_registry.json")
    registry = {}
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            registry = json.load(f).get("models", {})

    student_thinking = registry.get(args.student_model, {}).get("thinking", False)
    teacher_thinking = registry.get(args.teacher_model, {}).get("thinking", False)
    nothink_str = "<think>\n\n</think>\n\n"
    nothink_ids = student_tok.encode(nothink_str, add_special_tokens=False) if student_thinking else []
    nothink_ids_teacher = teacher_tok.encode(nothink_str, add_special_tokens=False) if teacher_thinking else []

    print(f"Student: {args.student_model}, Teacher: {args.teacher_model}, K={K}")

    # Load dataset
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    problems = [ds[i]["problem"] for i in range(args.num_problems)]

    # Phase 1: Student generate + logits
    print("Phase 1: Student generation...")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

    student_data = []
    for pi, problem in enumerate(problems):
        prompt = build_prompt(problem, student_tok)
        input_ids = student_tok.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=device)
        with torch.no_grad():
            output = student.generate(input_tensor, max_new_tokens=args.max_new_tokens,
                                       temperature=0.7, do_sample=True, pad_token_id=student_tok.eos_token_id)
        response_ids = output[0][len(input_ids):].tolist()
        if len(response_ids) < K + 10:
            continue

        full_ids = input_ids + nothink_ids + response_ids
        full_tensor = torch.tensor([full_ids], device=device)
        with torch.no_grad():
            s_logits = student(full_tensor).logits[0]
        resp_start = len(input_ids) + len(nothink_ids)
        s_log_probs = log_softmax(s_logits[resp_start - 1:-1].float(), dim=-1).cpu()

        student_data.append({
            "problem": problem, "input_ids": input_ids,
            "response_ids": response_ids, "s_log_probs": s_log_probs,
        })
        if (pi + 1) % 10 == 0:
            print(f"  {pi+1}/{args.num_problems}")

    del student; gc.collect(); torch.cuda.empty_cache()

    # Phase 2: Teacher scoring
    print(f"Phase 2: Teacher scoring ({len(student_data)} responses)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

    # Per-strategy accumulation
    strategies = ["prefix", "top_kl", "ent_student", "ent_teacher", "ent_and", "ent_or", "random", "middle", "last"]
    strat_kl_covered = {s: [] for s in strategies}  # list of (covered_kl, total_kl) per response
    strat_positions = {s: [] for s in strategies}    # average position of selected tokens
    strat_ent_overlap = {s: [] for s in strategies}  # % of selected tokens in global top-20% entropy

    for di, data in enumerate(student_data):
        input_ids = data["input_ids"]
        response_ids = data["response_ids"]
        s_lp = data["s_log_probs"]

        t_full_ids = input_ids + nothink_ids_teacher + response_ids
        t_resp_start = len(input_ids) + len(nothink_ids_teacher)
        t_tensor = torch.tensor([t_full_ids], device=device)
        with torch.no_grad():
            t_logits = teacher(t_tensor).logits[0]
        t_lp = log_softmax(t_logits[t_resp_start - 1:-1].float(), dim=-1).cpu()

        resp_len = min(len(response_ids), t_lp.shape[0], s_lp.shape[0])
        if resp_len < K + 10:
            continue

        vocab_size = min(s_lp.shape[-1], t_lp.shape[-1])
        s_lp = s_lp[:resp_len, :vocab_size]
        t_lp = t_lp[:resp_len, :vocab_size]

        # Compute per-position metrics
        s_probs = torch.exp(s_lp)
        t_probs = torch.exp(t_lp)
        kl_per_pos = (s_probs * (s_lp - t_lp)).sum(dim=-1).clamp(min=0)  # [resp_len]
        ent_s = -(s_probs * s_lp).sum(dim=-1)
        ent_t = -(t_probs * t_lp).sum(dim=-1)

        total_kl = kl_per_pos.sum().item()
        if total_kl < 1e-6:
            continue

        k = min(K, resp_len)

        # Select indices for each strategy
        for strat in strategies:
            if strat == "prefix":
                idx = torch.arange(k)
            elif strat == "top_kl":
                idx = torch.topk(kl_per_pos[:resp_len], k=k, largest=True).indices
            elif strat == "ent_student":
                idx = torch.topk(ent_s[:resp_len], k=k, largest=True).indices
            elif strat == "ent_teacher":
                idx = torch.topk(ent_t[:resp_len], k=k, largest=True).indices
            elif strat == "ent_and":
                score = ent_s[:resp_len] * ent_t[:resp_len]
                idx = torch.topk(score, k=k, largest=True).indices
            elif strat == "ent_or":
                score = ent_s[:resp_len] + ent_t[:resp_len]
                idx = torch.topk(score, k=k, largest=True).indices
            elif strat == "random":
                idx = torch.randperm(resp_len)[:k]
            elif strat == "middle":
                mid = resp_len // 2
                start = max(0, mid - k // 2)
                idx = torch.arange(start, min(start + k, resp_len))
            elif strat == "last":
                idx = torch.arange(max(0, resp_len - k), resp_len)

            covered_kl = kl_per_pos[idx].sum().item()
            strat_kl_covered[strat].append((covered_kl, total_kl))
            strat_positions[strat].append(idx.float().mean().item())

            # Compute entropy overlap: what fraction of selected tokens are in global top-20% entropy?
            ent_threshold_20 = torch.quantile(ent_s[:resp_len], 0.8)  # top 20% = above 80th percentile
            high_ent_mask = ent_s[:resp_len] >= ent_threshold_20
            n_high_ent_in_selection = high_ent_mask[idx].sum().item()
            strat_ent_overlap[strat].append(n_high_ent_in_selection / k * 100)

        if (di + 1) % 10 == 0:
            print(f"  {di+1}/{len(student_data)}")

    del teacher; gc.collect(); torch.cuda.empty_cache()

    # Results
    print(f"\n{'='*70}")
    print(f"KL Coverage Analysis: {args.student_model} → {args.teacher_model} (K={K})")
    print(f"{'='*70}")
    print(f"\n{'Strategy':<18} {'KL Cov %':<12} {'Ent Overlap %':<15} {'Avg Pos':<10} {'Contig':<8}")
    print("-" * 63)

    results = {}
    for strat in strategies:
        if not strat_kl_covered[strat]:
            continue
        coverages = [c / t * 100 for c, t in strat_kl_covered[strat]]
        avg_cov = np.mean(coverages)
        avg_ent_overlap = np.mean(strat_ent_overlap[strat]) if strat_ent_overlap[strat] else 0
        avg_pos = np.mean(strat_positions[strat])
        contiguous = strat in ("prefix", "middle", "last")
        print(f"{strat:<18} {avg_cov:<12.1f} {avg_ent_overlap:<15.1f} {avg_pos:<10.1f} {'Yes' if contiguous else 'No':<8}")
        results[strat] = {
            "kl_coverage_pct": avg_cov, "ent_overlap_pct": avg_ent_overlap,
            "avg_position": avg_pos, "contiguous": contiguous,
        }

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

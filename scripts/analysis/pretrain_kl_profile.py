"""
Pre-training KL profile analysis: compare teacher-student KL by position
across different model pairs BEFORE any distillation training.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/analysis/pretrain_kl_profile.py \
    --student_model meta-llama/Llama-3.2-1B-Instruct \
    --teacher_model meta-llama/Llama-3.1-8B-Instruct \
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
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = "cuda:0"

    # Load tokenizers
    student_tok = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    same_tokenizer = (student_tok.vocab_size == teacher_tok.vocab_size)

    # Check thinking mode via registry
    registry_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "model_registry.json")
    registry = {}
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            registry = json.load(f).get("models", {})

    student_thinking = registry.get(args.student_model, {}).get("thinking", _supports_thinking(student_tok))
    teacher_thinking = registry.get(args.teacher_model, {}).get("thinking", _supports_thinking(teacher_tok))

    nothink_str = "<think>\n\n</think>\n\n"
    nothink_ids = student_tok.encode(nothink_str, add_special_tokens=False) if student_thinking else []
    nothink_ids_teacher = teacher_tok.encode(nothink_str, add_special_tokens=False) if teacher_thinking else []

    print(f"Student: {args.student_model} (thinking={student_thinking})")
    print(f"Teacher: {args.teacher_model} (thinking={teacher_thinking})")
    print(f"Same tokenizer: {same_tokenizer}")
    print(f"Nothink IDs: student={nothink_ids}, teacher={nothink_ids_teacher}")

    # Load dataset
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    problems = [ds[i]["problem"] for i in range(args.num_problems)]

    # === Phase 1: Generate with student + get student logits ===
    print(f"\nPhase 1: Student generation + logits ({args.num_problems} problems)...")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    student_data = []  # list of dicts with input_ids, response_ids, s_log_probs
    for pi, problem in enumerate(problems):
        prompt = build_prompt(problem, student_tok)
        input_ids = student_tok.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=device)

        with torch.no_grad():
            output = student.generate(
                input_tensor,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                pad_token_id=student_tok.eos_token_id,
            )

        response_ids = output[0][len(input_ids):].tolist()
        if len(response_ids) < 5:
            continue

        # Get student logits
        full_ids = input_ids + nothink_ids + response_ids
        full_tensor = torch.tensor([full_ids], device=device)
        with torch.no_grad():
            s_logits = student(full_tensor).logits[0]

        resp_start = len(input_ids) + len(nothink_ids)
        s_log_probs = log_softmax(s_logits[resp_start - 1:-1].float(), dim=-1).cpu()

        student_data.append({
            "problem": problem,
            "input_ids": input_ids,
            "response_ids": response_ids,
            "s_log_probs": s_log_probs,
        })

        if (pi + 1) % 10 == 0:
            print(f"  Generated {pi + 1}/{args.num_problems}")

    # Free student
    del student
    gc.collect()
    torch.cuda.empty_cache()

    # === Phase 2: Score with teacher ===
    print(f"\nPhase 2: Teacher scoring ({len(student_data)} responses)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    all_positions_kl = []
    all_positions_teacher_ent = []
    all_positions_agreement = []

    for di, data in enumerate(student_data):
        input_ids = data["input_ids"]
        response_ids = data["response_ids"]
        s_log_probs_full = data["s_log_probs"]

        if same_tokenizer:
            t_full_ids = input_ids + nothink_ids_teacher + response_ids
            t_resp_start = len(input_ids) + len(nothink_ids_teacher)
        else:
            response_text = student_tok.decode(response_ids, skip_special_tokens=False)
            t_prompt = build_prompt(data["problem"], teacher_tok)
            t_input_ids = teacher_tok.encode(t_prompt, add_special_tokens=False)
            t_response_ids = teacher_tok.encode(response_text, add_special_tokens=False)
            t_full_ids = t_input_ids + nothink_ids_teacher + t_response_ids
            t_resp_start = len(t_input_ids) + len(nothink_ids_teacher)

        t_tensor = torch.tensor([t_full_ids], device=device)
        with torch.no_grad():
            t_logits = teacher(t_tensor).logits[0]

        t_log_probs = log_softmax(t_logits[t_resp_start - 1:-1].float(), dim=-1).cpu()

        resp_len = min(len(response_ids), t_log_probs.shape[0], s_log_probs_full.shape[0])
        s_lp = s_log_probs_full[:resp_len]
        t_lp = t_log_probs[:resp_len]

        if same_tokenizer:
            vocab_size = min(s_lp.shape[-1], t_lp.shape[-1])
            s_lp = s_lp[:, :vocab_size]
            t_lp = t_lp[:, :vocab_size]
        else:
            # For cross-tokenizer, skip KL (would need vocab mapping)
            continue

        # Per-position reverse KL: KL(p_s || p_t)
        s_probs = torch.exp(s_lp)
        per_pos_kl = (s_probs * (s_lp - t_lp)).sum(dim=-1)

        # Teacher entropy
        t_probs = torch.exp(t_lp)
        per_pos_t_ent = -(t_probs * t_lp).sum(dim=-1)

        # Agreement (top-1)
        per_pos_agree = (s_lp.argmax(dim=-1) == t_lp.argmax(dim=-1)).float()

        for pos in range(resp_len):
            all_positions_kl.append((pos, per_pos_kl[pos].item()))
            all_positions_teacher_ent.append((pos, per_pos_t_ent[pos].item()))
            all_positions_agreement.append((pos, per_pos_agree[pos].item()))

        if (di + 1) % 10 == 0:
            print(f"  Scored {di + 1}/{len(student_data)}")

    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    # === Results ===
    print(f"\n{'='*60}")
    print(f"Pre-training KL Profile: {args.student_model} → {args.teacher_model}")
    print(f"{'='*60}")

    bins = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 500)]
    print(f"\n{'Positions':<15} {'Mean KL':<12} {'Teacher Ent':<14} {'Agreement':<12} {'N tokens':<10}")
    print("-" * 63)

    results = {}
    for lo, hi in bins:
        kls = [v for p, v in all_positions_kl if lo <= p < hi]
        ents = [v for p, v in all_positions_teacher_ent if lo <= p < hi]
        agrees = [v for p, v in all_positions_agreement if lo <= p < hi]
        if kls:
            mean_kl = np.mean(kls)
            mean_ent = np.mean(ents)
            mean_agree = np.mean(agrees)
            n = len(kls)
            print(f"{lo}-{hi:<11} {mean_kl:<12.4f} {mean_ent:<14.4f} {mean_agree:<12.3f} {n:<10}")
            results[f"{lo}-{hi}"] = {"kl": mean_kl, "entropy": mean_ent, "agreement": mean_agree, "n": n}

    # First 100 vs rest
    kl_first100 = [v for p, v in all_positions_kl if p < 100]
    kl_rest = [v for p, v in all_positions_kl if p >= 100]
    if kl_first100 and kl_rest:
        print(f"\nFirst 100 tokens: mean KL = {np.mean(kl_first100):.4f}")
        print(f"Tokens 100+:     mean KL = {np.mean(kl_rest):.4f}")
        ratio = np.mean(kl_first100) / max(np.mean(kl_rest), 1e-8)
        print(f"Ratio (first100 / rest): {ratio:.2f}x")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "student": args.student_model,
                "teacher": args.teacher_model,
                "num_problems": args.num_problems,
                "bins": results,
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

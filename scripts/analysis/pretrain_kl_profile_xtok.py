"""
Cross-tokenizer KL profile (proxy).

Same goal as pretrain_kl_profile.py but works when student and teacher use
different tokenizers. Computes per-position |s_lp - t_lp| at the sampled
token (same proxy as kl_x_entropy_buckets.py). Teacher logprob is taken
from the teacher token whose character span covers the student token's
character midpoint, after decoding the response to text and re-tokenizing
with the teacher tokenizer.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/analysis/pretrain_kl_profile_xtok.py \\
    --student_model HuggingFaceTB/SmolLM2-1.7B-Instruct \\
    --teacher_model HuggingFaceTB/SmolLM3-3B \\
    --num_problems 50 --max_new_tokens 512 \\
    --output docs/kl_profile_smollm.json
"""

import argparse, gc, json, os, sys
import numpy as np
import torch
from torch.nn.functional import log_softmax
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from on_policy_distill_positional import build_prompt, _supports_thinking


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_model", required=True)
    ap.add_argument("--teacher_model", required=True)
    ap.add_argument("--num_problems", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = "cuda:0"

    s_tok = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    t_tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    s_thinking = _supports_thinking(s_tok)
    t_thinking = _supports_thinking(t_tok)
    nothink_str = "<think>\n\n</think>\n\n"
    s_nothink_ids = s_tok.encode(nothink_str, add_special_tokens=False) if s_thinking else []
    t_nothink_ids = t_tok.encode(nothink_str, add_special_tokens=False) if t_thinking else []

    print(f"Student: {args.student_model} (thinking={s_thinking}) vocab={s_tok.vocab_size}")
    print(f"Teacher: {args.teacher_model} (thinking={t_thinking}) vocab={t_tok.vocab_size}")

    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    problems = [ds[i]["problem"] for i in range(args.num_problems)]

    # Phase 1: student generation + sampled-token logprob
    print(f"\n[Phase 1] Generating with student ({args.num_problems} problems)...")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    trajs = []
    for pi, problem in enumerate(problems):
        prompt = build_prompt(problem, s_tok)
        prompt_ids = s_tok.encode(prompt, add_special_tokens=False)
        inp = torch.tensor([prompt_ids + s_nothink_ids], device=device)
        with torch.no_grad():
            out = student.generate(
                inp,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                pad_token_id=s_tok.eos_token_id,
            )
        resp_ids = out[0][inp.shape[1]:].tolist()
        if len(resp_ids) < 5:
            continue

        full = prompt_ids + s_nothink_ids + resp_ids
        with torch.no_grad():
            s_logits = student(torch.tensor([full], device=device)).logits[0]
        # logit at index t predicts token at t+1; for resp token at index r in full,
        # logprob = s_logits[r-1, resp_token_id]
        resp_start = len(prompt_ids) + len(s_nothink_ids)
        s_lps = []
        for ri, tid in enumerate(resp_ids):
            shift = resp_start + ri - 1
            lp = log_softmax(s_logits[shift].float(), dim=-1)[tid].item()
            s_lps.append(lp)

        trajs.append({
            "problem": problem,
            "prompt_ids": prompt_ids,
            "response_ids": resp_ids,
            "s_lps": s_lps,
        })
        if (pi + 1) % 10 == 0:
            print(f"  {pi+1}/{args.num_problems}")

    del student
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 2: teacher scoring with re-tokenization
    print(f"\n[Phase 2] Scoring with teacher ({len(trajs)} trajs)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)

    pos_kl = []  # list of (position, |s_lp - t_lp|)
    for di, tr in enumerate(trajs):
        prompt_ids = tr["prompt_ids"]
        resp_ids = tr["response_ids"]
        s_lps = tr["s_lps"]

        # Decode full text
        full_text = s_tok.decode(prompt_ids + resp_ids, skip_special_tokens=False)
        prompt_text = s_tok.decode(prompt_ids, skip_special_tokens=False)
        resp_char_start = len(prompt_text)

        # Student per-token char offsets (need offset_mapping; use fast tokenizer encode)
        try:
            s_enc = s_tok(full_text, return_offsets_mapping=True, add_special_tokens=False)
            s_offsets = s_enc["offset_mapping"]
        except Exception:
            print(f"  [skip {di}] student tokenizer doesn't support offsets")
            continue

        # Re-tokenize with teacher
        try:
            t_enc = t_tok(full_text, return_offsets_mapping=True, add_special_tokens=False)
        except Exception:
            print(f"  [skip {di}] teacher tokenizer doesn't support offsets")
            continue
        t_ids = t_enc["input_ids"]
        t_offsets = t_enc["offset_mapping"]

        # Find teacher response start
        t_resp_start = 0
        for idx, (cs, ce) in enumerate(t_offsets):
            if cs >= resp_char_start:
                t_resp_start = idx
                break

        # Insert nothink for teacher
        t_full_ids = t_ids[:t_resp_start] + t_nothink_ids + t_ids[t_resp_start:]
        # Validate vocab range
        t_vocab = teacher.config.vocab_size
        t_full_ids = [min(tid, t_vocab - 1) for tid in t_full_ids]

        try:
            with torch.no_grad():
                t_logits = teacher(torch.tensor([t_full_ids], device=device)).logits[0]
            t_log_probs = log_softmax(t_logits[:-1].float(), dim=-1)
        except (RuntimeError, IndexError) as e:
            print(f"  [skip {di}] teacher fwd failed: {e}")
            continue

        # For each student response token, find the teacher token that covers
        # the same char midpoint, and get teacher's log-prob of THAT teacher
        # token at the corresponding position.
        s_resp_start = len(prompt_ids)
        for ri in range(min(len(resp_ids), len(s_lps))):
            si = s_resp_start + ri
            if si >= len(s_offsets):
                break
            s_cs, s_ce = s_offsets[si]
            mid = (s_cs + s_ce) // 2
            # find teacher token covering mid
            ti_match = None
            for ti in range(t_resp_start, len(t_offsets)):
                t_cs, t_ce = t_offsets[ti]
                if t_cs <= mid < t_ce:
                    ti_match = ti
                    break
                if t_cs > mid:
                    ti_match = max(ti - 1, t_resp_start)
                    break
            if ti_match is None:
                continue
            # In t_full_ids, the teacher token at position ti_match has been shifted
            # by len(t_nothink_ids) (since nothink was inserted at t_resp_start)
            t_full_pos = ti_match + len(t_nothink_ids)
            # logits[shift_idx] predicts token at shift_idx+1
            shift_idx = t_full_pos - 1
            if shift_idx < 0 or shift_idx >= t_log_probs.shape[0]:
                continue
            t_token_id = t_full_ids[t_full_pos]
            t_lp = t_log_probs[shift_idx, t_token_id].item()
            s_lp = s_lps[ri]
            pos_kl.append((ri, abs(s_lp - t_lp)))

        if (di + 1) % 10 == 0:
            print(f"  scored {di+1}/{len(trajs)}, total positions: {len(pos_kl)}")

    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    # Aggregate
    print(f"\n{'='*60}\nResults: {args.student_model} -> {args.teacher_model}\n{'='*60}")
    bins = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 400), (400, 500)]
    print(f"\n{'Range':<12} {'Mean |Δ|':<12} {'N':<8}")
    print("-" * 35)
    bin_results = {}
    for lo, hi in bins:
        vals = [v for p, v in pos_kl if lo <= p < hi]
        if vals:
            m = float(np.mean(vals))
            print(f"{lo}-{hi:<8} {m:<12.4f} {len(vals):<8}")
            bin_results[f"{lo}-{hi}"] = {"mean_kl": m, "n": len(vals)}

    # First-100 vs rest
    f100 = [v for p, v in pos_kl if p < 100]
    rest = [v for p, v in pos_kl if p >= 100]
    ratio = (np.mean(f100) / max(np.mean(rest), 1e-8)) if (f100 and rest) else None
    if ratio is not None:
        print(f"\nFirst 100: mean |Δ| = {np.mean(f100):.4f}")
        print(f"Rest:      mean |Δ| = {np.mean(rest):.4f}")
        print(f"Ratio (first100 / rest): {ratio:.2f}x")

    # Cumulative KL fraction by position (for picking N at 45% rule)
    max_pos = max((p for p, _ in pos_kl), default=0)
    cum_curve = []
    if pos_kl:
        # average per position, then cumulative sum
        per_pos_sum = {}
        per_pos_n = {}
        for p, v in pos_kl:
            per_pos_sum[p] = per_pos_sum.get(p, 0.0) + v
            per_pos_n[p] = per_pos_n.get(p, 0) + 1
        per_pos_mean = {p: per_pos_sum[p] / per_pos_n[p] for p in per_pos_sum}
        total = sum(per_pos_mean.values())
        running = 0.0
        for p in sorted(per_pos_mean.keys()):
            running += per_pos_mean[p]
            cum_curve.append((p, running / max(total, 1e-8)))
        # Find positions for various coverage thresholds
        thresholds = [0.30, 0.40, 0.45, 0.50, 0.60]
        print(f"\nCumulative KL coverage thresholds:")
        for thr in thresholds:
            for p, frac in cum_curve:
                if frac >= thr:
                    print(f"  {int(thr*100)}% reached at position {p}")
                    break

    out = {
        "student": args.student_model,
        "teacher": args.teacher_model,
        "num_problems": args.num_problems,
        "max_new_tokens": args.max_new_tokens,
        "n_positions": len(pos_kl),
        "bins": bin_results,
        "first100_mean": float(np.mean(f100)) if f100 else None,
        "rest_mean": float(np.mean(rest)) if rest else None,
        "first100_rest_ratio": ratio,
        "cumulative_kl": [(int(p), float(c)) for p, c in cum_curve],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()

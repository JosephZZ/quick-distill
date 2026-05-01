"""
Compute full-vocab student entropy H(p) per response position for the
first 100 tokens, across many trajectories. Used to decide a principled
threshold (or top-K) for the hi_ent variant of prefix-100 distillation.

Inputs:
  - signal_analysis/trajectories.json (problem_idx + response_ids per traj)
  - NuminaMath-CoT (re-fetch prompts by problem_idx)
  - Qwen/Qwen2.5-Math-1.5B (undistilled student)

Outputs (under docs/entropy_distribution_prefix/):
  - per_position_entropy.jsonl  : one line per (traj, pos) with H, token, kl_proxy
  - distribution_summary.json   : percentiles + per-position-range stats
  - histogram_prefix100.png     : H histogram in first 100 positions
  - histogram_compare.png       : prefix vs 100-300 vs 300+ comparison

Decisions:
  - If H is bimodal in prefix → use threshold at the valley.
  - If long-tailed → top-K (and pick K from the elbow).
  - If smooth/unimodal → quantile threshold, but pick the quantile from
    the percent of "format-like" (low-H) tokens that need to be dropped.
"""
import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(problem: str, tokenizer, system_prompt: str) -> list:
    """Mirror on_policy_distill_positional.build_prompt — returns prompt token ids."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors=None
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", default="docs/signal_analysis/trajectories.json")
    ap.add_argument("--out_dir", default="docs/entropy_distribution_prefix")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    ap.add_argument("--max_pos", type=int, default=100,
                    help="Capture H for first N response positions per traj")
    ap.add_argument("--also_capture_after", type=int, default=400,
                    help="Also capture up to this position for comparison "
                         "(prefix vs mid vs late). 0 to disable.")
    ap.add_argument("--n_traj", type=int, default=100,
                    help="Max number of trajectories to process")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--system_prompt",
                    default=("Please reason step by step, and put your final "
                             "answer within \\boxed{}."))
    ap.add_argument("--dataset", default="AI-MO/NuminaMath-CoT")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    capture_pos = max(args.max_pos, args.also_capture_after)

    # ---- load trajectories ----
    with open(args.trajectories) as f:
        trajs = json.load(f)
    if args.n_traj > 0:
        trajs = trajs[: args.n_traj]
    print(f"[load] {len(trajs)} trajectories")

    # ---- load tokenizer + model ----
    print(f"[load] tokenizer + model {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.device).eval()

    # ---- load dataset (used to look up problem text by problem_idx) ----
    print(f"[load] dataset {args.dataset}")
    ds = load_dataset(args.dataset, split="train")
    print(f"  dataset size = {len(ds)}")

    # ---- per-trajectory: build prompt, forward, compute H per position ----
    out_jsonl = out_dir / "per_position_entropy.jsonl"
    n_pos_total = 0
    with open(out_jsonl, "w") as fout:
        for ti, t in enumerate(trajs):
            problem_idx = t["problem_idx"]
            response_ids = t["response_ids"]
            if not response_ids:
                continue
            problem = ds[problem_idx]["problem"]
            prompt_ids = build_prompt(problem, tokenizer, args.system_prompt)

            # Build full input: prompt + response[:capture_pos]
            resp_take = response_ids[:capture_pos]
            full_ids = prompt_ids + resp_take
            input_ids = torch.tensor([full_ids], device=args.device)

            with torch.no_grad():
                logits = model(input_ids).logits[0]  # [T, V]
            # Logits at position p predict token at p+1.
            # The first response token corresponds to logits at index
            # len(prompt_ids) - 1 (the token right before response[0]).
            resp_start = len(prompt_ids) - 1
            resp_logits = logits[resp_start : resp_start + len(resp_take)]
            # Compute H(p) = -sum p log p, in float32 for numerical safety
            log_probs = F.log_softmax(resp_logits.float(), dim=-1)
            probs = log_probs.exp()
            ent = -(probs * log_probs).sum(dim=-1)  # [resp_len]

            sampled_lp = log_probs.gather(
                -1, torch.tensor(resp_take, device=args.device).unsqueeze(-1)
            ).squeeze(-1)
            surprise = (-sampled_lp).cpu().tolist()
            ent = ent.cpu().tolist()
            for p, (h, sp) in enumerate(zip(ent, surprise)):
                tok_id = resp_take[p]
                tok_str = tokenizer.decode([tok_id])
                rec = {
                    "traj_idx": ti, "problem_idx": problem_idx, "pos": p,
                    "H": float(h), "surp": float(sp),
                    "tok_id": int(tok_id), "tok_str": tok_str,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_pos_total += 1
            if (ti + 1) % 20 == 0:
                print(f"  [{ti+1}/{len(trajs)}] total_positions={n_pos_total}")
    print(f"[save] {out_jsonl}  ({n_pos_total} (traj,pos) records)")

    # ---- summary ----
    H_by_range = {"0-49": [], "50-99": [], "100-199": [], "200-399": []}
    H_all_prefix = []
    surp_prefix = []
    H_records = []
    with open(out_jsonl) as f:
        for line in f:
            r = json.loads(line)
            H_records.append(r)
            p = r["pos"]
            if p < 50:   H_by_range["0-49"].append(r["H"])
            elif p < 100: H_by_range["50-99"].append(r["H"])
            elif p < 200: H_by_range["100-199"].append(r["H"])
            elif p < 400: H_by_range["200-399"].append(r["H"])
            if p < 100:
                H_all_prefix.append(r["H"])
                surp_prefix.append(r["surp"])

    summary = {"per_range": {}, "prefix_percentiles": {}, "n_traj": len(trajs)}
    for rng, vals in H_by_range.items():
        if not vals:
            continue
        a = np.array(vals)
        summary["per_range"][rng] = {
            "n": len(a),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "p10": float(np.percentile(a, 10)),
            "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)),
        }
    if H_all_prefix:
        a = np.array(H_all_prefix)
        for q in [5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95]:
            summary["prefix_percentiles"][f"p{q}"] = float(np.percentile(a, q))
        summary["prefix_corr_H_surp"] = float(np.corrcoef(a, np.array(surp_prefix))[0, 1])
    with open(out_dir / "distribution_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_dir/'distribution_summary.json'}")
    print(json.dumps(summary, indent=2))

    # ---- plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Histogram in prefix-100
        fig, ax = plt.subplots(figsize=(8, 4))
        if H_all_prefix:
            ax.hist(H_all_prefix, bins=80)
            for q, name in [(25, "p25"), (50, "p50"), (75, "p75")]:
                v = np.percentile(H_all_prefix, q)
                ax.axvline(v, color="r", ls="--", lw=0.8)
                ax.text(v, ax.get_ylim()[1] * 0.95, name, color="r", ha="center")
        ax.set_xlabel("Student full-vocab entropy H(p)")
        ax.set_ylabel("count")
        ax.set_title(f"Entropy distribution, prefix [0,{args.max_pos})  "
                     f"n={len(H_all_prefix)}")
        fig.tight_layout()
        fig.savefig(out_dir / "histogram_prefix100.png", dpi=140)
        plt.close(fig)
        print(f"[save] {out_dir/'histogram_prefix100.png'}")

        # Compare prefix vs mid vs late
        fig, ax = plt.subplots(figsize=(8, 4))
        for rng, vals in H_by_range.items():
            if vals:
                ax.hist(vals, bins=60, density=True, histtype="step",
                        label=f"{rng}  n={len(vals)}")
        ax.legend()
        ax.set_xlabel("Student full-vocab entropy H(p)")
        ax.set_ylabel("density")
        ax.set_title("Entropy distribution by position range")
        fig.tight_layout()
        fig.savefig(out_dir / "histogram_compare.png", dpi=140)
        plt.close(fig)
        print(f"[save] {out_dir/'histogram_compare.png'}")
    except Exception as e:
        print("[plot] skipped:", e)

    # ---- per-position-range token breakdown of low-H tail ----
    # what tokens fall in the bottom-50% / bottom-25% of entropy in prefix
    if H_all_prefix:
        prefix = [r for r in H_records if r["pos"] < args.max_pos]
        H_arr = np.array([r["H"] for r in prefix])
        for q_drop in [25, 50]:
            thr = float(np.percentile(H_arr, q_drop))
            below = [r for r in prefix if r["H"] <= thr]
            cnt = {}
            for r in below:
                cnt[r["tok_str"]] = cnt.get(r["tok_str"], 0) + 1
            top = sorted(cnt.items(), key=lambda x: -x[1])[:25]
            print(f"\n=== Bottom-{q_drop}% by H within prefix-{args.max_pos} "
                  f"(n={len(below)}, H<={thr:.3f}) — top 25 tokens ===")
            for t, c in top:
                print(f"  {t!r:25s}  {c}")
            with open(out_dir / f"low_H_bottom{q_drop}_tokens.json", "w") as f:
                json.dump(
                    {"threshold_H": thr, "n": len(below), "top_tokens": top},
                    f, indent=2, ensure_ascii=False,
                )

    print("\n[done] All outputs in:", out_dir)


if __name__ == "__main__":
    main()

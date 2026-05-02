#!/usr/bin/env python3
"""
After running KL profiles for InternLM, MiniCPM, (optional) SmolLM, decide which
(student, teacher) pair to train and emit a config JSON.

Selection rules (from user):
  1. Pair must have early-position KL ratio (first100/rest) >= ~2.
  2. If multiple qualify, prefer pair with smaller student model.
  3. If none of {InternLM, MiniCPM} qualify, fall back to SmolLM (largest KL).

Outputs JSON with: {student, teacher, prefix_n_45pct, qualifies, ratio}
where prefix_n_45pct = position p such that cumulative KL fraction first crosses 45%.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path("/zhi_backup/ziheng/quick-distillation/quick-distillation")
KLD = ROOT / "docs/kl_profile_xfamily"

CANDIDATES = [
    {
        "name": "internlm2.5-1.8B_internlm3-8B",
        "student": "internlm/internlm2_5-1_8b-chat",
        "teacher": "internlm/internlm3-8b-instruct",
        "student_size": 1.8,
        "file": KLD / "internlm2.5-1.8B_internlm3-8B.json",
    },
    {
        "name": "minicpm3-4B_minicpm4-8B",
        "student": "openbmb/MiniCPM3-4B",
        "teacher": "openbmb/MiniCPM4-8B",
        "student_size": 4.0,
        "file": KLD / "minicpm3-4B_minicpm4-8B.json",
    },
    {
        "name": "smollm2-1.7B_smollm3-3B",
        "student": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "teacher": "HuggingFaceTB/SmolLM3-3B",
        "student_size": 1.7,
        "file": KLD / "smollm2-1.7B_smollm3-3B.json",
    },
]

THRESHOLD = 2.0
TARGET_FRAC = 0.45


def load_kl(p):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"  [warn] cannot parse {p}: {e}")
        return None


def get_ratio(d):
    """first100 / rest ratio. Either from xtok schema or kl_profile_vllm schema."""
    r = d.get("first100_rest_ratio")
    if r is not None:
        return float(r)
    # vllm schema: per_position_kl_proxy + valid_count_per_position
    pp = d.get("per_position_kl_proxy")
    cnt = d.get("valid_count_per_position")
    if pp and cnt:
        f100_num = sum(p * c for p, c in zip(pp[:100], cnt[:100]))
        f100_den = sum(cnt[:100]) or 1
        rest_num = sum(p * c for p, c in zip(pp[100:], cnt[100:]))
        rest_den = sum(cnt[100:]) or 1
        return (f100_num / f100_den) / max(rest_num / rest_den, 1e-8)
    return None


def get_prefix_n_45pct(d):
    """Position where cumulative KL fraction first crosses TARGET_FRAC."""
    cc = d.get("cumulative_kl")  # xtok: list of [pos, frac]
    if cc:
        for pos, frac in cc:
            if frac >= TARGET_FRAC:
                return int(pos)
        return int(cc[-1][0])
    cf = d.get("cumulative_kl_fraction")  # vllm: dict {K: frac}
    if cf:
        items = sorted(((int(k), v) for k, v in cf.items()), key=lambda x: x[0])
        for K, frac in items:
            if frac >= TARGET_FRAC:
                return K
        return items[-1][0]
    return None


def main():
    print("=== KL profile pair selection ===")
    qualifying = []
    fallback = []
    for c in CANDIDATES:
        d = load_kl(c["file"])
        if d is None:
            print(f"  {c['name']:40s}  no KL file")
            continue
        r = get_ratio(d)
        n = get_prefix_n_45pct(d)
        c["ratio"] = r
        c["prefix_n_45pct"] = n
        c["data"] = d
        if r is None:
            print(f"  {c['name']:40s}  ratio=N/A  N={n}")
            continue
        print(f"  {c['name']:40s}  ratio={r:.2f}x  prefix_45%={n}")
        if r >= THRESHOLD * 0.95:  # "close to 2 or above"
            qualifying.append(c)
        else:
            fallback.append(c)

    selected = None
    if qualifying:
        # Prefer non-SmolLM; among ties, smallest student
        non_smol = [c for c in qualifying if not c["name"].startswith("smol")]
        pool = non_smol if non_smol else qualifying
        pool.sort(key=lambda c: c["student_size"])
        selected = pool[0]
        print(f"\nSelected (qualifies): {selected['name']} (ratio={selected['ratio']:.2f}, N={selected['prefix_n_45pct']})")
    elif fallback:
        # No pair qualifies — use SmolLM if available, else best by ratio
        smol = [c for c in fallback if c["name"].startswith("smol")]
        if smol:
            selected = smol[0]
        else:
            fallback.sort(key=lambda c: -(c["ratio"] or 0))
            selected = fallback[0]
        print(f"\nNo pair >= {THRESHOLD}x, fallback: {selected['name']} (ratio={selected['ratio']}, N={selected['prefix_n_45pct']})")
    else:
        print("\nNo KL profile data available; cannot select.")
        sys.exit(1)

    # Clamp prefix N to a reasonable range
    n_45 = selected["prefix_n_45pct"] or 200
    n_45 = max(50, min(n_45, 400))
    out = {
        "name": selected["name"],
        "student": selected["student"],
        "teacher": selected["teacher"],
        "student_size": selected["student_size"],
        "prefix_n_45pct": n_45,
        "ratio": selected["ratio"],
        "qualifies": (selected["ratio"] or 0) >= THRESHOLD * 0.95,
    }
    out_path = ROOT / "docs/kl_profile_xfamily/_selected_pair.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

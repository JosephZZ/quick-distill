#!/usr/bin/env python3
"""Pick the (student, teacher) pair with the LARGEST first100/rest KL ratio
among InternLM, MiniCPM, and SmolLM (if a profile exists for it).
Outputs JSON with the choice + 45% KL coverage prefix length."""
import json, sys
from pathlib import Path

ROOT = Path("/zhi_backup/ziheng/quick-distillation/quick-distillation")
KLD = ROOT / "docs/kl_profile_xfamily"

CANDIDATES = [
    ("internlm2.5-1.8B_internlm3-8B", "internlm/internlm2_5-1_8b-chat", "internlm/internlm3-8b-instruct", 1.8),
    ("minicpm3-4B_minicpm4-8B", "openbmb/MiniCPM3-4B", "openbmb/MiniCPM4-8B", 4.0),
    ("smollm2-1.7B_smollm3-3B", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM3-3B", 1.7),
]

def get_ratio(d):
    r = d.get("first100_rest_ratio")
    if r is not None: return float(r)
    pp = d.get("per_position_kl_proxy"); cnt = d.get("valid_count_per_position")
    if pp and cnt:
        f100n = sum(p*c for p,c in zip(pp[:100], cnt[:100])); f100d = sum(cnt[:100]) or 1
        restn = sum(p*c for p,c in zip(pp[100:], cnt[100:])); restd = sum(cnt[100:]) or 1
        return (f100n/f100d) / max(restn/restd, 1e-8)
    return None

def get_prefix_n(d, frac=0.45):
    cc = d.get("cumulative_kl")
    if cc:
        for pos, f in cc:
            if f >= frac: return int(pos)
        return int(cc[-1][0])
    cf = d.get("cumulative_kl_fraction")
    if cf:
        items = sorted(((int(k), v) for k,v in cf.items()), key=lambda x: x[0])
        for K, f in items:
            if f >= frac: return K
        return items[-1][0]
    return None

print("=== KL profile pair selection (largest ratio) ===")
results = []
for name, student, teacher, ssize in CANDIDATES:
    p = KLD / f"{name}.json"
    if not p.exists():
        print(f"  {name:40s}  no file ({p.name})"); continue
    d = json.loads(p.read_text())
    r = get_ratio(d); n = get_prefix_n(d)
    print(f"  {name:40s}  ratio={r if r is None else f'{r:.2f}'}x  prefix_45%={n}")
    if r is not None:
        results.append((name, student, teacher, ssize, r, n))

if not results:
    print("No KL data"); sys.exit(1)

results.sort(key=lambda x: -x[4])  # largest ratio first
name, student, teacher, ssize, ratio, n = results[0]
n_clamped = max(50, min(n or 200, 400))
out = {
    "name": name, "student": student, "teacher": teacher,
    "student_size": ssize, "prefix_n": n_clamped, "ratio": ratio,
}
out_path = KLD / "_selected_pair.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nSelected: {name} (ratio={ratio:.2f}x, N={n_clamped})")
print(f"Wrote {out_path}")

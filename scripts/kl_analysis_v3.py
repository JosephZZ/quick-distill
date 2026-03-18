"""
KL analysis v3: Compare student-teacher KL by position for multiple model checkpoints.
- Reuse existing per-token logprobs for step-50 models
- Generate + score new data for step-200 models
- Ranges: 0-50, 50-100, 100-150, 150-200, 200-300, ..., 900-1000
- Use vLLM for generation
- Plot per-position curves + save range tables
"""
import json, os, sys, time, torch, gc, subprocess, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DEVICE = "cuda:0"
TEACHER_MODEL = "Qwen/Qwen3-1.7B"
STUDENT_BASE = "Qwen/Qwen2.5-Math-1.5B"
N_PROBLEMS = 100
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
VLLM_GPU_UTIL = 0.85

OUTPUT_DIR = "docs/kl_position_analysis_v3"
V2_DIR = "docs/kl_position_analysis_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANGES = [
    (0, 50), (50, 100), (100, 150), (150, 200),
    (200, 300), (300, 400), (400, 500), (500, 600),
    (600, 700), (700, 800), (800, 900), (900, 1000),
]


def load_problems(n=100):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [item["problem"] for item in ds][:n]


def load_saved_logprobs(path):
    """Load per-token logprobs from jsonl."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_merged_model(base_name, lora_path, save_path):
    print(f"  Merging LoRA: {lora_path} -> {save_path}")
    model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    model.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    tokenizer.save_pretrained(save_path)
    del model
    gc.collect()


def generate_with_vllm(model_path, tokenizer_name, problems):
    prompt_file = "/tmp/kl_v3_prompts.json"
    with open(prompt_file, "w") as f:
        json.dump(problems, f)

    output_file = "/tmp/kl_v3_trajs.json"
    cmd = [
        sys.executable, "vllm_generate.py",
        "--model", model_path, "--tokenizer", tokenizer_name,
        "--problems_file", prompt_file, "--output_file", output_file,
        "--n_samples", "1", "--max_new_tokens", str(MAX_NEW_TOKENS),
        "--temperature", str(TEMPERATURE), "--gpu_memory_utilization", str(VLLM_GPU_UTIL),
        "--seed", "42",
    ]
    print(f"  Running vLLM generation...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    gen_time = time.time() - t0

    if result.returncode != 0:
        print(f"  vLLM FAILED: {result.stderr[-500:]}")
        return None, gen_time

    with open(output_file) as f:
        data = json.load(f)

    trajs = []
    for k in sorted(data.keys(), key=lambda x: int(x)):
        if data[k]:
            trajs.append(data[k][0])  # first (only) trajectory

    os.remove(prompt_file)
    os.remove(output_file)
    return trajs, gen_time


def score_with_model(model, trajs, device):
    all_lps = []
    for i, traj in enumerate(trajs):
        prompt_ids = traj["prompt_ids"]
        response_ids = traj["response_ids"]
        input_ids = torch.tensor([prompt_ids + response_ids], device=device)
        prompt_len = len(prompt_ids)

        with torch.no_grad():
            logits = model(input_ids).logits[0]
        response_logits = logits[prompt_len - 1: prompt_len - 1 + len(response_ids)]
        log_probs = torch.log_softmax(response_logits.float(), dim=-1)

        lps = [log_probs[t, tid].item() for t, tid in enumerate(response_ids)]
        all_lps.append(lps)

        if (i + 1) % 25 == 0:
            print(f"    Scored {i+1}/{len(trajs)}")
    return all_lps


def compute_ranges(student_lps_list, teacher_lps_list):
    """Compute mean |diff| for each range."""
    result = {}
    for start, end in RANGES:
        diffs = []
        for s_lps, t_lps in zip(student_lps_list, teacher_lps_list):
            seq_len = min(len(s_lps), len(t_lps))
            for pos in range(start, min(end, seq_len)):
                diffs.append(abs(s_lps[pos] - t_lps[pos]))
        if diffs:
            result[f"{start}-{end}"] = {"mean": float(np.mean(diffs)), "count": len(diffs)}
    return result


def compute_per_position(student_lps_list, teacher_lps_list, max_pos=1000):
    """Compute mean |diff| at each position."""
    kl_sums = np.zeros(max_pos)
    counts = np.zeros(max_pos)
    for s_lps, t_lps in zip(student_lps_list, teacher_lps_list):
        seq_len = min(len(s_lps), len(t_lps), max_pos)
        for pos in range(seq_len):
            kl_sums[pos] += abs(s_lps[pos] - t_lps[pos])
            counts[pos] += 1
    mean_kl = np.where(counts > 5, kl_sums / counts, np.nan)
    return mean_kl, counts


def main():
    print("=" * 60)
    print("KL Position Analysis v3")
    print("=" * 60)

    problems = load_problems(N_PROBLEMS)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE, trust_remote_code=True)

    # All models to analyze
    all_models = {}

    # --- Phase 1: Load existing step-50 data ---
    print("\n=== Phase 1: Load existing step-50 per-token data ===")
    for name in ["raw", "pos-200tok", "full-seq-50"]:
        path = os.path.join(V2_DIR, f"{name}_logprobs.jsonl")
        if os.path.exists(path):
            data = load_saved_logprobs(path)
            s_lps = [d["student_lps"] for d in data]
            t_lps = [d["teacher_lps"] for d in data]
            all_models[name] = {"student_lps": s_lps, "teacher_lps": t_lps}
            avg_len = np.mean([len(d["response_ids"]) for d in data])
            print(f"  Loaded {name}: {len(data)} trajs, avg len {avg_len:.0f}")
        else:
            print(f"  WARNING: {path} not found, skipping {name}")

    # --- Phase 2: Generate + score step-200 models ---
    print("\n=== Phase 2: Generate step-200 trajectories (vLLM) ===")

    step200_models = [
        ("pos-200tok-s200", "checkpoints/pos-limit-200tok/step_200"),
        ("full-seq-s200", "checkpoints/full-seq-3584tok/step_200"),
    ]

    step200_trajs = {}
    for model_name, lora_path in step200_models:
        print(f"\n--- {model_name} ---")
        merged_path = f"/tmp/kl_merged_{model_name}"
        save_merged_model(STUDENT_BASE, lora_path, merged_path)
        trajs, gen_time = generate_with_vllm(merged_path, STUDENT_BASE, problems)
        shutil.rmtree(merged_path, ignore_errors=True)

        if trajs is None:
            continue
        avg_len = np.mean([len(t["response_ids"]) for t in trajs])
        print(f"  Generated {len(trajs)} trajs in {gen_time:.0f}s, avg len {avg_len:.0f}")
        step200_trajs[model_name] = trajs

    # Score step-200 with student models
    print("\n=== Phase 3: Score step-200 with student models ===")
    for model_name, lora_path in step200_models:
        if model_name not in step200_trajs:
            continue
        print(f"\n--- Student scoring: {model_name} ---")
        student = AutoModelForCausalLM.from_pretrained(STUDENT_BASE, torch_dtype=torch.bfloat16, trust_remote_code=True)
        student = PeftModel.from_pretrained(student, lora_path)
        student = student.merge_and_unload().to(DEVICE)
        student.eval()

        s_lps = score_with_model(student, step200_trajs[model_name], DEVICE)
        del student; gc.collect(); torch.cuda.empty_cache()

        all_models[model_name] = {"student_lps": s_lps, "teacher_lps": None, "trajs": step200_trajs[model_name]}

    # Score step-200 with teacher
    print("\n=== Phase 4: Score step-200 with teacher ===")
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).to(DEVICE)
    teacher.eval()

    for model_name in step200_trajs:
        print(f"\n--- Teacher scoring: {model_name} ---")
        t_lps = score_with_model(teacher, step200_trajs[model_name], DEVICE)
        all_models[model_name]["teacher_lps"] = t_lps

    del teacher; gc.collect(); torch.cuda.empty_cache()

    # Save step-200 per-token data
    print("\n=== Phase 5: Save per-token data ===")
    for model_name in step200_trajs:
        save_data = []
        for i in range(len(step200_trajs[model_name])):
            save_data.append({
                "response_ids": step200_trajs[model_name][i]["response_ids"],
                "student_lps": all_models[model_name]["student_lps"][i],
                "teacher_lps": all_models[model_name]["teacher_lps"][i],
            })
        save_path = os.path.join(OUTPUT_DIR, f"{model_name}_logprobs.jsonl")
        with open(save_path, "w") as f:
            for d in save_data:
                f.write(json.dumps(d) + "\n")
        print(f"  Saved {model_name} to {save_path}")

    # === Phase 6: Analysis ===
    print("\n=== Phase 6: Analysis ===")

    # Compute per-position and range stats for all models
    model_names_ordered = ["raw", "pos-200tok", "pos-200tok-s200", "full-seq-50", "full-seq-s200"]
    model_labels = {
        "raw": "Raw (no distill)",
        "pos-200tok": "Pos-200tok (step 100)",
        "pos-200tok-s200": "Pos-200tok (step 200)",
        "full-seq-50": "Full-seq (step 50)",
        "full-seq-s200": "Full-seq (step 200)",
    }
    colors = {
        "raw": "#e74c3c",
        "pos-200tok": "#3498db",
        "pos-200tok-s200": "#1a5276",
        "full-seq-50": "#2ecc71",
        "full-seq-s200": "#145a32",
    }

    per_pos_data = {}
    range_data = {}

    for name in model_names_ordered:
        if name not in all_models:
            continue
        m = all_models[name]
        per_pos, counts = compute_per_position(m["student_lps"], m["teacher_lps"])
        per_pos_data[name] = (per_pos, counts)
        range_data[name] = compute_ranges(m["student_lps"], m["teacher_lps"])

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # Left: full curve smoothed
    ax = axes[0]
    window = 15
    for name in model_names_ordered:
        if name not in per_pos_data:
            continue
        kl, counts = per_pos_data[name]
        smoothed = np.full_like(kl, np.nan)
        for i in range(len(kl)):
            s, e = max(0, i - window // 2), min(len(kl), i + window // 2 + 1)
            vals = kl[s:e]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                smoothed[i] = np.mean(vals)
        x = np.arange(len(smoothed))
        mask = ~np.isnan(smoothed) & (counts > 5)
        ax.plot(x[mask], smoothed[mask], color=colors[name], label=model_labels[name], linewidth=1.5)

    ax.axvline(x=200, color="gray", linestyle="--", alpha=0.5, label="Position 200")
    ax.set_xlabel("Token Position", fontsize=12)
    ax.set_ylabel("Mean |log p_s - log p_t|", fontsize=12)
    ax.set_title("Per-Position KL (smoothed, window=15)", fontsize=13)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 700)
    ax.grid(True, alpha=0.3)

    # Middle: first 50 tokens unsmoothed
    ax = axes[1]
    for name in model_names_ordered:
        if name not in per_pos_data:
            continue
        kl, counts = per_pos_data[name]
        x = np.arange(min(50, len(kl)))
        vals = kl[:50]
        mask = ~np.isnan(vals)
        ax.plot(x[mask], vals[mask], color=colors[name], label=model_labels[name], linewidth=1.5, marker="o", markersize=2)

    ax.set_xlabel("Token Position", fontsize=12)
    ax.set_ylabel("Mean |log p_s - log p_t|", fontsize=12)
    ax.set_title("First 50 Tokens (unsmoothed)", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: bar chart by range
    ax = axes[2]
    range_labels = [f"{s}-{e}" for s, e in RANGES if e <= 700]
    x_pos = np.arange(len(range_labels))
    width = 0.15
    offsets = np.linspace(-2 * width, 2 * width, len(model_names_ordered))

    for idx, name in enumerate(model_names_ordered):
        if name not in range_data:
            continue
        vals = []
        for rl in range_labels:
            if rl in range_data[name]:
                vals.append(range_data[name][rl]["mean"])
            else:
                vals.append(0)
        ax.bar(x_pos + offsets[idx], vals, width, label=model_labels[name], color=colors[name], alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(range_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean |log p_s - log p_t|", fontsize=12)
    ax.set_title("KL by Position Range", fontsize=13)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "kl_comparison_all.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved plot to {fig_path}")
    plt.close()

    # --- Print range table ---
    print("\n" + "=" * 100)
    print("Mean |student_lp - teacher_lp| by position range")
    print("=" * 100)

    range_strs = [f"{s}-{e}" for s, e in RANGES]
    header = f"{'Model':>22} | " + " | ".join(f"{r:>8}" for r in range_strs)
    print(header)
    print("-" * len(header))

    for name in model_names_ordered:
        if name not in range_data:
            continue
        vals = []
        for r in range_strs:
            if r in range_data[name]:
                vals.append(f"{range_data[name][r]['mean']:>8.3f}")
            else:
                vals.append(f"{'—':>8}")
        print(f"{model_labels[name]:>22} | " + " | ".join(vals))

    # Save results
    results = {}
    for name in model_names_ordered:
        if name not in range_data:
            continue
        results[name] = {
            "ranges": range_data[name],
            "label": model_labels[name],
        }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

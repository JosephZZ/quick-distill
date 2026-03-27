"""
Compare KL divergence by position between teacher and different student models.
Models: raw (no distill), pos-200tok (step 100), full-seq (step 50)
Teacher: Qwen3-1.7B
Generate 100 problems × 1 trajectory with vLLM, score with teacher HF forward.
Save per-token log probs + per-position KL curve + 100-range table.
"""
import json, os, sys, time, torch, gc, subprocess, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import save_file

DEVICE = "cuda:0"
TEACHER_MODEL = "Qwen/Qwen3-1.7B"
STUDENT_BASE = "Qwen/Qwen2.5-Math-1.5B"
N_PROBLEMS = 100
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
VLLM_GPU_UTIL = 0.85

OUTPUT_DIR = "docs/kl_position_analysis_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_problems(n=100):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [item["problem"] for item in ds][:n]


def build_prompt(problem, tokenizer):
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def save_merged_model(base_name, lora_path, save_path):
    """Merge LoRA into base and save for vLLM."""
    print(f"  Merging LoRA: {lora_path} -> {save_path}")
    model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    model.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    tokenizer.save_pretrained(save_path)
    del model
    gc.collect()
    return save_path


def generate_with_vllm(model_path, tokenizer_name, problems, tokenizer):
    """Generate trajectories using vLLM subprocess."""
    # Write problems file (vllm_generate.py expects raw problem strings)
    prompt_file = "/tmp/kl_analysis_prompts.json"
    with open(prompt_file, "w") as f:
        json.dump(problems, f)

    output_file = "/tmp/kl_analysis_trajs.json"

    cmd = [
        sys.executable, "vllm_generate.py",
        "--model", model_path,
        "--tokenizer", tokenizer_name,
        "--problems_file", prompt_file,
        "--output_file", output_file,
        "--n_samples", "1",
        "--max_new_tokens", str(MAX_NEW_TOKENS),
        "--temperature", str(TEMPERATURE),
        "--gpu_memory_utilization", str(VLLM_GPU_UTIL),
        "--seed", "42",
    ]

    env = os.environ.copy()
    print(f"  Running vLLM generation...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    gen_time = time.time() - t0

    if result.returncode != 0:
        print(f"  vLLM FAILED: {result.stderr[-500:]}")
        return None, gen_time

    # Parse output - vllm_generate.py outputs json with problem_key -> [trajs]
    with open(output_file) as f:
        data = json.load(f)

    trajs = []
    for prob_key in sorted(data.keys(), key=lambda x: int(x.split("_")[-1]) if "_" in x else 0):
        traj_list = data[prob_key]
        if traj_list:
            t = traj_list[0]
            trajs.append({
                "prompt_ids": t["prompt_ids"],
                "response_ids": t["response_ids"],
            })

    os.remove(prompt_file)
    os.remove(output_file)

    return trajs, gen_time


def score_with_model(model, trajs, device):
    """Get log probs for each trajectory's response tokens."""
    all_lps = []
    for i, traj in enumerate(trajs):
        prompt_ids = traj["prompt_ids"]
        response_ids = traj["response_ids"]
        full_ids = prompt_ids + response_ids
        input_ids = torch.tensor([full_ids], device=device)
        prompt_len = len(prompt_ids)

        with torch.no_grad():
            logits = model(input_ids).logits[0]

        response_logits = logits[prompt_len - 1: prompt_len - 1 + len(response_ids)]
        log_probs = torch.log_softmax(response_logits.float(), dim=-1)

        lps = []
        for t, token_id in enumerate(response_ids):
            lps.append(log_probs[t, token_id].item())

        all_lps.append(lps)
        if (i + 1) % 25 == 0:
            print(f"    Scored {i+1}/{len(trajs)}")

    return all_lps


def main():
    print("=" * 60)
    print("KL Position Analysis v2 (vLLM + per-token saving)")
    print("=" * 60)

    problems = load_problems(N_PROBLEMS)
    print(f"Loaded {len(problems)} problems")

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE, trust_remote_code=True)

    models_config = [
        ("raw", None),
        ("pos-200tok", "checkpoints/pos-limit-200tok/step_100"),
        ("full-seq-50", "checkpoints/full-seq-3584tok/step_50"),
    ]

    all_data = {}  # model_name -> list of {student_lps, teacher_lps, response_ids}

    # === Phase 1: Generate trajectories with vLLM (no teacher needed) ===
    print("\n=== Phase 1: Generation with vLLM ===")
    all_trajs = {}

    for model_name, lora_path in models_config:
        print(f"\n--- Generating for: {model_name} ---")

        if lora_path is None:
            model_path = STUDENT_BASE
        else:
            merged_path = f"/tmp/kl_merged_{model_name}"
            model_path = save_merged_model(STUDENT_BASE, lora_path, merged_path)

        trajs, gen_time = generate_with_vllm(model_path, STUDENT_BASE, problems, tokenizer)

        if trajs is None:
            print(f"  FAILED for {model_name}, skipping")
            continue

        avg_len = np.mean([len(t["response_ids"]) for t in trajs])
        print(f"  Generated {len(trajs)} trajs in {gen_time:.0f}s, avg len: {avg_len:.0f}")
        all_trajs[model_name] = trajs

        # Cleanup merged model
        if lora_path is not None:
            shutil.rmtree(model_path, ignore_errors=True)

    # === Phase 2: Score all trajectories with student models ===
    print("\n=== Phase 2: Score with student models ===")

    all_student_lps = {}

    for model_name, lora_path in models_config:
        if model_name not in all_trajs:
            continue
        print(f"\n--- Scoring with student: {model_name} ---")

        if lora_path is None:
            student = AutoModelForCausalLM.from_pretrained(
                STUDENT_BASE, torch_dtype=torch.bfloat16, trust_remote_code=True
            ).to(DEVICE)
        else:
            student = AutoModelForCausalLM.from_pretrained(
                STUDENT_BASE, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            student = PeftModel.from_pretrained(student, lora_path)
            student = student.merge_and_unload()
            student = student.to(DEVICE)

        student.eval()
        lps = score_with_model(student, all_trajs[model_name], DEVICE)
        all_student_lps[model_name] = lps

        del student
        gc.collect()
        torch.cuda.empty_cache()

    # === Phase 3: Score all trajectories with teacher ===
    print("\n=== Phase 3: Score with teacher ===")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(DEVICE)
    teacher.eval()

    all_teacher_lps = {}
    for model_name in all_trajs:
        print(f"\n--- Teacher scoring: {model_name} ---")
        lps = score_with_model(teacher, all_trajs[model_name], DEVICE)
        all_teacher_lps[model_name] = lps

    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    # === Phase 4: Save per-token data ===
    print("\n=== Phase 4: Save per-token data ===")
    for model_name in all_trajs:
        save_data = []
        for i in range(len(all_trajs[model_name])):
            save_data.append({
                "response_ids": all_trajs[model_name][i]["response_ids"],
                "student_lps": all_student_lps[model_name][i],
                "teacher_lps": all_teacher_lps[model_name][i],
            })
        save_path = os.path.join(OUTPUT_DIR, f"{model_name}_logprobs.jsonl")
        with open(save_path, "w") as f:
            for d in save_data:
                f.write(json.dumps(d) + "\n")
        print(f"  Saved {len(save_data)} trajectories to {save_path}")

    # === Phase 5: Analyze ===
    print("\n=== Phase 5: Analysis ===")

    # Per-position KL for each model
    max_pos = 800
    kl_per_pos = {}  # model_name -> array of mean KL at each position
    count_per_pos = {}

    for model_name in all_trajs:
        kl_sums = np.zeros(max_pos)
        counts = np.zeros(max_pos)
        for s_lps, t_lps in zip(all_student_lps[model_name], all_teacher_lps[model_name]):
            seq_len = min(len(s_lps), len(t_lps), max_pos)
            for pos in range(seq_len):
                kl_sums[pos] += abs(s_lps[pos] - t_lps[pos])
                counts[pos] += 1
        # Mean KL (only where count > 10)
        mean_kl = np.where(counts > 10, kl_sums / counts, np.nan)
        kl_per_pos[model_name] = mean_kl
        count_per_pos[model_name] = counts

    # --- Plot: per-position KL curve (smoothed) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = {"raw": "#e74c3c", "pos-200tok": "#3498db", "full-seq-50": "#2ecc71"}
    labels = {"raw": "Raw (no distill)", "pos-200tok": "Pos-200tok (step 100)", "full-seq-50": "Full-seq (step 50)"}

    # Smoothed curve (window=10)
    window = 10
    for model_name in ["raw", "pos-200tok", "full-seq-50"]:
        if model_name not in kl_per_pos:
            continue
        kl = kl_per_pos[model_name]
        # Smooth
        valid = ~np.isnan(kl)
        smoothed = np.full_like(kl, np.nan)
        for i in range(len(kl)):
            start = max(0, i - window // 2)
            end = min(len(kl), i + window // 2 + 1)
            vals = kl[start:end]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                smoothed[i] = np.mean(vals)

        x = np.arange(len(smoothed))
        mask = ~np.isnan(smoothed) & (count_per_pos[model_name] > 10)
        ax1.plot(x[mask], smoothed[mask], color=colors[model_name], label=labels[model_name], linewidth=1.5)

    ax1.axvline(x=200, color="gray", linestyle="--", alpha=0.5, label="Position 200")
    ax1.set_xlabel("Token Position", fontsize=12)
    ax1.set_ylabel("Mean |log p_student - log p_teacher|", fontsize=12)
    ax1.set_title("Per-Position KL Divergence (smoothed, window=10)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 600)
    ax1.grid(True, alpha=0.3)

    # Zoomed into 0-50
    for model_name in ["raw", "pos-200tok", "full-seq-50"]:
        if model_name not in kl_per_pos:
            continue
        kl = kl_per_pos[model_name]
        x = np.arange(min(50, len(kl)))
        vals = kl[:50]
        mask = ~np.isnan(vals)
        ax2.plot(x[mask], vals[mask], color=colors[model_name], label=labels[model_name], linewidth=1.5, marker="o", markersize=3)

    ax2.set_xlabel("Token Position", fontsize=12)
    ax2.set_ylabel("Mean |log p_student - log p_teacher|", fontsize=12)
    ax2.set_title("Per-Position KL (first 50 tokens, unsmoothed)", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "kl_by_position_curve.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved curve plot to {fig_path}")
    plt.close()

    # --- Table: 100-token ranges ---
    ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 600)]

    print("\n" + "=" * 80)
    print("COMPARISON: Mean |student_lp - teacher_lp| by 100-token range")
    print("=" * 80)

    range_labels = [f"{s}-{e}" for s, e in ranges]
    header = f"{'Model':>15} | " + " | ".join(f"{r:>10}" for r in range_labels)
    print(header)
    print("-" * len(header))

    range_results = {}
    for model_name in ["raw", "pos-200tok", "full-seq-50"]:
        if model_name not in kl_per_pos:
            continue
        row = {}
        vals_str = []
        for start, end in ranges:
            kl_vals = []
            for s_lps, t_lps in zip(all_student_lps[model_name], all_teacher_lps[model_name]):
                seq_len = min(len(s_lps), len(t_lps))
                for pos in range(start, min(end, seq_len)):
                    kl_vals.append(abs(s_lps[pos] - t_lps[pos]))
            if kl_vals:
                mean_v = np.mean(kl_vals)
                row[f"{start}-{end}"] = {"mean": mean_v, "count": len(kl_vals)}
                vals_str.append(f"{mean_v:>10.4f}")
            else:
                vals_str.append(f"{'N/A':>10}")
        print(f"{model_name:>15} | " + " | ".join(vals_str))
        range_results[model_name] = row

    # Save all results
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    save_results = {}
    for model_name in all_trajs:
        save_results[model_name] = {
            "range_results": range_results.get(model_name, {}),
            "avg_response_len": float(np.mean([len(t["response_ids"]) for t in all_trajs[model_name]])),
            "n_problems": len(all_trajs[model_name]),
        }
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"Per-token data saved to {OUTPUT_DIR}/*_logprobs.jsonl")
    print(f"Plot saved to {fig_path}")


if __name__ == "__main__":
    main()

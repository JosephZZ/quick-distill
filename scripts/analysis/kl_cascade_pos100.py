"""
Quick KL cascade analysis for pos-100 checkpoint.
Generates trajectories from pos-100 distilled student, scores with teacher,
computes per-position-range KL, and saves results.

Usage: CUDA_VISIBLE_DEVICES=2 python scripts/analysis/kl_cascade_pos100.py
"""
import json, os, gc, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

STUDENT_BASE = "Qwen/Qwen2.5-Math-1.5B"
TEACHER_MODEL = "Qwen/Qwen3-1.7B"
LORA_PATH = "checkpoints/dft-pos100-math/step_200"  # best step for pos-100
N_PROBLEMS = 100
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
DEVICE = "cuda:0"
OUTPUT_FILE = "docs/kl_cascade_pos100.json"

RANGES = [
    (0, 50), (50, 100), (100, 150), (150, 200),
    (200, 300), (300, 400), (400, 500), (500, 600), (600, 700),
]

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def build_prompt(problem, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_trajectories(model, tokenizer, problems):
    """Generate one trajectory per problem."""
    trajs = []
    model.eval()
    for i, prob in enumerate(problems):
        prompt = build_prompt(prob, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        prompt_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=1.0,
            )
        resp_ids = out[0, prompt_len:]
        trajs.append({
            "problem": prob,
            "prompt": prompt,
            "prompt_len": prompt_len,
            "response_ids": resp_ids.tolist(),
            "response_len": len(resp_ids),
        })
        if (i + 1) % 20 == 0:
            print(f"  Generated {i+1}/{len(problems)}, last len={len(resp_ids)}")
    return trajs


def compute_kl_per_position(model, tokenizer, trajs, model_name="model"):
    """Compute per-token KL for each trajectory. Returns list of per-token KL arrays."""
    model.eval()
    all_kl = []
    for i, traj in enumerate(trajs):
        prompt = traj["prompt"]
        resp_ids = traj["response_ids"]
        prompt_len = traj["prompt_len"]

        # Build full sequence
        full_text = prompt + tokenizer.decode(resp_ids, skip_special_tokens=False)
        inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)
        input_ids = inputs.input_ids

        with torch.no_grad():
            logits = model(**inputs).logits  # (1, seq_len, vocab)

        # Get log probs for response tokens
        resp_logits = logits[0, prompt_len - 1: prompt_len - 1 + len(resp_ids)]  # (resp_len, vocab)
        log_probs = torch.log_softmax(resp_logits, dim=-1)

        all_kl.append(log_probs.cpu())

        if (i + 1) % 20 == 0:
            print(f"  Scored {i+1}/{len(trajs)} with {model_name}")

    return all_kl


def compute_range_kl(student_logprobs_list, teacher_logprobs_list, trajs):
    """Compute mean KL per position range."""
    range_kl = {f"{s}-{e}": [] for s, e in RANGES}

    for s_lp, t_lp, traj in zip(student_logprobs_list, teacher_logprobs_list, trajs):
        resp_len = min(len(s_lp), len(t_lp))
        # Compute per-token reverse KL: sum_x q(x) * [log q(x) - log p(x)]
        # where q = teacher, p = student
        for start, end in RANGES:
            if start >= resp_len:
                continue
            actual_end = min(end, resp_len)
            t_slice = t_lp[start:actual_end]
            s_slice = s_lp[start:actual_end]
            # Reverse KL per token
            t_probs = torch.exp(t_slice)
            kl_per_token = (t_probs * (t_slice - s_slice)).sum(dim=-1)  # (n_tokens,)
            kl_per_token = kl_per_token.clamp(min=0)
            range_kl[f"{start}-{end}"].extend(kl_per_token.tolist())

    # Average per range
    results = {}
    for key in range_kl:
        vals = range_kl[key]
        results[key] = float(np.mean(vals)) if vals else 0.0
    return results


def main():
    print("Loading problems...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = [item["problem"] for item in ds][:N_PROBLEMS]

    # --- Raw student ---
    print("\n=== Raw Student (no distillation) ===")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE, trust_remote_code=True)
    student_raw = AutoModelForCausalLM.from_pretrained(
        STUDENT_BASE, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(DEVICE)

    print("Generating with raw student...")
    raw_trajs = generate_trajectories(student_raw, tokenizer, problems)
    print(f"Mean response length: {np.mean([t['response_len'] for t in raw_trajs]):.0f}")

    print("Scoring raw trajectories with raw student...")
    raw_student_logprobs = compute_kl_per_position(student_raw, tokenizer, raw_trajs, "raw_student")

    del student_raw
    gc.collect()
    torch.cuda.empty_cache()

    # --- Distilled student (pos-100) ---
    print("\n=== Distilled Student (pos-100) ===")
    student_dist = AutoModelForCausalLM.from_pretrained(
        STUDENT_BASE, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    student_dist = PeftModel.from_pretrained(student_dist, LORA_PATH)
    student_dist = student_dist.merge_and_unload().to(DEVICE)

    print("Generating with distilled student...")
    dist_trajs = generate_trajectories(student_dist, tokenizer, problems)
    print(f"Mean response length: {np.mean([t['response_len'] for t in dist_trajs]):.0f}")

    print("Scoring distilled trajectories with distilled student...")
    dist_student_logprobs = compute_kl_per_position(student_dist, tokenizer, dist_trajs, "dist_student")

    del student_dist
    gc.collect()
    torch.cuda.empty_cache()

    # --- Teacher scoring both sets ---
    print("\n=== Teacher Scoring ===")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(DEVICE)
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)

    # For Qwen3-1.7B, need to re-tokenize with teacher tokenizer
    # But since both are Qwen family, tokenizer should be compatible
    # Use student tokenizer for both (same vocab)
    print("Scoring raw trajectories with teacher...")
    raw_teacher_logprobs = compute_kl_per_position(teacher, tokenizer, raw_trajs, "teacher_on_raw")

    print("Scoring distilled trajectories with teacher...")
    dist_teacher_logprobs = compute_kl_per_position(teacher, tokenizer, dist_trajs, "teacher_on_dist")

    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    # --- Compute KL per range ---
    print("\n=== Computing KL per range ===")
    raw_kl = compute_range_kl(raw_student_logprobs, raw_teacher_logprobs, raw_trajs)
    dist_kl = compute_range_kl(dist_student_logprobs, dist_teacher_logprobs, dist_trajs)

    print("\nRange      | Raw     | Pos-100 | Reduction")
    print("-" * 50)
    for start, end in RANGES:
        key = f"{start}-{end}"
        r = raw_kl[key]
        d = dist_kl[key]
        red = (r - d) / r * 100 if r > 0 else 0
        print(f"{key:10s} | {r:.3f}   | {d:.3f}   | {red:.1f}%")

    results = {
        "raw_kl": raw_kl,
        "pos100_kl": dist_kl,
        "n_problems": N_PROBLEMS,
        "lora_path": LORA_PATH,
        "raw_mean_len": float(np.mean([t['response_len'] for t in raw_trajs])),
        "dist_mean_len": float(np.mean([t['response_len'] for t in dist_trajs])),
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

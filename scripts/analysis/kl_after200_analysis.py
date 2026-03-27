"""
Compare KL divergence after position 200 between teacher and different student models.
Models: raw (no distill), pos-200tok (step 100), full-seq (step 50)
Teacher: Qwen3-1.7B
Generate 100 problems × 1 trajectory, then score with teacher.
"""
import json, os, sys, time, torch, gc, argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

GPU = int(os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
DEVICE = f"cuda:0"  # mapped device
TEACHER_MODEL = "Qwen/Qwen3-1.7B"
STUDENT_BASE = "Qwen/Qwen2.5-Math-1.5B"
N_PROBLEMS = 100
MAX_NEW_TOKENS = 1024  # enough to get past 200 tokens, don't need full 3584
TEMPERATURE = 0.7

def build_prompt(problem, tokenizer):
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def load_problems(n=100):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = [item["problem"] for item in ds]
    return problems[:n]

def merge_lora(base_model_name, lora_path, device):
    """Load base + LoRA, merge, return merged model."""
    print(f"  Merging LoRA from {lora_path}...")
    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    model = model.to(device)
    model.eval()
    return model

def generate_trajectories(model, tokenizer, problems, device):
    """Generate 1 trajectory per problem, return list of (prompt_ids, response_ids)."""
    trajs = []
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for i, prob in enumerate(problems):
        prompt_text = build_prompt(prob, tokenizer)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True, temperature=TEMPERATURE, top_p=0.95,
            )

        response_ids = output[0, prompt_len:].tolist()
        prompt_ids = inputs["input_ids"][0].tolist()
        trajs.append({"prompt_ids": prompt_ids, "response_ids": response_ids, "problem": prob})

        if (i + 1) % 20 == 0:
            print(f"    Generated {i+1}/{len(problems)}")

    return trajs

def score_with_teacher(teacher_model, tokenizer, trajs, device):
    """Get teacher log probs for each trajectory's response tokens."""
    results = []

    for i, traj in enumerate(trajs):
        prompt_ids = traj["prompt_ids"]
        response_ids = traj["response_ids"]
        full_ids = prompt_ids + response_ids

        input_ids = torch.tensor([full_ids], device=device)
        prompt_len = len(prompt_ids)

        with torch.no_grad():
            logits = teacher_model(input_ids).logits[0]  # (seq_len, vocab)

        # Get log probs for response tokens
        # logits[t] predicts token at position t+1
        response_logits = logits[prompt_len - 1: prompt_len - 1 + len(response_ids)]  # (resp_len, vocab)
        log_probs = torch.log_softmax(response_logits.float(), dim=-1)

        teacher_lps = []
        for t, token_id in enumerate(response_ids):
            teacher_lps.append(log_probs[t, token_id].item())

        results.append({
            "problem": traj["problem"],
            "response_ids": response_ids,
            "teacher_log_probs": teacher_lps,
        })

        if (i + 1) % 20 == 0:
            print(f"    Scored {i+1}/{len(trajs)}")

    return results

def score_with_student(student_model, tokenizer, trajs, device):
    """Get student log probs for each trajectory's response tokens."""
    results = []

    for i, traj in enumerate(trajs):
        prompt_ids = traj["prompt_ids"]
        response_ids = traj["response_ids"]
        full_ids = prompt_ids + response_ids

        input_ids = torch.tensor([full_ids], device=device)
        prompt_len = len(prompt_ids)

        with torch.no_grad():
            logits = student_model(input_ids).logits[0]

        response_logits = logits[prompt_len - 1: prompt_len - 1 + len(response_ids)]
        log_probs = torch.log_softmax(response_logits.float(), dim=-1)

        student_lps = []
        for t, token_id in enumerate(response_ids):
            student_lps.append(log_probs[t, token_id].item())

        results.append(student_lps)

        if (i + 1) % 20 == 0:
            print(f"    Student-scored {i+1}/{len(trajs)}")

    return results

def analyze_kl(scored_trajs, student_lps_list, label):
    """Analyze |student_lp - teacher_lp| by position range."""
    ranges = [(0, 50), (50, 100), (100, 200), (200, 500), (200, 1024)]

    kl_by_range = {r: [] for r in ranges}

    for traj, student_lps in zip(scored_trajs, student_lps_list):
        teacher_lps = traj["teacher_log_probs"]
        seq_len = min(len(teacher_lps), len(student_lps))

        for start, end in ranges:
            for pos in range(start, min(end, seq_len)):
                diff = abs(student_lps[pos] - teacher_lps[pos])
                kl_by_range[(start, end)].append(diff)

    print(f"\n=== {label} ===")
    print(f"{'Range':>12} | {'Mean |diff|':>12} | {'Median':>8} | {'Count':>8}")
    print("-" * 50)
    result = {}
    for r in ranges:
        vals = kl_by_range[r]
        if vals:
            mean_v = np.mean(vals)
            med_v = np.median(vals)
            print(f"  {r[0]:>3}-{r[1]:<4} | {mean_v:>12.4f} | {med_v:>8.4f} | {len(vals):>8}")
            result[f"{r[0]}-{r[1]}"] = {"mean": mean_v, "median": med_v, "count": len(vals)}

    return result

def main():
    print("Loading problems...")
    problems = load_problems(N_PROBLEMS)
    print(f"Loaded {len(problems)} problems")

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_BASE, trust_remote_code=True)

    models_config = [
        ("raw", None),
        ("pos-200tok", "checkpoints/pos-limit-200tok/step_100"),
        ("full-seq-50", "checkpoints/full-seq-3584tok/step_50"),
    ]

    all_results = {}

    # Load teacher once
    print("\nLoading teacher model...")
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).to(DEVICE)
    teacher.eval()

    for model_name, lora_path in models_config:
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"{'='*60}")

        # Load student model
        if lora_path is None:
            print("  Loading raw student model...")
            student = AutoModelForCausalLM.from_pretrained(STUDENT_BASE, torch_dtype=torch.bfloat16, trust_remote_code=True).to(DEVICE)
            student.eval()
        else:
            student = merge_lora(STUDENT_BASE, lora_path, DEVICE)

        # Generate trajectories
        print(f"  Generating {N_PROBLEMS} trajectories...")
        t0 = time.time()
        trajs = generate_trajectories(student, tokenizer, problems, DEVICE)
        gen_time = time.time() - t0

        avg_len = np.mean([len(t["response_ids"]) for t in trajs])
        print(f"  Generated in {gen_time:.0f}s, avg response length: {avg_len:.0f} tokens")

        # Score with student (on its own trajectories)
        print(f"  Scoring with student model...")
        student_lps = score_with_student(student, tokenizer, trajs, DEVICE)

        # Free student
        del student
        gc.collect()
        torch.cuda.empty_cache()

        # Score with teacher
        print(f"  Scoring with teacher model...")
        scored = score_with_teacher(teacher, tokenizer, trajs, DEVICE)

        # Analyze
        result = analyze_kl(scored, student_lps, model_name)
        all_results[model_name] = {
            "kl_by_range": result,
            "avg_response_len": avg_len,
            "n_problems": len(problems),
        }

    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    # Save results
    output_path = "docs/kl_after200_comparison.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON: Mean |student_lp - teacher_lp| by position range")
    print("="*80)
    ranges = ["0-50", "50-100", "100-200", "200-500", "200-1024"]
    header = f"{'Model':>15} | " + " | ".join(f"{r:>10}" for r in ranges)
    print(header)
    print("-" * len(header))
    for model_name in ["raw", "pos-200tok", "full-seq-50"]:
        res = all_results[model_name]["kl_by_range"]
        vals = []
        for r in ranges:
            if r in res:
                vals.append(f"{res[r]['mean']:>10.4f}")
            else:
                vals.append(f"{'N/A':>10}")
        print(f"{model_name:>15} | " + " | ".join(vals))

    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()

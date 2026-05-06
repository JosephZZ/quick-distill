"""
Compare student vs teacher generation from scratch on the same problems.
Uses vLLM for fast generation. Focuses on content/reasoning differences.
"""

import json
import os
import subprocess
import sys
import tempfile

N_SAMPLES = 1          # 1 per problem, we pick 10 problems
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
N_PROBLEMS = 10

STUDENT_MODEL = "Qwen/Qwen2.5-Math-1.5B"
TEACHER_MODEL = "Qwen/Qwen3-1.7B"

OUTPUT_DIR = "docs/scratch_comparison"


def generate_with_vllm(model, tokenizer_name, problems, output_file):
    """Call vllm_generate.py as subprocess."""
    problems_file = tempfile.mktemp(suffix=".json")
    with open(problems_file, "w") as f:
        json.dump(problems, f)

    cmd = [
        sys.executable, "vllm_generate.py",
        "--model", model,
        "--tokenizer", tokenizer_name,
        "--problems_file", problems_file,
        "--output_file", output_file,
        "--n_samples", str(N_SAMPLES),
        "--max_new_tokens", str(MAX_NEW_TOKENS),
        "--temperature", str(TEMPERATURE),
        "--gpu_memory_utilization", "0.85",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"vllm_generate failed for {model}")

    os.remove(problems_file)


def decode_trajectories(traj_file, tokenizer_name):
    """Load trajectories and decode response tokens."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    with open(traj_file) as f:
        all_trajs = json.load(f)

    decoded = {}
    for key, trajs in all_trajs.items():
        texts = []
        for t in trajs:
            text = tok.decode(t["response_ids"], skip_special_tokens=True)
            texts.append(text)
        decoded[key] = texts
    return decoded


def generate_teacher_hf(model_name, problems, output_file):
    """Generate with teacher using HF generate (vLLM doesn't support Qwen3)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    DEVICE = "cuda:0"  # CUDA_VISIBLE_DEVICES=1 maps GPU1 to cuda:0

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    # Build prompts with nothink
    def _supports_thinking(tokenizer):
        try:
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            return True
        except TypeError:
            return False

    def _supports_system_role(tokenizer):
        try:
            tokenizer.apply_chat_template(
                [{"role": "system", "content": "test"}, {"role": "user", "content": "test"}],
                tokenize=False, add_generation_prompt=True,
            )
            return True
        except Exception:
            return False

    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    all_trajectories = {}
    for i, problem in enumerate(problems):
        if _supports_system_role(tok):
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": problem}]
        else:
            messages = [{"role": "user", "content": system_prompt + "\n\n" + problem}]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if _supports_thinking(tok):
            kwargs["enable_thinking"] = False
        prompt_text = tok.apply_chat_template(messages, **kwargs)
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)

        input_ids = torch.tensor([prompt_ids], device=DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=0.95,
            )
        response_ids = outputs[0, len(prompt_ids):].tolist()
        # Strip special tokens
        eos_id = tok.eos_token_id
        special_ids = {eos_id, tok.pad_token_id, 151645, 151643}
        while response_ids and response_ids[-1] in special_ids:
            response_ids.pop()

        all_trajectories[str(i)] = [{"prompt_ids": prompt_ids, "response_ids": response_ids}]
        print(f"  Problem {i+1}/{len(problems)}: {len(response_ids)} tokens")

    with open(output_file, "w") as f:
        json.dump(all_trajectories, f)

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def main():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    all_problems = [row["problem"] for row in ds]

    # Pick N_PROBLEMS spread across the dataset
    step = len(all_problems) // N_PROBLEMS
    problems = [all_problems[i * step] for i in range(N_PROBLEMS)]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save problems for reference
    with open(os.path.join(OUTPUT_DIR, "problems.json"), "w") as f:
        json.dump(problems, f, indent=2)

    student_file = os.path.join(OUTPUT_DIR, "student_trajs.json")
    teacher_file = os.path.join(OUTPUT_DIR, "teacher_trajs.json")

    # Generate with student (vLLM) - skip if already exists
    if os.path.exists(student_file):
        print("=== Student trajectories already exist, skipping ===")
    else:
        print("=== Generating with STUDENT ===")
        generate_with_vllm(STUDENT_MODEL, STUDENT_MODEL, problems, student_file)

    # Generate with teacher (HF generate, since vLLM doesn't support Qwen3)
    print("\n=== Generating with TEACHER (HF generate) ===")
    generate_teacher_hf(TEACHER_MODEL, problems, teacher_file)

    # Decode and compare
    print("\n=== Decoding ===")
    student_decoded = decode_trajectories(student_file, STUDENT_MODEL)
    teacher_decoded = decode_trajectories(teacher_file, TEACHER_MODEL)

    # Write comparison
    output_path = os.path.join(OUTPUT_DIR, "comparison.txt")
    with open(output_path, "w") as f:
        for i, problem in enumerate(problems):
            key = str(i)
            s_texts = student_decoded.get(key, ["[NO OUTPUT]"])
            t_texts = teacher_decoded.get(key, ["[NO OUTPUT]"])

            f.write(f"\n{'='*100}\n")
            f.write(f"PROBLEM {i+1}: {problem}\n")
            f.write(f"{'='*100}\n")

            f.write(f"\n[STUDENT]\n")
            f.write(s_texts[0] + "\n")

            f.write(f"\n[TEACHER]\n")
            f.write(t_texts[0] + "\n")

    print(f"\nSaved comparison to {output_path}")

    # Also print to stdout
    with open(output_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()

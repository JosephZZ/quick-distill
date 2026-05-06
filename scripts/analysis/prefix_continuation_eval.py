"""
Prefix continuation evaluation on MATH-500:
  Experiment 1: Student prefix (100 tok) → Teacher continues (3000 tok) → Eval
  Experiment 2: Teacher prefix (100 tok) → Student continues (3000 tok) → Eval

Uses vLLM for both generation phases (run with new conda env that supports Qwen3).
Each phase runs as a subprocess so GPU memory is fully freed between models.
"""

import json
import os
import sys
import re
import subprocess
import tempfile
import argparse
from collections import Counter

import numpy as np
from datasets import load_dataset

# ── answer extraction (copied from eval_math500.py) ─────────────────────────

_repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_stub_dir = os.path.join(_repo_dir, "math_eval_stub")
_fallback_dir = os.path.join(_repo_dir, "math_evaluation")
if os.path.isdir(_stub_dir):
    sys.path.insert(0, _stub_dir)
else:
    sys.path.insert(0, _fallback_dir)
from grader import math_equal


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def remove_boxed(s):
    if s is None:
        return None
    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] == left:
            return s[len(left):]
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except (AssertionError, IndexError):
        return None


def extract_answer(solution_str):
    try:
        s = last_boxed_only_string(solution_str)
        if s is None:
            return None
        return remove_boxed(s)
    except Exception:
        return None


# ── vLLM generation via subprocess ───────────────────────────────────────────

def vllm_generate(model, prompts_data, max_new_tokens, temperature, n_samples=1,
                  gpu_mem=0.85, max_model_len=4096):
    """Generate completions using vLLM in a subprocess. Returns list of response texts."""
    # Write prompts to temp file
    prompts_file = tempfile.mktemp(suffix=".json")
    output_file = tempfile.mktemp(suffix=".json")

    with open(prompts_file, "w") as f:
        json.dump(prompts_data, f)

    # Use inline vLLM script
    script = f'''
import json, sys
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

with open("{prompts_file}") as f:
    prompts = json.load(f)

llm = LLM(
    model="{model}",
    dtype="bfloat16",
    max_model_len={max_model_len},
    seed=42,
    trust_remote_code=True,
    gpu_memory_utilization={gpu_mem},
    enforce_eager=True,
)

sampling_params = SamplingParams(
    temperature={temperature},
    top_p=0.95 if {temperature} > 0 else 1.0,
    max_tokens={max_new_tokens},
    n={n_samples},
)

outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

results = []
for output in outputs:
    completions = []
    for comp in output.outputs:
        completions.append({{
            "text": comp.text,
            "token_ids": list(comp.token_ids),
        }})
    results.append(completions)

with open("{output_file}", "w") as f:
    json.dump(results, f)

print(f"Generated {{len(results)}} outputs")
'''
    script_file = tempfile.mktemp(suffix=".py")
    with open(script_file, "w") as f:
        f.write(script)

    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, script_file],
        capture_output=True, text=True, env=env,
    )
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.returncode != 0:
        print(f"STDERR (last 1000): {result.stderr[-1000:]}")
        raise RuntimeError(f"vLLM generation failed for {model}")

    with open(output_file) as f:
        results = json.load(f)

    os.remove(prompts_file)
    os.remove(output_file)
    os.remove(script_file)
    return results


# ── main ─────────────────────────────────────────────────────────────────────

def build_prompt(problem, tok):
    """Build chat prompt string for a problem. tok is a pre-loaded tokenizer."""
    instruction = "Let's think step by step and output the final answer within \\boxed{}."
    question = problem + " " + instruction

    chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": question}],
            enable_thinking=False, **chat_kwargs,
        )
    except TypeError:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": question}], **chat_kwargs,
        )
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_len", type=int, default=100)
    parser.add_argument("--cont_len", type=int, default=3000)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="docs/prefix_continuation_eval")
    args = parser.parse_args()

    STUDENT = "Qwen/Qwen2.5-Math-1.5B"
    TEACHER = "Qwen/Qwen3-1.7B"

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print("Loading MATH-500...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = [row["problem"] for row in ds]
    ground_truths = [extract_answer(row["solution"]) for row in ds]

    from transformers import AutoTokenizer
    student_tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER, trust_remote_code=True)

    # ── Phase 1: Generate prefixes ───────────────────────────────────────
    # Student prefixes
    student_prefix_file = os.path.join(args.output_dir, "student_prefixes.json")
    if not os.path.exists(student_prefix_file):
        print("\n=== Phase 1a: Student generates prefixes ===")
        student_prompts = [build_prompt(p, student_tok) for p in problems]
        student_results = vllm_generate(
            STUDENT, student_prompts, args.prefix_len, args.temperature,
            n_samples=1, gpu_mem=0.85, max_model_len=2048,
        )
        student_prefixes = []
        for r in student_results:
            text = r[0]["text"]
            token_ids = r[0]["token_ids"][:args.prefix_len]
            student_prefixes.append({"text": text, "token_ids": token_ids})
        with open(student_prefix_file, "w") as f:
            json.dump(student_prefixes, f)
        print(f"Saved {len(student_prefixes)} student prefixes")
    else:
        print("Student prefixes already exist, loading...")
        with open(student_prefix_file) as f:
            student_prefixes = json.load(f)

    # Teacher prefixes
    teacher_prefix_file = os.path.join(args.output_dir, "teacher_prefixes.json")
    if not os.path.exists(teacher_prefix_file):
        print("\n=== Phase 1b: Teacher generates prefixes ===")
        teacher_prompts = [build_prompt(p, teacher_tok) for p in problems]
        teacher_results = vllm_generate(
            TEACHER, teacher_prompts, args.prefix_len, args.temperature,
            n_samples=1, gpu_mem=0.85, max_model_len=2048,
        )
        teacher_prefixes = []
        for r in teacher_results:
            text = r[0]["text"]
            token_ids = r[0]["token_ids"][:args.prefix_len]
            teacher_prefixes.append({"text": text, "token_ids": token_ids})
        with open(teacher_prefix_file, "w") as f:
            json.dump(teacher_prefixes, f)
        print(f"Saved {len(teacher_prefixes)} teacher prefixes")
    else:
        print("Teacher prefixes already exist, loading...")
        with open(teacher_prefix_file) as f:
            teacher_prefixes = json.load(f)

    # ── Phase 2: Cross-continuations ─────────────────────────────────────
    # Exp 1: Student prefix → Teacher continues
    exp1_file = os.path.join(args.output_dir, "exp1_student_prefix_teacher_cont.json")
    if not os.path.exists(exp1_file):
        print("\n=== Phase 2a: Teacher continues from student prefix ===")
        # Build prompts: teacher_prompt + nothink + student_prefix_text
        nothink_str = "<think>\n\n</think>\n\n"
        nothink_ids = teacher_tok.encode(nothink_str, add_special_tokens=False)
        if len(nothink_ids) > 6:
            nothink_ids = []
        nothink_text = teacher_tok.decode(nothink_ids) if nothink_ids else ""

        exp1_prompts = []
        for i, problem in enumerate(problems):
            teacher_prompt = build_prompt(problem, teacher_tok)
            prefix_text = student_tok.decode(student_prefixes[i]["token_ids"], skip_special_tokens=True)
            # Re-encode with teacher tokenizer to get a clean prompt
            full_prompt = teacher_prompt + nothink_text + prefix_text
            exp1_prompts.append(full_prompt)

        exp1_results = vllm_generate(
            TEACHER, exp1_prompts, args.cont_len, args.temperature,
            n_samples=args.n_samples, gpu_mem=0.85, max_model_len=4096,
        )
        # Combine prefix + continuation (all n_samples per problem)
        exp1_responses = []
        for i, r in enumerate(exp1_results):
            prefix_text = student_tok.decode(student_prefixes[i]["token_ids"], skip_special_tokens=True)
            samples = [prefix_text + comp["text"] for comp in r]
            exp1_responses.append(samples)
        with open(exp1_file, "w") as f:
            json.dump(exp1_responses, f)
    else:
        print("Exp1 results exist, loading...")
        with open(exp1_file) as f:
            exp1_responses = json.load(f)

    # Exp 2: Teacher prefix → Student continues
    exp2_file = os.path.join(args.output_dir, "exp2_teacher_prefix_student_cont.json")
    if not os.path.exists(exp2_file):
        print("\n=== Phase 2b: Student continues from teacher prefix ===")
        exp2_prompts = []
        for i, problem in enumerate(problems):
            student_prompt = build_prompt(problem, student_tok)
            prefix_text = teacher_tok.decode(teacher_prefixes[i]["token_ids"], skip_special_tokens=True)
            full_prompt = student_prompt + prefix_text
            exp2_prompts.append(full_prompt)

        exp2_results = vllm_generate(
            STUDENT, exp2_prompts, args.cont_len, args.temperature,
            n_samples=args.n_samples, gpu_mem=0.85, max_model_len=4096,
        )
        exp2_responses = []
        for i, r in enumerate(exp2_results):
            prefix_text = teacher_tok.decode(teacher_prefixes[i]["token_ids"], skip_special_tokens=True)
            samples = [prefix_text + comp["text"] for comp in r]
            exp2_responses.append(samples)
        with open(exp2_file, "w") as f:
            json.dump(exp2_responses, f)
    else:
        print("Exp2 results exist, loading...")
        with open(exp2_file) as f:
            exp2_responses = json.load(f)

    # ── Phase 3: Evaluate ────────────────────────────────────────────────
    print("\n=== Phase 3: Evaluation ===")

    def normalize_answer_for_vote(ans):
        if ans is None:
            return None
        return re.sub(r"\s+", " ", str(ans)).strip()

    def evaluate(all_responses, label):
        """all_responses: list of lists, [n_problems][n_samples] of response strings."""
        n_problems = len(all_responses)
        n_samples = len(all_responses[0]) if all_responses else 1

        pass_correct = 0
        avg_correct_sum = 0.0
        maj_correct = 0

        for i, samples in enumerate(all_responses):
            gt = ground_truths[i]
            sample_results = []
            answers = []
            for resp in samples:
                pred = extract_answer(resp)
                is_correct = pred is not None and math_equal(pred, gt)
                sample_results.append(is_correct)
                answers.append(normalize_answer_for_vote(pred))

            # pass@k: any sample correct
            if any(sample_results):
                pass_correct += 1

            # avg@k: fraction of samples correct
            avg_correct_sum += sum(sample_results) / len(sample_results)

            # maj@k: majority vote
            valid_answers = [a for a in answers if a is not None]
            if valid_answers:
                counter = Counter(valid_answers)
                majority_ans = counter.most_common(1)[0][0]
                # Check if majority answer is correct
                if math_equal(majority_ans, gt):
                    maj_correct += 1

        avg_at_k = avg_correct_sum / n_problems * 100
        maj_at_k = maj_correct / n_problems * 100
        pass_at_k = pass_correct / n_problems * 100

        print(f"  {label}: avg@{n_samples}={avg_at_k:.2f}% | maj@{n_samples}={maj_at_k:.2f}% | pass@{n_samples}={pass_at_k:.2f}%")
        return {"label": label, "n_samples": n_samples,
                "avg_at_k": round(avg_at_k, 2), "maj_at_k": round(maj_at_k, 2),
                "pass_at_k": round(pass_at_k, 2)}

    results = {}
    results["exp1"] = evaluate(exp1_responses, "Student prefix → Teacher cont")
    results["exp2"] = evaluate(exp2_responses, "Teacher prefix → Student cont")

    # Save summary
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to {summary_file}")


if __name__ == "__main__":
    main()

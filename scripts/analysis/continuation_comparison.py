"""
Continuation comparison: student generates first 100 tokens, then both student and teacher
continue independently. Compare the style of continuations side-by-side.
"""

import json
import gc
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

DEVICE = "cuda:0"  # use CUDA_VISIBLE_DEVICES=1 to select physical GPU 1
STUDENT_MODEL = "Qwen/Qwen2.5-Math-1.5B"
TEACHER_MODEL = "Qwen/Qwen3-1.7B"
PREFIX_LEN = 100       # student generates first N tokens
CONT_LEN = 200         # continuation length
N_SAMPLES = 8
TEMPERATURE = 0.7


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


def build_prompt(problem, tokenizer, system_prompt=None):
    if system_prompt is None:
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    if _supports_system_role(tokenizer):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
        ]
    else:
        messages = [{"role": "user", "content": system_prompt + "\n\n" + problem}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if _supports_thinking(tokenizer):
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def generate_continuation(model, input_ids, max_new_tokens, temperature):
    """Generate continuation from given input_ids prefix."""
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
        )
    # Return only new tokens
    return outputs[0, input_ids.shape[1]:].tolist()


def main():
    # Load problems
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = [row["problem"] for row in ds]

    # Pick N_SAMPLES diverse problems (spread across dataset)
    step = len(problems) // N_SAMPLES
    selected = [problems[i * step] for i in range(N_SAMPLES)]

    # --- Phase 1: Load student, generate prefix + student continuation ---
    print("=== Loading student model ===")
    student_tok = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(DEVICE)
    student.eval()

    results = []
    for i, problem in enumerate(selected):
        print(f"\n--- Problem {i+1}/{N_SAMPLES} ---")
        print(f"Q: {problem[:100]}...")

        prompt_text = build_prompt(problem, student_tok)
        prompt_ids = student_tok.encode(prompt_text, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], device=DEVICE)

        # Generate prefix (first PREFIX_LEN tokens)
        prefix_tokens = generate_continuation(student, input_ids, PREFIX_LEN, TEMPERATURE)
        if len(prefix_tokens) < PREFIX_LEN:
            print(f"  Student finished early ({len(prefix_tokens)} tokens), using all")

        # Generate student continuation from prefix
        prefix_input = torch.tensor([prompt_ids + prefix_tokens], device=DEVICE)
        student_cont = generate_continuation(student, prefix_input, CONT_LEN, TEMPERATURE)

        results.append({
            "problem": problem,
            "prompt_ids": prompt_ids,
            "prefix_tokens": prefix_tokens,
            "student_cont": student_cont,
        })

        prefix_text = student_tok.decode(prefix_tokens)
        student_cont_text = student_tok.decode(student_cont)
        print(f"  Prefix ({len(prefix_tokens)} tok): {prefix_text[:150]}...")
        print(f"  Student cont ({len(student_cont)} tok): {student_cont_text[:150]}...")

    # Offload student
    del student
    gc.collect()
    torch.cuda.empty_cache()

    # --- Phase 2: Load teacher, generate teacher continuation from same prefix ---
    print("\n=== Loading teacher model ===")
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(DEVICE)
    teacher.eval()

    # Qwen3 nothink prefix
    nothink_str = "<think>\n\n</think>\n\n"
    nothink_ids = teacher_tok.encode(nothink_str, add_special_tokens=False)
    if len(nothink_ids) > 6:
        nothink_ids = []

    for i, r in enumerate(results):
        print(f"\n--- Teacher continuing problem {i+1}/{N_SAMPLES} ---")

        # Re-encode prompt with teacher tokenizer
        prompt_text = build_prompt(r["problem"], teacher_tok)
        teacher_prompt_ids = teacher_tok.encode(prompt_text, add_special_tokens=False)

        # Decode student prefix using student tokenizer, re-encode with teacher tokenizer
        prefix_text = student_tok.decode(r["prefix_tokens"])
        teacher_prefix_ids = teacher_tok.encode(prefix_text, add_special_tokens=False)

        # Build teacher input: teacher_prompt + nothink + teacher_prefix
        teacher_input_ids = teacher_prompt_ids + nothink_ids + teacher_prefix_ids
        teacher_input = torch.tensor([teacher_input_ids], device=DEVICE)

        teacher_cont_ids = generate_continuation(teacher, teacher_input, CONT_LEN, TEMPERATURE)
        teacher_cont_text = teacher_tok.decode(teacher_cont_ids)
        r["teacher_cont"] = teacher_cont_text

        print(f"  Teacher cont ({len(teacher_cont_ids)} tok): {teacher_cont_text[:150]}...")

    # --- Phase 2.5: Teacher generates from scratch (no student prefix) ---
    print("\n=== Teacher generating from scratch ===")
    for i, r in enumerate(results):
        print(f"\n--- Teacher from scratch problem {i+1}/{N_SAMPLES} ---")

        prompt_text = build_prompt(r["problem"], teacher_tok)
        teacher_prompt_ids = teacher_tok.encode(prompt_text, add_special_tokens=False)

        # Teacher prompt + nothink, then generate full response
        teacher_input_ids = teacher_prompt_ids + nothink_ids
        teacher_input = torch.tensor([teacher_input_ids], device=DEVICE)

        teacher_full_ids = generate_continuation(
            teacher, teacher_input, PREFIX_LEN + CONT_LEN, TEMPERATURE
        )
        teacher_full_text = teacher_tok.decode(teacher_full_ids)
        r["teacher_full"] = teacher_full_text
        r["teacher_full_len"] = len(teacher_full_ids)

        print(f"  Teacher full ({len(teacher_full_ids)} tok): {teacher_full_text[:200]}...")

    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    # --- Phase 3: Print side-by-side comparison ---
    print("\n" + "=" * 100)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 100)

    for i, r in enumerate(results):
        prefix_text = student_tok.decode(r["prefix_tokens"])
        student_cont_text = student_tok.decode(r["student_cont"])
        teacher_cont_text = r["teacher_cont"]

        print(f"\n{'='*100}")
        print(f"PROBLEM {i+1}: {r['problem'][:200]}")
        print(f"{'='*100}")
        print(f"\n[SHARED PREFIX - {len(r['prefix_tokens'])} tokens]")
        print(prefix_text)
        print(f"\n[STUDENT CONTINUATION - {len(r['student_cont'])} tokens]")
        print(student_cont_text)
        print(f"\n[TEACHER CONTINUATION from prefix]")
        print(teacher_cont_text)
        print(f"\n[TEACHER FROM SCRATCH - {r['teacher_full_len']} tokens]")
        print(r["teacher_full"])

    # Save to file
    output_path = "docs/continuation_comparison.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for i, r in enumerate(results):
            prefix_text = student_tok.decode(r["prefix_tokens"])
            student_cont_text = student_tok.decode(r["student_cont"])
            teacher_cont_text = r["teacher_cont"]

            f.write(f"\n{'='*100}\n")
            f.write(f"PROBLEM {i+1}: {r['problem']}\n")
            f.write(f"{'='*100}\n")
            f.write(f"\n[SHARED PREFIX - {len(r['prefix_tokens'])} tokens]\n")
            f.write(prefix_text + "\n")
            f.write(f"\n[STUDENT CONTINUATION - {len(r['student_cont'])} tokens]\n")
            f.write(student_cont_text + "\n")
            f.write(f"\n[TEACHER CONTINUATION from prefix]\n")
            f.write(teacher_cont_text + "\n")
            f.write(f"\n[TEACHER FROM SCRATCH - {r['teacher_full_len']} tokens]\n")
            f.write(r["teacher_full"] + "\n")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

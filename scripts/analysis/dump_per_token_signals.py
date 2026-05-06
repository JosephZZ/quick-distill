#!/usr/bin/env python3
"""Dump per-example per-token (KL_rev, H_s, H_t) on the same 100 MATH trajectories
used for signal_analysis_v2, so we can play with different top-k selection rules offline.

Output: docs/signal_analysis_v2/per_token_signals.npz
"""
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

STUDENT_MODEL = "Qwen/Qwen2.5-Math-1.5B"
TEACHER_MODEL = "Qwen/Qwen3-1.7B"
OUT_DIR = "/zhi_backup/ziheng/quick-distillation/docs/signal_analysis_v2"
TRAJ_PATH = os.path.join(OUT_DIR, "trajectories.json")
OUT_PATH = os.path.join(OUT_DIR, "per_token_signals.npz")

device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES remaps physical id

def build_prompt(problem_str, tok):
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": problem_str},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    print("loading trajectories", flush=True)
    trajectories = json.load(open(TRAJ_PATH))
    n = len(trajectories)
    max_len = max(t["response_len"] for t in trajectories)
    print(f"  n={n}, max_len={max_len}", flush=True)

    print("loading MATH-500", flush=True)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = [item["problem"] for item in ds][:100]

    print("loading tokenizer", flush=True)
    tok = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)

    print("loading student", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    print("loading teacher", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    kl_rev = np.full((n, max_len), np.nan, dtype=np.float32)
    h_s = np.full((n, max_len), np.nan, dtype=np.float32)
    h_t = np.full((n, max_len), np.nan, dtype=np.float32)
    resp_lens = np.zeros(n, dtype=np.int32)
    prompt_lens = np.zeros(n, dtype=np.int32)

    t0 = time.time()
    for i, traj in enumerate(trajectories):
        problem = problems[traj["problem_idx"]]
        prompt = build_prompt(problem, tok)
        prompt_ids = tok(prompt, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]
        prompt_lens[i] = prompt_len
        resp_len = traj["response_len"]
        resp_lens[i] = resp_len

        response_ids_tensor = torch.tensor([traj["response_ids"]])
        full_ids = torch.cat([prompt_ids, response_ids_tensor], dim=1).to(device)

        with torch.no_grad():
            s_logits = student(full_ids).logits[0]
            t_logits = teacher(full_ids).logits[0]

        for j in range(resp_len):
            logit_idx = prompt_len + j - 1
            s_lp = F.log_softmax(s_logits[logit_idx].float(), dim=-1)
            t_lp = F.log_softmax(t_logits[logit_idx].float(), dim=-1)
            s_p = torch.exp(s_lp)
            t_p = torch.exp(t_lp)

            mask_s = s_p > 1e-10
            mask_t = t_p > 1e-10

            kl_r = torch.sum(s_p[mask_s] * (s_lp[mask_s] - t_lp[mask_s])).item()
            ent_s = -torch.sum(s_p[mask_s] * s_lp[mask_s]).item()
            ent_t = -torch.sum(t_p[mask_t] * t_lp[mask_t]).item()

            kl_rev[i, j] = max(kl_r, 0.0)
            h_s[i, j] = ent_s
            h_t[i, j] = ent_t

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"  {i+1}/{n}  elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    print(f"done in {time.time()-t0:.1f}s", flush=True)
    np.savez(
        OUT_PATH,
        kl_rev=kl_rev.astype(np.float16),
        h_s=h_s.astype(np.float16),
        h_t=h_t.astype(np.float16),
        resp_len=resp_lens,
        prompt_len=prompt_lens,
    )
    print(f"wrote {OUT_PATH}", flush=True)

if __name__ == "__main__":
    main()

"""
Cross-family KL profile via vLLM (robust to custom modeling code in HF).

Supports BOTH same-tokenizer pairs (Qwen/Qwen) and cross-tokenizer pairs
(InternLM2.5 vs InternLM3, MiniCPM3 vs MiniCPM4, SmolLM2 vs SmolLM3, etc.).

For one (student, teacher) pair:
  1. Student vLLM generates trajectories on N math problems with logprobs.
  2. Tear down student, load teacher vLLM.
  3a. Same-tok: teacher scores each (prompt + sampled_response) using
      prompt_logprobs to get its log-probability of every sampled token.
  3b. Cross-tok: decode student trajectory to text -> re-tokenize with teacher
      -> teacher scores its own re-tokenized prompt+response with
      prompt_logprobs -> for each student response token, find the teacher
      token whose char span covers the student token's char midpoint, and use
      teacher's prompt_logprob of THAT teacher token.
  4. Per-position |s_lp - t_lp| computed (proxy for KL on the sampled token).

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/analysis/kl_profile_vllm.py \\
    --student_model internlm/internlm2_5-1_8b-chat \\
    --teacher_model internlm/internlm3-8b-instruct \\
    --num_problems 50 --max_new_tokens 512 \\
    --output docs/kl_profile_xfamily/internlm2.5-1.8B_internlm3-8B.json
"""

import argparse
import json
import os
import sys
import gc
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def encode_with_offsets(tok, text: str):
    """Return (input_ids, [(cs, ce), ...]) char offsets for each token.
    Uses fast-tokenizer offsets when available; otherwise reconstructs offsets
    via progressive prefix-decode (works for slow Python tokenizers like
    InternLM3 / MiniCPM3-4B which lack return_offsets_mapping support)."""
    try:
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    except (NotImplementedError, ValueError):
        ids = tok.encode(text, add_special_tokens=False)
        offsets = []
        prev_end = 0
        text_len = len(text)
        for i in range(len(ids)):
            prefix = tok.decode(ids[: i + 1], skip_special_tokens=False)
            end = min(len(prefix), text_len)
            if end < prev_end:
                end = prev_end
            offsets.append((prev_end, end))
            prev_end = end
        return ids, offsets


def build_chat_prompt(problem: str, tokenizer) -> str:
    """Apply chat template; system+user with math-style instruction."""
    sys_msg = "Please reason step by step, and put your final answer within \\boxed{}."
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": problem}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        msgs = [{"role": "user", "content": f"{sys_msg}\n\n{problem}"}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_model", required=True)
    ap.add_argument("--teacher_model", required=True)
    ap.add_argument("--num_problems", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.55)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    s_tok = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    t_tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    same_tok = (s_tok.vocab_size == t_tok.vocab_size and s_tok.get_vocab() == t_tok.get_vocab())
    print(f"[KL profile] student={args.student_model}  teacher={args.teacher_model}  same_tok={same_tok}")

    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    problems = [ds[i]["problem"] for i in range(args.num_problems)]
    s_prompts = [build_chat_prompt(p, s_tok) for p in problems]

    # ---- Phase 1: student generation with sampled-token logprobs ----
    print(f"\n[Phase 1] Spinning up student vLLM ({args.student_model}) ...")
    from vllm import LLM, SamplingParams
    student = LLM(
        model=args.student_model,
        tokenizer=args.student_model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        enforce_eager=False,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        logprobs=1,
    )
    print(f"[Phase 1] Generating {len(s_prompts)} trajectories ...")
    outs = student.generate(s_prompts, sp)

    trajs = []
    for o in outs:
        out0 = o.outputs[0]
        token_ids = list(out0.token_ids)
        s_lps = []
        for step_logprobs, tid in zip(out0.logprobs, token_ids):
            if step_logprobs is None or tid not in step_logprobs:
                s_lps.append(None)
            else:
                s_lps.append(float(step_logprobs[tid].logprob))
        trajs.append({
            "prompt_text": o.prompt,
            "prompt_ids": list(o.prompt_token_ids),
            "response_ids": token_ids,
            "student_lps": s_lps,
        })
    avg_resp_len = float(np.mean([len(t["response_ids"]) for t in trajs]))
    print(f"[Phase 1] Done. avg response len = {avg_resp_len:.1f}")

    del student
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Phase 2: teacher scoring ----
    print(f"\n[Phase 2] Spinning up teacher vLLM ({args.teacher_model}) ...")
    teacher = LLM(
        model=args.teacher_model,
        tokenizer=args.teacher_model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        enforce_eager=False,
    )

    sp2 = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)

    if same_tok:
        # Same tokenizer: teacher scores prompt + student_response_ids directly.
        score_prompts = []
        score_lens = []
        for t in trajs:
            full_ids = t["prompt_ids"] + t["response_ids"]
            score_prompts.append(full_ids)
            score_lens.append((len(t["prompt_ids"]), len(t["response_ids"])))

        print(f"[Phase 2] Scoring {len(score_prompts)} trajectories with teacher (same-tok) ...")
        score_outs = teacher.generate(
            [{"prompt_token_ids": ids} for ids in score_prompts],
            sampling_params=sp2,
        )

        for traj, so, (plen, rlen) in zip(trajs, score_outs, score_lens):
            t_lps = []
            for k in range(rlen):
                full_idx = plen + k
                sampled_tid = traj["response_ids"][k]
                ent = so.prompt_logprobs[full_idx] if full_idx < len(so.prompt_logprobs) else None
                if ent is None or sampled_tid not in ent:
                    t_lps.append(None)
                else:
                    t_lps.append(float(ent[sampled_tid].logprob))
            traj["teacher_lps"] = t_lps
            traj["pos_for_lp"] = list(range(rlen))  # student-position -> response position k
    else:
        # Cross tokenizer: decode student full-text, build teacher prompt+resp,
        # score with teacher prompt_logprobs, then for each student response
        # token find the teacher token covering its char midpoint.
        # IMPORTANT: re-build the teacher prompt with the teacher's own chat template,
        # then append the decoded student response text to it. This avoids feeding
        # the student's chat-formatted prompt verbatim to the teacher.
        teacher_inputs = []  # list of dicts with everything needed for alignment
        for traj, problem in zip(trajs, problems):
            # 1) Decode student response text from its sampled token IDs
            resp_text = s_tok.decode(traj["response_ids"], skip_special_tokens=False)

            # 2) Build teacher chat prompt (its own template)
            t_prompt_text = build_chat_prompt(problem, t_tok)
            full_teacher_text = t_prompt_text + resp_text

            # 3) Encode full teacher input with offsets (fast or slow tokenizer)
            t_full_ids, t_offsets = encode_with_offsets(t_tok, full_teacher_text)
            t_prompt_len = len(t_tok.encode(t_prompt_text, add_special_tokens=False))
            t_resp_char_start = len(t_prompt_text)

            # 4) For each student response token, compute its char span
            #    relative to the start of the response text. Tokenize the
            #    response standalone with offsets to get spans.
            try:
                _, s_resp_offsets = encode_with_offsets(s_tok, resp_text)
            except Exception:
                s_resp_offsets = None

            teacher_inputs.append({
                "t_full_ids": t_full_ids,
                "t_offsets": t_offsets,
                "t_prompt_len": t_prompt_len,
                "t_resp_char_start": t_resp_char_start,
                "s_resp_offsets": s_resp_offsets,  # offsets into resp_text only
            })

        score_prompts = [ti["t_full_ids"] for ti in teacher_inputs]
        print(f"[Phase 2] Scoring {len(score_prompts)} trajectories with teacher (cross-tok) ...")
        score_outs = teacher.generate(
            [{"prompt_token_ids": ids} for ids in score_prompts],
            sampling_params=sp2,
        )

        for traj, ti, so in zip(trajs, teacher_inputs, score_outs):
            # For each student response token, find teacher token whose offset
            # in t_offsets covers the same char midpoint within the response.
            t_full_ids = ti["t_full_ids"]
            t_offsets = ti["t_offsets"]
            t_prompt_len = ti["t_prompt_len"]
            t_resp_char_start = ti["t_resp_char_start"]
            s_resp_offsets = ti["s_resp_offsets"] or []

            t_lps = []
            for k, sampled_tid in enumerate(traj["response_ids"]):
                if k >= len(s_resp_offsets):
                    t_lps.append(None)
                    continue
                s_cs, s_ce = s_resp_offsets[k]
                if s_cs == s_ce:  # zero-width
                    t_lps.append(None)
                    continue
                mid = (s_cs + s_ce) // 2  # midpoint relative to resp_text
                target_char = t_resp_char_start + mid

                ti_match = None
                for ti_idx in range(t_prompt_len, len(t_offsets)):
                    t_cs, t_ce = t_offsets[ti_idx]
                    if t_cs <= target_char < t_ce:
                        ti_match = ti_idx
                        break
                    if t_cs > target_char:
                        ti_match = max(ti_idx - 1, t_prompt_len)
                        break
                if ti_match is None or ti_match >= len(t_full_ids):
                    t_lps.append(None)
                    continue

                # Teacher's logprob of the teacher token at position ti_match
                # within the full sequence so.prompt_logprobs.
                t_token_id = t_full_ids[ti_match]
                ent = so.prompt_logprobs[ti_match] if ti_match < len(so.prompt_logprobs) else None
                if ent is None or t_token_id not in ent:
                    t_lps.append(None)
                else:
                    t_lps.append(float(ent[t_token_id].logprob))
            traj["teacher_lps"] = t_lps

    # ---- Per-position aggregation ----
    print("\n[Aggregate] Per-position KL proxy = |s_lp - t_lp|")
    max_pos = args.max_new_tokens
    sums = np.zeros(max_pos)
    counts = np.zeros(max_pos)
    for t in trajs:
        for pos, (s, tt) in enumerate(zip(t["student_lps"], t["teacher_lps"])):
            if pos >= max_pos:
                break
            if s is None or tt is None:
                continue
            sums[pos] += abs(s - tt)
            counts[pos] += 1
    per_pos_mean = (sums / np.maximum(counts, 1)).tolist()
    valid = counts.tolist()

    # First 100 vs rest mean
    f100_num = sums[:100].sum(); f100_n = counts[:100].sum()
    rest_num = sums[100:].sum(); rest_n = counts[100:].sum()
    f100_mean = float(f100_num / max(f100_n, 1))
    rest_mean = float(rest_num / max(rest_n, 1))
    ratio = f100_mean / max(rest_mean, 1e-8)

    cum_total = float(sums.sum())
    cum_by_prefix = {}
    if cum_total > 0:
        for K in [10, 25, 50, 100, 150, 200, 300, 500]:
            if K <= max_pos:
                cum_by_prefix[K] = float(sums[:K].sum() / cum_total)

    out = {
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
        "same_tokenizer": bool(same_tok),
        "num_problems": args.num_problems,
        "max_new_tokens": args.max_new_tokens,
        "avg_response_len": avg_resp_len,
        "per_position_kl_proxy": per_pos_mean,
        "valid_count_per_position": valid,
        "first100_mean": f100_mean,
        "rest_mean": rest_mean,
        "first100_rest_ratio": float(ratio),
        "cumulative_kl_fraction": cum_by_prefix,
        "total_kl_proxy": cum_total,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[KL profile] Saved -> {args.output}")
    print(f"  first100 mean = {f100_mean:.4f}, rest mean = {rest_mean:.4f}, ratio = {ratio:.2f}x")
    print(f"  total = {cum_total:.2f}, cumulative fraction by prefix:")
    for K, frac in cum_by_prefix.items():
        print(f"    first {K:>4} tokens: {frac*100:.1f}%")


if __name__ == "__main__":
    main()

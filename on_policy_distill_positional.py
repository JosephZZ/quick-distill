"""
Position-Limited On-policy self-distillation: 
Modified to only distill first N tokens for efficiency testing.

Based on original on_policy_distill.py with position limitation added.
"""

import json
import argparse
import gc
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.nn.functional import log_softmax, kl_div
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from datasets import load_dataset
from tqdm import tqdm
import wandb


# --- Format-token classification (used by token_select_mode="format_mask") ---
# A token is "format" if its decoded string is structural punctuation, a
# LaTeX command, a math operator, or a pure number. Reasoning words
# ("planning") and English continuation words are NOT format and remain in
# the loss. Classification is keyed on the **student-emitted** token, so this
# is the same definition used in scripts/analysis/format_mask_threshold.py.

_PLANNING_WORDS = {
    "to","let","we","first","the","since","note","recall","now","next","then",
    "so","thus","hence","therefore","given","consider","suppose","assume",
    "because","if","for","by","from","using","applying","substituting",
    "simplifying","solving","calculating","computing","evaluating","finally",
    "step","answer","solution",
}
_MATH_OPS = {"+","-","*","/","=","<",">","^","_","≤","≥","≠","±","×","÷"}
_LATEX_HINTS = ("frac","sqrt","sum","int","lim","boxed","text","cdot","times")
_STRUCTURAL_LITERALS = {"**","##","#","---","```",":",";",",",".","!","?","(",")","[","]","{","}"}
_NUMBER_RE = re.compile(r'^-?\d+\.?\d*$')


def _classify_token_str(s: str) -> str:
    st = s.strip()
    if st.lower() in _PLANNING_WORDS:
        return "planning"
    if s in ("\n","\r\n","\r") or st == "":
        return "structural"
    if st in _STRUCTURAL_LITERALS:
        return "structural"
    if st.startswith("**") or st.startswith("##"):
        return "structural"
    if st.startswith("\\") and len(st) > 1:
        return "math_latex"
    if st in _MATH_OPS:
        return "math_operator"
    if st.replace(".","").replace(",","").isdigit():
        return "math_number"
    if any(x in st for x in _LATEX_HINTS):
        return "math_latex"
    if _NUMBER_RE.match(st):
        return "math_number"
    if len(st) <= 2 and not st.isalnum():
        return "structural"
    return "continuation"


# Categories considered "format" for the format_mask ablation.
# Recommendation from scripts/analysis/format_mask_threshold.py:
# {structural, math_latex, math_operator, math_number} -> 58.6% of tokens,
# 42.2% of total KL removed. Leaves planning + continuation (word-level
# reasoning content) for distillation.
_FORMAT_CATS = {"structural", "math_latex", "math_operator", "math_number"}


def build_format_token_mask(tokenizer, cats=None):
    """Return a BoolTensor of shape [vocab_size]: True iff token id is in 'format' cats.

    cats: iterable of category names treated as 'format' (masked out of loss).
          Default = _FORMAT_CATS (structural+math_latex+math_operator+math_number).
    """
    if cats is None:
        cats = _FORMAT_CATS
    cats = set(cats)
    vocab_size = len(tokenizer)
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    n_fmt = 0
    cat_counts = {"planning":0,"structural":0,"math_latex":0,"math_operator":0,"math_number":0,"continuation":0}
    for tid in range(vocab_size):
        try:
            s = tokenizer.decode([tid])
        except Exception:
            continue
        c = _classify_token_str(s)
        cat_counts[c] = cat_counts.get(c, 0) + 1
        if c in cats:
            mask[tid] = True
            n_fmt += 1
    print(f"[format_mask] cats={sorted(cats)}; format={n_fmt}/{vocab_size} ({n_fmt/vocab_size*100:.1f}%)")
    print(f"[format_mask] Per-category counts: {cat_counts}")
    return mask


def _supports_thinking(tokenizer):
    """Check if tokenizer supports enable_thinking parameter (Qwen3, Gemma3, etc.)."""
    try:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        return True
    except TypeError:
        return False


def _supports_system_role(tokenizer):
    """Check if tokenizer's chat template supports system role."""
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "test"}, {"role": "user", "content": "test"}],
            tokenize=False, add_generation_prompt=True,
        )
        return True
    except Exception:
        return False


def build_prompt(problem: str, tokenizer, system_prompt: str = None, enable_thinking: bool = False) -> str:
    if system_prompt is None:
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    if _supports_system_role(tokenizer):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
        ]
    else:
        # Prepend system prompt to user message for models without system role (e.g. Gemma 2)
        messages = [
            {"role": "user", "content": system_prompt + "\n\n" + problem},
        ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if _supports_thinking(tokenizer):
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **kwargs)


def query_teacher_hf(teacher_model, trajectories, nothink_ids=None, device="cuda:1"):
    """Compute teacher logprobs using HF forward pass. One trajectory at a time."""
    all_logprobs = []
    for traj in trajectories:
        prompt_ids = traj["prompt_ids"]
        response_ids = traj["response_ids"]

        # Concatenate: prompt + nothink + response
        full_ids = prompt_ids + nothink_ids + response_ids
        resp_start = len(prompt_ids) + len(nothink_ids)

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = teacher_model(input_ids=input_ids)
            logits = outputs.logits

        # Get log probs for response tokens
        shift_logits = logits[0, :-1, :]  # [seq-1, vocab]
        shift_labels = input_ids[0, 1:]   # [seq-1]
        log_probs_all = log_softmax(shift_logits.float(), dim=-1)
        sampled_log_probs = log_probs_all.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        start = resp_start - 1
        end = start + len(response_ids)
        response_logprobs = sampled_log_probs[start:end].float().cpu().tolist()
        all_logprobs.append(response_logprobs)
        
        del input_ids, outputs
        torch.cuda.empty_cache()

    return all_logprobs


def query_teacher_hf_logits(teacher_model, traj, nothink_ids=None, device="cuda:1"):
    """Compute teacher log-probs (full vocab) for response positions. Returns [resp_len, vocab] on CPU."""
    prompt_ids = traj["prompt_ids"]
    response_ids = traj["response_ids"]

    # Concatenate: prompt + nothink + response
    full_ids = prompt_ids + nothink_ids + response_ids
    resp_start = len(prompt_ids) + len(nothink_ids)

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = teacher_model(input_ids=input_ids)
        logits = outputs.logits

    # shift_logits[i] predicts token at position i+1
    shift_logits = logits[0, :-1, :]  # [seq-1, vocab]
    log_probs_all = log_softmax(shift_logits.float(), dim=-1)

    start = resp_start - 1
    end = start + len(response_ids)
    teacher_log_probs = log_probs_all[start:end].cpu()  # [resp_len, vocab] on CPU

    del input_ids, outputs
    torch.cuda.empty_cache()

    return teacher_log_probs


def query_teacher_hf_logits_batch(teacher_model, trajs, nothink_ids, position_limit, device="cuda:1", micro_bs=0):
    """Batch teacher scoring. Returns list of [effective_len, vocab] tensors on CPU.
    micro_bs: if > 0, process in micro-batches to avoid OOM on long sequences."""
    pad_token_id = 0
    # Build padded batch
    all_full_ids = []
    resp_starts = []
    effective_lens = []
    for traj in trajs:
        prompt_ids = traj["prompt_ids"]
        response_ids = traj["response_ids"]
        resp_len = len(response_ids)
        effective_len = min(resp_len, position_limit) if position_limit > 0 else resp_len
        # Only keep tokens up to position_limit in response
        full_ids = prompt_ids + nothink_ids + response_ids[:effective_len]
        all_full_ids.append(full_ids)
        resp_starts.append(len(prompt_ids) + len(nothink_ids))
        effective_lens.append(effective_len)

    # If micro_bs > 0, process in chunks to avoid OOM
    if micro_bs > 0 and len(trajs) > micro_bs:
        results = []
        for mb_start in range(0, len(trajs), micro_bs):
            mb_end = min(mb_start + micro_bs, len(trajs))
            mb_ids = all_full_ids[mb_start:mb_end]
            mb_resp_starts = resp_starts[mb_start:mb_end]
            mb_eff_lens = effective_lens[mb_start:mb_end]

            max_len = max(len(ids) for ids in mb_ids)
            padded = [ids + [pad_token_id] * (max_len - len(ids)) for ids in mb_ids]
            attention_mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in mb_ids]

            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            attn_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = teacher_model(input_ids=input_ids, attention_mask=attn_mask)
                logits = outputs.logits

            for i in range(mb_end - mb_start):
                shift_logits = logits[i, :-1, :]
                log_probs_all = log_softmax(shift_logits.float(), dim=-1)
                start = mb_resp_starts[i] - 1
                end = start + mb_eff_lens[i]
                results.append(log_probs_all[start:end].cpu())

            del input_ids, attn_mask, outputs, logits
            torch.cuda.empty_cache()
        return results

    # Original batch path
    max_len = max(len(ids) for ids in all_full_ids)
    padded = [ids + [pad_token_id] * (max_len - len(ids)) for ids in all_full_ids]
    attention_mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in all_full_ids]

    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    attn_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = teacher_model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs.logits  # [B, seq, vocab]

    # Extract per-traj teacher logprobs for response positions
    results = []
    for i in range(len(trajs)):
        shift_logits = logits[i, :-1, :]
        log_probs_all = log_softmax(shift_logits.float(), dim=-1)
        start = resp_starts[i] - 1
        end = start + effective_lens[i]
        results.append(log_probs_all[start:end].cpu())

    del input_ids, attn_mask, outputs, logits
    torch.cuda.empty_cache()
    return results


def query_teacher_cross_tokenizer(teacher_model, trajs, student_tokenizer, teacher_tokenizer,
                                    nothink_ids, position_limit, device="cuda:1"):
    """Score trajectories with teacher using correct cross-tokenizer handling.
    1. Decode student token IDs → text using student tokenizer
    2. Re-tokenize with teacher tokenizer (with chat template)
    3. Teacher forward pass on teacher token IDs
    4. Map teacher log-probs back to student token positions via character alignment
    Returns list of [effective_len, teacher_vocab] tensors on CPU."""
    results = []
    for traj in trajs:
        prompt_ids = traj["prompt_ids"]
        response_ids = traj["response_ids"]
        resp_len = len(response_ids)
        effective_len = min(resp_len, position_limit) if position_limit > 0 else resp_len

        # Decode student's full sequence to text
        full_student_ids = prompt_ids + response_ids[:effective_len]
        full_text = student_tokenizer.decode(full_student_ids, skip_special_tokens=False)

        # Get character offsets for student tokens (response portion only)
        # Decode prompt separately to find where response starts in text
        prompt_text = student_tokenizer.decode(prompt_ids, skip_special_tokens=False)
        resp_text_start = len(prompt_text)

        # Re-tokenize the full text with teacher tokenizer
        teacher_enc = teacher_tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
        teacher_ids = teacher_enc["input_ids"]

        # Prepend nothink_ids after finding the prompt/response boundary
        # Find which teacher token starts the response (by character offset)
        offsets = teacher_enc["offset_mapping"]
        t_resp_start = 0
        for idx, (s, e) in enumerate(offsets):
            if s >= resp_text_start:
                t_resp_start = idx
                break

        # Insert nothink_ids at the response boundary
        teacher_full_ids = teacher_ids[:t_resp_start] + nothink_ids + teacher_ids[t_resp_start:]
        t_resp_start_adj = t_resp_start + len(nothink_ids)

        # Validate token IDs are within teacher model's vocab
        t_vocab_size = getattr(teacher_model.config, 'vocab_size', None) or getattr(teacher_model.config, 'text_config', None) and teacher_model.config.text_config.vocab_size or 262208
        teacher_full_ids = [min(tid, t_vocab_size - 1) for tid in teacher_full_ids]

        # Teacher forward pass (with error handling for edge cases)
        input_ids_t = torch.tensor([teacher_full_ids], dtype=torch.long, device=device)
        try:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = teacher_model(input_ids=input_ids_t)
                logits = outputs.logits
            shift_logits = logits[0, :-1, :]
            t_log_probs = log_softmax(shift_logits.float(), dim=-1)
        except (RuntimeError, IndexError) as e:
            # Skip this trajectory if teacher scoring fails
            del input_ids_t
            torch.cuda.empty_cache()
            results.append(torch.zeros(effective_len, t_vocab_size))
            continue

        # Map teacher positions back to student positions via character offsets
        # For each student response token, find the teacher token that covers the same text
        s_enc = student_tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
        s_offsets = s_enc["offset_mapping"]
        s_resp_start = len(prompt_ids)  # student token index where response starts

        # Teacher token offsets (excluding nothink insertion)
        t_offsets = offsets  # original teacher offsets before nothink insertion

        # Build student-to-teacher position mapping
        mapped_teacher_lps = []
        for si in range(s_resp_start, min(s_resp_start + effective_len, len(s_offsets))):
            s_char_start, s_char_end = s_offsets[si]
            s_char_mid = (s_char_start + s_char_end) // 2

            # Find teacher token covering this character position
            best_ti = t_resp_start  # default
            for ti in range(t_resp_start, len(t_offsets)):
                t_cs, t_ce = t_offsets[ti]
                if t_cs <= s_char_mid < t_ce:
                    best_ti = ti
                    break
                if t_cs > s_char_mid:
                    best_ti = max(ti - 1, t_resp_start)
                    break

            # Teacher log-probs at this position (adjusted for nothink insertion)
            # shift_logits[pos] predicts token at pos+1, so the log-prob for position
            # best_ti is at shift index (best_ti + len(nothink_ids) - 1) for teacher_full_ids
            t_shift_idx = best_ti + len(nothink_ids) - 1
            t_shift_idx = max(0, min(t_shift_idx, t_log_probs.shape[0] - 1))
            mapped_teacher_lps.append(t_log_probs[t_shift_idx].cpu())

        vocab_size = t_log_probs.shape[-1]
        if mapped_teacher_lps:
            result_tensor = torch.stack(mapped_teacher_lps)  # [mapped_len, teacher_vocab]
            # Ensure exact effective_len output
            if result_tensor.shape[0] < effective_len:
                # Pad with uniform distribution
                pad = torch.full((effective_len - result_tensor.shape[0], vocab_size),
                                  -float('inf'))
                pad[:, 0] = 0.0  # dummy: put all prob on token 0
                result_tensor = torch.cat([result_tensor, pad], dim=0)
            elif result_tensor.shape[0] > effective_len:
                result_tensor = result_tensor[:effective_len]
            results.append(result_tensor)
        else:
            results.append(torch.zeros(effective_len, vocab_size))

        del input_ids_t, outputs, logits
        torch.cuda.empty_cache()

    return results


def generate_hf(student, tokenizer, problems, n_samples, max_new_tokens, temperature, gen_batch_size=0, system_prompt=None, enable_thinking=False):
    """Generate trajectories using HF model.generate() directly — no subprocess or disk I/O."""
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or 0
    special_ids = {eos_id, pad_id, 151645, 151643}

    # Build all prompts: each problem repeated n_samples times
    all_prompts = []
    problem_indices = []  # track which problem each prompt belongs to
    for i, problem in enumerate(problems):
        prompt_text = build_prompt(problem, tokenizer, system_prompt=system_prompt, enable_thinking=enable_thinking)
        for _ in range(n_samples):
            all_prompts.append(prompt_text)
            problem_indices.append(i)

    # Tokenize all prompts (left-pad for batch generation)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Generate in sub-batches if gen_batch_size > 0
    batch_sz = gen_batch_size if gen_batch_size > 0 else len(all_prompts)
    all_trajectories = {}

    def _generate_batch(prompts_subset, base_offset):
        """Generate for a batch of prompts, splitting on OOM."""
        inputs = tokenizer(prompts_subset, return_tensors="pt", padding=True).to(student.device)
        try:
            with torch.no_grad():
                outputs = student.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                )
            # Parse outputs into trajectories grouped by problem
            for seq_idx in range(len(prompts_subset)):
                global_idx = base_offset + seq_idx
                prob_idx = problem_indices[global_idx]
                full_output = outputs[seq_idx].tolist()
                pad_len = (inputs["attention_mask"][seq_idx] == 0).sum().item()
                input_len = inputs["attention_mask"][seq_idx].sum().item()
                prompt_ids = full_output[pad_len:pad_len + input_len]
                response_ids = full_output[pad_len + input_len:]
                while response_ids and response_ids[-1] in special_ids:
                    response_ids.pop()
                if len(response_ids) == 0:
                    continue
                prob_key = str(prob_idx)
                if prob_key not in all_trajectories:
                    all_trajectories[prob_key] = []
                all_trajectories[prob_key].append({
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                    "full_ids": prompt_ids + response_ids,
                })
            del outputs, inputs
        except torch.cuda.OutOfMemoryError:
            del inputs
            torch.cuda.empty_cache()
            if len(prompts_subset) <= 1:
                print(f"  WARNING: OOM on single prompt (generate), skipping")
                return
            mid = len(prompts_subset) // 2
            print(f"  WARNING: OOM during generate (batch={len(prompts_subset)}), splitting to {mid}+{len(prompts_subset)-mid}")
            _generate_batch(prompts_subset[:mid], base_offset)
            _generate_batch(prompts_subset[mid:], base_offset + mid)

    student.eval()
    for batch_start in range(0, len(all_prompts), batch_sz):
        batch_end = min(batch_start + batch_sz, len(all_prompts))
        batch_prompts = all_prompts[batch_start:batch_end]
        _generate_batch(batch_prompts, batch_start)
        torch.cuda.empty_cache()

    student.train()
    torch.cuda.empty_cache()
    tokenizer.padding_side = "right"  # restore default
    return all_trajectories


###############################################################################
# SGLang persistent server helpers
###############################################################################

_sglang_process = None
_sglang_port = None


def start_sglang_server(model_path, tokenizer_name, gpu_memory_utilization=0.50, port=30000, gpu_id=None):
    """Launch a persistent SGLang server. Returns (process, port)."""
    global _sglang_process, _sglang_port
    if _sglang_process is not None:
        return _sglang_process, _sglang_port

    import requests as _req
    env = os.environ.copy()
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    # Ensure ninja and conda bin on PATH for JIT compilation
    conda_bin = os.path.dirname(sys.executable)
    env["PATH"] = f"/usr/local/cuda-12.6/bin:{conda_bin}:/home/ziheng/.local/bin:" + env.get("PATH", "")
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(_get_physical_gpu_id(gpu_id))

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--tokenizer-path", tokenizer_name,
        "--port", str(port),
        "--mem-fraction-static", str(gpu_memory_utilization),
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--disable-cuda-graph",
        "--context-length", "8192",
    ]
    print(f"  Starting SGLang server on port {port} (gpu_util={gpu_memory_utilization}, gpu={gpu_id})...")
    sglang_log = open(os.path.join(os.environ.get("HOME", "."), "sglang_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=sglang_log, stderr=subprocess.STDOUT)

    # Wait for server to be ready (up to 120s)
    for i in range(120):
        try:
            r = _req.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                print(f"  SGLang server ready (took {i+1}s)")
                _sglang_process = proc
                _sglang_port = port
                return proc, port
        except Exception:
            pass
        time.sleep(1)
        if proc.poll() is not None:
            raise RuntimeError(f"SGLang server exited with code {proc.returncode}")

    proc.kill()
    raise RuntimeError("SGLang server failed to start within 120s")


def sglang_release_memory(port=None):
    """Tell SGLang to release GPU memory so training can use it."""
    import requests as _req
    port = port or _sglang_port
    r = _req.post(f"http://localhost:{port}/release_memory_occupation", json={}, timeout=30)
    if r.status_code != 200:
        print(f"  WARNING: SGLang release_memory failed: {r.text}")
        return False
    print("  SGLang: released GPU memory")
    return True


def sglang_resume_memory(port=None):
    """Tell SGLang to resume GPU memory occupation for inference."""
    import requests as _req
    port = port or _sglang_port
    r = _req.post(f"http://localhost:{port}/resume_memory_occupation", json={}, timeout=30)
    if r.status_code != 200:
        print(f"  WARNING: SGLang resume_memory failed: {r.text}")
        return False
    print("  SGLang: resumed GPU memory")
    return True


def update_sglang_weights(model_path, port=None):
    """Update the running SGLang server's model weights from a saved checkpoint."""
    import requests as _req
    port = port or _sglang_port
    r = _req.post(f"http://localhost:{port}/update_weights_from_disk",
                  json={"model_path": os.path.abspath(model_path)}, timeout=120)
    if r.status_code != 200 or not r.json().get("success", False):
        print(f"  WARNING: SGLang weight update failed: {r.text}")
        return False
    return True


def generate_chunk_sglang(problems, n_samples, max_new_tokens, temperature,
                          tokenizer, system_prompt=None, port=None):
    """Generate trajectories using the persistent SGLang server."""
    import requests as _req
    port = port or _sglang_port

    # Build prompts
    prompts = []
    for problem in problems:
        prompt = build_prompt(problem, tokenizer, system_prompt)
        prompts.extend([prompt] * n_samples)

    # Send batch request
    r = _req.post(f"http://localhost:{port}/generate", json={
        "text": prompts,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "skip_special_tokens": False,
        },
    }, timeout=3600)

    if r.status_code != 200:
        print(f"  SGLang generate failed: {r.status_code} {r.text[:200]}")
        return None

    outputs = r.json()

    # Parse into trajectory format matching vLLM output
    all_trajectories = {}
    for i, problem in enumerate(problems):
        trajs = []
        for j in range(n_samples):
            idx = i * n_samples + j
            text = outputs[idx]["text"] if isinstance(outputs[idx], dict) else outputs[idx]
            prompt_text = prompts[idx]
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            response_ids = tokenizer.encode(text, add_special_tokens=False)
            trajs.append({
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "text": text,
            })
        all_trajectories[str(i)] = trajs

    return all_trajectories


def stop_sglang_server():
    """Stop the persistent SGLang server."""
    global _sglang_process, _sglang_port
    if _sglang_process is not None:
        _sglang_process.kill()
        _sglang_process.wait()
        _sglang_process = None
        _sglang_port = None
        gc.collect()
        torch.cuda.empty_cache()


def generate_chunk_vllm(model_path, tokenizer_name, problems, n_samples, max_new_tokens, temperature, output_file, gpu_id=0, max_retries=3, mem_threshold_mb=500, gpu_memory_utilization=0.90, system_prompt=None):
    """Generate trajectories for a chunk of problems using vLLM subprocess."""
    output_file = os.path.abspath(output_file)
    if os.path.exists(model_path):
        model_path = os.path.abspath(model_path)
    problems_file = output_file + ".problems.json"
    with open(problems_file, "w") as f:
        json.dump(problems, f)

    env = os.environ.copy()
    # Always set CUDA_VISIBLE_DEVICES for vLLM subprocess to the correct physical GPU
    # Map through parent's CUDA_VISIBLE_DEVICES if set (e.g., "0,1" with gpu_id=1 → "1")
    physical_gpu = _get_physical_gpu_id(gpu_id)
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)

    cmd = [
        sys.executable, "vllm_generate.py",
        "--model", model_path,
        "--tokenizer", tokenizer_name,
        "--problems_file", problems_file,
        "--output_file", output_file,
        "--n_samples", str(n_samples),
        "--max_new_tokens", str(max_new_tokens),
        "--temperature", str(temperature),
        "--gpu_memory_utilization", str(gpu_memory_utilization),
    ]
    if system_prompt is not None:
        cmd.extend(["--system_prompt", system_prompt])

    for attempt in range(max_retries):
        if attempt > 0:
            print(f"  Retry {attempt+1}/{max_retries}...")
        result = subprocess.run(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr,
                                cwd=os.path.dirname(os.path.abspath(__file__)))
        # Always clean up after vLLM subprocess
        _kill_orphan_vllm(gpu_id, mem_threshold_mb=mem_threshold_mb)
        if result.returncode == 0:
            break
        print(f"  vLLM generate attempt {attempt+1} failed (exit code {result.returncode})")

    os.remove(problems_file)

    if result.returncode != 0:
        print(f"  vLLM generate failed after {max_retries} attempts")
        return None

    with open(output_file, "r") as f:
        all_trajectories = json.load(f)

    return all_trajectories


def _get_physical_gpu_id(logical_id=0):
    """Get the physical GPU ID from CUDA_VISIBLE_DEVICES, or return logical_id if unset."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is not None and cvd.strip():
        ids = [x.strip() for x in cvd.split(",")]
        if logical_id < len(ids):
            return ids[logical_id]
    return str(logical_id)


def _kill_orphan_vllm(gpu_id, mem_threshold_mb=500):
    """Kill any leftover VLLM::EngineCore processes on the given GPU and wait for memory release."""
    physical_id = _get_physical_gpu_id(gpu_id)
    import signal
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader",
             f"--id={physical_id}"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.strip().split("\n"):
            if "VLLM" in line or "vllm" in line:
                pid = int(line.split(",")[0].strip())
                try:
                    os.kill(pid, signal.SIGKILL)
                    print(f"  Killed orphan vLLM process {pid}")
                except ProcessLookupError:
                    pass
    except Exception:
        pass

    # Wait for GPU memory to drop below threshold (up to 15s)
    for _ in range(30):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                 f"--id={physical_id}"],
                capture_output=True, text=True, timeout=5,
            )
            mem_mb = int(result.stdout.strip())
            if mem_mb < mem_threshold_mb:
                return
        except Exception:
            pass
        time.sleep(0.5)
    print(f"  Warning: GPU {physical_id} memory not fully freed")


def run_eval_math500(model_path, output_dir, tokenizer_name, n_samples=4, gpu_id=0):
    """Run MATH-500 eval using vLLM as a subprocess."""
    env = os.environ.copy()
    # Only override CUDA_VISIBLE_DEVICES if not already set
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, "eval_math500.py",
        "--model", model_path,
        "--output_dir", output_dir,
        "--n_samples", str(n_samples),
        "--temperature", "0.7",
        "--max_model_len", "4096",
        "--gpu_memory_utilization", "0.70",
    ]

    print(f"  Running MATH-500 eval (avg@{n_samples}) on GPU {gpu_id}...")
    result = subprocess.run(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr,
                            cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"  Eval failed (exit code {result.returncode})")
        return None

    summary_file = os.path.join(output_dir, "summary.json")
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            return json.load(f)
    return None


def save_merged_model(student, tokenizer, merged_path):
    """Save merged LoRA model."""
    os.makedirs(merged_path, exist_ok=True)
    student.merge_adapter()
    # Get merged state dict, strip PEFT key prefixes, and filter out LoRA-only keys
    merged_sd = student.base_model.model.state_dict()
    clean_sd = {}
    for k, v in merged_sd.items():
        # Skip LoRA-specific parameters (already merged into base weights)
        if any(lora_key in k for lora_key in ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']):
            continue
        clean_k = k.replace('base_model.model.', '') if k.startswith('base_model.model.') else k
        # Strip PEFT's .base_layer. wrapper from key names
        clean_k = clean_k.replace('.base_layer.', '.')
        # Clone to CPU to break shared memory (tied embeddings) and avoid GPU OOM
        clean_sd[clean_k] = v.detach().cpu().clone()
    # Unmerge adapter so LoRA training can continue correctly
    student.unmerge_adapter()
    # Save config + weights + tokenizer
    student.base_model.model.config.save_pretrained(merged_path)
    from safetensors.torch import save_file
    save_file(clean_sd, os.path.join(merged_path, "model.safetensors"))
    tokenizer.save_pretrained(merged_path)
    del clean_sd
    return merged_path


def build_vocab_mapping(student_tokenizer, teacher_tokenizer):
    """Build mapping from teacher vocab to student vocab for cross-tokenizer distillation.
    Returns (teacher_to_student_idx, valid_mask):
      - teacher_to_student_idx: LongTensor [teacher_vocab] mapping teacher ids to student ids
      - valid_mask: BoolTensor [teacher_vocab] indicating which teacher ids have a student equivalent
    Only tokens present in both vocabs are used for KL; others are masked out."""
    s_vocab = student_tokenizer.get_vocab()  # str → id
    t_vocab = teacher_tokenizer.get_vocab()  # str → id

    t_size = max(t_vocab.values()) + 1
    mapping = torch.zeros(t_size, dtype=torch.long)
    mask = torch.zeros(t_size, dtype=torch.bool)

    mapped = 0
    for tok_str, t_id in t_vocab.items():
        if tok_str in s_vocab:
            mapping[t_id] = s_vocab[tok_str]
            mask[t_id] = True
            mapped += 1

    # Also compute the sorted unique student IDs that have mappings
    mapped_student_ids = sorted(set(mapping[mask].tolist()))
    mapped_student_ids = torch.tensor(mapped_student_ids, dtype=torch.long)

    print(f"  Vocab mapping: {mapped}/{len(t_vocab)} teacher tokens mapped to student ({mapped/len(t_vocab):.1%})")
    print(f"  Mapped student token coverage: {len(mapped_student_ids)}/{len(s_vocab)} ({len(mapped_student_ids)/len(s_vocab):.1%})")
    return mapping, mask, mapped_student_ids


def remap_teacher_logprobs(t_log_probs, vocab_mapping, valid_mask, student_vocab_size, mapped_student_ids=None):
    """Remap teacher log-probs from teacher vocab order to student vocab order.
    t_log_probs: [seq_len, teacher_vocab] log-probabilities
    If mapped_student_ids is provided, returns [seq_len, len(mapped_student_ids)] log-probs
    over ONLY the shared vocabulary (both teacher and student distributions renormalized).
    Otherwise returns [seq_len, student_vocab] with unmapped positions at -inf."""
    device = t_log_probs.device

    # Work on CPU to avoid CUDA assert issues, then move back
    t_lp = t_log_probs.float().cpu()
    mapping = vocab_mapping
    mask = valid_mask

    # Truncate if teacher vocab > mapping size
    t_size = min(t_lp.shape[-1], mapping.shape[0])
    t_lp = t_lp[..., :t_size]
    mapping_t = mapping[:t_size]
    mask_t = mask[:t_size]

    seq_len = t_lp.shape[0]
    valid_idx = mask_t.nonzero(as_tuple=True)[0]

    # Step 1: extract mapped teacher logprobs and renormalize over mapped set only
    t_lp_mapped = t_lp[:, valid_idx]  # [seq_len, n_mapped_teacher]
    t_lp_normed = log_softmax(t_lp_mapped, dim=-1)

    # Step 2: scatter into student vocab positions
    result = torch.full((seq_len, student_vocab_size), float('-inf'))
    student_ids = mapping_t[valid_idx]  # student token IDs for each mapped teacher token
    result[:, student_ids] = t_lp_normed

    # Step 3: if mapped_student_ids provided, slice to shared vocab only
    # This ensures KL is computed only over shared tokens (no penalty for unmapped tokens)
    if mapped_student_ids is not None:
        result = result[:, mapped_student_ids]
        # Renormalize over shared vocab
        result = log_softmax(result, dim=-1)

    return result.to(device)


def load_student(base_model_name, lora_path, device, lora_config=None, full_finetune=False):
    """Load student model (base + optional LoRA, or full finetune)."""
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    if full_finetune:
        student = base.to(device)
        student.gradient_checkpointing_enable()
        return student
    if lora_path and os.path.exists(lora_path):
        student = PeftModel.from_pretrained(base, lora_path, is_trainable=True).to(device)
    else:
        student = get_peft_model(base, lora_config).to(device)
    student.enable_input_require_grads()
    student.gradient_checkpointing_enable()
    return student


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="checkpoints/on-policy-distill-positional")
    parser.add_argument("--dataset", type=str, default="AI-MO/NuminaMath-CoT")
    parser.add_argument("--num_problems", type=int, default=1000)
    parser.add_argument("--bs", type=int, default=16, help="Training batch size (number of trajectories per optimizer step)")
    parser.add_argument("--mini_bs", type=int, default=0, help="Mini-batch size for gradient accumulation (0=same as bs, i.e. no accumulation)")
    parser.add_argument("--n_samples", type=int, default=16, help="Trajectories per problem")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated linear module names (last segment) to wrap with LoRA.")
    parser.add_argument("--loss_type", type=str, default="reverse_kl",
                        choices=["reverse_kl", "dft_distill", "dft_distill_deadzone"])
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=20)  # Changed to save every 20 steps
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--eval_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="dft-distill-positional")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--student_gpu", type=int, default=0)
    parser.add_argument("--teacher_gpu", type=int, default=1)
    parser.add_argument("--vllm_gpu", type=int, default=None,
                       help="GPU for vLLM/SGLang generation (default: same as student_gpu). "
                            "Set to a separate GPU to avoid model offloading.")
    
    # New parameter: position limit
    parser.add_argument("--position_limit", type=int, default=50,
                       help="Only distill first N tokens (0=disabled, default=50)")
    parser.add_argument(
        "--token_select_mode",
        type=str,
        default="prefix",
        choices=["prefix", "top_kl", "top_entropy_student", "top_entropy_teacher", "random",
                 "middle", "last", "reopold", "ent_and", "ent_or", "ent_kl_and", "format_mask",
                 "hi_kl_hi_surp", "hi_kl_hi_surp_topk", "hi_kl_hi_ent_topk", "raw_product_topk", "hi_surp", "hi_ent"],
        help="Token selection: prefix, top_kl, top_entropy_student/teacher, random, middle, last, "
             "reopold, ent_and (teacher*student entropy product), ent_or (teacher+student entropy sum), "
             "ent_kl_and (high entropy AND high KL), "
             "format_mask (full-seq with student-emitted format tokens masked out), "
             "hi_kl_hi_surp (full-seq, keep only positions where per-batch KL > p75 AND -s_lp > p75), "
             "hi_kl_hi_surp_topk (per-trajectory top-K by joint score = KL * surprise; K = position_limit), "
             "hi_kl_hi_ent_topk (per-trajectory top-K by joint score = KL * full-vocab entropy; "
             "K = position_limit, or floor(response_len * top_k_frac) if top_k_frac > 0), "
             "hi_surp (drop low-surprise positions: keep where -s_lp > hi_surp_quantile p; "
             "if position_limit>0, restrict the rule to first N positions, else full-seq), "
             "hi_ent (same shape as hi_surp but uses full-vocab entropy "
             "H(p)=-sum p log p instead of -log p_sampled; threshold = hi_ent_quantile)",
    )
    parser.add_argument("--format_mask_cats", type=str, default=None,
                       help="format_mask: comma-separated category names to treat as 'format' "
                            "(masked out of loss). Choices: structural,math_latex,math_operator,"
                            "math_number,planning,continuation. Default = "
                            "'structural,math_latex,math_operator,math_number'. "
                            "E.g. --format_mask_cats=structural,math_latex masks only those two.")
    parser.add_argument("--hi_kl_quantile", type=float, default=0.75,
                       help="hi_kl_hi_surp: quantile threshold on per-position KL (default 0.75)")
    parser.add_argument("--hi_surp_quantile", type=float, default=0.75,
                       help="hi_kl_hi_surp / hi_surp: quantile threshold on per-position surprise (default 0.75)")
    parser.add_argument("--hi_ent_quantile", type=float, default=0.25,
                       help="hi_ent: drop positions where full-vocab entropy H(p) <= this quantile "
                            "of valid positions in the active region (default 0.25 = drop bottom 25%%, "
                            "chosen from prefix-100 H distribution analysis: p25=0.031 ≈ deterministic "
                            "format/structural tokens). Ignored if --hi_ent_threshold > 0.")
    parser.add_argument("--hi_ent_threshold", type=float, default=0.0,
                       help="hi_ent: absolute H threshold (drop positions where H(p) <= threshold). "
                            "If > 0, takes precedence over --hi_ent_quantile. "
                            "Reference: H=0.01 -> top-1 prob ≈ 99%% (deterministic format/structural).")
    parser.add_argument("--top_k_frac", type=float, default=0.0,
                       help="hi_kl_hi_*_topk: if >0, K per trajectory = floor(response_len * top_k_frac); "
                            "else K = position_limit (default 0.0 = use position_limit)")
    parser.add_argument("--reopold_beta", type=float, default=0.2,
                       help="REOPOLD: fraction of highest-entropy tokens to keep (default 0.2 = top 20%%)")
    parser.add_argument("--reward_clip", type=float, default=0.0,
                       help="Clip per-token KL from above at this value (0=disabled). REOPOLD uses ~5.0")
    parser.add_argument("--progressive_position", action="store_true",
                       help="Linearly increase position_limit from 1 to num_problems over training")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume from a LoRA checkpoint directory (e.g. checkpoints/.../step_100)")
    parser.add_argument("--teacher_micro_bs", type=int, default=0,
                       help="Micro-batch size for teacher scoring (0=all at once, >0 for OOM prevention with long sequences)")
    parser.add_argument("--gen_batch_size", type=int, default=0,
                       help="Batch size for HF generation (0=all at once, >0 to generate in sub-batches)")
    parser.add_argument("--use_vllm", action="store_true",
                       help="Use vLLM subprocess for generation (faster but requires GPU offload)")
    parser.add_argument("--use_sglang", action="store_true",
                       help="Use persistent SGLang server for generation (fastest, no offload needed)")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.90,
                       help="GPU memory utilization for vLLM/SGLang (default 0.90)")
    parser.add_argument("--full_finetune", action="store_true",
                       help="Full finetune (no LoRA). All parameters are trainable.")
    parser.add_argument("--fresh_scheduler", action="store_true",
                       help="When resuming, use a fresh LR schedule instead of restoring the old one")
    parser.add_argument("--problem_field", type=str, default="problem",
                       help="Field name for problem text in dataset (default: 'problem' for NuminaMath)")
    parser.add_argument("--system_prompt", type=str, default=None,
                       help="System prompt for generation (default: math reasoning prompt)")
    parser.add_argument("--single_gpu", action="store_true",
                       help="Single-GPU mode: only one model on GPU at a time (student/teacher/sglang take turns)")
    parser.add_argument("--enable_thinking", action="store_true",
                       help="Enable thinking/CoT mode for models that support it (Qwen3, Gemma3). "
                            "Student generates with thinking, teacher scores with thinking.")

    args = parser.parse_args()
    if args.vllm_gpu is None:
        args.vllm_gpu = args.student_gpu
    # Auto-detect single-GPU mode when all GPUs are the same
    single_gpu = args.single_gpu or (args.student_gpu == args.teacher_gpu == args.vllm_gpu)
    if single_gpu:
        print("Single-GPU mode: models will take turns on the GPU")
    _fullseq_modes = ("prefix", "reopold", "ent_and", "ent_or", "ent_kl_and", "format_mask",
                      "hi_kl_hi_surp", "hi_surp", "hi_ent")
    if args.token_select_mode not in _fullseq_modes and args.position_limit <= 0:
        if args.token_select_mode in ("hi_kl_hi_ent_topk", "raw_product_topk") and args.top_k_frac > 0.0:
            pass  # fraction-based K is valid
        else:
            raise ValueError("For non-prefix token selection modes, --position_limit must be > 0 (used as top-K).")
    if args.token_select_mode in ("reopold", "ent_and", "ent_or", "ent_kl_and", "format_mask",
                                  "hi_kl_hi_surp") and args.position_limit <= 0:
        args.position_limit = 0  # these modes use dynamic selection, not fixed K

    # Compute n_problems per step from bs and n_samples
    if args.bs % args.n_samples != 0:
        print(f"WARNING: bs={args.bs} is not divisible by n_samples={args.n_samples}. "
              f"Rounding down: n_problems={args.bs // args.n_samples} "
              f"(effective bs={args.bs // args.n_samples * args.n_samples})")
    n_problems_per_step = args.bs // args.n_samples
    if n_problems_per_step == 0:
        raise ValueError(f"bs={args.bs} < n_samples={args.n_samples}, need bs >= n_samples")

    # Mini-batch size for gradient accumulation
    mini_bs = args.mini_bs if args.mini_bs > 0 else args.bs
    if args.bs % mini_bs != 0:
        print(f"WARNING: bs={args.bs} is not divisible by mini_bs={mini_bs}. "
              f"Rounding up: accum_steps={-(-args.bs // mini_bs)}")
    accum_steps = -(-args.bs // mini_bs)  # ceil division
    print(f"bs={args.bs}, n_samples={args.n_samples} → n_problems={n_problems_per_step}/step, "
          f"mini_bs={mini_bs}, accum_steps={accum_steps}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load dataset (supports HF hub names and local JSONL/JSON files)
    if os.path.exists(args.dataset):
        dataset = load_dataset("json", data_files=args.dataset, split="train")
    else:
        dataset = load_dataset(args.dataset, split="train", streaming=False)
    problems = random.sample(list(dataset), min(args.num_problems, len(dataset)))
    indices = list(range(len(problems)))

    run_name = args.wandb_run_name or f"pos-distill-{args.position_limit}-{args.loss_type}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)

    # Precompute the per-vocab format-token boolean mask for format_mask mode.
    # Built once at startup; moved to student_device when needed.
    is_format_token_id = None
    if args.token_select_mode == "format_mask":
        _cats = None
        if getattr(args, "format_mask_cats", None):
            _cats = [c.strip() for c in args.format_mask_cats.split(",") if c.strip()]
        is_format_token_id = build_format_token_mask(tokenizer, cats=_cats)

    # --- Thinking mode detection via model_registry.json ---
    _registry_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_registry.json")
    _registry = {}
    if os.path.exists(_registry_path):
        with open(_registry_path) as f:
            _registry = json.load(f).get("models", {})

    def _model_has_thinking(model_name, tok):
        """Check if model supports thinking mode. Uses registry first, falls back to runtime detection."""
        if model_name in _registry:
            return _registry[model_name].get("thinking", False)
        # Try to resolve snapshot paths: .../models--org--name/snapshots/... → org/name
        import re as _re
        _snap_match = _re.search(r'models--(.+?)--(.+?)/snapshots/', model_name)
        if _snap_match:
            _resolved = f"{_snap_match.group(1)}/{_snap_match.group(2)}"
            if _resolved in _registry:
                return _registry[_resolved].get("thinking", False)
        # Model not in registry — detect at runtime and warn
        detected = _supports_thinking(tok)
        print(f"  WARNING: {model_name} not in model_registry.json! "
              f"Runtime detection says thinking={detected}. "
              f"Please add this model to the registry.")
        return detected

    teacher_has_thinking = _model_has_thinking(args.teacher_model, teacher_tokenizer)
    student_has_thinking = _model_has_thinking(args.student_model, tokenizer)
    print(f"  Thinking mode: student={student_has_thinking}, teacher={teacher_has_thinking}, "
          f"enable_thinking={args.enable_thinking}")

    # Build nothink_ids: suppress thinking UNLESS --enable_thinking is set
    _nothink_str = "<think>\n\n</think>\n\n"
    if args.enable_thinking:
        # Thinking enabled — no nothink prefix needed
        nothink_ids_teacher = []
        nothink_ids = []
        print(f"  CoT mode: thinking enabled, no nothink prefix")
    else:
        if teacher_has_thinking:
            nothink_ids_teacher = teacher_tokenizer.encode(_nothink_str, add_special_tokens=False)
        else:
            nothink_ids_teacher = []
        if student_has_thinking:
            nothink_ids = tokenizer.encode(_nothink_str, add_special_tokens=False)
        else:
            nothink_ids = []
        if nothink_ids or nothink_ids_teacher:
            print(f"  Nothink IDs: student={nothink_ids}, teacher={nothink_ids_teacher}")

    # Build cross-tokenizer vocab mapping if student and teacher use different tokenizers
    vocab_mapping = None
    valid_mask = None
    mapped_student_ids = None
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        # Check if vocabs are nearly identical (e.g. Qwen2 vs Qwen3)
        s_vocab = tokenizer.get_vocab()
        t_vocab = teacher_tokenizer.get_vocab()
        same_id_count = sum(1 for k in t_vocab if k in s_vocab and t_vocab[k] == s_vocab[k])
        overlap_ratio = same_id_count / max(len(s_vocab), len(t_vocab))
        if overlap_ratio > 0.99:
            print(f"  Near-identical tokenizers ({overlap_ratio:.1%} same-ID overlap) — using vocab truncation, no remap needed.")
            vocab_mapping = None  # skip cross-tokenizer path
            # will truncate to min vocab at loss computation
        else:
            print("  Different tokenizers detected — building vocab mapping for cross-tokenizer KL...")
            vocab_mapping, valid_mask, mapped_student_ids = build_vocab_mapping(tokenizer, teacher_tokenizer)
    else:
        print("  Same tokenizer — no vocab mapping needed.")

    # Load teacher model
    teacher_device = f"cuda:{args.teacher_gpu}"
    print(f"Loading teacher model {args.teacher_model}...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    if not single_gpu:
        teacher_model = teacher_model.to(teacher_device)
        print(f"Teacher model loaded on {teacher_device}.")
    else:
        print(f"Teacher model loaded on CPU (single-GPU mode, will move to {teacher_device} for scoring).")
    teacher_model.eval()

    # Initialize LoRA config (skip if full finetune)
    lora_config = None
    if not args.full_finetune:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
        )

    # Prepare optimizer and scheduler
    student_device = f"cuda:{args.student_gpu}"
    # 1 step = 1 chunk = n_problems_per_step problems × n_samples trajectories = bs trajectories
    chunks = [indices[i:i+n_problems_per_step] for i in range(0, len(indices), n_problems_per_step)]
    total_steps = len(chunks)
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}, Chunks: {len(chunks)}, "
          f"Problems: {len(problems)}")
    # Auto-clamp max_new_tokens only for prefix distillation.
    # For top-K token selection modes we need full-trajectory rollout.
    if args.token_select_mode == "prefix" and args.position_limit > 0 and args.max_new_tokens > args.position_limit:
        print(f"Auto-clamping max_new_tokens from {args.max_new_tokens} to {args.position_limit} (matches position_limit)")
        args.max_new_tokens = args.position_limit
    print(
        f"Loss: {args.loss_type}, Token select: {args.token_select_mode}, "
        f"Position limit/top-K: {args.position_limit}, Max new tokens: {args.max_new_tokens}"
    )

    log_file = open(os.path.join(args.output_dir, "train_log.jsonl"), "w")
    accum_loss = 0.0
    accum_ce = 0.0
    accum_kl = 0.0
    accum_tokens = 0
    n_trajs_total = 0
    # gen_model_path: updated every chunk via merge (on-policy)
    gen_model_path = args.student_model  # Start with base model
    merged_path = None  # Will be set after first chunk
    step = 0

    # Initialize student on GPU 0 — stays in memory for the entire run
    print("Loading student model...")
    resume_step = 0
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"  Resuming from {args.resume_from}...")
        student = load_student(args.student_model, args.resume_from, student_device, lora_config,
                               full_finetune=args.full_finetune)
        # Load optimizer/scheduler state if available
        opt_path = os.path.join(args.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            resume_state = torch.load(opt_path, map_location=student_device)
            resume_step = resume_state["step"]
            print(f"  Resuming from step {resume_step}")
    else:
        student = load_student(args.student_model, None, student_device, lora_config,
                               full_finetune=args.full_finetune)
    tp = sum(p.numel() for p in student.parameters() if p.requires_grad)
    ap = sum(p.numel() for p in student.parameters())
    print(f"  Student LoRA: {tp:,} / {ap:,} trainable ({tp/ap*100:.2f}%)")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if args.resume_from and resume_step > 0:
        opt_path = os.path.join(args.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            resume_state = torch.load(opt_path, map_location=student_device)
            optimizer.load_state_dict(resume_state["optimizer"])
            if args.fresh_scheduler:
                resume_step = 0  # treat as new run from old weights
                print(f"  Restored optimizer state (fresh scheduler: lr will cosine from {args.lr} over {total_steps} steps)")
            else:
                scheduler.load_state_dict(resume_state["scheduler"])
                print(f"  Restored optimizer/scheduler state")

    for chunk_idx, chunk_indices in enumerate(chunks):
        # Skip chunks that were already completed before resume
        # (but not when fresh_scheduler — that's a new run from old weights)
        if not args.fresh_scheduler and chunk_idx + 1 <= resume_step:
            step = chunk_idx + 1
            continue

        chunk_problems = [problems[i][args.problem_field] for i in chunk_indices]

        # ---- Phase 1: Generate trajectories ----
        gen_start = time.time()

        if args.use_sglang:
            # SGLang colocate path: SGLang and student share the same GPU
            # via enable_memory_saver (release/resume GPU memory)
            colocate = (args.vllm_gpu == args.student_gpu)
            print(f"  Chunk {chunk_idx+1}/{len(chunks)}: generating {len(chunk_problems)} × {args.n_samples} trajectories (SGLang, {'colocate' if colocate else 'separate GPU'}{', single-GPU restart' if single_gpu else ''})...")

            # 1. Save merged model for SGLang weight update
            merged_gen_path = os.path.join(args.output_dir, "_sglang_merged")
            if args.full_finetune:
                student.save_pretrained(merged_gen_path)
                tokenizer.save_pretrained(merged_gen_path)
            else:
                save_merged_model(student, tokenizer, merged_gen_path)

            # 2. If colocate, offload student to CPU to free GPU for SGLang
            if colocate:
                student.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()

            # 3. Start/manage SGLang server
            _sgl_gpu = args.vllm_gpu if args.vllm_gpu is not None else args.student_gpu
            if single_gpu:
                # Single-GPU: kill old server, start fresh each step to fully free GPU
                stop_sglang_server()
                gc.collect()
                torch.cuda.empty_cache()
                start_sglang_server(merged_gen_path, args.student_model,
                                    gpu_memory_utilization=args.vllm_gpu_util,
                                    port=30000 + _sgl_gpu,
                                    gpu_id=_sgl_gpu)
            elif _sglang_process is None:
                start_sglang_server(merged_gen_path, args.student_model,
                                    gpu_memory_utilization=args.vllm_gpu_util,
                                    port=30000 + _sgl_gpu,
                                    gpu_id=_sgl_gpu)
            else:
                if colocate:
                    sglang_resume_memory()
                update_sglang_weights(merged_gen_path)

            # 4. Generate
            all_trajectories = generate_chunk_sglang(
                chunk_problems, args.n_samples, args.max_new_tokens,
                args.temperature, tokenizer, system_prompt=args.system_prompt,
            )

            # 5. Release GPU for next phase
            if single_gpu:
                # Kill SGLang completely to free all GPU memory
                stop_sglang_server()
                gc.collect()
                torch.cuda.empty_cache()
            elif colocate:
                sglang_release_memory()
                gc.collect()
                torch.cuda.empty_cache()
                student.to(student_device)

            # 6. Cleanup merged checkpoint
            if os.path.exists(merged_gen_path):
                shutil.rmtree(merged_gen_path)

        elif args.use_vllm:
            # vLLM path: only offload models that share a GPU with vLLM
            offload_student = (args.vllm_gpu == args.student_gpu)
            offload_teacher = (args.vllm_gpu == args.teacher_gpu)
            offload_desc = "no offload" if not (offload_student or offload_teacher) else \
                           f"offload {'student+teacher' if offload_student and offload_teacher else 'student' if offload_student else 'teacher'}"
            print(f"  Chunk {chunk_idx+1}/{len(chunks)}: generating {len(chunk_problems)} × {args.n_samples} trajectories (vLLM, {offload_desc})...")

            # 1. Save student model for vLLM (merge LoRA if needed)
            merged_gen_path = os.path.join(args.output_dir, "_vllm_merged")
            if args.full_finetune:
                student.save_pretrained(merged_gen_path)
                tokenizer.save_pretrained(merged_gen_path)
            else:
                save_merged_model(student, tokenizer, merged_gen_path)

            # 2. Offload only models that share GPU with vLLM
            if offload_student:
                student.to("cpu")
            if offload_teacher:
                teacher_model.to("cpu")
            # Single-GPU: also offload optimizer state to CPU so vLLM has room
            # for KV cache. Critical for chunk_idx > 0 where Adam states + cached
            # backward buffers occupy 20+ GB even after model.to(cpu).
            if single_gpu:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor) and v.is_cuda:
                            state[k] = v.cpu()
            if offload_student or offload_teacher or single_gpu:
                for _ in range(3):
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

            # 3. Run vLLM subprocess on vllm_gpu
            traj_output = os.path.join(args.output_dir, "_vllm_trajs.json")
            all_trajectories = generate_chunk_vllm(
                merged_gen_path, merged_gen_path, chunk_problems,
                args.n_samples, args.max_new_tokens, args.temperature,
                traj_output, gpu_id=args.vllm_gpu, max_retries=2,
                mem_threshold_mb=500, gpu_memory_utilization=args.vllm_gpu_util,
                system_prompt=args.system_prompt,
            )

            # 4. Move models back if they were offloaded
            if offload_student or offload_teacher:
                gc.collect()
                torch.cuda.empty_cache()
            if not single_gpu:
                if offload_student:
                    student.to(student_device)
                if offload_teacher:
                    teacher_model.to(teacher_device)
            # single_gpu: both stay on CPU; teacher loaded for scoring, student
            # for training. Optimizer state also stays on CPU and is reloaded
            # along with student before training (Phase 3, line ~1390).

            # 5. Cleanup
            if os.path.exists(merged_gen_path):
                shutil.rmtree(merged_gen_path)
            if os.path.exists(traj_output):
                os.remove(traj_output)
        else:
            # HF generate path
            print(f"  Chunk {chunk_idx+1}/{len(chunks)}: generating {len(chunk_problems)} × {args.n_samples} trajectories (HF)...")
            all_trajectories = generate_hf(
                student, tokenizer, chunk_problems,
                args.n_samples, args.max_new_tokens, args.temperature,
                gen_batch_size=args.gen_batch_size,
                system_prompt=args.system_prompt,
                enable_thinking=args.enable_thinking,
            )

        gen_time = time.time() - gen_start

        if all_trajectories is None:
            print(f"  Generation failed for chunk {chunk_idx+1}, skipping...")
            if single_gpu:
                # Ensure student is back on GPU for next iteration
                student.to(student_device)
            step += 1
            continue

        total_trajs = sum(len(trajs) for trajs in all_trajectories.values())
        n_trajs_total += total_trajs
        print(f"Generated {total_trajs} trajectories for {len(chunk_problems)} problems ({gen_time:.0f}s)")

        # Clean up after generation before scoring
        gc.collect()
        torch.cuda.empty_cache()

        # ---- Phase 2: Score all trajectories in this chunk with teacher ----
        # Single-GPU: swap student out (if on GPU), load teacher in
        if single_gpu:
            student.to("cpu")  # no-op if already on CPU (sglang/vllm paths)
            # Also offload optimizer states to CPU to free GPU memory
            # (Adam momentum/variance buffers can be ~2x model size)
            if args.full_finetune:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor) and v.is_cuda:
                            state[k] = v.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            teacher_model.to(teacher_device)

        # Compute current position limit (progressive or fixed)
        step += 1
        if args.progressive_position:
            current_pos_limit = step  # step 1 → 1 token, step N → N tokens
        else:
            current_pos_limit = args.position_limit

        score_start = time.time()
        all_chunk_trajs = []
        all_chunk_teacher_lps = []
        for prob_idx_str in sorted(all_trajectories.keys(), key=int):
            trajs = all_trajectories[prob_idx_str]
            if len(trajs) == 0:
                continue
            if vocab_mapping is not None:
                # Cross-tokenizer: re-tokenize text for teacher scoring
                teacher_lps = query_teacher_cross_tokenizer(
                    teacher_model, trajs, tokenizer, teacher_tokenizer,
                    nothink_ids_teacher, current_pos_limit, device=teacher_device,
                )
            else:
                # For prefix and prefix-bounded hi_surp/hi_ent (position_limit>0),
                # only score the active prefix region (saves teacher compute).
                if args.token_select_mode == "prefix":
                    query_pos_limit = current_pos_limit
                elif args.token_select_mode in ("hi_surp", "hi_ent") and args.position_limit > 0:
                    query_pos_limit = current_pos_limit
                else:
                    query_pos_limit = 0
                teacher_lps = query_teacher_hf_logits_batch(
                    teacher_model, trajs, nothink_ids_teacher, query_pos_limit, device=teacher_device,
                    micro_bs=args.teacher_micro_bs,
                )
            all_chunk_trajs.extend(trajs)
            all_chunk_teacher_lps.extend(teacher_lps)
        score_time = time.time() - score_start

        if len(all_chunk_trajs) == 0:
            continue

        # Clean up after scoring before training
        gc.collect()
        torch.cuda.empty_cache()

        # ---- Phase 3: Train on all trajectories in this chunk (one optimizer step) ----
        # Single-GPU: swap teacher out, reload student for training
        if single_gpu:
            teacher_model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            student.to(student_device)
            # Reload optimizer states back to GPU (offloaded in Phase 1 vLLM
            # block or Phase 2). Done unconditionally so LoRA mode also works.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and not v.is_cuda:
                        state[k] = v.to(student_device)

        train_start = time.time()
        optimizer.zero_grad()
        n_trajs = len(all_chunk_trajs)

        # Build padded batch for student forward
        pad_id = 0
        all_input_ids = []
        resp_starts = []
        response_lens = []
        distill_counts = []
        for traj in all_chunk_trajs:
            prompt_ids = traj["prompt_ids"]
            response_ids = traj["response_ids"]
            resp_len = len(response_ids)
            if args.token_select_mode == "prefix":
                # Legacy positional distillation: only keep prefix tokens.
                distill_len = min(resp_len, current_pos_limit) if current_pos_limit > 0 else resp_len
                full_ids = prompt_ids + nothink_ids + response_ids[:distill_len]
            elif args.token_select_mode == "reopold":
                # REOPOLD: keep full response, select by entropy percentage later
                distill_len = resp_len
                full_ids = prompt_ids + nothink_ids + response_ids
            else:
                # Full trajectory rollout, select top-K positions later.
                distill_len = min(resp_len, current_pos_limit)
                full_ids = prompt_ids + nothink_ids + response_ids
            all_input_ids.append(full_ids)
            resp_starts.append(len(prompt_ids) + len(nothink_ids))
            response_lens.append(resp_len)
            distill_counts.append(distill_len)

        # Gradient accumulation: split bs trajectories into mini-batches of mini_bs
        step_loss_val = 0.0
        step_kl = 0.0
        step_ce = 0.0
        step_tokens = 0

        for mb_start in range(0, n_trajs, mini_bs):
            mb_end = min(mb_start + mini_bs, n_trajs)
            mb_size = mb_end - mb_start
            mb_ids = all_input_ids[mb_start:mb_end]
            mb_resp_starts = resp_starts[mb_start:mb_end]
            mb_resp_lens = response_lens[mb_start:mb_end]
            mb_distill_counts = distill_counts[mb_start:mb_end]
            mb_teacher = all_chunk_teacher_lps[mb_start:mb_end]

            max_len = max(len(ids) for ids in mb_ids)
            padded = [ids + [pad_id] * (max_len - len(ids)) for ids in mb_ids]
            masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in mb_ids]

            input_ids = torch.tensor(padded, dtype=torch.long, device=student_device)
            attn_mask = torch.tensor(masks, dtype=torch.long, device=student_device)
            outputs = None

            try:
                outputs = student(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
                logits_mb = outputs.logits  # [mb_size, max_len, vocab]
                shift_logits = logits_mb[:, :-1, :]  # [mb_size, max_len-1, vocab]
                shift_labels = input_ids[:, 1:]       # [mb_size, max_len-1]

                # Determine max response length across mini-batch for padding
                max_resp_len = max(mb_resp_lens)

                # Pad teacher log-probs to same length and stack into batch
                teacher_vocab_size = mb_teacher[0].shape[-1]
                t_log_probs_padded = torch.full(
                    (mb_size, max_resp_len, teacher_vocab_size),
                    float('-inf'), device=student_device,
                )
                # Build a mask for valid response positions [mb_size, max_resp_len]
                resp_valid_mask = torch.zeros(mb_size, max_resp_len, dtype=torch.bool, device=student_device)
                for i in range(mb_size):
                    rlen = mb_resp_lens[i]
                    t_lp = mb_teacher[i]
                    actual_t_len = min(rlen, t_lp.shape[0])
                    t_log_probs_padded[i, :actual_t_len] = t_lp[:actual_t_len].to(student_device)
                    resp_valid_mask[i, :rlen] = True

                # Extract student logits at response positions into [mb_size, max_resp_len, vocab]
                student_vocab_size = shift_logits.shape[-1]
                s_logits_resp = torch.zeros(
                    mb_size, max_resp_len, student_vocab_size,
                    device=student_device, dtype=shift_logits.dtype,
                )
                labels_resp = torch.zeros(
                    mb_size, max_resp_len, device=student_device, dtype=torch.long,
                )
                for i in range(mb_size):
                    start = mb_resp_starts[i] - 1
                    rlen = mb_resp_lens[i]
                    s_logits_resp[i, :rlen] = shift_logits[i, start:start+rlen]
                    labels_resp[i, :rlen] = shift_labels[i, start:start+rlen]

                s_log_probs_resp = log_softmax(s_logits_resp.float(), dim=-1)  # [mb_size, max_resp_len, vocab]

                # Build selection mask [mb_size, max_resp_len] for which positions to distill
                sel_mask = torch.zeros(mb_size, max_resp_len, dtype=torch.bool, device=student_device)

                if args.token_select_mode == "prefix":
                    # Select first k positions per trajectory
                    for i in range(mb_size):
                        k = mb_distill_counts[i]
                        sel_mask[i, :k] = True
                elif args.token_select_mode == "top_kl":
                    with torch.no_grad():
                        kl_per_pos = (torch.exp(t_log_probs_padded) * (t_log_probs_padded - s_log_probs_resp.detach())).sum(dim=-1)
                        kl_per_pos[~resp_valid_mask] = float('-inf')
                        for i in range(mb_size):
                            k = mb_distill_counts[i]
                            topk_idx = torch.topk(kl_per_pos[i, :mb_resp_lens[i]], k=k, largest=True).indices
                            sel_mask[i].scatter_(0, topk_idx, True)
                elif args.token_select_mode == "top_entropy_student":
                    with torch.no_grad():
                        ps = torch.exp(s_log_probs_resp.detach())
                        ent_per_pos = -(ps * s_log_probs_resp.detach()).sum(dim=-1)
                        ent_per_pos[~resp_valid_mask] = float('-inf')
                        for i in range(mb_size):
                            k = mb_distill_counts[i]
                            topk_idx = torch.topk(ent_per_pos[i, :mb_resp_lens[i]], k=k, largest=True).indices
                            sel_mask[i].scatter_(0, topk_idx, True)
                elif args.token_select_mode == "top_entropy_teacher":
                    with torch.no_grad():
                        pt = torch.exp(t_log_probs_padded)
                        ent_per_pos = -(pt * t_log_probs_padded).sum(dim=-1)
                        ent_per_pos[~resp_valid_mask] = float('-inf')
                        for i in range(mb_size):
                            k = mb_distill_counts[i]
                            topk_idx = torch.topk(ent_per_pos[i, :mb_resp_lens[i]], k=k, largest=True).indices
                            sel_mask[i].scatter_(0, topk_idx, True)
                elif args.token_select_mode == "random":
                    for i in range(mb_size):
                        k = mb_distill_counts[i]
                        rlen = mb_resp_lens[i]
                        perm = torch.randperm(rlen, device=student_device)[:k]
                        sel_mask[i].scatter_(0, perm, True)
                elif args.token_select_mode == "middle":
                    for i in range(mb_size):
                        k = mb_distill_counts[i]
                        rlen = mb_resp_lens[i]
                        if rlen <= k:
                            sel_mask[i, :rlen] = True
                        else:
                            mid = rlen // 2
                            start = max(0, mid - k // 2)
                            end = min(rlen, start + k)
                            start = max(0, end - k)
                            sel_mask[i, start:end] = True
                elif args.token_select_mode == "last":
                    for i in range(mb_size):
                        k = mb_distill_counts[i]
                        rlen = mb_resp_lens[i]
                        if rlen <= k:
                            sel_mask[i, :rlen] = True
                        else:
                            sel_mask[i, rlen-k:rlen] = True
                elif args.token_select_mode == "reopold":
                    # REOPOLD: entropy-guided dynamic masking — keep top beta% highest-entropy tokens
                    with torch.no_grad():
                        ps = torch.exp(s_log_probs_resp.detach())
                        ent_per_pos = -(ps * s_log_probs_resp.detach()).sum(dim=-1)
                        ent_per_pos[~resp_valid_mask] = float('-inf')
                        for i in range(mb_size):
                            rlen = mb_resp_lens[i]
                            k = max(1, int(rlen * args.reopold_beta))  # dynamic: beta% of response length
                            topk_idx = torch.topk(ent_per_pos[i, :rlen], k=k, largest=True).indices
                            sel_mask[i].scatter_(0, topk_idx, True)
                elif args.token_select_mode == "ent_and":
                    # AND: product of teacher*student entropy — both must be high
                    with torch.no_grad():
                        ps = torch.exp(s_log_probs_resp.detach())
                        ent_s = -(ps * s_log_probs_resp.detach()).sum(dim=-1)
                        pt = torch.exp(t_log_probs_padded)
                        ent_t = -(pt * t_log_probs_padded).sum(dim=-1)
                        score = ent_s * ent_t  # product → high only when BOTH are high
                        score[~resp_valid_mask] = float('-inf')
                        for i in range(mb_size):
                            k = mb_distill_counts[i]
                            topk_idx = torch.topk(score[i, :mb_resp_lens[i]], k=k, largest=True).indices
                            sel_mask[i].scatter_(0, topk_idx, True)
                elif args.token_select_mode == "ent_or":
                    # OR: sum of teacher+student entropy — either high is enough
                    with torch.no_grad():
                        ps = torch.exp(s_log_probs_resp.detach())
                        ent_s = -(ps * s_log_probs_resp.detach()).sum(dim=-1)
                        pt = torch.exp(t_log_probs_padded)
                        ent_t = -(pt * t_log_probs_padded).sum(dim=-1)
                        score = ent_s + ent_t  # sum → high when EITHER is high
                        score[~resp_valid_mask] = float('-inf')
                        for i in range(mb_size):
                            k = mb_distill_counts[i]
                            topk_idx = torch.topk(score[i, :mb_resp_lens[i]], k=k, largest=True).indices
                            sel_mask[i].scatter_(0, topk_idx, True)
                elif args.token_select_mode == "ent_kl_and":
                    # High entropy AND high KL — tokens where both uncertainty and divergence are large
                    with torch.no_grad():
                        ps = torch.exp(s_log_probs_resp.detach())
                        ent_s = -(ps * s_log_probs_resp.detach()).sum(dim=-1)
                        pt = torch.exp(t_log_probs_padded)
                        ent_t = -(pt * t_log_probs_padded).sum(dim=-1)
                        kl_per_pos = (ps * (s_log_probs_resp.detach() - t_log_probs_padded)).sum(dim=-1)
                        # Normalize each signal to [0,1] range before combining
                        ent_combined = ent_s * ent_t
                        score = ent_combined * kl_per_pos.clamp(min=0)  # triple product
                        score[~resp_valid_mask] = float('-inf')
                        for i in range(mb_size):
                            k = mb_distill_counts[i]
                            topk_idx = torch.topk(score[i, :mb_resp_lens[i]], k=k, largest=True).indices
                            sel_mask[i].scatter_(0, topk_idx, True)
                elif args.token_select_mode == "format_mask":
                    # Full-seq supervision but mask out positions where the
                    # student emitted a "format" token (structural / latex /
                    # math_operator / math_number — see _FORMAT_CATS). Mask is
                    # keyed on student emission, matching the offline analysis
                    # in scripts/analysis/format_mask_threshold.py.
                    _is_fmt_dev = is_format_token_id.to(student_device, non_blocking=True)
                    # labels_resp[i, t] is the student-emitted token id at
                    # response position t (already padded with 0 beyond rlen).
                    fmt_at_pos = _is_fmt_dev[labels_resp]  # [mb_size, max_resp_len], bool
                    # keep = valid response position AND not a format token
                    sel_mask = resp_valid_mask & (~fmt_at_pos)
                    # Diagnostic counts (printed once early in training)
                    if step <= 1 and mb_start == 0:
                        n_valid = int(resp_valid_mask.sum().item())
                        n_kept  = int(sel_mask.sum().item())
                        n_fmt   = int((resp_valid_mask & fmt_at_pos).sum().item())
                        print(f"[format_mask] mb {mb_size}x{max_resp_len}: "
                              f"valid={n_valid} kept={n_kept} masked_format={n_fmt} "
                              f"({n_fmt/max(n_valid,1)*100:.1f}% of valid)")
                elif args.token_select_mode == "hi_kl_hi_surp":
                    # Principled selection: keep only positions where the
                    # student is genuinely uncertain (-s_lp > p75) AND the
                    # teacher disagrees with the sampled token (|s_lp - t_lp|
                    # > p75). Tests the hypothesis that prefix-100 wins
                    # because position 0-100 over-concentrates this bucket
                    # (see docs/position_x_bucket.md and
                    # docs/kl_x_entropy_buckets.md).
                    #
                    # Thresholds are per-batch quantiles (over valid response
                    # positions) so the rule auto-adapts as the student
                    # distribution shifts during training.
                    with torch.no_grad():
                        # s_lp / t_lp at sampled tokens
                        s_lp_at = s_log_probs_resp.gather(
                            -1, labels_resp.unsqueeze(-1)
                        ).squeeze(-1)  # [mb, T]
                        t_lp_at = t_log_probs_padded.gather(
                            -1, labels_resp.clamp(max=t_log_probs_padded.shape[-1]-1).unsqueeze(-1)
                        ).squeeze(-1)
                        t_lp_at = torch.nan_to_num(t_lp_at, nan=-20.0, posinf=0.0, neginf=-20.0)
                        kl_pos   = (s_lp_at - t_lp_at).abs()
                        surp_pos = -s_lp_at
                        valid_kl   = kl_pos[resp_valid_mask]
                        valid_surp = surp_pos[resp_valid_mask]
                        if valid_kl.numel() > 0:
                            kl_thr   = torch.quantile(valid_kl.float(),   args.hi_kl_quantile)
                            surp_thr = torch.quantile(valid_surp.float(), args.hi_surp_quantile)
                        else:
                            kl_thr   = torch.tensor(float('inf'), device=student_device)
                            surp_thr = torch.tensor(float('inf'), device=student_device)
                    sel_mask = resp_valid_mask & (kl_pos > kl_thr) & (surp_pos > surp_thr)
                    if step <= 1 and mb_start == 0:
                        n_valid = int(resp_valid_mask.sum().item())
                        n_kept  = int(sel_mask.sum().item())
                        print(f"[hi_kl_hi_surp] mb {mb_size}x{max_resp_len}: "
                              f"valid={n_valid} kept={n_kept} "
                              f"({n_kept/max(n_valid,1)*100:.1f}% of valid)  "
                              f"kl_thr={kl_thr.item():.3f}  surp_thr={surp_thr.item():.3f}")
                elif args.token_select_mode == "hi_kl_hi_surp_topk":
                    # Budget-controlled hi_kl_hi_surp: per-trajectory top-K
                    # positions ranked by joint score = KL * surprise. K is
                    # taken from --position_limit so this slots directly into
                    # the K=100 token-selection family (prefix-100, top-KL-100,
                    # random-100, etc.) for an apples-to-apples comparison.
                    # Joint score multiplicative form requires both factors
                    # large; either alone being small pushes the score down.
                    with torch.no_grad():
                        s_lp_at = s_log_probs_resp.gather(
                            -1, labels_resp.unsqueeze(-1)
                        ).squeeze(-1)  # [mb, T]
                        t_lp_at = t_log_probs_padded.gather(
                            -1, labels_resp.clamp(max=t_log_probs_padded.shape[-1]-1).unsqueeze(-1)
                        ).squeeze(-1)
                        t_lp_at = torch.nan_to_num(t_lp_at, nan=-20.0, posinf=0.0, neginf=-20.0)
                        kl_pos   = (s_lp_at - t_lp_at).abs()
                        surp_pos = -s_lp_at
                        # Joint score: both must be large. Clamp surprise
                        # at >=0 (it's -log p so always >=0 for valid p<=1
                        # but numerical noise can make it slightly <0).
                        score = kl_pos * surp_pos.clamp(min=0.0)
                        score[~resp_valid_mask] = float('-inf')
                        for i in range(mb_size):
                            k = mb_distill_counts[i]
                            rlen = mb_resp_lens[i]
                            if k > 0 and rlen > 0:
                                k_eff = min(k, rlen)
                                topk_idx = torch.topk(
                                    score[i, :rlen], k=k_eff, largest=True
                                ).indices
                                sel_mask[i].scatter_(0, topk_idx, True)
                    if step <= 1 and mb_start == 0:
                        n_valid = int(resp_valid_mask.sum().item())
                        n_kept  = int(sel_mask.sum().item())
                        print(f"[hi_kl_hi_surp_topk] mb {mb_size}x{max_resp_len}: "
                              f"valid={n_valid} kept={n_kept} "
                              f"({n_kept/max(n_valid,1)*100:.1f}% of valid)  "
                              f"K_per_traj={mb_distill_counts}")
                elif args.token_select_mode == "hi_kl_hi_ent_topk":
                    # Per-trajectory top-K by joint score = KL * full-vocab entropy
                    # of the student. Replaces surprise (-log p_sampled) with
                    # H(p) = -sum_v p_v log p_v which is a vocab-wide measure
                    # of the student's uncertainty at this position.
                    # K is taken from --position_limit, OR if --top_k_frac > 0,
                    # K_i = floor(response_len_i * top_k_frac) per trajectory
                    # (e.g., top_k_frac=0.5 -> half the response length).
                    with torch.no_grad():
                        s_lp_at = s_log_probs_resp.gather(
                            -1, labels_resp.unsqueeze(-1)
                        ).squeeze(-1)  # [mb, T]
                        t_lp_at = t_log_probs_padded.gather(
                            -1, labels_resp.clamp(max=t_log_probs_padded.shape[-1]-1).unsqueeze(-1)
                        ).squeeze(-1)
                        t_lp_at = torch.nan_to_num(t_lp_at, nan=-20.0, posinf=0.0, neginf=-20.0)
                        kl_pos = (s_lp_at - t_lp_at).abs()
                        # Full-vocab entropy of student at each position
                        s_lp_full = s_log_probs_resp.detach()
                        ps = torch.exp(s_lp_full)
                        ent_pos = -(ps * s_lp_full).sum(dim=-1)  # [mb, T]
                        score = kl_pos * ent_pos.clamp(min=0.0)
                        score[~resp_valid_mask] = float('-inf')
                        for i in range(mb_size):
                            rlen = mb_resp_lens[i]
                            if args.top_k_frac > 0.0:
                                k = int(rlen * args.top_k_frac)
                            else:
                                k = mb_distill_counts[i]
                            if k > 0 and rlen > 0:
                                k_eff = min(k, rlen)
                                topk_idx = torch.topk(
                                    score[i, :rlen], k=k_eff, largest=True
                                ).indices
                                sel_mask[i].scatter_(0, topk_idx, True)
                    if step <= 1 and mb_start == 0:
                        n_valid = int(resp_valid_mask.sum().item())
                        n_kept  = int(sel_mask.sum().item())
                        ks_used = [
                            int(rl * args.top_k_frac) if args.top_k_frac > 0.0
                            else int(mb_distill_counts[i])
                            for i, rl in enumerate(mb_resp_lens)
                        ]
                        print(f"[hi_kl_hi_ent_topk] mb {mb_size}x{max_resp_len}: "
                              f"valid={n_valid} kept={n_kept} "
                              f"({n_kept/max(n_valid,1)*100:.1f}% of valid)  "
                              f"top_k_frac={args.top_k_frac}  K_per_traj={ks_used}")
                elif args.token_select_mode == "raw_product_topk":
                    # Per-trajectory top-K by joint score = KL(s||t) * H_s * H_t.
                    # All three factors are non-negative nats, so raw product ranks
                    # positions where the teacher disagrees, the student is unsure,
                    # AND the teacher has a meaningful alternative distribution.
                    # K = args.position_limit, or floor(rlen * top_k_frac) if set.
                    with torch.no_grad():
                        # Reverse KL per position (full vocab)
                        ps = torch.exp(s_log_probs_resp.detach())
                        s_lp_full = s_log_probs_resp.detach()
                        t_lp_full = t_log_probs_padded
                        mask_s = ps > 1e-10
                        # KL(s||t) = sum p_s (lp_s - lp_t), computed safely
                        kl_rev = (ps * (s_lp_full - t_lp_full)).sum(dim=-1).clamp(min=0.0)  # [mb, T]
                        # Student entropy
                        ent_s = -(ps * s_lp_full).sum(dim=-1).clamp(min=0.0)  # [mb, T]
                        # Teacher entropy
                        pt = torch.exp(t_lp_full)
                        ent_t = -(pt * t_lp_full).sum(dim=-1).clamp(min=0.0)  # [mb, T]
                        score = kl_rev * ent_s * ent_t
                        score[~resp_valid_mask] = float('-inf')
                        for i in range(mb_size):
                            rlen = mb_resp_lens[i]
                            if args.top_k_frac > 0.0:
                                k = int(rlen * args.top_k_frac)
                            else:
                                k = mb_distill_counts[i]
                            if k > 0 and rlen > 0:
                                k_eff = min(k, rlen)
                                topk_idx = torch.topk(
                                    score[i, :rlen], k=k_eff, largest=True
                                ).indices
                                sel_mask[i].scatter_(0, topk_idx, True)
                    if step <= 1 and mb_start == 0:
                        n_valid = int(resp_valid_mask.sum().item())
                        n_kept  = int(sel_mask.sum().item())
                        ks_used = [
                            int(rl * args.top_k_frac) if args.top_k_frac > 0.0
                            else int(mb_distill_counts[i])
                            for i, rl in enumerate(mb_resp_lens)
                        ]
                        print(f"[raw_product_topk] mb {mb_size}x{max_resp_len}: "
                              f"valid={n_valid} kept={n_kept} "
                              f"({n_kept/max(n_valid,1)*100:.1f}% of valid)  "
                              f"top_k_frac={args.top_k_frac}  K_per_traj={ks_used}")
                elif args.token_select_mode == "hi_ent":
                    # Drop low-entropy positions, where entropy is the full-vocab
                    # H(p) = -sum_v p_v log p_v of the student. Vocab-wide measure
                    # of uncertainty, independent of which token was sampled.
                    # Threshold (per-batch quantile of H over valid positions
                    # within the active region) chosen from prefix-100 H
                    # distribution analysis (docs/entropy_distribution_prefix/):
                    # - p25 ≈ 0.031: tokens where student is essentially
                    #   deterministic (top-1 prob > 95%) — format/structural tail.
                    # - p50 ≈ 0.44, p75 ≈ 1.28 (long right tail).
                    # If position_limit > 0, restrict the rule to the first N
                    # response positions; else full-seq.
                    with torch.no_grad():
                        s_lp_full = s_log_probs_resp.detach()
                        ps = torch.exp(s_lp_full)
                        ent_pos = -(ps * s_lp_full).sum(dim=-1)  # [mb, T]
                        ent_pos = ent_pos.clamp(min=0.0)  # numerical safety
                        if args.position_limit > 0:
                            T = ent_pos.shape[1]
                            pos_idx = torch.arange(T, device=ent_pos.device).unsqueeze(0)
                            region_mask = resp_valid_mask & (pos_idx < args.position_limit)
                        else:
                            region_mask = resp_valid_mask
                        valid_ent = ent_pos[region_mask]
                        if args.hi_ent_threshold > 0.0:
                            # absolute threshold mode: fixed H cutoff (e.g. 0.01 ≈ top-1 prob 99%)
                            ent_thr = torch.tensor(
                                float(args.hi_ent_threshold), device=student_device
                            )
                            thr_mode = "abs"
                        elif valid_ent.numel() > 0:
                            ent_thr = torch.quantile(
                                valid_ent.float(), args.hi_ent_quantile
                            )
                            thr_mode = "q"
                        else:
                            ent_thr = torch.tensor(float('inf'), device=student_device)
                            thr_mode = "inf"
                    sel_mask = region_mask & (ent_pos > ent_thr)
                    if step <= 1 and mb_start == 0:
                        n_valid  = int(resp_valid_mask.sum().item())
                        n_region = int(region_mask.sum().item())
                        n_kept   = int(sel_mask.sum().item())
                        print(f"[hi_ent] mb {mb_size}x{max_resp_len}: "
                              f"valid={n_valid} region={n_region} kept={n_kept} "
                              f"({n_kept/max(n_region,1)*100:.1f}% of region)  "
                              f"ent_thr={ent_thr.item():.5f} ({thr_mode})  "
                              f"quantile={args.hi_ent_quantile} "
                              f"abs={args.hi_ent_threshold}  "
                              f"pos_limit={args.position_limit}")
                elif args.token_select_mode == "hi_surp":
                    # Drop low-surprise (low-entropy at sampled token) positions.
                    # If position_limit > 0, restrict the rule to the first N
                    # response positions (prefix-bounded variant); else full-seq.
                    # Threshold = per-batch quantile of -s_lp over valid positions
                    # within the active region. Keeps positions where
                    # -s_lp > threshold.
                    with torch.no_grad():
                        s_lp_at = s_log_probs_resp.gather(
                            -1, labels_resp.unsqueeze(-1)
                        ).squeeze(-1)  # [mb, T]
                        surp_pos = -s_lp_at  # [mb, T], surprise (>=0 typically)
                        # Region mask: full-seq if position_limit==0, else prefix.
                        if args.position_limit > 0:
                            T = surp_pos.shape[1]
                            pos_idx = torch.arange(T, device=surp_pos.device).unsqueeze(0)
                            region_mask = resp_valid_mask & (pos_idx < args.position_limit)
                        else:
                            region_mask = resp_valid_mask
                        valid_surp = surp_pos[region_mask]
                        if valid_surp.numel() > 0:
                            surp_thr = torch.quantile(
                                valid_surp.float(), args.hi_surp_quantile
                            )
                        else:
                            surp_thr = torch.tensor(float('inf'), device=student_device)
                    sel_mask = region_mask & (surp_pos > surp_thr)
                    if step <= 1 and mb_start == 0:
                        n_valid  = int(resp_valid_mask.sum().item())
                        n_region = int(region_mask.sum().item())
                        n_kept   = int(sel_mask.sum().item())
                        print(f"[hi_surp] mb {mb_size}x{max_resp_len}: "
                              f"valid={n_valid} region={n_region} kept={n_kept} "
                              f"({n_kept/max(n_region,1)*100:.1f}% of region)  "
                              f"surp_thr={surp_thr.item():.3f}  "
                              f"pos_limit={args.position_limit}")
                else:
                    raise ValueError(f"Unknown token_select_mode={args.token_select_mode}")

                # Count selected tokens
                n_selected = sel_mask.sum().item()
                step_tokens += int(n_selected)

                mb_loss = torch.tensor(0.0, device=student_device)
                if vocab_mapping is not None:
                    # Cross-tokenizer: reverse KL over shared vocabulary
                    # Process per-trajectory since remap_teacher_logprobs works per-trajectory
                    for i in range(mb_size):
                        sel_idx_i = sel_mask[i].nonzero(as_tuple=True)[0]
                        if len(sel_idx_i) == 0:
                            continue
                        t_lp_sel = t_log_probs_padded[i].index_select(0, sel_idx_i)
                        t_log_probs_shared = remap_teacher_logprobs(
                            t_lp_sel, vocab_mapping, valid_mask,
                            student_vocab_size, mapped_student_ids)
                        t_log_probs_shared = torch.nan_to_num(t_log_probs_shared, nan=-20.0, posinf=0.0, neginf=-20.0)
                        t_log_probs_shared = t_log_probs_shared.clamp(min=-20.0, max=0.0)

                        _msi_dev = mapped_student_ids.to(student_device)
                        _msi_dev = _msi_dev.clamp(max=student_vocab_size - 1)
                        s_logits_sel = s_logits_resp[i].index_select(0, sel_idx_i)
                        s_logits_shared = s_logits_sel[:, _msi_dev]
                        s_log_probs_shared = log_softmax(s_logits_shared.float(), dim=-1)

                        # Compute loss over shared vocab
                        t_lp_shared = t_log_probs_shared.to(student_device)
                        if args.loss_type == "reverse_kl":
                            loss_traj = kl_div(
                                t_lp_shared, s_log_probs_shared,
                                log_target=True, reduction="batchmean",
                            )
                        elif args.loss_type in ("dft_distill", "dft_distill_deadzone"):
                            per_pos_kl = (torch.exp(t_lp_shared) * (t_lp_shared - s_log_probs_shared)).sum(dim=-1)
                            _msi_local = mapped_student_ids.to(limited_labels.device)
                            _lm_local = (limited_labels.unsqueeze(-1) == _msi_local.unsqueeze(0)).long().argmax(-1)
                            _lm_local = _lm_local.clamp(max=s_log_probs_shared.shape[-1] - 1)
                            s_prob_sampled = s_log_probs_shared.gather(-1, _lm_local.unsqueeze(-1)).squeeze(-1).exp().detach()
                            if args.loss_type == "dft_distill_deadzone":
                                with torch.no_grad():
                                    mask = (per_pos_kl.detach() > 0.1).float()
                                s_prob_sampled = s_prob_sampled * mask
                            loss_traj = (per_pos_kl * s_prob_sampled).mean()
                        else:
                            raise ValueError(f"Unknown loss_type: {args.loss_type}")
                        # Clamp loss to reasonable range
                        loss_traj = loss_traj.clamp(max=50.0)
                        mb_loss = mb_loss + loss_traj / n_trajs

                        with torch.no_grad():
                            limited_labels_i = labels_resp[i].index_select(0, sel_idx_i)
                            _msi = mapped_student_ids.to(limited_labels_i.device)
                            _lm = (limited_labels_i.unsqueeze(-1) == _msi.unsqueeze(0)).long().argmax(-1)
                            _lm = _lm.clamp(max=s_log_probs_shared.shape[-1] - 1)
                            s_lps = s_log_probs_shared.gather(-1, _lm.unsqueeze(-1)).squeeze(-1)
                            t_lps = t_log_probs_shared.to(student_device).gather(-1, _lm.unsqueeze(-1)).squeeze(-1)
                            step_kl += (s_lps - t_lps).mean().item()
                            step_ce += (-s_lps).mean().item()
                else:
                    # Same-tokenizer: batch KL over selected positions
                    # Align vocab sizes if needed
                    min_vocab = min(t_log_probs_padded.shape[-1], s_log_probs_resp.shape[-1])
                    t_lp = t_log_probs_padded[..., :min_vocab]
                    s_lp = s_log_probs_resp[..., :min_vocab]

                    # Sanitize teacher log-probs: padding positions (beyond
                    # response length) are -inf, which produces NaN under
                    # `0 * inf` when later masked. Clamp to a finite floor.
                    t_lp = torch.nan_to_num(t_lp, nan=-20.0, posinf=0.0, neginf=-20.0)
                    t_lp = t_lp.clamp(min=-20.0, max=0.0)

                    # Compute per-position KL: [mb_size, max_resp_len]
                    if args.loss_type == "reverse_kl":
                        per_pos_kl = (torch.exp(s_lp) * (s_lp - t_lp)).sum(dim=-1)
                    elif args.loss_type in ("dft_distill", "dft_distill_deadzone"):
                        per_pos_kl_raw = (torch.exp(t_lp) * (t_lp - s_lp)).sum(dim=-1)
                        s_prob_sampled = s_lp.gather(-1, labels_resp.unsqueeze(-1)).squeeze(-1).exp().detach()
                        if args.loss_type == "dft_distill_deadzone":
                            with torch.no_grad():
                                dz_mask = (per_pos_kl_raw.detach() > 0.1).float()
                            s_prob_sampled = s_prob_sampled * dz_mask
                        per_pos_kl = per_pos_kl_raw * s_prob_sampled
                    else:
                        raise ValueError(f"Unknown loss_type: {args.loss_type}")

                    # Reward clipping (REOPOLD): cap extreme per-token KL values
                    if args.reward_clip > 0:
                        per_pos_kl = per_pos_kl.clamp(max=args.reward_clip)
                    # Mask: only selected positions contribute. Use `where`
                    # rather than `*` so any inf/NaN at unmasked positions
                    # cannot leak through `0 * inf = NaN`.
                    per_pos_kl = torch.where(sel_mask, per_pos_kl, torch.zeros_like(per_pos_kl))
                    # Average over selected positions, then over batch
                    n_sel_per_traj = sel_mask.float().sum(dim=-1).clamp(min=1)
                    loss_per_traj = per_pos_kl.sum(dim=-1) / n_sel_per_traj
                    mb_loss = loss_per_traj.sum() / n_trajs

                    with torch.no_grad():
                        s_lps = s_lp.gather(-1, labels_resp.unsqueeze(-1)).squeeze(-1)
                        t_lps = t_lp.gather(-1, labels_resp.unsqueeze(-1)).squeeze(-1)
                        sel_s = (s_lps * sel_mask.float()).sum() / n_selected if n_selected > 0 else 0
                        sel_t = (t_lps * sel_mask.float()).sum() / n_selected if n_selected > 0 else 0
                        step_kl += (sel_s - sel_t).item() if isinstance(sel_s, torch.Tensor) else (sel_s - sel_t)
                        step_ce += (-sel_s).item() if isinstance(sel_s, torch.Tensor) else (-sel_s)

                mb_loss.backward()
                step_loss_val += mb_loss.item()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                err_type = "OOM" if "out of memory" in str(e).lower() else "RuntimeError"
                print(f"  WARNING: {err_type} on mini-batch (seq_len={max_len}), skipping: {str(e)[:100]}")
                student.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()

            del input_ids, attn_mask, outputs

        # Gradient clipping and optimization (once per chunk)
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        # Stats
        step_loss = step_loss_val
        step_ce /= n_trajs
        step_kl /= n_trajs
        accum_loss += step_loss
        accum_ce += step_ce
        accum_kl += step_kl
        accum_tokens += step_tokens
        train_time = time.time() - train_start

        # Logging
        if step % args.log_steps == 0:
            pos_info = f" pos_limit={current_pos_limit}" if args.progressive_position else ""
            print(f"  step={step} loss={step_loss:.4f} kl={step_kl:.4f} tokens={step_tokens} lr={scheduler.get_last_lr()[0]:.2e} gen={gen_time:.0f}s score={score_time:.0f}s train={train_time:.1f}s{pos_info}", flush=True)
            wandb.log({
                "train/loss": step_loss,
                "train/kl": step_kl,
                "train/ce": step_ce,
                "train/lr": scheduler.get_last_lr()[0],
                "train/gen_time": gen_time,
                "train/score_time": score_time,
                "train/step_tokens": step_tokens,
            }, step=step)

        # Periodic checkpoint saves
        if step % args.save_steps == 0:
            save_dir = os.path.join(args.output_dir, f"step_{step}")
            if args.full_finetune:
                student.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
            else:
                student.save_pretrained(save_dir)
            # Skip saving optimizer state (5.8GB per checkpoint, not needed for eval-only workflow)
            # optimizer_state = {
            #     'optimizer': optimizer.state_dict(),
            #     'scheduler': scheduler.state_dict(),
            #     'step': step,
            # }
            # torch.save(optimizer_state, os.path.join(save_dir, "optimizer.pt"))
            print(f"  Saved {'full' if args.full_finetune else 'LoRA'} checkpoint: {save_dir}")

        # Periodic eval (at every save_steps) — vLLM eval on GPU 1
        do_eval = args.eval_steps > 0 and step % args.eval_steps == 0

        if do_eval:
            # For full finetune, model is already merged; for LoRA, merge first
            if args.full_finetune:
                eval_merged_path = os.path.join(args.output_dir, f"step_{step}")
            else:
                eval_merged_path = os.path.join(args.output_dir, "_tmp_merged")
                if not os.path.exists(eval_merged_path):
                    save_merged_model(student, tokenizer, eval_merged_path)
            eval_dir = os.path.join(args.output_dir, f"eval_step_{step}")
            summary = run_eval_math500(
                eval_merged_path, eval_dir, args.student_model,
                n_samples=args.eval_samples, gpu_id=args.student_gpu,
            )
            _kill_orphan_vllm(args.student_gpu, mem_threshold_mb=5000)
            if summary:
                acc = summary["accuracy"]
                print(f"  MATH-500 pass@{args.eval_samples}: {acc:.4f} ({summary['correct']}/{summary['total']})")
                wandb.log({"eval/math500_accuracy": acc, "step": step})

        # Clean up temporary merged model if eval created one
        tmp_merged = os.path.join(args.output_dir, "_tmp_merged")
        if os.path.exists(tmp_merged):
            shutil.rmtree(tmp_merged)

    # Final checkpoint and merged model
    final_dir = os.path.join(args.output_dir, f"step_{step}")
    if args.full_finetune:
        student.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        merged_path = final_dir  # Already a full model
    else:
        student.save_pretrained(final_dir)
        merged_path = os.path.join(args.output_dir, "_merged_latest")
        if os.path.exists(merged_path):
            shutil.rmtree(merged_path)
        save_merged_model(student, tokenizer, merged_path)
    print(f"  Final model: {merged_path}")

    eval_dir = os.path.join(args.output_dir, "eval_final")
    summary = run_eval_math500(
        merged_path, eval_dir, args.student_model,
        n_samples=args.eval_samples, gpu_id=args.teacher_gpu,
    )
    _kill_orphan_vllm(args.teacher_gpu, mem_threshold_mb=5000)
    if summary:
        acc = summary["accuracy"]
        print(f"MATH-500 final avg@{args.eval_samples}: {acc:.4f} ({summary['correct']}/{summary['total']})")
        wandb.log({"eval/math500_accuracy": acc, "step": step})

    # Clean up temporary files
    for p in Path(args.output_dir).glob("_tmp*"):
        if p.is_dir():
            shutil.rmtree(p)
        else:
            os.remove(p)

    del teacher_model
    gc.collect()
    torch.cuda.empty_cache()

    log_file.close()
    wandb.finish()
    print(f"\nTraining complete! Final model: {merged_path}")
    print(f"Position limit used: {args.position_limit} tokens")


if __name__ == "__main__":
    main()

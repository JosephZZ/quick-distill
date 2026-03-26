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


def build_prompt(problem: str, tokenizer, system_prompt: str = None) -> str:
    if system_prompt is None:
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if _supports_thinking(tokenizer):
        kwargs["enable_thinking"] = False
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


def generate_hf(student, tokenizer, problems, n_samples, max_new_tokens, temperature, gen_batch_size=0, system_prompt=None):
    """Generate trajectories using HF model.generate() directly — no subprocess or disk I/O."""
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or 0
    special_ids = {eos_id, pad_id, 151645, 151643}

    # Build all prompts: each problem repeated n_samples times
    all_prompts = []
    problem_indices = []  # track which problem each prompt belongs to
    for i, problem in enumerate(problems):
        prompt_text = build_prompt(problem, tokenizer, system_prompt=system_prompt)
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


def start_sglang_server(model_path, tokenizer_name, gpu_memory_utilization=0.50, port=30000):
    """Launch a persistent SGLang server. Returns (process, port)."""
    global _sglang_process, _sglang_port
    if _sglang_process is not None:
        return _sglang_process, _sglang_port

    import requests as _req
    env = os.environ.copy()

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--tokenizer-path", tokenizer_name,
        "--port", str(port),
        "--mem-fraction-static", str(gpu_memory_utilization),
        "--trust-remote-code",
        "--dtype", "bfloat16",
    ]
    print(f"  Starting SGLang server on port {port} (gpu_util={gpu_memory_utilization})...")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
        all_trajectories[problem] = trajs

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
    # Only override CUDA_VISIBLE_DEVICES if not already set (avoid clobbering parent's GPU assignment)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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
    
    # New parameter: position limit
    parser.add_argument("--position_limit", type=int, default=50,
                       help="Only distill first N tokens (0=disabled, default=50)")
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

    args = parser.parse_args()

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
    # Qwen3/Gemma3 thinking models need nothink prefix; others get empty list
    _nothink_str = "<think>\n\n</think>\n\n"
    _test_ids = teacher_tokenizer.encode(_nothink_str, add_special_tokens=False)
    # If the tokenizer doesn't know <think> token, it will split into subwords (many ids)
    nothink_ids = _test_ids if len(_test_ids) <= 6 else []

    # Load teacher model on GPU 1 using HF
    teacher_device = f"cuda:{args.teacher_gpu}"
    print(f"Loading teacher model {args.teacher_model} on {teacher_device}...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(teacher_device)
    teacher_model.eval()
    print("Teacher model loaded.")

    # Initialize LoRA config (skip if full finetune)
    lora_config = None
    if not args.full_finetune:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    # Prepare optimizer and scheduler
    student_device = f"cuda:{args.student_gpu}"
    # 1 step = 1 chunk = n_problems_per_step problems × n_samples trajectories = bs trajectories
    chunks = [indices[i:i+n_problems_per_step] for i in range(0, len(indices), n_problems_per_step)]
    total_steps = len(chunks)
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}, Chunks: {len(chunks)}, "
          f"Problems: {len(problems)}")
    # Auto-clamp max_new_tokens to position_limit when using positional loss
    if args.position_limit > 0 and args.max_new_tokens > args.position_limit:
        print(f"Auto-clamping max_new_tokens from {args.max_new_tokens} to {args.position_limit} (matches position_limit)")
        args.max_new_tokens = args.position_limit
    print(f"Loss: {args.loss_type}, Position limit: {args.position_limit}, Max new tokens: {args.max_new_tokens}")

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
            # SGLang path: persistent server, no offloading needed
            print(f"  Chunk {chunk_idx+1}/{len(chunks)}: generating {len(chunk_problems)} × {args.n_samples} trajectories (SGLang)...")

            # 1. Save merged model and update server weights
            merged_gen_path = os.path.join(args.output_dir, "_sglang_merged")
            if args.full_finetune:
                student.save_pretrained(merged_gen_path)
                tokenizer.save_pretrained(merged_gen_path)
            else:
                save_merged_model(student, tokenizer, merged_gen_path)

            # 2. Start server on first call, or update weights
            if _sglang_process is None:
                start_sglang_server(merged_gen_path, args.student_model,
                                    gpu_memory_utilization=args.vllm_gpu_util,
                                    port=30000 + args.student_gpu)
            else:
                update_sglang_weights(merged_gen_path)

            # 3. Generate (no offload needed — server is a separate process)
            all_trajectories = generate_chunk_sglang(
                chunk_problems, args.n_samples, args.max_new_tokens,
                args.temperature, tokenizer, system_prompt=args.system_prompt,
            )

            # 4. Cleanup merged checkpoint
            if os.path.exists(merged_gen_path):
                shutil.rmtree(merged_gen_path)

        elif args.use_vllm:
            # vLLM path: offload models to CPU, run vLLM subprocess, reload
            print(f"  Chunk {chunk_idx+1}/{len(chunks)}: generating {len(chunk_problems)} × {args.n_samples} trajectories (vLLM)...")

            # 1. Save student model for vLLM (merge LoRA if needed)
            merged_gen_path = os.path.join(args.output_dir, "_vllm_merged")
            if args.full_finetune:
                student.save_pretrained(merged_gen_path)
                tokenizer.save_pretrained(merged_gen_path)
            else:
                save_merged_model(student, tokenizer, merged_gen_path)

            # 2. Offload both models to CPU to free GPU for vLLM
            student.to("cpu")
            teacher_model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # free IPC memory

            # 3. Run vLLM subprocess
            traj_output = os.path.join(args.output_dir, "_vllm_trajs.json")
            all_trajectories = generate_chunk_vllm(
                merged_gen_path, args.student_model, chunk_problems,
                args.n_samples, args.max_new_tokens, args.temperature,
                traj_output, gpu_id=args.student_gpu, max_retries=2,
                mem_threshold_mb=500, gpu_memory_utilization=args.vllm_gpu_util,
                system_prompt=args.system_prompt,
            )

            # 4. Move models back to GPU (clear cache first — vLLM may leave residual memory)
            gc.collect()
            torch.cuda.empty_cache()
            student.to(student_device)
            teacher_model.to(teacher_device)

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
            )

        gen_time = time.time() - gen_start

        if all_trajectories is None:
            print(f"  Generation failed for chunk {chunk_idx+1}, skipping...")
            step += 1
            continue

        total_trajs = sum(len(trajs) for trajs in all_trajectories.values())
        n_trajs_total += total_trajs
        print(f"Generated {total_trajs} trajectories for {len(chunk_problems)} problems ({gen_time:.0f}s)")

        # ---- Phase 2: Score all trajectories in this chunk with teacher ----
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
            teacher_lps = query_teacher_hf_logits_batch(
                teacher_model, trajs, nothink_ids, current_pos_limit, device=teacher_device,
                micro_bs=args.teacher_micro_bs,
            )
            all_chunk_trajs.extend(trajs)
            all_chunk_teacher_lps.extend(teacher_lps)
        score_time = time.time() - score_start

        if len(all_chunk_trajs) == 0:
            continue

        # ---- Phase 3: Train on all trajectories in this chunk (one optimizer step) ----
        train_start = time.time()
        optimizer.zero_grad()
        n_trajs = len(all_chunk_trajs)

        # Build padded batch for student forward
        pad_id = 0
        all_input_ids = []
        resp_starts = []
        effective_lens = []
        for traj in all_chunk_trajs:
            prompt_ids = traj["prompt_ids"]
            response_ids = traj["response_ids"]
            resp_len = len(response_ids)
            eff_len = min(resp_len, current_pos_limit) if current_pos_limit > 0 else resp_len
            full_ids = prompt_ids + nothink_ids + response_ids[:eff_len]
            all_input_ids.append(full_ids)
            resp_starts.append(len(prompt_ids) + len(nothink_ids))
            effective_lens.append(eff_len)

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
            mb_eff_lens = effective_lens[mb_start:mb_end]
            mb_teacher = all_chunk_teacher_lps[mb_start:mb_end]

            max_len = max(len(ids) for ids in mb_ids)
            padded = [ids + [pad_id] * (max_len - len(ids)) for ids in mb_ids]
            masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in mb_ids]

            input_ids = torch.tensor(padded, dtype=torch.long, device=student_device)
            attn_mask = torch.tensor(masks, dtype=torch.long, device=student_device)
            outputs = None

            try:
                outputs = student(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
                logits_mb = outputs.logits

                mb_loss = torch.tensor(0.0, device=student_device)
                for i in range(mb_size):
                    shift_logits_i = logits_mb[i, :-1, :]
                    shift_labels_i = input_ids[i, 1:]
                    start = mb_resp_starts[i] - 1
                    eff_len = mb_eff_lens[i]

                    t_log_probs = mb_teacher[i]
                    s_log_probs_resp = log_softmax(shift_logits_i[start:start+eff_len].float(), dim=-1)
                    limited_labels = shift_labels_i[start:start+eff_len]
                    step_tokens += eff_len

                    loss_traj = kl_div(
                        t_log_probs.to(student_device),
                        s_log_probs_resp,
                        log_target=True,
                        reduction="batchmean",
                    )
                    mb_loss = mb_loss + loss_traj / n_trajs

                    with torch.no_grad():
                        s_lps = s_log_probs_resp.gather(-1, limited_labels.unsqueeze(-1)).squeeze(-1)
                        t_lps_sampled = t_log_probs.to(student_device).gather(-1, limited_labels.unsqueeze(-1)).squeeze(-1)
                        step_kl += (s_lps - t_lps_sampled).mean().item()
                        step_ce += (-s_lps).mean().item()

                mb_loss.backward()
                step_loss_val += mb_loss.item()
            except torch.cuda.OutOfMemoryError:
                print(f"  WARNING: OOM on mini-batch (seq_len={max_len}), skipping")
                student.zero_grad(set_to_none=True)
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
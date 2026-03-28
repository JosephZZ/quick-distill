"""
Generate trajectories using vLLM for a batch of problems.
Saves results to a JSON file. Designed to be called as a subprocess
so GPU memory is fully freed when it exits.
"""

import json
import argparse

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# vLLM 0.11 caches tokenizer.all_special_tokens_extended; Qwen2Tokenizer omits it in transformers>=5.0.
try:
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer as _Q2Tok

    if not hasattr(_Q2Tok, "all_special_tokens_extended"):
        _Q2Tok.all_special_tokens_extended = property(
            lambda self: list(getattr(self, "all_special_tokens", []) or [])
        )
except Exception:
    pass
try:
    from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast as _Q2TokF

    if not hasattr(_Q2TokF, "all_special_tokens_extended"):
        _Q2TokF.all_special_tokens_extended = property(
            lambda self: list(getattr(self, "all_special_tokens", []) or [])
        )
except Exception:
    pass


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


def build_prompt(problem: str, tokenizer, system_prompt: str = None) -> str:
    if system_prompt is None:
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    if _supports_system_role(tokenizer):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
        ]
    else:
        messages = [
            {"role": "user", "content": system_prompt + "\n\n" + problem},
        ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if _supports_thinking(tokenizer):
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--problems_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--system_prompt", type=str, default=None,
                       help="System prompt for generation (default: math reasoning prompt)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    with open(args.problems_file) as f:
        problems = json.load(f)

    prompts = [build_prompt(p, tokenizer, system_prompt=args.system_prompt) for p in problems]

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        dtype="bfloat16",
        max_model_len=args.max_new_tokens + 512,
        seed=args.seed,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_new_tokens,
        n=args.n_samples,
    )

    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    special_ids = {eos_id, pad_id, 151645, 151643}

    all_trajectories = {}
    for i, output in enumerate(outputs):
        prompt_ids = list(output.prompt_token_ids)
        trajectories = []
        for completion in output.outputs:
            response_ids = list(completion.token_ids)
            while response_ids and response_ids[-1] in special_ids:
                response_ids.pop()
            if len(response_ids) == 0:
                continue
            trajectories.append({
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "full_ids": prompt_ids + response_ids,
            })
        all_trajectories[str(i)] = trajectories

    with open(args.output_file, "w") as f:
        json.dump(all_trajectories, f)

    total = sum(len(t) for t in all_trajectories.values())
    print(f"Generated {total} trajectories for {len(problems)} problems")

    # Explicit cleanup: destroy vLLM engine and free GPU
    del llm.llm_engine
    del llm
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()

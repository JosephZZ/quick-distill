"""
Generate trajectories using vLLM for a batch of problems.
Saves results to a JSON file. Designed to be called as a subprocess
so GPU memory is fully freed when it exits.
"""

import json
import argparse

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def build_prompt(problem: str, tokenizer, system_prompt: str = None) -> str:
    if system_prompt is None:
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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

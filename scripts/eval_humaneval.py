"""Evaluate models on HumanEval/MBPP using evalplus with custom vLLM gpu_memory_utilization."""
import argparse, json, os, sys

# vLLM 0.11 / torch-triton compatibility shim:
try:
    import triton.compiler.compiler as _triton_compiler
    if not hasattr(_triton_compiler, "triton_key"):
        def _triton_key():
            return "unknown"
        _triton_compiler.triton_key = _triton_key
except Exception:
    pass

from vllm import LLM, SamplingParams
from evalplus.data import get_human_eval_plus, get_mbpp_plus


def get_problems(dataset):
    if dataset == "humaneval":
        return get_human_eval_plus()
    elif dataset == "mbpp":
        return get_mbpp_plus()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def build_prompt(task_id, problem, dataset, tokenizer):
    """Build prompt for code generation."""
    # Both HumanEval and MBPP provide ready-to-use prompts
    return problem["prompt"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="humaneval", choices=["humaneval", "mbpp"])
    parser.add_argument("--output_dir", type=str, default="evalplus_results")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.50)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    problems = get_problems(args.dataset)
    print(f"Loaded {len(problems)} problems from {args.dataset}")

    # Init vLLM
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        max_model_len=2048,
    )
    tokenizer = llm.get_tokenizer()

    # Build prompts
    task_ids = sorted(problems.keys())
    prompts = []
    for tid in task_ids:
        prompts.append(build_prompt(tid, problems[tid], args.dataset, tokenizer))

    # Generate
    sampling_params = SamplingParams(
        temperature=args.temperature if args.temperature > 0 else 0,
        top_p=1.0 if args.temperature == 0 else 0.95,
        max_tokens=args.max_tokens,
        n=args.n_samples,
    )

    print(f"Generating {len(prompts)} prompts × {args.n_samples} samples...")
    outputs = llm.generate(prompts, sampling_params)

    # Save in evalplus format
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model.replace("/", "--")
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{model_name}.jsonl")

    with open(output_file, "w") as f:
        for tid, output in zip(task_ids, outputs):
            for sample in output.outputs:
                completion = sample.text
                # Stop at common end markers
                for stop in ["\nclass ", "\ndef ", "\n# ", "\nif __name__", "\nprint("]:
                    if stop in completion:
                        completion = completion[:completion.index(stop)]
                entry = {"task_id": tid, "completion": completion}
                f.write(json.dumps(entry) + "\n")

    print(f"Saved {len(task_ids) * args.n_samples} completions to {output_file}")
    print(f"\nTo evaluate, run:")
    print(f"  evalplus.evaluate --dataset {args.dataset} --samples {output_file}")


if __name__ == "__main__":
    main()

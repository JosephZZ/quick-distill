"""
Evaluate function calling accuracy on BFCL-format test data.
Uses vLLM for generation, then compares with ground truth.

Metrics:
- name_acc: correct function name(s) called
- full_acc: correct name + all required arguments match
"""

import json
import argparse
import os
import re
import sys
import subprocess
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a helpful assistant with access to functions. "
    "When the user's request can be fulfilled by calling a function, "
    "respond with a JSON array of function calls like: "
    '[{"name": "function_name", "arguments": {"arg1": "value1"}}]\n'
    "If no function is needed, respond normally."
)


def parse_function_calls(text):
    """Parse function calls from model output. Returns list of {name, arguments} dicts."""
    text = text.strip()

    # Try parsing as JSON array directly
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict) and "name" in parsed:
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code blocks
    code_block = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if code_block:
        try:
            parsed = json.loads(code_block.group(1).strip())
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict) and "name" in parsed:
                return [parsed]
        except json.JSONDecodeError:
            pass

    # Try extracting <tool_call> tags (hermes format)
    tool_calls = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
    if tool_calls:
        results = []
        for tc in tool_calls:
            try:
                parsed = json.loads(tc)
                if isinstance(parsed, dict) and "name" in parsed:
                    results.append(parsed)
            except json.JSONDecodeError:
                pass
        if results:
            return results

    # Try finding JSON objects/arrays in text
    for match in re.finditer(r'(\[[\s\S]*?\]|\{[\s\S]*?\})', text):
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list) and len(parsed) > 0:
                if all(isinstance(item, dict) and "name" in item for item in parsed):
                    return parsed
            elif isinstance(parsed, dict) and "name" in parsed:
                return [parsed]
        except json.JSONDecodeError:
            continue

    return []


def normalize_value(v):
    """Normalize a value for comparison."""
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, bool):
        return v
    if isinstance(v, list):
        return [normalize_value(x) for x in v]
    return v


def check_name_match(predicted_calls, ground_truth):
    """Check if the predicted function names match ground truth.
    ground_truth is a list of possible answer sets (any one is correct).
    Each answer set is a list of {func_name: {args}} dicts.
    """
    for gt_option in ground_truth:
        gt_names = set()
        for call_dict in gt_option:
            gt_names.update(call_dict.keys())

        pred_names = set(c.get("name", "") for c in predicted_calls)
        if gt_names == pred_names:
            return True
    return False


def check_full_match(predicted_calls, ground_truth):
    """Check if predicted function calls fully match ground truth.
    ground_truth format: [[{func_name: {arg: [possible_values]}}]]
    """
    for gt_option in ground_truth:
        if _match_one_gt(predicted_calls, gt_option):
            return True
    return False


def _match_one_gt(predicted_calls, gt_calls):
    """Check if predicted calls match one ground truth option."""
    # Build predicted lookup: name -> arguments
    pred_by_name = {}
    for call in predicted_calls:
        name = call.get("name", "")
        args = call.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except:
                args = {}
        pred_by_name[name] = args

    # Check each gt call
    for gt_call_dict in gt_calls:
        for func_name, expected_args in gt_call_dict.items():
            if func_name not in pred_by_name:
                return False

            pred_args = pred_by_name[func_name]

            # Check each expected argument
            for arg_name, possible_values in expected_args.items():
                if arg_name not in pred_args:
                    # Check if arg has a default / is optional
                    if possible_values == [""] or possible_values == [None]:
                        continue
                    return False

                pred_val = normalize_value(pred_args[arg_name])

                # possible_values is a list of acceptable values
                matched = False
                for pv in possible_values:
                    if normalize_value(pv) == pred_val:
                        matched = True
                        break
                    # Also try string comparison
                    if str(normalize_value(pv)) == str(pred_val):
                        matched = True
                        break
                if not matched:
                    return False

    return True


def generate_responses(model_path, problems, output_dir, max_new_tokens=512,
                       temperature=0.0, gpu_id=1, gpu_memory_utilization=0.85,
                       batch_size=8):
    """Generate responses using vLLM."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Build prompts
    prompts = []
    for p in problems:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": p["problem"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompts.append(prompt)

    print(f"Loading model {model_path} with vLLM...")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=2048,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        temperature=temperature if temperature > 0 else 0,
        top_p=1.0 if temperature == 0 else 0.95,
        max_tokens=max_new_tokens,
    )

    print(f"Generating {len(prompts)} responses with vLLM...")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    responses = [output.outputs[0].text for output in outputs]
    print(f"Generated {len(responses)} responses")

    # Free GPU memory
    del llm
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return responses


def evaluate(problems, responses):
    """Evaluate function calling accuracy."""
    results = {
        "total": len(problems),
        "name_correct": 0,
        "full_correct": 0,
        "parse_failures": 0,
        "by_category": {},
    }

    details = []
    for i, (prob, resp) in enumerate(zip(problems, responses)):
        cat = prob.get("category", "unknown")
        gt = prob["ground_truth"]
        # Normalize gt to list-of-lists format: [[{func: args}, ...], ...]
        # BFCL format is [{func: args}, ...] — wrap as single option
        if gt and isinstance(gt[0], dict):
            gt = [gt]

        if cat not in results["by_category"]:
            results["by_category"][cat] = {"total": 0, "name_correct": 0, "full_correct": 0}
        results["by_category"][cat]["total"] += 1

        # Parse response
        calls = parse_function_calls(resp)
        if not calls:
            results["parse_failures"] += 1
            details.append({
                "id": prob.get("id", i),
                "category": cat,
                "name_match": False,
                "full_match": False,
                "parsed_calls": [],
                "response": resp[:200],
            })
            continue

        name_match = check_name_match(calls, gt)
        full_match = check_full_match(calls, gt)

        if name_match:
            results["name_correct"] += 1
            results["by_category"][cat]["name_correct"] += 1
        if full_match:
            results["full_correct"] += 1
            results["by_category"][cat]["full_correct"] += 1

        details.append({
            "id": prob.get("id", i),
            "category": cat,
            "name_match": name_match,
            "full_match": full_match,
            "parsed_calls": calls,
            "response": resp[:200],
        })

    results["name_acc"] = results["name_correct"] / results["total"] if results["total"] > 0 else 0
    results["full_acc"] = results["full_correct"] / results["total"] if results["total"] > 0 else 0
    results["parse_rate"] = 1 - results["parse_failures"] / results["total"] if results["total"] > 0 else 0

    for cat_data in results["by_category"].values():
        cat_data["name_acc"] = cat_data["name_correct"] / cat_data["total"] if cat_data["total"] > 0 else 0
        cat_data["full_acc"] = cat_data["full_correct"] / cat_data["total"] if cat_data["total"] > 0 else 0

    return results, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model path or HF name")
    parser.add_argument("--eval_data", type=str, default="data/funcall/eval_bfcl.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.70)
    parser.add_argument("--categories", type=str, default="simple,multiple",
                        help="BFCL categories to evaluate (comma-separated)")
    args = parser.parse_args()

    # Load eval data
    with open(args.eval_data) as f:
        all_problems = [json.loads(l) for l in f]

    # Filter by category
    cats = set(args.categories.split(","))
    problems = [p for p in all_problems if p.get("category", "unknown") in cats]
    print(f"Evaluating {len(problems)} problems (categories: {cats})")

    # Generate
    responses = generate_responses(
        args.model, problems, args.output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        gpu_id=args.gpu_id,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    if responses is None:
        print("Generation failed!")
        return

    # Evaluate
    results, details = evaluate(problems, responses)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.output_dir, "details.json"), "w") as f:
        json.dump(details, f, indent=2)

    # Print
    print(f"\n=== Results ===")
    print(f"Total: {results['total']}")
    print(f"Parse rate: {results['parse_rate']:.1%}")
    print(f"Name accuracy: {results['name_acc']:.1%} ({results['name_correct']}/{results['total']})")
    print(f"Full accuracy: {results['full_acc']:.1%} ({results['full_correct']}/{results['total']})")
    print(f"\nBy category:")
    for cat, data in results["by_category"].items():
        print(f"  {cat}: name={data['name_acc']:.1%} full={data['full_acc']:.1%} ({data['total']} problems)")


if __name__ == "__main__":
    main()

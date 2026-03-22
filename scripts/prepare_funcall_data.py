"""
Prepare function calling training data from glaive-function-calling-v2
and BFCL eval data into a unified format.

Training: glaive single-turn function calls → JSONL with 'problem' field
Eval: BFCL v3 simple/multiple/parallel → JSONL with 'problem', 'ground_truth', 'category'
"""

import json
import os
import random
import re
import argparse
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a helpful assistant with access to functions. "
    "When the user's request can be fulfilled by calling a function, "
    "respond with a JSON array of function calls like: "
    '[{"name": "function_name", "arguments": {"arg1": "value1"}}]\n'
    "If no function is needed, respond normally."
)


def parse_glaive_tools(system_text):
    """Extract tool definitions from glaive system prompt."""
    # Remove the prefix
    text = system_text.replace("SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n", "")
    text = text.replace("SYSTEM: You are a helpful assistant, with no access to external functions.\n", "")
    text = text.strip()
    if not text:
        return None

    # Parse JSON tool definitions (may have multiple separated by newlines)
    tools = []
    # Try to find JSON objects
    depth = 0
    start = None
    for i, c in enumerate(text):
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    tool = json.loads(text[start:i+1])
                    tools.append(tool)
                except json.JSONDecodeError:
                    pass
                start = None
    return tools if tools else None


def parse_glaive_single_turn(chat_text):
    """Parse function call examples from glaive chat format.
    Returns (query, function_call_str) or None.
    Accepts: single USER → ASSISTANT with functioncall (may have clarification steps)."""
    parts = chat_text.split("\n\n\n")

    user_msgs = []
    function_calls = []

    for part in parts:
        part = part.strip()
        if part.startswith("USER:"):
            user_msgs.append(part[5:].strip())
        elif "<functioncall>" in part:
            # Use greedy match to capture full JSON with nested braces
            fc_match = re.search(r'<functioncall>\s*(\{.+\})\s*(?:<\|endoftext\|>)?', part, re.DOTALL)
            if fc_match:
                function_calls.append(fc_match.group(1))

    # Accept examples with exactly one function call and at least one user message
    if len(function_calls) != 1 or len(user_msgs) == 0:
        return None

    # Use the last user message before the function call as the query
    query = user_msgs[-1] if len(user_msgs) > 1 else user_msgs[0]
    return query, function_calls[0]


def format_tools_for_prompt(tools):
    """Format tool definitions as a readable string for the prompt."""
    lines = []
    for tool in tools:
        name = tool.get("name", "unknown")
        desc = tool.get("description", "")
        params = tool.get("parameters", {})
        lines.append(f"Function: {name}")
        lines.append(f"  Description: {desc}")
        if "properties" in params:
            lines.append("  Parameters:")
            for pname, pinfo in params["properties"].items():
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                req = pname in params.get("required", [])
                lines.append(f"    - {pname} ({ptype}{'*' if req else ''}): {pdesc}")
        lines.append("")
    return "\n".join(lines)


def format_tools_for_prompt_json(tools):
    """Format tool definitions as JSON for the prompt (compact)."""
    # Keep it compact but readable
    formatted = []
    for tool in tools:
        formatted.append({
            "name": tool.get("name", tool.get("function", {}).get("name", "unknown")),
            "description": tool.get("description", tool.get("function", {}).get("description", "")),
            "parameters": tool.get("parameters", tool.get("function", {}).get("parameters", {})),
        })
    return json.dumps(formatted, indent=2)


def prepare_glaive_training(num_train=3200, seed=42):
    """Load glaive and extract single-turn function calling examples."""
    from datasets import load_dataset
    print("Loading glaive-function-calling-v2...")
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")

    examples = []
    for ex in ds:
        system = ex["system"]
        chat = ex["chat"]

        tools = parse_glaive_tools(system)
        if not tools:
            continue

        parsed = parse_glaive_single_turn(chat)
        if parsed is None:
            continue

        query, fc_str = parsed

        # Validate function call JSON - glaive uses single quotes around arguments
        try:
            # Replace single quotes wrapping the arguments JSON string
            fc_clean = re.sub(r"'(\{[^']*\})'", r'\1', fc_str)
            fc_clean = fc_clean.replace("'", '"')  # fallback
            fc = json.loads(fc_clean)
        except json.JSONDecodeError:
            try:
                fc = json.loads(fc_str)
            except json.JSONDecodeError:
                continue
        if "name" not in fc:
            continue
        # Normalize arguments
        args = fc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except:
                continue
        fc["arguments"] = args

        # Format the problem string
        tools_str = format_tools_for_prompt_json(tools)
        problem = f"Available functions:\n{tools_str}\n\nUser request: {query}"

        # Ground truth answer
        answer = json.dumps([{"name": fc["name"], "arguments": fc["arguments"]}])

        examples.append({
            "problem": problem,
            "answer": answer,
            "source": "glaive",
        })

    print(f"Extracted {len(examples)} single-turn function calling examples from glaive")

    random.seed(seed)
    random.shuffle(examples)

    train = examples[:num_train]
    print(f"Training set: {len(train)} examples")
    return train


def prepare_bfcl_eval():
    """Load BFCL v3 test data (simple + multiple + parallel + parallel_multiple)."""
    bfcl_dir = Path.home() / ".cache/huggingface/hub/datasets--gorilla-llm--Berkeley-Function-Calling-Leaderboard"
    # Find the snapshot dir
    snapshots = list((bfcl_dir / "snapshots").iterdir()) if (bfcl_dir / "snapshots").exists() else []
    if not snapshots:
        print("BFCL data not found in cache. Downloading...")
        from datasets import load_dataset
        try:
            load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard", split="train")
        except:
            pass
        snapshots = list((bfcl_dir / "snapshots").iterdir())

    snap_dir = snapshots[0]

    eval_examples = []
    categories = ["simple", "multiple", "parallel", "parallel_multiple"]

    for cat in categories:
        q_file = snap_dir / f"BFCL_v3_{cat}.json"
        a_file = snap_dir / f"possible_answer/BFCL_v3_{cat}.json"

        if not q_file.exists() or not a_file.exists():
            print(f"  Skipping {cat}: files not found")
            continue

        with open(q_file) as f:
            questions = [json.loads(l) for l in f]
        with open(a_file) as f:
            answers = [json.loads(l) for l in f]

        # Build answer lookup
        answer_map = {a["id"]: a["ground_truth"] for a in answers}

        for q in questions:
            qid = q["id"]
            user_content = q["question"][0][0]["content"]  # first turn, first message
            tools = q["function"]
            gt = answer_map.get(qid)

            if gt is None:
                continue

            tools_str = format_tools_for_prompt_json(tools)
            problem = f"Available functions:\n{tools_str}\n\nUser request: {user_content}"

            eval_examples.append({
                "id": qid,
                "problem": problem,
                "ground_truth": gt,
                "category": cat,
            })

        print(f"  {cat}: {len([e for e in eval_examples if e['category'] == cat])} examples")

    print(f"Total eval: {len(eval_examples)} examples")
    return eval_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/funcall")
    parser.add_argument("--num_train", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare training data
    train = prepare_glaive_training(num_train=args.num_train, seed=args.seed)
    train_file = os.path.join(args.output_dir, "train.jsonl")
    with open(train_file, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved training data: {train_file}")

    # Prepare eval data
    eval_data = prepare_bfcl_eval()
    eval_file = os.path.join(args.output_dir, "eval_bfcl.jsonl")
    with open(eval_file, "w") as f:
        for ex in eval_data:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved eval data: {eval_file}")

    # Print stats
    print(f"\n=== Summary ===")
    print(f"Training: {len(train)} problems → {train_file}")
    print(f"Eval: {len(eval_data)} problems → {eval_file}")
    print(f"System prompt: {SYSTEM_PROMPT[:80]}...")


if __name__ == "__main__":
    main()

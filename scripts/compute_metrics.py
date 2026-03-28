#!/usr/bin/env python3
"""Compute avg@k and maj@k from eval results.jsonl"""

import json
import sys
from pathlib import Path
from collections import Counter


def compute_metrics(results_file):
    """Compute pass@k, avg@k, maj@k from results.jsonl"""
    total_problems = 0
    pass_correct = 0
    total_samples = 0
    total_correct_samples = 0
    maj_correct = 0

    with open(results_file, 'r') as f:
        for line in f:
            result = json.loads(line.strip())
            total_problems += 1

            responses = result.get('responses', [])
            is_correct_list = [r['is_correct'] for r in responses]

            # pass@k: any correct
            if any(is_correct_list):
                pass_correct += 1

            # avg@k: average of all samples
            total_samples += len(is_correct_list)
            total_correct_samples += sum(is_correct_list)

            # maj@k: majority vote
            c = Counter(is_correct_list)
            maj = c.most_common(1)[0][0]  # True or False
            # For majority vote, we need ground truth comparison
            # maj is the majority prediction (True means majority predicted correct)
            # But we need to check if majority is actually correct
            # In this setup, is_correct=True means prediction matches GT
            # So if maj=True, majority of samples are correct → majority vote is correct
            if maj:
                maj_correct += 1

    pass_at_k = pass_correct / total_problems if total_problems > 0 else 0
    avg_at_k = total_correct_samples / total_samples if total_samples > 0 else 0
    maj_at_k = maj_correct / total_problems if total_problems > 0 else 0

    return {
        'total_problems': total_problems,
        'n_samples_per_problem': total_samples // total_problems if total_problems > 0 else 0,
        'pass_at_k': pass_at_k,
        'avg_at_k': avg_at_k,
        'maj_at_k': maj_at_k,
        'pass_correct': pass_correct,
        'total_correct_samples': total_correct_samples,
        'maj_correct': maj_correct,
    }


def update_summary(eval_dir):
    """Update summary.json with avg@k and maj@k"""
    results_file = Path(eval_dir) / 'results.jsonl'
    summary_file = Path(eval_dir) / 'summary.json'

    if not results_file.exists():
        print(f"No results.jsonl found in {eval_dir}")
        return

    if not summary_file.exists():
        print(f"No summary.json found in {eval_dir}")
        return

    # Compute metrics
    metrics = compute_metrics(results_file)

    # Load existing summary
    with open(summary_file, 'r') as f:
        summary = json.load(f)

    # Add new metrics
    summary['pass_at_k'] = round(metrics['pass_at_k'], 4)
    summary['avg_at_k'] = round(metrics['avg_at_k'], 4)
    summary['maj_at_k'] = round(metrics['maj_at_k'], 4)
    summary['pass_correct'] = metrics['pass_correct']
    summary['avg_correct_samples'] = metrics['total_correct_samples']
    summary['maj_correct'] = metrics['maj_correct']

    # Save updated summary
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Updated {summary_file}:")
    print(f"  pass@4: {metrics['pass_at_k']:.4f} ({metrics['pass_correct']}/{metrics['total_problems']})")
    print(f"  avg@4:  {metrics['avg_at_k']:.4f} ({metrics['total_correct_samples']}/{metrics['total_problems'] * metrics['n_samples_per_problem']})")
    print(f"  maj@4:  {metrics['maj_at_k']:.4f} ({metrics['maj_correct']}/{metrics['total_problems']})")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Update specific eval directory
        update_summary(sys.argv[1])
    else:
        # Update all math fullseq evals
        import glob
        pattern = '/sg-pvc/quick-distillation/checkpoints/scale-*-math-fullseq/eval_step_*'
        for eval_dir in glob.glob(pattern):
            if Path(eval_dir).is_dir():
                print(f"\nProcessing {eval_dir}...")
                update_summary(eval_dir)

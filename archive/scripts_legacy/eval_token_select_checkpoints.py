#!/usr/bin/env python3
"""
Evaluate all checkpoints for a token-select experiment.
"""
import sys
import os
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_token_select_checkpoints.py <experiment_name> [--gpu GPU_ID]")
        sys.exit(1)
    
    exp_name = sys.argv[1]
    gpu_id = 1
    if "--gpu" in sys.argv:
        idx = sys.argv.index("--gpu")
        if idx + 1 < len(sys.argv):
            gpu_id = int(sys.argv[idx+1])
    
    outdir = f"checkpoints/{exp_name}"
    if not os.path.exists(outdir):
        print(f"Directory {outdir} not found")
        return 1
    
    print(f"Evaluating {exp_name} on GPU {gpu_id}")
    
    for step in [50, 100, 150, 200]:
        lora_path = f"{outdir}/step_{step}"
        if not os.path.exists(lora_path):
            print(f"  No checkpoint at step_{step}, skipping")
            continue
            
        eval_dir = f"{outdir}/eval_step_{step}"
        if os.path.exists(f"{eval_dir}/summary.json"):
            print(f"  Eval for step_{step} already exists, skipping")
            continue
            
        merged_path = f"{outdir}/_eval_merged_step_{step}"
        
        print(f"  Merging step_{step}...")
        try:
            subprocess.run([
                "python", "-c", f'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("/sg-pvc/hfmodels/Qwen_Qwen2.5-Math-1.5B", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "{lora_path}")
merged = model.merge_and_unload()
merged.save_pretrained("{merged_path}")
AutoTokenizer.from_pretrained("/sg-pvc/hfmodels/Qwen_Qwen2.5-Math-1.5B", trust_remote_code=True).save_pretrained("{merged_path}")
print("Merge completed")
'''
            ], check=True, cwd=os.getcwd())
        except Exception as e:
            print(f"  Merge failed: {e}")
            continue
            
        print(f"  Running MATH-500 eval for step_{step}...")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        try:
            subprocess.run([
                "python", "eval_math500.py",
                "--model", merged_path,
                "--output_dir", eval_dir,
                "--n_samples", "4",
                "--temperature", "0.7",
                "--max_model_len", "4096",
                "--gpu_memory_utilization", "0.65"
            ], env=env, check=True, cwd=os.getcwd())
            print(f"  Eval for step_{step} completed")
        except Exception as e:
            print(f"  Eval failed: {e}")
        finally:
            if os.path.exists(merged_path):
                import shutil
                shutil.rmtree(merged_path, ignore_errors=True)
    
    print(f"Evaluation for {exp_name} finished.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

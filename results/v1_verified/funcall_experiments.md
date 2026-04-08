# Function Calling (BFCL) Experiments

## Qwen3-4B (student) → Qwen3-8B (teacher), pos-100

**Code**: `on_policy_distill_positional.py` (new code)
**Server**: scai5 GPU 1+3
**Config**: LoRA r=32, alpha=64, lr=5e-5, bs=16, n_samples=1, 3200 problems, 200 steps
**Eval**: BFCL 600 problems, simple+multiple categories

**Baseline**: Qwen3-4B = 73.00% full_acc (438/600)

| Step | full_correct | full_acc |
|------|-------------|----------|
| 50 | 454 | 75.67% |
| 100 | 443 | 73.83% |
| **150** | **456** | **76.00%** |
| 200 | 456 | 76.00% |

**Result: POSITIVE.** Best = 76.00% (+3.0pp over baseline).
Unlike math where Qwen3-4B→Qwen3-8B was negative (-3.8pp), funcall benefits from distillation.

## Qwen2.5-Math-1.5B (student) → Qwen3-1.7B (teacher), pos-100

**Code**: `on_policy_distill_positional.py` (new code)
**Server**: scai3 GPU 0+1
**Baseline**: 2.70% full_acc (16/600)

| Step | name_acc | full_acc | parse_fail |
|------|----------|----------|------------|
| 50 | 23.0% | 0.5% | 73.2% |
| **100** | **46.0%** | **6.3%** | 48.3% |

**Result: Weak but improving.** Only 2 checkpoints (training ran 200 steps but only saved at 50/100?).
Parse failures dominate — model slowly learning JSON format. Needs more training steps.
Better than V1 (0% full_acc) but still very weak compared to 4B student (76%).

## Notes
- V1 code funcall was broken (0% full_acc) due to template incompatibility
- New code with system_prompt="" may fix the issue
- Gemma funcall not yet started on scai

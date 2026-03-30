# Timing and Efficiency Results

Hardware: NVIDIA RTX A6000 (48GB), single GPU for both student and teacher.
Student: Qwen2.5-Math-1.5B, LoRA r=32. Batch size 16, n_samples=1, 3200 math problems.
Pos-100 uses HF generate (100 tokens), full-seq uses vLLM (max 3584 tokens).

## Per-Step Timing (averaged over 10 steps)

| Config | Gen (s) | Score (s) | Train (s) | **Total (s/step)** |
|--------|---------|-----------|-----------|-------------------|
| **Pos-100, Q3-1.7B teacher** | **5** | **1** | **2** | **~8** |
| Full-seq, Q3-1.7B teacher | 100-731 | 3-10 | 5-12 | **~120-740** |
| **Pos-100, Q3-4B teacher** | **5** | **1** | **2** | **~8** |
| Full-seq, Q3-4B teacher | 97-733 | 3-9 | 6-10 | **~110-750** |
| **Pos-100, Q3-8B teacher** | **5** | **1** | **2** | **~8** |
| Full-seq, Q3-8B teacher | 96-230 | 7-11 | 1-6 | **~110-250** (OOM issues) |

## Speedup Summary

| Teacher | Pos-100 avg | Full-seq avg | **Speedup** |
|---------|------------|-------------|------------|
| Q3-1.7B | ~8s | ~280s | **~35x** |
| Q3-4B | ~8s | ~210s | **~26x** |
| Q3-8B | ~8s | ~170s | **~21x** |

**Key finding**: Pos-100 is 21-35x faster per step than full-seq distillation. The dominant cost in full-seq is generation (100-730s vs 5s), since autoregressive generation of full sequences (avg ~1000 tokens) is extremely expensive. Teacher scoring and training time are similar between methods.

## Notes

- Full-seq generation time is highly variable (100-730s) due to variable sequence lengths and vLLM batching behavior.
- Pos-100 generation time is constant (~5s) since only 100 tokens are generated via HF generate.
- Full-seq with Q3-8B teacher OOMed on some steps (47GB GPU insufficient for 8B teacher + 1.5B student + vLLM), making it impractical on a single A6000.
- Pos-100 with Q3-8B teacher runs fine (~8s/step) since only short sequences need to be processed.
- Total training time for 200 steps: Pos-100 ≈ 27 minutes vs Full-seq ≈ 9-40 hours.

## GPU Memory

- Pos-100: Student (1.5B) + Teacher (1.7B/4B) fit comfortably on single 48GB GPU with room for training.
- Full-seq: Requires vLLM with gpu_memory_utilization=0.7-0.9, leaving limited memory for training.
- Full-seq + Q3-8B: **OOMs on 48GB GPU** — cannot fit 8B teacher + vLLM + student training on single GPU.
- Pos-100 + Q3-8B: Works fine on single 48GB GPU since no vLLM overhead and short sequences.

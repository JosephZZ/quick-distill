# Experiment Tracker (Live)
Last updated: 2026-04-01 08:30 PDT

## Verified Results (V1 Original Code)

### 1.7B Teacher Math Position Sweep

| N | Best avg@4 | Best Step | GPU | Time/step | Student Mem | Teacher Mem |
|---|-----------|-----------|-----|-----------|-------------|-------------|
| 5 | 58.20% | 150 | A100×2 | ~5s | ~5GB | ~4GB |
| 10 | 58.75% | 50 | A100×2 | ~6s | ~5GB | ~4GB |
| 20 | 57.65% | 50 | A100×2 | ~7s | ~5GB | ~4GB |
| 50 | 61.70% | 150 | A100×2 | ~8s | ~5GB | ~4GB |
| 100 | **63.15%** | 150 | A100×2 | ~11s | ~5GB | ~4GB |
| 150 | **63.90%** | 200 | A100×2 | ~14s | ~5GB | ~4GB |
| 200 | 62.45% | 100 | A100×2 | ~17s | ~5GB | ~4GB |
| Fullseq s50 | 61.30% | 50 | A100×2 | **~130s** | ~13GB | ~4GB |

### 4B Teacher Math

| N | Best avg@4 | Best Step | GPU | Time/step |
|---|-----------|-----------|-----|-----------|
| 50 | 61.70% | 150 | A100×2 | ~8s |
| 100 | **64.20%** | 200 | A100×2 | ~11s |

### 1.7B Coding (pos-100)

| Step | HE | HE+ | MBPP | MBPP+ |
|------|-----|------|------|-------|
| 150 | **40.2%** | 34.8% | 50.0% | 44.2% |
| 200 | 39.6% | 34.8% | 49.7% | 44.2% |

### 1.7B Fullseq Funcall (step 50)

| Metric | Value |
|--------|-------|
| full_acc | **0.5%** |
| name_acc | 7.5% |
| parse_rate | 8.7% |

## Training In Progress

| Experiment | Server | GPU | Step | Time/step | Mem | save_steps |
|-----------|--------|-----|------|-----------|-----|------------|
| 1.7B FS math | scai3 | A100 1+2 | 70/200 | 130s | 17+4GB | 10 |
| 1.7B FS coding | scai3 | A100 7 | 30/200 | 267s | 36GB | 50 |
| 1.7B FS funcall | scai5 | L40S 2 | 100/200 | 80s | 16GB | 10 |
| 4B pos-50 coding | scai4 | A6000 2+3 | running | ~8s | 5+9GB | 50 |
| Gemma pos-50 coding | UCLACG | A6000 0 | ~3 steps | — | ~8GB | 50 |
| Gemma pos-100 coding | UCLACG | A6000 1 | ~2 steps | — | ~8GB | 50 |

## Queued (waiting for GPU)

| Experiment | Blocked by |
|-----------|------------|
| 4B pos-50 funcall | After 4B pos-50 coding (scai4) |
| Gemma pos-50/100 funcall | After Gemma coding (UCLACG) |
| 1.7B pos-50/100 funcall | Need new code + A100 GPU |
| 4B fullseq (all tasks) | No free GPU |
| Gemma fullseq (all tasks) | No free GPU |

## OOM / Failures

| Experiment | GPU | Issue |
|-----------|-----|-------|
| V100 coding/funcall | scai1 V100 16GB | OOM: 16GB too small for coding gen |
| V1 Gemma | any | System role not supported (need new code) |
| V1 funcall data | scai1/scai3 | load_dataset doesn't handle .jsonl (patched) |
| scai3 FS math save | A100 | No space left (cleaned 187GB, resolved) |

## Mini-batch Size Notes

All V1 experiments use mini_bs=1 (gradient accumulation 16 steps to reach bs=16).
This matches the original successful experiments.

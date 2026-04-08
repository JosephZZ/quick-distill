# Training Status (Last updated: 2026-04-01)

## Currently Running

| Experiment | Server | GPU | ~Step | Time/step | save_steps | PID |
|-----------|--------|-----|-------|-----------|------------|-----|
| 1.7B fullseq math | scai3 | A100 1+2 | ~100/200 | ~130s | 10 | 860835 |
| 1.7B fullseq coding | scai3 | A100 7 | ~50/200 | ~267s | 50 | 785603 |
| 4B fullseq math | scai4 | GPU 5 | ~10/200 | ~177s | — | — |
| Gemma fullseq math | UCLACG | A6000 0 | just started | — | — | — |

## Queued (on scai3 GPU 7, sequential after fullseq coding)

1. 1.7B pos-50 funcall
2. 1.7B fullseq coding (V1 re-run)
3. 1.7B fullseq funcall (V1 re-run)

## Completed V1 Experiments

| Experiment | Server | Checkpoints | Eval Done? |
|-----------|--------|-------------|------------|
| 1.7B pos-5/10/20/50/100/150/200 math | scai3 | ✅ | ✅ |
| 1.7B pos-100 coding | scai3 | ✅ step 150,200 | ✅ |
| 1.7B fullseq math step 50 | scai3 | ✅ | ✅ |
| 1.7B fullseq funcall step 50 | scai5 | ✅ | ✅ (0.5%) |
| 4B pos-50 math | scai4 | ✅ | ✅ |
| 4B pos-100 math | scai4 | ✅ | ✅ |

## Known Issues
- **V1 fullseq funcall**: Gives catastrophically bad results (0.5% full_acc). Root cause unknown.
- **Checkpoint dir not found**: scai3 V1 fullseq math/coding checkpoints — process running from /home/antarachugh/ but no checkpoint files found. May need to check write permissions or disk space.
- **scai5 fullseq funcall**: Was at step ~130, status unclear (no running process found).

# Regime A: repeated single-matrix warm control

The unchanged V1 kernels repeatedly execute the real `blk.0.ffn_down.weight` matrix (M=5,120, K=17,408). There are 200 warmups and 5,000 CUDA-event-timed AB/BA pairs using the same two prequantized Q8_1 vectors.

| Kernel | Median ms | Mean ms | p5 ms | p95 ms | Std ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q2 | 0.016384 | 0.016573 | 0.015360 | 0.017408 | 0.002306 |
| TQ1 V1 | 0.027648 | 0.028005 | 0.026752 | 0.028736 | 0.001820 |

- TQ1/Q2 median ratio: **1.687500×**
- Frozen V1 ratio: **1.729730×**
- Control reproduction: **PASS**

This confirms the cache-resident single-matrix behavior without tuning for it.

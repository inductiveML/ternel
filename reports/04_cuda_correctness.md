# CUDA correctness

## Workload

- Tensor: `blk.0.ffn_down.weight`
- Shape: M=5,120, K=17,408 (89,128,960 weights)
- Activation vectors: 1,000 deterministic vectors (500 FP16-origin, 500 BF16-origin)
- Runtime activation path: Prism F32-to-Q8_1
- Output and accumulation: FP32
- Full CUDA outputs compared: 5,120,000
- Independently streamed reference rows: 32 stratified rows/vector

The original and TQ1 streaming CPU references are exactly equal on every
reference point. The model-wide exhaustive decoder separately proves their
logical weights and scale bits are identical for all groups.

| Comparison | Max abs | Mean abs | RMSE | Max relative | Cosine | Non-finite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TQ1 CUDA vs Prism CUDA (full) | 5.72204589844e-06 | 7.0360459722e-08 | 9.84332522182e-08 | 0.238418579102 | 1 | 0 |
| Prism CUDA vs CPU reference | 4.76837158203e-07 | 6.02997897658e-08 | 8.36925749697e-08 | 0.00107169744475 | 1 | 0 |
| TQ1 CUDA vs CPU reference | 4.76837158203e-07 | 5.912251072e-08 | 8.23282634322e-08 | 0.000936621916953 | 1 | 0 |

**Numerical gate: PASS.** TQ1 does not exceed the baseline error thresholds.

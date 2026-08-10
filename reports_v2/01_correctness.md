# V2 all-tensor correctness

**Gate: PASS.**

Every one of the 497 selected real tensors was evaluated with two matched inputs: one FP16-origin and one BF16-origin vector, both promoted to F32 and quantized once by Prism to the identical Q8_1 buffer used by Q2 and TQ1. This compared 8,304,640 F32 outputs. All per-tensor checksums were finite and nontrivial.

| Metric | Observed TQ1 vs Q2 | Limit |
| --- | ---: | ---: |
| Max absolute error | 1.43051147461e-06 | 0.000105722045898 |
| Mean absolute error | 4.89654238278e-08 | 1.07036045972e-06 |
| RMSE | 6.9674753174e-08 | 1.09843325222e-06 |
| Max relative error | 0.0928571428571 | diagnostic |
| Cosine similarity | 1 | >= 0.999999 |
| Non-finite values | 0 | 0 |

The frozen exhaustive representation proof remains unchanged: the selected 25,621,954,560 logical weights and 200,171,520 raw FP16 scale groups match tensor-by-tensor hashes with zero mismatches. This V2 pass is regression protection for the multi-tensor launch path, not a replacement for that proof.

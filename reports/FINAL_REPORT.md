VERDICT: PACKING_ONLY_KERNEL_FAIL

# BONSAI-TQ1-G128 final report

| Metric | Existing Bonsai | TQ1_G128 | Delta |
| --- | ---: | ---: | ---: |
| Bits/weight incl. scale | 2.125 | 1.750 | -17.6471% |
| Quantized tensor bytes | 7,143,546,880 | 5,882,920,960 | -17.6471% |
| Projected whole-model bytes | 7,165,121,600 | 5,904,495,680 | -17.5939% |
| Weight parity | source | 0 mismatches / 26,893,352,960 | exact |
| Scale parity | source | 0 mismatches / 210,104,320 | bit-exact |
| Major MLP GEMV latency (warm) | 0.016576 ms | 0.028672 ms | +72.97% |
| Major MLP throughput (warm) | 308880317 rows/s | 178571426 rows/s | -42.19% |
| Physical weight bandwidth (warm) | 1428.263 GB/s | 680.000 GB/s | -52.39% |
| Max numerical error vs reference | 4.76837158203e-07 | 4.76837158203e-07 | equal max |

## Answers

1. **Did ~7 GB actually become ~6 GB without altering the model?** Yes. The
   verified 7,165,121,600-byte file projects to 5,904,495,680 bytes with the
   exact same logical weights and raw FP16 scales. The packed sidecar itself is
   5,883,168,530 bytes because it contains the Q2 tensors plus its manifest,
   not the unchanged GGUF tensors.

2. **Can CUDA consume the new representation directly?** Yes. The kernel reads
   the 19,496,960-byte packed real-layer allocation directly; it has no full
   unpacked weight buffer or representation conversion.

3. **Is it at least as fast as the current kernel?** No under the required
   worse-cache-state rule. V1 is 5.41% faster
   cold-ish, but 72.97% slower warm, so the required
   worst ratio is 1.729730×.

4. **What exactly limits performance?** V0 was dominated by serialized
   divergent constant-LUT decoding. V1 removes that failure and becomes
   competitive when weights come from DRAM, but when the baseline weights are
   L2-resident its cheap 2-bit `byte_perm` decoder outruns TQ1's base-3 lookup,
   rolling-byte assembly, and DP4A path. Hardware counters were unavailable,
   so this attribution is an inference from the isolated warm/cold timings,
   decoder change, and generated resource counts—not a counter measurement.

5. **Is a full llama.cpp integration justified?** No. Packing is validated,
   but the warm-cache performance gate fails after the allowed V1 pass.

6. **Is a later sparse mask+sign kernel worth testing?** Not as a basic storage
   replacement. Exact zero density is 29.6951831996%
   (7,986,030,430 zeros), median groups have 90 nonzeros, and only
   1.3489270473% of
   groups have at most 80 nonzeros. At the median, mask+sign+scale needs about
   30 bytes/group versus TQ1's 28. A compute-oriented sparse research kernel
   would be a separate, lower-priority experiment.

## Gate record

- Packing: PASS
- Exhaustive losslessness: PASS
- Prism wrapper vs public graph: PASS
- Numerical correctness: PASS
- Performance: FAIL (>1.20× worst-cache ratio)
- Primary tensor: `blk.0.ffn_down.weight`, 5,120×17,408
- Prism commit: `9ca265a57f85f2117942490f421f64a226dd9847`

# TQ1_LUT23_SMEM_STREAMK implementation repair

## Outcome

The hot packed-code loop is now fully unrolled while `__launch_bounds__(256, 5)`
forces the RTX 6000 Ada compiler into the five-CTA residency envelope required by
the 680-CTA `K_CHUNK=512` grid. This replaces the original explicit
`#pragma unroll 1`, which left loop control and repeated slot-address arithmetic
inside the phase responsible for 83.82% of the original measured cycles.

The final build uses 48 registers/thread, 14,084 bytes of shared memory, zero
stack bytes, zero local memory, and zero spill loads or stores for all three
K_CHUNK specializations.

## Identity

- Source-bundle SHA-256: `f90f7b0f3f9362271997c6cded05cc5ed0f594af2051b0a0787b2d588eb99ae3`
- Benchmark-binary SHA-256: `7887b022a58bf9a24a9697dd01fa30b5c6812f85ed53f02736f4bfc76fe4f654`
- Build: `cmake --build build_lut23 --target benchmark-lut23-primary -j2`
- Raw measurement: `artifacts/results_lut23/revision_final_shared.json`

## Correctness

- Exhaustive reorder remains at zero ternary mismatches and zero FP16 scale-byte
  mismatches across 26,893,352,960 values and 210,104,320 groups.
- 128 random activation vectors plus 8 edge vectors passed.
- Maximum candidate/reference absolute error: `9.53674316406e-7` (limit
  `1.5e-6`).
- Non-finite outputs: zero.
- No unpacked weight allocation, activation change, separate LUT kernel, or
  separate reduction kernel was introduced.

## Shared-GPU development timing

The user explicitly authorized timing while foreign PID `2448364` retained
1,802 MiB and intermittently used the GPU. The paired CUDA-event data are useful
implementation evidence, but they do **not** replace the earlier clean
qualification record. Bimodal p95 values caused by the foreign workload are not
used as clean performance claims.

| K_CHUNK | New median | New/Q2 (bootstrap 95% CI) | New/V1 (bootstrap 95% CI) |
| ---: | ---: | ---: | ---: |
| 512 | 0.022400 ms | 1.2785× [1.2476, 1.2915] | 0.7595× [0.7584, 0.7633] |
| 1024 | 0.025376 ms | 1.4658× [1.4436, 1.4685] | 0.8688× [0.8607, 0.8808] |
| 2048 | 0.029632 ms | 1.8121× [1.8102, 1.8176] | 1.1140× [1.1114, 1.1154] |

Best development setting: **K_CHUNK=512**. In this shared run it clears both
preregistered warm-stop thresholds (`new/Q2 <= 1.30`, `new/V1 <= 0.90`). The
original K=512 clean result was 1.6853× Q2 and 0.9995× V1.

## Remaining qualification boundary

Nsight Compute hardware counters remain inaccessible with
`ERR_NVGPUCTRPERM`; therefore shared wavefronts/request and the other mandatory
counter gates remain unmeasured. The official frozen verdict is not changed by
this shared-GPU development run. No ring or full-model claim is made here.

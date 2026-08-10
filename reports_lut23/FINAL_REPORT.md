VERDICT: IMPLEMENTATION_NOT_QUALIFIED

# TQ1_LUT23_SMEM_STREAMK final falsification

The native kernel is lossless and the repaired warm result clears the continuation threshold. A final compliant revision removes one CTA barrier per g128 LUT build by loading contiguous activation intervals per warp and broadcasting them in registers. In the decisive six-real-matrix ring, it beats Q2 and improves on frozen V1, but the V1 gain is only 2.5%, missing the preregistered 10% requirement. Nsight Compute counters also remain unavailable, and the development run shared the GPU with a foreign process. The full-model traversal is therefore **NOT RUN**.

Even if the counter and shared-GPU qualification failures were waived, the performance branch would freeze as `STOP_FROZEN_MODEL_BRANCH` at the V1 ring gate. No further packed-kernel rescue is proposed.

## Identity and build

- Frozen Prism llama.cpp commit: `9ca265a57f85f2117942490f421f64a226dd9847`
- Bonsai GGUF revision: `abbae723028d71be674e71e1a71201a6f43fab22`
- GGUF SHA-256: `868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757`
- Reordered sidecar SHA-256: `421ff2ee4b7b8bd6f64774d916b8a6250ac3e1f71205b31aaabe4b8c5e274624`
- Workspace source-bundle SHA-256: `26c5526f05af4d56bf6fea012fb124df956af3b644065fa007e4c3773c4b2803`
- Kernel source SHA-256: `7b107488f7117fed409ec94ffbc7bb7642a6e4aab6ad2f7b37af6c244563fcbb`
- Benchmark binary SHA-256: `a7a82df0c16c62f60d9b3bd6c46fc57276fc98dd1243100a926db1e89f3dc89b`
- Build command: `cmake -S native/lut23 -B build_lut23 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 && cmake --build build_lut23 --target benchmark-lut23-primary -j2`
- The workspace is not a Git checkout, so the source-bundle hash is the reproducible experiment identity.

## Environment

- GPU: NVIDIA RTX 6000 Ada Generation, compute capability 8.9, 49,140 MiB
- Driver / CUDA compiler: 570.133.20 / CUDA 12.8 (`V12.8.93`)
- CMake / Python / uv: 3.28.3 / 3.12.13 / 0.11.19
- Current observed graphics/memory clocks: 2730 / 9501 MHz
- Foreign process during the latest development timing: PID 2473328, using 1802 MiB. It produced visible high-latency tails and prevents a clean qualification claim.

## Correctness and implementation

- Exhaustive reorder: 498 tensors, 210,104,320 g128 groups, 26,893,352,960 ternary values.
- Ternary/code mismatches: **0**; raw FP16 scale mismatches: **0**; illegal regular/tail codes: **0/0**.
- CUDA suite: 128 deterministic random vectors plus 8 edge vectors.
- Maximum TQ1/reference absolute error: `9.53674316406e-7` (limit `1.5e-6`); cosine similarity `1.0`; non-finites `0`.
- Six-matrix ring maximum TQ1/Q2 absolute delta: `9.53674316406e-7`; non-finites `0`.
- `uv run pytest -q`: **64 passed**.
- Compiler result for all three K_CHUNK specializations: 48 registers/thread, 13,060 bytes shared memory, 0-byte stack, 0 spill loads, and 0 spill stores.
- Contiguous per-warp Q8_1 loads plus register broadcasts replace the old shared activation staging array, removing one CTA-wide barrier per g128 group without changing the code-position-major weight layout or the FP32 LUT arithmetic.
- The benchmark allocates only packed Q2, packed TQ1, reordered codes/scales, Q8_1 activations, FP32 outputs, and the small Stream-K workspace. It allocates no unpacked weight copy.

## Warm primary tensor

Tensor: `blk.0.ffn_down.weight`, M=5120, K=17408. Ratios are new/baseline; brackets are paired-bootstrap 95% confidence intervals.

| K_CHUNK | New median | New/Q2 | New/V1 | Warm continuation |
| ---: | ---: | ---: | ---: | --- |
| 512 | 0.021472 ms | 1.2312x [1.2294, 1.2335] | 0.7518x [0.7372, 0.7551] | PASS |
| 1024 | 0.022432 ms | 1.3665x [1.3638, 1.3684] | 0.8375x [0.8367, 0.8399] | Q2 stop threshold exceeded |
| 2048 | 0.027712 ms | 1.6848x [1.6816, 1.6881] | 1.0358x [1.0297, 1.0377] | FAIL |

Best warm K_CHUNK: **512**. It satisfies both preregistered continuation checks (`new/Q2 <= 1.30`, `new/V1 <= 0.90`).

## Six-matrix streaming ring

The ring contains `blk.0` through `blk.5` FFN-down tensors, all M=5120 by K=17408. Each timed sample traverses all six distinct resident packed extents; all six launches are inside one CUDA-event interval. K_CHUNK=512 is the best streaming setting.

| Comparison | Baseline median | New median | Ratio of medians | Bootstrap 95% CI | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Q2 vs new | 0.179248 ms | 0.166752 ms | 0.9303x | [0.9268, 0.9582] | PASS: upper <= 1.10 |
| V1 vs new | 0.173360 ms | 0.169024 ms | 0.9750x | [0.9613, 0.9833] | **FAIL: upper > 0.90** |

The paired new/Q2 p95 ratio is 1.0943 because foreign work intermittently preempted one side of a pair. The ratio-of-medians result is stable, but this run is development evidence rather than a clean final qualification. Even the favorable 95% confidence bound versus V1 is 0.9613, well outside the 0.90 gate.

## Profiler diagnosis

Nsight Compute fails with `ERR_NVGPUCTRPERM`, so the mandatory shared-wavefront, DRAM, L2, stall, occupancy, and SASS counter gates are not measured. Static compiler evidence proves zero local-memory spills, but it cannot establish shared wavefronts/request `<=1.10`.

Internal K_CHUNK=512 phase accounting attributes 16.00% of cycles to LUT construction, 79.25% to packed-code consumption, and 4.75% to fused Stream-K reduction. The barrier repair reduced the LUT fraction, but the packed-code/shared-lookup phase remains dominant. Rejected shared-GPU load-staging variants were slower than the direct decoder.

## Stopping gate

| Stage | Result | Status |
| --- | --- | --- |
| Exhaustive lossless parity | 0 code/scale mismatches | PASS |
| Numerical correctness | max abs `9.53674316406e-7` | PASS |
| Static spills | 0 loads / 0 stores | PASS |
| Mandatory hardware counters | protected by host policy | NOT QUALIFIED |
| Warm primary continuation | 1.2312x Q2 / 0.7518x V1 | PASS |
| Six-matrix ring | 0.9303x Q2 / 0.9750x V1 | **FAIL V1 GATE** |
| Forced-cold diagnostic | stopped at ring gate | NOT RUN |
| Frozen-V2 full-model traversal | stopped at ring gate | NOT RUN |

Raw measurements are in `artifacts/results_lut23/dev_warp_broadcast_warm_shared.json`, `artifacts/results_lut23/dev_warp_broadcast_ring_k512_shared.json`, `artifacts/results_lut23/dev_warp_broadcast_ring_k1024_shared.json`, `artifacts/results_lut23/dev_warp_broadcast_ring_k2048_shared.json`, and `artifacts/results_lut23/dev_warp_broadcast_phase.json`.

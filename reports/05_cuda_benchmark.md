# CUDA benchmark

**VALIDITY: VALID.**

## Protocol

The real `blk.0.ffn_down.weight` matrix was measured with 200 warmups followed
by five repeats of 1,000 CUDA-event-timed launches per kernel. Inputs rotate
through the same 1,000 prequantized Q8_1 vectors. Cold-ish trials scrub
201,330,688 bytes (>2× the queried 100,663,296-byte L2)
outside the timed interval. Output checksums are consumed and no unpacked TQ1
matrix exists.

| Cache state | Kernel | Median ms | p5 ms | p95 ms | Physical GB/s | Original-byte GB/s | Rows/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Warm | Prism Q2_0 | 0.016576 | 0.016384 | 0.017408 | 1428.263 | 1428.263 | 308880317 |
| Warm | TQ1 V1 | 0.028672 | 0.027648 | 0.029568 | 680.000 | 825.714 | 178571426 |
| Cold-ish | Prism Q2_0 | 0.037888 | 0.037728 | 0.038882 | 624.865 | 624.865 | 135135129 |
| Cold-ish | TQ1 V1 | 0.035840 | 0.034816 | 0.036864 | 544.000 | 660.571 | 142857139 |

- Warm TQ1/baseline ratio: **1.729730×**
- Cold-ish TQ1/baseline ratio: **0.945946×**
- Required worse ratio: **1.729730×**
- 95% bootstrap CI, warm: [1.605948, 1.736434]
- 95% bootstrap CI, cold-ish: [0.945946, 0.945946]

The shared conversion-plus-GEMV warm medians are
0.019392 ms (Prism) and
0.030720 ms (TQ1).

- Launches per core GEMV: 1
- Timed launches per kernel/cache state: 5,000
- Baseline / TQ1 V1 registers per thread: 56 / 56
- Static shared memory: 384 / 16 bytes
- Theoretical register-limited occupancy: 75% / 75%; achieved occupancy unavailable
- Before: P0, 2505 MHz graphics,
  10001 MHz memory, 62.92 W,
  2% GPU utilization
- After: P0, 2505 MHz graphics,
  10001 MHz memory, 63.53 W,
  2% GPU utilization

## Single optimization pass

V0 used 32 divergent constant-memory LUT reads per 32-weight chunk and measured
27.100777× the baseline. Nsight Compute was attempted once,
but the host denied performance-counter access with `ERR_NVGPUCTRPERM`.
The single V1 repair switched to a read-only/L1 packed LUT, consumed each
five-trit entry once through a rolling byte buffer, and used signed DP4A. It cut
registers from 152 to
56 per thread with no spills and improved
the worst ratio to 1.729730×. No further tuning was performed.

Hardware-counter DRAM/L2 hit-rate and load-efficiency measurements are **NOT
AVAILABLE** because counter access was denied. The physical-byte rates above
are workload bytes divided by event time, not hardware-counter throughput.

**Performance gate: FAIL (>1.20× in the worse cache state).**

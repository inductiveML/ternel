# Regime B: full model-order streaming benchmark

**Primary V2 ratio: 1.256425×.**

Each complete traversal launches all 497 real matrix tensors once in model order with no explicit cache flush. The Q2 stream is 6,805,831,680 bytes (67.610× L2); TQ1 is 5,604,802,560 bytes (55.679× L2). Six full warmups precede 40 CUDA-event-timed AB/BA pairs. Checksums consume every traversal outside its end event.

| Stream | Median ms | Mean ms | p5 ms | p95 ms | Std ms | Physical GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q2 | 8.972416 | 9.001300 | 8.953750 | 9.071088 | 0.050831 | 758.528 |
| TQ1 V1 | 11.273168 | 11.671856 | 11.122242 | 12.698555 | 0.666663 | 497.181 |

- Ratio-of-medians TQ1/Q2: **1.256425×**
- Paired-ratio median: **1.255074×**
- Paired bootstrap 95% CI: **[1.249732, 1.292363]** (100,000 resamples)
- AB / BA ratios: 1.256961× / 1.256226×
- Q2 / TQ1 CV: 0.5647% / 5.7117%
- CUDA-process audit: 601 direct-NVML samples at a 10 ms requested interval; maximum observed gap 176.8 ms; zero foreign PIDs and zero monitor errors
- GPU before / after: P0 / P0, graphics 2505 / 2505 MHz, memory 10001 / 10001 MHz, power 63.16 / 62.04 W
- Timing trust checks: **PASS**
- Classification: **STREAMING_TQ1_FAIL**

The claim-bearing run preceded the forced-cold and per-tensor diagnostic passes. No TQ1 kernel modification was made before or after the primary measurement.

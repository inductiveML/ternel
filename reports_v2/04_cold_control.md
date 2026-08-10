# Regime C: forced cold-ish control

Before every timed `blk.0.ffn_down.weight` launch, the harness writes 201,330,688 unrelated bytes (>2× L2). The scrub is stream-ordered before the begin event and excluded from latency. This diagnostic intentionally mirrors the V1 cold-ish control; it is not the primary workload.

| Kernel | Median ms | Mean ms | p5 ms | p95 ms | Std ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q2 | 0.042880 | 0.042566 | 0.041984 | 0.043008 | 0.001436 |
| TQ1 V1 | 0.040960 | 0.041319 | 0.039936 | 0.042912 | 0.002419 |

- TQ1/Q2 median ratio: **0.955224×**
- TQ1 latency delta: **-4.478%**

The cache spectrum is therefore 1.6875× warm, 1.2564× model-streaming, and 0.9552× forced-cold.

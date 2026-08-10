V1 VERDICT (FROZEN): PACKING_ONLY_KERNEL_FAIL
V2 VERDICT: STREAMING_TQ1_FAIL

| Metric                         |              Q2 |             TQ1 |     Delta |
| ------------------------------ | --------------: | --------------: | --------: |
| Weight storage                 | 7,143,546,880 B | 5,882,920,960 B | -17.6471% |
| Whole-model projected bytes    | 7,165,121,600 B | 5,904,495,680 B | -17.5939% |
| Single-matrix warm latency     | 0.016384 ms | 0.027648 ms | +68.7500% |
| Forced-cold latency            | 0.042880 ms | 0.040960 ms | -4.4776% |
| Full-stream traversal latency  | 8.972416 ms | 11.273168 ms | +25.6425% |
| Full-stream TQ1/Q2 ratio       |       1.000000× | 1.256425× | +25.6425% |
| Full-stream effective GB/s     | 758.528 | 497.181 | -34.4546% |
| Numerical max error            |               0 | 1.43051147461e-06 | within gate |
| Oracle-hybrid latency          | 9.677776 ms | 9.603712 ms | -0.7653% |
| Oracle-hybrid projected memory | 7,165,121,600 B | 6,842,315,840 B | -4.5052% |

1. **Was the previous warm-matrix failure caused primarily by cache residency?** No. Model-order streaming still exceeds the 1.05× parity boundary, so cache residency does not explain away the execution penalty.
2. **Does actual model-order traversal behave more like the warm or cold regime?** It behaves closer to forced-cold: 1.2564× streaming versus 0.9552× cold and 1.6875× warm.
3. **Does TQ1 give 17.6% lower memory essentially for free?** No. The complete traversal costs 1.2564× baseline, outside the <=1.05× free-cost gate.
4. **Is TQ1 faster under realistic streaming?** No; it is 25.64% slower.
5. **Is there a tensor-size crossover?** Only tentatively: the sole 337,715,200-byte LM head is 0.66% faster and the size/ratio Spearman correlation is -0.906, but one tensor does not establish a clean size threshold. The smaller wins are geometry-specific FFN-down matrices.
6. **Would a mixed Q2/TQ1 model dominate either uniform format?** No clearly useful mixed Pareto point survives the preregistered usefulness threshold.
7. **Is full llama.cpp integration justified?** No. Stop; full llama.cpp integration is not justified by this result.

The V1 report files remain byte-for-byte unchanged. V2 used the existing validated V1 TQ1 kernel with no pre-primary tuning and no representation change.

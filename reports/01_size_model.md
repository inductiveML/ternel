# TQ1_G128 size model

## Block arithmetic

| Metric | Existing Q2_0/g128 | TQ1_G128 | Delta |
| --- | ---: | ---: | ---: |
| Values/group | 128 | 128 | 0 |
| Scale bytes/group | 2 | 2 | 0 |
| Symbol bytes/group | 32 | 26 | -6 |
| Total bytes/group | 34 | 28 | -6 |
| Bits/weight including scale | 2.125 | 1.75 | -0.375 |

Five trits are encoded per byte because `3^5 = 243`; the last three trits use
one canonical tail byte. Scales are copied byte-for-byte.

## Exact model projection

| Metric | Bytes |
| --- | ---: |
| Existing Q2_0 tensor payload | 7,143,546,880 |
| Projected TQ1 tensor payload | 5,882,920,960 |
| Unchanged F32 tensors | 10,582,016 |
| Existing GGUF metadata/alignment | 10,992,704 |
| Existing complete file | 7,165,121,600 |
| Projected replacement file | 5,904,495,680 |

- Quantized-payload reduction: **17.6470588235%**
- Whole-file reduction: **17.5939222022%**
- Projected size: **5.904495680 GB**
  (5.498990119 GiB)

## Gate 1

**PASS**: the actual alphabet is ternary, quantized storage falls by at least
15%, and the projected replacement file is below 6.1 decimal GB.

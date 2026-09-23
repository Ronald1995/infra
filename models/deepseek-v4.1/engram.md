# 基础流程
```text
                       INPUT TOKEN HISTORY
                              │
                              ▼
                 raw tokenizer IDs [T]
                              │
                     normalize / compress
                              │
                              ▼
                 compressed token IDs
                              │
             ┌────────────────┼────────────────┐
             │                │                │
           2-gram           3-gram           4-gram
             │                │                │
          8 hashes         8 hashes         8 hashes
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                      hash_ids [T,24]
                              │
                ┌─────────────┴─────────────┐
                │                           │
         layer1 table                 layer14 table
       ~384M × 256                   ~384M × 256
                │                           │
                ▼                           ▼
             lookup                      lookup
                │
                ▼
                         [T,24,256]
                              │
                           flatten
                              ▼
                          [T,6144]
                              │
                             Wkv
                              ▼
                         [T,25600]
                         /        \
                        /          \
                       ▼            ▼
              key [T,4,5120]   value [T,5120]
                       │            │
                       │            │
mHC residual ──────────┤            │
[T,4,5120]             │            │
                       ▼            │
             normalized similarity  │
                       │            │
                  signed sqrt       │
                       │            │
                    sigmoid         │
                       │            │
                       ▼            │
                   gate [T,4]       │
                       │            │
                       └──────┬─────┘
                              ▼
                   gate × shared value
                              │
                              ▼
                    residual + memory
                       [T,4,5120]
                              │
                            mHC pre
                              │
                              ▼
                   Attention input [T,5120]
```
# offload流程

```text
                         GPU MAIN STREAM
────────────────────────────────────────────────────────────

input_ids [T]
    │
    ▼
token_map lookup
    │
    ▼
compressed IDs
    │
    ▼
Triton ngram hash
    │
    ▼
hash_ids [T,2,24]
    │
    ├─────────────────────────────────────────────┐
    │                                             │
    ▼                                             │
start Engram prefetch                        continue forward
                                                  │
                                                  ▼
                                              layer 0
                                                  │
                                                  ▼
                                              ...
                                                  │
                                      wait only when Engram needed
                                                  │
                                                  ▼
                                             layer 1/14


                   GPU PREFETCH CUDA STREAM
────────────────────────────────────────────────────────────

hash_ids [T,24]
       │
       ▼
_engram_lookup_kernel
       │
       │ UVA memory loads
       │
       ▼

              CPU PINNED HOST MEMORY
       ┌─────────────────────────────┐
       │ FP8 Engram table shard      │
       │ UE8M0 scales                │
       └─────────────────────────────┘
       │
       │ GPU initiated read
       ▼
GPU registers
       │
       │ FP8 × scale
       ▼
BF16
       │
       ▼
staged_rows [T,local_heads,256]
       │
       └────────── ready


                       GPU MAIN STREAM
────────────────────────────────────────────────────────────

when layer 1/14 arrives:
       │
       ▼
wait_stream(prefetch_stream)
       │
       ▼
TP/DP row gather if needed
       │
       ▼
[T,24,256]
       │
       ▼
flatten [T,6144]
       │
       ▼
Wkv
       │
       ├── keys  [T,4,5120]
       └── value [T,5120]
              │
              ▼
      Engram gate Triton
              │
              ▼
residual + gate * value
[T,4,5120]
```
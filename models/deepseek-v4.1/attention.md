# 主体结构
```text
mHC PRE output
[T,5120]
    │
    │
    ├──────────────────────────────────────────────────────────────────────────┐
    │                                                                          │
    │                                                                          │
    ▼                                                                          ▼
┌──────────────────────────────┐                                 ┌─────────────────────────────┐
│ Main Q / SWA-KV Projection   │                                 │ Compressor Projection       │
│                              │                                 │ only on KV-source layers    │
│ fused_wqa_wkv                │                                 │ L2/L8/L14/L20              │
└──────────────┬───────────────┘                                 └─────────────┬───────────────┘
               │                                                               │
               ▼                                                               ▼
       qr_kv [T,1792]                                            hidden_states [T,5120]
               │                                                               │
       ┌───────┴────────┐                                                      │
       ▼                ▼                                                      ▼
 qr [T,1280]       kv [T,512]                                      fused_wkv_wgate
       │                │                                                      │
       │                │                                    ┌─────────────────┴──────────────┐
       │                │                                    │                                │
       ▼                ▼                                    ▼                                ▼
    q_norm           kv_norm                           C1A / ratio=1                   C2A / ratio=2
       │                │                              [T,512]                         [T,1024]
       ▼                ▼                                    │                      KV + gate score
 qr [T,1280]       kv [T,512]                               │                                │
       │                │                                    │                                ▼
       │                │                                    │                     2-token softmax pooling
       │                │                                    │                                │
       ▼                │                                    └──────────────┬─────────────────┘
     wq_b               │                                                   ▼
       │                │                                          compressed latent
       ▼                │                                               [T,512]
Q [T,local_heads,512]   │                                                   │
       │                │                                      only group-boundary rows
       │                │                                      are valid for C2A
       │                │                                                   │
       │                │                         ┌─────────────────────────┴─────────────────────────┐
       │                │                         │                                                   │
       │                │                         ▼                                                   ▼
       │                │             compressed main KV path                              Indexer-K path
       │                │                         │                                                   │
       │                │                         │ RoPE + quant                                      │ wk
       │                │                         ▼                                                   ▼
       │                │             compressed KV cache                                  index K [T,128]
       │                │                         │                                                   │
       │                │              C1A: 1 state/token                                            │ k_norm
       │                │              C2A: 1 state/2 tokens                                         │
       │                │                         │                                                   │ RoPE + quant
       │                │                         │                                                   ▼
       │                │                         │                                          Index K cache
       │                │                         │                                           [...,128]
       │                │                         │                                                   │
       │                │                         │                                                   │
       │                │                         │                                                   │
       │                │                         │                                                   │
       │                │                         │                                                   │
       │                │                         │                INDEXER QUERY PATH                 │
       │                │                         │                                                   │
       │                │                         │          qr [T,1280]                              │
       │                │                         │                │                                  │
       │                │                         │                │ indexer.wq_b                     │
       │                │                         │                ▼                                  │
       │                │                         │          [T,32,128]                               │
       │                │                         │                │                                  │
       │                │                         │                │ RoPE + quant                     │
       │                │                         │                ▼                                  │
       │                │                         │            Index Q                                │
       │                │                         │          [T,32,128]                               │
       │                │                         │                │                                  │
       │                │                         │                │                                  │
       │                │                         │ hidden [T,5120] │                                  │
       │                │                         │        │       │                                  │
       │                │                         │ weights_proj    │                                  │
       │                │                         │        │       │                                  │
       │                │                         │        ▼       │                                  │
       │                │                         │ index weights   │                                  │
       │                │                         │    [T,32]       │                                  │
       │                │                         │        │       │                                  │
       │                │                         │        └───┬───┘                                  │
       │                │                         │            │                                      │
       │                │                         │            ▼                                      │
       │                │                         │   ┌───────────────────────┐                        │
       │                │                         │   │ Sparse Indexer Score  │◄───────────────────────┘
       │                │                         │   │                       │
       │                │                         │   │ Q × historical K      │
       │                │                         │   │ 32 index heads        │
       │                │                         │   │ + per-head weights    │
       │                │                         │   └──────────┬────────────┘
       │                │                         │              │
       │                │                         │              ▼
       │                │                         │        sparse scores
       │                │                         │              │
       │                │                         │              │ TopK
       │                │                         │              ▼
       │                │                         │     topk_indices [T,512]
       │                │                         │              │
       │                │                         │              │
       │                │                         └──────────────┼──────────────────────────────┐
       │                │                                        │                              │
       │                │                                        │                              │
       │                │                                        │                              │
       │                │                                        │                              │
       │                │                                        │                              │
       │                │                                        │                              │
       │                │                      COMPRESSED LONG-RANGE ATTENTION                   │
       │                │                                        │                              │
       │                │                                        ▼                              │
       │                │                             compressed KV cache                        │
       │                │                                        │                              │
       │                │                            gather topk_indices                         │
       │                │                                        │                              │
       │                │                                        ▼                              │
       │                │                             ≤512 compressed KV                         │
       │                │                                                                       │
       │                │                                                                       │
       │                │                                                                       │
       │                │                         SWA PATH                                      │
       │                │                                                                       │
       │                ▼                                                                       │
       │         KV RoPE + quant                                                                │
       │                │                                                                       │
       │                │ slot_mapping                                                          │
       │                ▼                                                                       │
       │       per-layer SWA KV cache                                                           │
       │            window = 128                                                                │
       │                │                                                                       │
       │                │                                                                       │
       │                ▼                                                                       │
       │      DeepseekSparseSWAMetadata                                                         │
       │                │                                                                       │
       │       ┌────────┴────────┐                                                              │
       │       ▼                 ▼                                                              │
       │ decode_swa_indices   swa_lens                                                          │
       │ [T, ≤128]            [T]                                                               │
       │       │                                                                                │
       │       │ normal:                                                                       │
       │       │ start=max(pos-127,0)                                                           │
       │       │                                                                                │
       │       │ bounded replay:                                                                │
       │       │ start=max(pos-127,replay_start)                                                 │
       │       │                                                                                │
       │       ▼                                                                                │
       │ recent SWA KV ≤128                                                                     │
       │       │                                                                                │
       │       │                                                                                │
       └───────┼─────────────────────────────────────┐                                          │
               │                                     │                                          │
               │                                     │                                          │
               ▼                                     ▼                                          │
      ┌─────────────────┐                    ┌───────────────────┐                               │
      │ SWA KV ≤128     │                    │ Compressed KV     │◄──────────────────────────────┘
      │ exact recent KV │                    │ TopK ≤512         │
      └────────┬────────┘                    └─────────┬─────────┘
               │                                       │
               └──────────────────┬────────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ One Sparse MLA Attention │
                    │                          │
                    │ Q                        │
                    │    ×                     │
                    │ {SWA KV ∪ TopK KV}       │
                    │                          │
                    │ + attention sink         │
                    │                          │
                    │ ONE shared softmax       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                     attention output
                    [T,local_heads,512]
                                 │
                                 ▼
                               wo_a
                                 │
                     grouped low-rank projection
                                 │
                                 ▼
                               wo_b
                                 │
                                 ▼
                           [T,5120]
                                 │
                                 ▼
                             mHC POST
```

# swa bouned replay

```text
                   PREFIX CACHE HIT
                         H
                         │
                         ▼
              replay_start = H - 128
                         │
                         ▼
                rewind computed count
                         │
                         ▼
                replay [H-128, H)
                         │
             ┌───────────┴────────────┐
             │                        │
             ▼                        ▼
   compressed/indexer cache         SWA cache
       already cached               missing
             │                        │
             ▼                        ▼
    slot = PAD_SLOT_ID             real slot
             │                        │
             ▼                        ▼
       don't rewrite               rebuild
                                      │
                                      ▼
                               each replay token
                                      │
                                      ▼
                         start = max(
                             pos - 127,
                             replay_start
                         )
                                      │
                                      ▼
                          visible logical positions
                                      │
                                      ▼
                                block_table
                                      │
                                      ▼
                           physical SWA slot IDs
                                      │
                                      ▼
                        prefill/decode SWA indices
                                      │
                                      ▼
                              Sparse MLA reads
                                      │
                                      ▼
                         SWA gradually fills up
                                      │
                                      ▼
                         after replay finishes

                         SWA = last 128 KV
                                      │
                                      ▼
                              normal decode
```
# 整体流程
```text
                         layer 输入状态
────────────────────────────────────────────────────────

previous sublayer output x
[T,5120]

mHC residual
[T,4,5120]

previous pre_mix
[T,4]

previous post_mix
[T,4,1]

previous res_mix
[T,4,4]


                         │
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ 1. Previous sublayer mHC POST   │
        │                                 │
        │ res_mix × residual              │
        │        +                        │
        │ post_mix × x                    │
        └────────────────┬────────────────┘
                         │
                         ▼
              updated residual
               [T,4,5120]
                         │
                         │
                         ├───────────────────────────────┐
                         │                               │
                         │                               │
                         ▼                               ▼
        ┌───────────────────────────┐     ┌───────────────────────────┐
        │ 2. mHC coefficient GEMM  │     │ 3. current layer input   │
        │                           │     │                           │
        │ flatten residual          │     │ previous pre_mix          │
        │ [T,4,5120]                │     │ [T,4]                     │
        │      ↓                    │     │      │                    │
        │ [T,20480]                 │     │      │ weighted collapse  │
        │      │                    │     │      ▼                    │
        │      │ @ hc_fn.T          │     │ residual                 │
        │      │ [20480,24]         │     │ [T,4,5120]                │
        │      ▼                    │     │      │                    │
        │ mixes [T,24]              │     │      ▼ sum over HC dim   │
        └────────────┬──────────────┘     │ [T,5120]                 │
                     │                    │      │                    │
                     │                    │      ▼ RMSNorm            │
                     │                    │ layer_input              │
                     │                    │ [T,5120]                 │
                     │                    └────────────┬──────────────┘
                     │                                 │
                     ▼                                 │
       ┌─────────────────────────────┐                  │
       │ 4. split 24 coefficients   │                  │
       │                             │                  │
       │ mixes[:, 0:4]               │                  │
       │      ↓ sigmoid              │                  │
       │ next_pre_mix [T,4]          │                  │
       │                             │                  │
       │ mixes[:, 4:8]               │                  │
       │      ↓ sigmoid × 2          │                  │
       │ post_mix [T,4,1]            │                  │
       │                             │                  │
       │ mixes[:, 8:24]              │                  │
       │      ↓ reshape              │                  │
       │ [T,4,4]                     │                  │
       │      ↓ softmax + Sinkhorn   │                  │
       │ res_mix [T,4,4]             │                  │
       └──────────────┬──────────────┘                  │
                      │                                 │
                      │                                 ▼
                      │                       ┌───────────────────────┐
                      │                       │ 5. Attention          │
                      │                       │                       │
                      │                       │ input [T,5120]        │
                      │                       │       ↓               │
                      │                       │ Sparse MLA            │
                      │                       │       ↓               │
                      │                       │ output [T,5120]       │
                      │                       └───────────┬───────────┘
                      │                                   │
                      │                                   │
                      │       attention post_mix          │
                      │       attention res_mix           │
                      │                                   │
                      └───────────────────────┬───────────┘
                                              │
                                              ▼
                     ┌─────────────────────────────────────┐
                     │ 6. Attention POST                   │
                     │                                     │
                     │ residual [T,4,5120]                 │
                     │          │                          │
                     │          ├─ res_mix [T,4,4]         │
                     │          │                          │
                     │          ▼                          │
                     │ mixed residual [T,4,5120]           │
                     │                                     │
                     │ attention output [T,5120]           │
                     │          │                          │
                     │          ├─ post_mix [T,4,1]        │
                     │          ▼                          │
                     │ post term [T,4,5120]                │
                     │                                     │
                     │ mixed residual + post term          │
                     └────────────────┬────────────────────┘
                                      │
                                      ▼
                         residual_after_attn
                           [T,4,5120]
                                      │
                                      │
                                      ├────────────────────────────┐
                                      │                            │
                                      ▼                            ▼
                    ┌──────────────────────────┐     ┌────────────────────────┐
                    │ 7. FFN coefficient GEMM│     │ 8. FFN input collapse │
                    │                          │     │                        │
                    │ flatten                  │     │ attn_pre_mix [T,4]     │
                    │ [T,4,5120]               │     │        │               │
                    │      ↓                   │     │        │ × residual    │
                    │ [T,20480]                │     │        ▼               │
                    │      │                   │     │ [T,4,5120]             │
                    │      │ @ hc_ffn_fn.T     │     │        │               │
                    │      ▼                   │     │        ▼ sum HC dim    │
                    │ mixes [T,24]             │     │ [T,5120]              │
                    └────────────┬─────────────┘     │        │               │
                                 │                   │        ▼ RMSNorm       │
                                 ▼                   │ FFN input [T,5120]    │
                   ┌───────────────────────────┐      └────────────┬───────────┘
                   │ 9. split FFN coefficients│                   │
                   │                           │                   ▼
                   │ ffn_pre_mix [T,4]         │         ┌───────────────────┐
                   │ ffn_post_mix [T,4,1]      │         │ 10. MoE FFN       │
                   │ ffn_res_mix [T,4,4]       │         │                   │
                   └─────────────┬─────────────┘         │ input [T,5120]    │
                                 │                       │       ↓           │
                                 │                       │ MoE top-k experts │
                                 │                       │       ↓           │
                                 │                       │ output [T,5120]   │
                                 │                       └─────────┬─────────┘
                                 │                                 │
                                 └───────────────────┬─────────────┘
                                                     │
                                                     ▼
                          layer 输出 / carry to next layer
──────────────────────────────────────────────────────────────────

x                  = FFN output
                     [T,5120]

residual           = residual_after_attn
                     [T,4,5120]

pre_mix            = ffn_pre_mix
                     [T,4]

post_mix           = ffn_post_mix
                     [T,4,1]

res_mix            = ffn_res_mix
                     [T,4,4]

                                                     │
                                                     │
                                                     ▼
                           下一层 Attention seam
                                                     │
                                                     ▼
                    FFN POST + next Attention PRE
                          在这里继续融合
```

# mix计算

```text
                     PRE

R0 ──×p0──┐
R1 ──×p1──┤
R2 ──×p2──┼── sum ──→ X_in [H]
R3 ──×p3──┘

             4 streams → 1 stream


                 ATTENTION / FFN

X_in [H] ─────────────→ X_out [H]


                     POST

旧 R0 ─┐
旧 R1 ─┼── res_mix [4×4] ──→ M0 M1 M2 M3
旧 R2 ─┤
旧 R3 ─┘

X_out ── post_mix [4] ─────→ q0X q1X q2X q3X

                               │
                               ▼
                         elementwise add
                               │
                               ▼
                       R'0 R'1 R'2 R'3

             1 stream → 4 streams
             + old 4 streams → new 4 streams
```
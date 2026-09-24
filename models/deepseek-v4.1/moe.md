# 主要结构
```text
hidden [T,5120]
      │
      ▼
GateLinear
[T,384]
      │
      ▼
sqrtsoftplus
      │
      ├── + noaux correction bias
      │          ↓
      │        Top-6
      │          │
      └──────────┘
            gather unbiased score
                  │
             normalize × 1.5
                  │
                  ▼
         6 routed FP4 experts
                  │
                  ▼
          weighted routed sum
                  │
                  │
shared expert ────┤
                  ▼
             MoE output
[T,5120]
```
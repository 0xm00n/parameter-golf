# RDQuant: Per-Layer Rate-Distortion Quantization

## Summary

Per-layer R-D optimized mixed quantization with dead-zone support, built on the proven MLP3x+SmearGate+BigramHash+SWA recipe.

**val_bpb: TBD** | **Artifact: TBD** | 8xH100 SXM, 600s

## Key Techniques

### Base Recipe (proven)
1. **3x MLP** (hidden=1536)
2. **SmearGate** — learned gate blending token with previous token's embedding
3. **BigramHash** — 4096-bucket hash embedding for token-pair context
4. **Orthogonal init + muP** scaling
5. **SWA** over last 50% of training
6. **Sliding window eval** (stride=64)
7. **Muon WD + AdamW** with tuned hyperparameters

### Novel: Per-Layer R-D Quantization with Dead-Zone
- Greedy per-layer quantization allocation based on rate-distortion analysis
- Each layer independently assigned int4/int5/int6/int8 with configurable dead-zone
- Dead-zone quantization: values within threshold mapped to zero, creating more zeros for better zstd compression
- Configurable via `QUANT_CONFIG` environment variable (JSON)

## R-D Allocation Results

Per-layer sensitivity analysis on a 4L x 832W test model (20.2M params):

| Sensitivity | Layers | int6→int4 BPB cost |
|---|---|---|
| **Cheapest** | blocks.2/3 attention (c_q, c_k, c_v) | +0.0000 to +0.0001 |
| **Mid** | blocks.1 attention, blocks.2/3 MLP fc | +0.0001 to +0.0004 |
| **Most expensive** | blocks.0 MLP, blocks.0/1 attn.proj | +0.0014 to +0.0035 |

Full aggressive allocation (23 downgrades): **11.60 MB**, verified BPB +0.0015 vs int6 baseline.
With 4.4 MB headroom, we selectively downgrade only the cheapest layers.

## Hyperparameters

9 layers, 512 dim, MLP 3x, seq2048, batch=786K, warmdown=3000, matrix_lr=0.02, grad_clip=0.3, muon_momentum=0.99.

## Run Command

```bash
QUANT_CONFIG='{}' \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

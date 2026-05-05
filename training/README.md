# Training QwenRecurrentModel

This directory contains training scripts for fine-tuning the converted `QwenRecurrentModel` (Qwen3-4B restructured into Mythos RDT topology).

## Prerequisites

- **GPU:** Minimum 24GB VRAM (A100/H100 recommended for multi-GPU FSDP)
- **Python 3.12+** with dependencies from `pyproject.toml`
- **Extra:** `pip install loguru` (used by training scripts)

```bash
pip install -e ".[training]" loguru
```

## Model Overview

The converted checkpoint `converted_qwen3_4b_recurrent.pt` contains:

| Component | Params | Notes |
|---|---|---|
| Embedding + Head (tied) | ~389M | From Qwen3-4B, do not touch aggressively |
| 5 unique QwenTransformerBlocks | ~505M | Averaged from 36 Qwen3 layers |
| LTIInjection | ~5K | Random init, needs training |
| ACTHalting | ~2.6K | Random init, needs training |
| LoRAAdapter | ~143K | Zero init, needs training |
| **Total** | **~894M** | ~505M unique layer params |

## Two-Phase Fine-Tuning Strategy

The model's recurrent modules (LTI/ACT/LoRA) are randomly initialized. Training them together with the averaged Qwen backbone at the same learning rate causes gradient interference. Use two phases:

### Phase 1: Recurrent-Module Warmup

Trains only the new recurrent modules (~153K params) while freezing the Qwen backbone.

```bash
python training/qwen_recurrent_finetune.py \
  --phase recurrent_only \
  --converted-ckpt converted_qwen3_4b_recurrent.pt \
  --steps 10000 \
  --seq-len 1024 \
  --micro-batch 4 \
  --lr 1e-4 \
  --warmup 500 \
  --dataset roneneldan/TinyStories \
  --ckpt-dir checkpoints/phase1
```

**What to monitor:**
- `rho(A)` should converge from 0.85 toward 0.7-0.95 (stable recurrence)
- Loss should decrease steadily within 500-1000 steps
- Halting distribution should spread across iterations (not all at loop 1)

### Phase 2: Full Differential LR Fine-Tune

Unfreezes all components with differentiated learning rates.

```bash
python training/qwen_recurrent_finetune.py \
  --phase full_differential \
  --converted-ckpt converted_qwen3_4b_recurrent.pt \
  --resume-checkpoints checkpoints/phase1 \
  --steps 50000 \
  --seq-len 2048 \
  --micro-batch 4 \
  --lr 3e-4 \
  --warmup 2000 \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --ckpt-dir checkpoints/phase2
```

**Learning rate groups:**

| Parameter Group | LR | Weight Decay | Why |
|---|---|---|---|
| Recurrent modules (injection/act/lora/norm) | 5e-5 | 0.01 | Random init, need full adaptation |
| Prelude + Coda blocks | 1e-5 | 0.05 | Averaged from 2 layers, moderate adjustment |
| Recurrent transformer block | 5e-6 | 0.05 | Averaged from 28 layers, most fragile |
| Embedding/head | 1e-5 | 0.0 | Token representation, no decay |
| All RMSNorm weights | frozen | — | Norms are sensitive to perturbation |

## Multi-GPU (FSDP)

```bash
torchrun --nproc_per_node=8 \
  training/qwen_recurrent_finetune.py \
  --phase full_differential \
  --converted-ckpt converted_qwen3_4b_recurrent.pt \
  --steps 50000 \
  --seq-len 2048 \
  --micro-batch 8
```

FSDP is automatically enabled when launched via `torchrun` (detects `RANK` env var). Uses `FULL_SHARD` with bf16/fp16 mixed precision.

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--phase` | `recurrent_only` | `recurrent_only` or `full_differential` |
| `--converted-ckpt` | required | Path to converted checkpoint |
| `--resume-checkpoints` | none | Directory to resume latest checkpoint from |
| `--steps` | 15000 | Total training steps |
| `--seq-len` | 1024 | Sequence length |
| `--micro-batch` | 4 | Per-GPU micro batch size |
| `--lr` | 1e-4 | Base learning rate (scaled per group in Phase 2) |
| `--warmup` | 500 | Warmup steps |
| `--weight-decay` | 0.05 | Weight decay |
| `--grad-clip` | 1.0 | Gradient clipping norm |
| `--dataset` | `roneneldan/TinyStories` | HF dataset name |
| `--dataset-config` | none | HF dataset config name (e.g. `sample-10BT`) |
| `--ckpt-dir` | `checkpoints` | Checkpoint directory |
| `--ckpt-every` | 2000 | Save checkpoint every N steps |
| `--log-every` | 10 | Log every N steps |

## Dataset Recommendations

For coherence restoration (small data, fast iteration):

| Dataset | HF Name | Size | Notes |
|---|---|---|---|
| TinyStories | `roneneldan/TinyStories` | ~2M stories | Simple language, good for loop smoke-test |
| FineWeb-Edu 10BT | `HuggingFaceFW/fineweb-edu` (config: `sample-10BT`) | 10B tokens | High-quality web text, good balance |
| Dolmino-100B | `datalab-all/dolmino-gguf-100b` | 100B tokens | Very clean, for serious fine-tuning |

## Hardware Requirements

| GPU VRAM | Feasible? | Notes |
|---|---|---|
| < 24GB | No | Model weights alone are ~1.8GB fp16, gradients + optimizer state need ~6x that |
| 24GB (RTX 4090) | Yes | seq_len=512-1024, micro_batch=2-4, grad_accum=16 |
| 40GB (A100) | Yes | seq_len=2048, micro_batch=8, grad_accum=8 |
| 80GB (A100/H100) | Comfortable | seq_len=4096, micro_batch=16, grad_accum=4 |
| 8x 80GB | Ideal | Full FSDP, seq_len=4096, large global batch |

## Validation After Fine-Tuning

```bash
python scripts/validate_qwen_recurrent.py \
  --checkpoint checkpoints/phase2/step_050000.pt \
  --generate \
  --prompts "Once upon a time" "def fibonacci(n):" "The capital of France is"
```

Expected output progression:
- **Before fine-tune:** Gibberish, no coherent patterns
- **After Phase 1:** Some repetitive patterns, better token distribution
- **After Phase 2:** Coherent sentences, grammatically correct text

## Stability Checklist

During training, monitor these in the logs:

1. **`rho(A)` < 1.0** — LTI spectral radius must stay under 1.0 at all times
2. **`rho(A)` > 0.3** — If A collapses toward 0, the recurrence is dead
3. **Halting spread** — Tokens should halt across multiple loop iterations, not all at once
4. **Loss monotonicity** — Loss should trend downward; spikes > 3x indicate instability
5. **Gradient norm** — Should be 0.1-5.0; > 10.0 suggests LR is too high

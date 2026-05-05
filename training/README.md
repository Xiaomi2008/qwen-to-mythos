# Training QwenRecurrentModel

This directory contains the Qwen-to-Mythos fine-tuning path for `QwenRecurrentModel`, a Qwen3-4B checkpoint restructured into a Recurrent-Depth Transformer topology.

## Prerequisites

- **GPU:** 32GB VRAM is the practical single-GPU floor for full differential fine-tuning at `seq_len=512`; more VRAM is needed for longer contexts.
- **Python:** 3.12+ with dependencies from `pyproject.toml`.
- **Extra:** `loguru` for training logs.

```bash
pip install -e . loguru
```

## Converted Model

Create the recurrent checkpoint with:

```bash
python scripts/convert_qwen_to_recurrent.py \
  --qwen-path Qwen/Qwen3-4B \
  --output converted_qwen3_4b_recurrent_v3.pt
```

The converted model uses a coherent contiguous mapping:

| Component | Source layers | Notes |
|---|---:|---|
| Prelude | Qwen L0-L3 | Copied verbatim |
| Recurrent stack | Qwen L4-L7 | Four blocks looped seven times |
| Coda | Qwen L32-L35 | Copied verbatim |
| Projection LoRAs | Qwen L4-L31 deltas | Rank-SVD initialization per `(loop, block, projection)` |

The alignment is `P + K*T = N - E`: with `N=36`, `P=4`, `K=4`, `T=7`, and `E=4`, the recurrent span covers the 28 middle Qwen layers. Loop 0 uses zero LoRA deltas, while loops 1-6 approximate Qwen L8-L31 with low-rank projection updates.

Approximate parameter layout:

| Component | Params | Trainability |
|---|---:|---|
| Embedding + tied head | ~389M | Phase 2 only |
| 12 QwenTransformerBlocks | ~1.21B | Frozen in Phase 1, trained in Phase 2 |
| Projection LoRAs | ~25.7M | Phase 1 and Phase 2 |
| LTI / ACT / recurrent norm | <1M | Phase 1 and Phase 2 |
| **Total** | **~1.63B** | RMSNorm weights remain frozen in Phase 2 |

## Fine-Tuning Plan

Use two phases. Phase 1 is a stability warmup for the new recurrent components. Phase 2 is a full-parameter recovery run with differential learning rates.

### Phase 1: Recurrent Warmup

Phase 1 freezes the copied Qwen backbone and trains only LTI, ACT, recurrent norm, and projection LoRAs.

```bash
python training/qwen_recurrent_finetune.py \
  --phase recurrent_only \
  --converted-ckpt converted_qwen3_4b_recurrent_v3.pt \
  --steps 1000 \
  --seq-len 512 \
  --micro-batch 1 \
  --grad-accum 4 \
  --lr 1e-4 \
  --warmup 50 \
  --dataset roneneldan/TinyStories \
  --ckpt-dir checkpoints/qwen_recurrent_phase1 \
  --ckpt-every 250 \
  --log-every 10 \
  --num-workers 2
```

Healthy Phase 1 behavior:

- Loss drops quickly from the damaged-conversion baseline.
- `rho(A)` stays below `1.0`, typically around `0.85-0.90`.
- No NaNs, repeated loss spikes, or runaway gradient norms.

### Phase 2: FineWeb-Edu Recovery

Phase 2 starts from Phase 1 model weights, creates a fresh optimizer, and trains most non-norm parameters with differential learning rates.

```bash
python training/qwen_recurrent_finetune.py \
  --phase full_differential \
  --converted-ckpt converted_qwen3_4b_recurrent_v3.pt \
  --resume-checkpoints checkpoints/qwen_recurrent_phase1 \
  --resume-model-only \
  --steps 5000 \
  --seq-len 512 \
  --micro-batch 1 \
  --grad-accum 4 \
  --lr 3e-5 \
  --warmup 200 \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset-config sample-10BT \
  --ckpt-dir checkpoints/qwen_recurrent_phase2 \
  --ckpt-every 1000 \
  --log-every 10 \
  --num-workers 2
```

The `--resume-model-only` flag is important when moving from Phase 1 to Phase 2: the Phase 1 optimizer only contains recurrent-module parameters, while Phase 2 uses different optimizer groups.

Learning-rate groups in Phase 2:

| Group | LR multiplier | Params | Purpose |
|---|---:|---:|---|
| Recurrent modules | `1.00x` | ~25.7M | Adapt LTI, ACT, recurrent norm, and projection LoRAs |
| Prelude + coda blocks | `0.33x` | ~807M | Lightly adapt copied boundary blocks |
| Recurrent Qwen blocks | `0.17x` | ~404M | Conservatively adapt reused L4-L7 blocks |
| Embedding/head | `0.33x` | ~389M | Keep token space aligned |
| RMSNorm weights | frozen | - | Avoid norm destabilization |

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--phase` | `recurrent_only` | `recurrent_only` or `full_differential` |
| `--converted-ckpt` | required | Converted Qwen recurrent checkpoint |
| `--resume-checkpoints` | none | Directory containing `step_*.pt` checkpoints |
| `--resume-model-only` | false | Load model weights without optimizer state |
| `--steps` | 15000 | Total optimizer steps |
| `--seq-len` | 1024 | Training sequence length |
| `--micro-batch` | 4 | Per-GPU micro batch size |
| `--grad-accum` | auto | Gradient accumulation steps |
| `--lr` | `1e-4` | Base learning rate |
| `--warmup` | 500 | Warmup steps |
| `--weight-decay` | 0.05 | Weight decay |
| `--grad-clip` | 1.0 | Gradient clipping norm |
| `--dataset` | `roneneldan/TinyStories` | Hugging Face dataset name |
| `--dataset-config` | none | Hugging Face dataset config, such as `sample-10BT` |
| `--ckpt-dir` | `checkpoints` | Checkpoint output directory |
| `--ckpt-every` | 2000 | Save every N steps |
| `--log-every` | 10 | Log every N steps |
| `--num-workers` | 4 | DataLoader worker count |
| `--tokenizer` | `Qwen/Qwen3-4B` | Tokenizer repo or local tokenizer path |

## Hardware Notes

Full differential checkpoints include optimizer state and are large. A single Phase 2 checkpoint for this model is roughly 19GB. Use a conservative checkpoint cadence, for example every 1000 steps, unless disk space is abundant.

| Hardware | Suggested settings |
|---|---|
| 32GB VRAM | `seq_len=512`, `micro_batch=1`, `grad_accum=4` |
| 40GB VRAM | `seq_len=1024`, `micro_batch=1-2` |
| 80GB VRAM | `seq_len=2048+`, larger micro batches |
| Multi-GPU FSDP | Use `torchrun`; FSDP activates when `RANK` is present |

## Validation

Validate checkpoint integrity and run optional generation:

```bash
python scripts/validate_qwen_recurrent.py \
  --checkpoint checkpoints/qwen_recurrent_phase2/step_0005000.pt \
  --generate \
  --prompts "Once upon a time" "def fibonacci(n):" "The capital of France is"
```

Expected progression:

- **Converted checkpoint:** structurally valid, but may show conversion damage.
- **After Phase 1:** much better token distribution and basic coherence.
- **After Phase 2:** more Qwen-like general text behavior, with the recurrent topology healed enough for longer recovery runs.

## Stability Checklist

Monitor these in the logs:

1. `rho(A) < 1.0`: the recurrent carry path must remain contractive.
2. `rho(A) > 0.3`: very low retention means the recurrence is effectively dead.
3. Loss should trend down over hundreds of steps; short-term noise is normal.
4. Repeated loss spikes above 3x the running average usually mean the LR is too high.
5. Gradient norms should settle after warmup; persistent large values suggest reducing LR or freezing more backbone parameters.

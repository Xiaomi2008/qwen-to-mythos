#!/usr/bin/env python3
"""
QwenRecurrentModel fine-tuning with two-phase differential learning rates.

Phase 1 (recurrent_only): Trains only LTI/ACT/LoRA modules while freezing
the Qwen backbone. Stabilizes the recurrent dynamics before touching averaged
weights.

Phase 2 (full_differential): Unfreezes all components with differentiated
learning rates — higher for recurrent modules, lower for averaged blocks.

Single GPU:
    python training/qwen_recurrent_finetune.py --phase recurrent_only \\
        --converted-ckpt converted_qwen3_4b_recurrent.pt

Multi-GPU (FSDP):
    torchrun --nproc_per_node=8 training/qwen_recurrent_finetune.py \\
        --phase full_differential --converted-ckpt converted_qwen3_4b_recurrent.pt
"""

import argparse
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from contextlib import nullcontext
from loguru import logger
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import DataLoader

from open_mythos.qwen_recurrent import (
    QwenRecurrentBlock,
    QwenRecurrentConfig,
    QwenRecurrentModel,
    QwenTransformerBlock,
)
from training.qwen_finetune_dataset import QwenStreamingDataset


# ---------------------------------------------------------------------------
# LR schedule: linear warmup -> cosine decay
# ---------------------------------------------------------------------------


def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup -> half-cosine decay to min_lr."""
    if step < warmup:
        return max_lr * step / warmup
    if step >= total:
        return min_lr
    decay = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


# ---------------------------------------------------------------------------
# Checkpointing (reused from 3b_fine_web_edu.py)
# ---------------------------------------------------------------------------


def _list_ckpts(ckpt_dir: str) -> list[str]:
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith("step_") and f.endswith(".pt")
    )


def save_checkpoint(
    model,
    optimizer,
    step: int,
    cfg,
    vocab_size: int,
    ckpt_dir: str,
    ddp: bool,
    master: bool,
    keep_last: int = 3,
) -> None:
    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if not master:
        return

    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
    tmp_path = final_path + ".tmp"
    torch.save(
        {
            "step": step,
            "model": model_state,
            "optimizer": optim_state,
            "cfg": cfg,
            "vocab_size": vocab_size,
        },
        tmp_path,
    )
    os.replace(tmp_path, final_path)

    for old in _list_ckpts(ckpt_dir)[:-keep_last]:
        try:
            os.remove(old)
        except OSError as exc:
            logger.warning(f"Failed to prune old checkpoint {old}: {exc}")

    logger.success(f"Checkpoint saved -> {final_path}")


def load_checkpoint(model, optimizer, path: str, ddp: bool) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            model.load_state_dict(ckpt["model"])
            optim_state = FSDP.optim_state_dict_to_load(
                model=model,
                optim=optimizer,
                optim_state_dict=ckpt["optimizer"],
            )
            optimizer.load_state_dict(optim_state)
    else:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    return int(ckpt["step"])


# ---------------------------------------------------------------------------
# Parameter group construction (differential LR)
# ---------------------------------------------------------------------------


def build_param_groups(model: QwenRecurrentModel, phase: str, base_lr: float, wd: float):
    """
    Build optimizer parameter groups with differential learning rates.

    Phase "recurrent_only":
        - Trainable: recurrent.injection, recurrent.act, recurrent.lora, recurrent.norm
        - Frozen: all transformer blocks, embedding, head, other norms

    Phase "full_differential":
        - Recurrent modules: lr=base_lr (highest)
        - Prelude + Coda blocks: lr=base_lr * 0.33
        - Recurrent transformer block: lr=base_lr * 0.17
        - Embedding/head: lr=base_lr * 0.33
        - All RMSNorm weights: frozen
    """
    if phase == "recurrent_only":
        # Freeze everything except recurrent modules
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Unfreeze recurrent modules only
        for module in [
            model.recurrent.injection,
            model.recurrent.act,
            model.recurrent.lora,
            model.recurrent.norm,
        ]:
            for param in module.parameters():
                param.requires_grad = True

        trainable = [p for p in model.parameters() if p.requires_grad]
        logger.info(
            f"Phase=recurrent_only: {len(trainable)} trainable param tensors, "
            f"{sum(p.numel() for p in trainable):,} params, lr={base_lr}"
        )
        return [{"params": trainable, "lr": base_lr, "weight_decay": wd}]

    elif phase == "full_differential":
        # Collect params by category
        recurrent_module_params = []
        for module in [
            model.recurrent.injection,
            model.recurrent.act,
            model.recurrent.lora,
            model.recurrent.norm,
        ]:
            recurrent_module_params.extend(module.parameters())

        prelude_coda_params = []
        for block in model.prelude:
            for n, p in block.named_parameters():
                if "norm" not in n:
                    prelude_coda_params.append(p)
        for block in model.coda:
            for n, p in block.named_parameters():
                if "norm" not in n:
                    prelude_coda_params.append(p)

        recurrent_block_params = [
            p
            for n, p in model.recurrent.block.named_parameters()
            if "norm" not in n
        ]

        # Embedding/head (tied, same weight)
        embed_params = [model.embed.weight]

        # Freeze all norms
        for name, param in model.named_parameters():
            if "norm" in name:
                param.requires_grad = False

        groups = [
            {
                "params": recurrent_module_params,
                "lr": base_lr,
                "weight_decay": 0.01,
            },
            {
                "params": prelude_coda_params,
                "lr": base_lr * 0.33,
                "weight_decay": wd,
            },
            {
                "params": recurrent_block_params,
                "lr": base_lr * 0.17,
                "weight_decay": wd,
            },
            {
                "params": embed_params,
                "lr": base_lr * 0.33,
                "weight_decay": 0.0,
            },
        ]
        for i, g in enumerate(groups):
            pname = ["recurrent_mod", "prelude_coda", "recurrent_block", "embed"][i]
            logger.info(
                f"  Group {i} ({pname}): {len(g['params'])} tensors, "
                f"{sum(p.numel() for p in g['params']):,} params, lr={g['lr']:.2e}"
            )
        return groups

    else:
        raise ValueError(f"Unknown phase: {phase}")


# ---------------------------------------------------------------------------
# Stability monitoring
# ---------------------------------------------------------------------------


def log_stability(model: QwenRecurrentModel, step: int, master: bool):
    """Log LTI spectral radius and other stability metrics."""
    if not master:
        return
    with torch.no_grad():
        A = model.recurrent.injection.get_A()
        rho = A.max().item()
        rho_min = A.min().item()
        rho_mean = A.mean().item()

    logger.info(f"step {step:6d} | rho(A)=[{rho_min:.4f}, {rho:.4f}] mean={rho_mean:.4f}")

    if rho >= 1.0:
        logger.warning(f"rho(A) = {rho:.4f} >= 1.0, recurrence may be unstable!")
    if rho_min < 0.1:
        logger.warning(f"rho(A).min = {rho_min:.4f} < 0.1, hidden state may be decaying too fast!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune QwenRecurrentModel with differential LR"
    )
    parser.add_argument(
        "--phase",
        default="recurrent_only",
        choices=["recurrent_only", "full_differential"],
        help="Training phase",
    )
    parser.add_argument("--converted-ckpt", required=True, help="Path to converted checkpoint")
    parser.add_argument("--resume-checkpoints", default=None, help="Dir to resume latest ckpt from")
    parser.add_argument("--steps", type=int, default=15000, help="Total training steps")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--micro-batch", type=int, default=4, help="Per-GPU micro batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--warmup", type=int, default=500, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument(
        "--dataset",
        default="roneneldan/TinyStories",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="HuggingFace dataset config (e.g. sample-10BT)",
    )
    parser.add_argument("--ckpt-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--ckpt-every", type=int, default=2000, help="Save every N steps")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-4B",
        help="Tokenizer repo",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Distributed init
    # ------------------------------------------------------------------
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank = local_rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    master = rank == 0

    if master:
        logger.info(
            f"GPUs: {torch.cuda.device_count()}  |  World size: {world_size}  |  Device: {device}"
        )
        logger.info(f"Phase: {args.phase}")

    # ------------------------------------------------------------------
    # Dataset (also gives us vocab_size)
    # ------------------------------------------------------------------
    dataset = QwenStreamingDataset(
        dataset_name=args.dataset,
        seq_len=args.seq_len,
        rank=rank,
        world_size=world_size,
        tokenizer_name=args.tokenizer,
        dataset_config=args.dataset_config,
    )
    vocab_size = dataset.vocab_size

    if master:
        logger.info(f"Tokenizer: {args.tokenizer}  |  Vocab size: {vocab_size:,}")

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    grad_accum = max(1, 256 // (world_size * args.micro_batch))
    global_batch_tok = world_size * args.micro_batch * grad_accum * args.seq_len

    if master:
        logger.info(
            f"seq_len={args.seq_len} | micro_batch={args.micro_batch} | "
            f"grad_accum={grad_accum} | global_batch_tokens={global_batch_tok:,} | "
            f"total_steps={args.steps:,} | warmup={args.warmup}"
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cfg = QwenRecurrentConfig()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = args.seq_len

    model = QwenRecurrentModel(cfg)

    # Load converted checkpoint
    if master:
        logger.info(f"Loading converted checkpoint: {args.converted_ckpt}")
    sd = torch.load(args.converted_ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if master:
        logger.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    if ddp:
        mp_policy = MixedPrecision(
            param_dtype=amp_dtype,
            reduce_dtype=amp_dtype,
            buffer_dtype=amp_dtype,
        )
        wrap_policy = ModuleWrapPolicy({QwenTransformerBlock, QwenRecurrentBlock})
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            auto_wrap_policy=wrap_policy,
            device_id=local_rank,
        )
    else:
        model = model.to(device)
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            if "cuda" in device
            else nullcontext()
        )

    amp_ctx = nullcontext() if ddp else amp_ctx  # type: ignore[possibly-undefined]

    if master:
        n_params = sum(p.numel() for p in model.parameters())
        trainable_before = sum(1 for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Parameters: {n_params:,}  |  Trainable (before phase): {trainable_before}  |  AMP: {amp_dtype}"
        )

    # ------------------------------------------------------------------
    # Optimizer (differential LR groups)
    # ------------------------------------------------------------------
    param_groups = build_param_groups(model, args.phase, args.lr, args.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, betas=(0.9, 0.95), fused=True
    )

    # ------------------------------------------------------------------
    # Resume from latest checkpoint (if any)
    # ------------------------------------------------------------------
    ckpt_dir_for_resume = args.resume_checkpoints or args.ckpt_dir
    start_step = 0
    existing_ckpts = _list_ckpts(ckpt_dir_for_resume)
    if existing_ckpts:
        latest = existing_ckpts[-1]
        if master:
            logger.info(f"Resuming from checkpoint: {latest}")
        start_step = load_checkpoint(model, optimizer, latest, ddp)
        if master:
            logger.success(f"Resumed at step {start_step}")

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------
    loader = DataLoader(
        dataset, batch_size=args.micro_batch, num_workers=4, pin_memory=True
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if master:
        os.makedirs(args.ckpt_dir, exist_ok=True)

    model.train()
    data_iter = iter(loader)
    t0 = time.perf_counter()
    step = start_step
    loss_history = []

    while step < args.steps:
        # LR schedule
        cur_lr = get_lr(step, args.warmup, args.steps, args.lr, args.lr * 0.1)
        for g in optimizer.param_groups:
            # Scale each group's base LR by its ratio to the global base_lr
            group_base = g.get("initial_lr", args.lr)
            scale = group_base / args.lr
            g["lr"] = cur_lr * scale

        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            sync = (
                nullcontext()
                if (not ddp or micro_step == grad_accum - 1)
                else model.no_sync()
            )
            with sync, amp_ctx:
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                )
                loss = loss / grad_accum

            loss.backward()
            loss_accum += loss.item()

        # Gradient clipping
        if ddp:
            grad_norm = model.clip_grad_norm_(args.grad_clip)
        else:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        step += 1

        loss_history.append(loss_accum)

        # Logging
        if master and step % args.log_every == 0:
            dt = time.perf_counter() - t0
            tok_per_sec = global_batch_tok * args.log_every / dt
            tokens_seen = step * global_batch_tok

            # Loss spike detection
            msg = (
                f"step {step:6d}/{args.steps} | loss {loss_accum:.4f} "
                f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e} "
                f"| {tok_per_sec / 1e6:.2f}M tok/s "
                f"| {tokens_seen / 1e9:.2f}B tokens"
            )

            if len(loss_history) >= 3 and loss_history[-3] > 0:
                running_avg = sum(loss_history[-20:]) / min(20, len(loss_history))
                if loss_accum > running_avg * 3:
                    msg += f" | WARN: loss spike (avg={running_avg:.4f})"

            logger.info(msg)

            # Stability monitoring
            log_stability(model, step, master)

            t0 = time.perf_counter()

        # Checkpointing
        if step % args.ckpt_every == 0:
            save_checkpoint(
                model, optimizer, step, cfg, vocab_size,
                args.ckpt_dir, ddp, master,
            )

    # Final checkpoint
    if step > start_step and step % args.ckpt_every != 0:
        save_checkpoint(
            model, optimizer, step, cfg, vocab_size,
            args.ckpt_dir, ddp, master,
        )

    if ddp:
        dist.barrier()
        dist.destroy_process_group()

    if master:
        logger.success(f"Training complete after {step} steps.")


if __name__ == "__main__":
    main()

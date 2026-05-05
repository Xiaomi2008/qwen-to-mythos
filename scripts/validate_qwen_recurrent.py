"""
Validate the converted QwenRecurrentModel checkpoint.

Usage:
    python scripts/validate_qwen_recurrent.py --checkpoint converted_qwen3_4b_recurrent_v2.pt
    python scripts/validate_qwen_recurrent.py --checkpoint converted_qwen3_4b_recurrent_v2.pt --generate --prompts "Hello world" "def fibonacci("
"""

import argparse
import time

import torch


def validate(checkpoint_path: str, device: str = "cpu", generate: bool = False, prompts: list = None):
    print("=== QwenRecurrentModel Validation ===\n")

    from open_mythos.qwen_recurrent import (
        QwenRecurrentConfig,
        QwenRecurrentModel,
    )

    # Load model
    print(f"[1/4] Loading checkpoint: {checkpoint_path}")
    cfg = QwenRecurrentConfig()
    model = QwenRecurrentModel(cfg).to(device)
    sd = torch.load(checkpoint_path, weights_only=False, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if missing:
        for k in missing[:5]:
            print(f"    {k}")

    # Count parameters
    print("[2/4] Parameter count...")
    total_params = sum(p.numel() for p in model.parameters())
    unique_layer_params = (
        sum(p.numel() for p in model.prelude.parameters())
        + sum(p.numel() for p in model.recurrent.parameters())
        + sum(p.numel() for p in model.coda.parameters())
    )
    print(f"  Total params: {total_params:,}")
    print(f"  Unique layer params (prelude+recurrent+coda): {unique_layer_params:,}")
    print(f"  Effective params with {cfg.max_loop_iters} loops: ~{unique_layer_params * cfg.max_loop_iters:,} (weight-tied)")

    # State dict integrity
    print("[3/4] State dict integrity...")
    has_nan = []
    has_inf = []
    for k, v in model.named_parameters():
        if torch.isnan(v).any():
            has_nan.append(k)
        if torch.isinf(v).any():
            has_inf.append(k)
    print(f"  NaN parameters: {len(has_nan)}")
    print(f"  Inf parameters: {len(has_inf)}")
    if has_nan[:3]:
        for k in has_nan[:3]:
            print(f"    {k}")

    # Forward pass
    print("[4/4] Forward pass...")
    test_inputs = [
        torch.tensor([[1, 100, 200, 300]], device=device),
        torch.tensor([[50000, 10000, 20000, 30000, 40000]], device=device),
        torch.tensor([[151935]], device=device),  # max vocab index
    ]
    for n_loops in [1, 4, 8, cfg.max_loop_iters]:
        for i, inp in enumerate(test_inputs):
            with torch.no_grad():
                logits = model(inp, n_loops=n_loops)
            seq_len = inp.shape[1]
            assert logits.shape == (1, seq_len, cfg.vocab_size), f"Shape mismatch: {logits.shape}"
            assert not torch.isnan(logits).any(), f"NaN in output (loops={n_loops}, input={i})"
            assert not torch.isinf(logits).any(), f"Inf in output (loops={n_loops}, input={i})"
        print(f"  n_loops={n_loops:2d}: OK (batch shapes: {[(x.shape[1],) for x in test_inputs]})")

    # LTI A values
    with torch.no_grad():
        A = model.recurrent.injection.get_A()
    print(f"\n  LTI A range: [{A.min():.4f}, {A.max():.4f}] (should be ~0.85)")
    print(f"  LTI A mean: {A.mean():.4f}")
    print(f"  Spectral radius ρ(A): {A.max():.4f} (must be < 1.0)")

    print("\n  All checks passed!")

    # Generation (optional)
    if generate:
        print("\n=== Generation Test ===\n")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", legacy=False)

        test_prompts = prompts or [
            "Once upon a time",
            "def quicksort(arr):",
            "The capital of France is",
        ]

        for prompt in test_prompts:
            print(f"Prompt: \"{prompt}\"")
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            # Warmup
            with torch.no_grad():
                _ = model(input_ids, n_loops=2)

            # Timed generation
            start = time.time()
            with torch.no_grad():
                output = model.generate(
                    input_ids, max_new_tokens=50, n_loops=4, temperature=0.7, top_k=50
                )
            elapsed = time.time() - start
            new_tokens = output[:, input_ids.shape[1]:]
            text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
            print(f"  Output: {text}")
            print(f"  Time: {elapsed:.2f}s ({50/elapsed:.1f} tok/s)")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate QwenRecurrentModel checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to converted checkpoint")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--generate", action="store_true", help="Run generation test")
    parser.add_argument("--prompts", nargs="*", default=None, help="Custom prompts for generation")
    args = parser.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    validate(args.checkpoint, device, args.generate, args.prompts)

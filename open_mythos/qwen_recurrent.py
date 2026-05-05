"""
Qwen-to-Recurrent adapter — wraps Qwen3 transformer blocks in Mythos topology.

Takes a standard Qwen3-4B model (36 stacked layers) and restructures it into
a Recurrent-Depth Transformer: prelude → recurrent loop → coda.

Keeps Qwen's architecture intact (dim, GQA, dense SwiGLU, vocab, RoPE) and
only adds the Mythos structural ideas: layer grouping, LTI injection,
ACT halting, and LoRA depth adapters.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from open_mythos.main import (
    ACTHalting,
    LoRAAdapter,
    LTIInjection,
    RMSNorm,
    apply_rope,
    loop_index_embedding,
    precompute_rope_freqs,
)


@dataclass
class QwenRecurrentConfig:
    """Configuration for the QwenRecurrentModel, mirroring Qwen3-4B."""

    # Qwen3-4B native dimensions
    vocab_size: int = 151936
    dim: int = 2560
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128  # Qwen3: head_dim=128 (Q output = 32*128 = 4096, not dim)
    intermediate_size: int = 9728
    n_layers: int = 36  # original, used for reference

    # Positional encoding
    max_seq_len: int = 40960
    rope_theta: float = 1_000_000

    # Normalization
    rms_norm_eps: float = 1e-6

    # Mythos structural choices
    prelude_layers: int = 2
    coda_layers: int = 2
    # Qwen L0-L3 → prelude (2 blocks, averaged pairs)
    # Qwen L4-L31 → recurrent (1 block, averaged 28 layers)
    # Qwen L32-L35 → coda (2 blocks, averaged pairs)

    # Recurrent loop
    max_loop_iters: int = 16
    act_threshold: float = 0.99
    loop_dim: int = 320  # dim // 8, channels for loop-index embedding

    # LoRA depth adaptation
    lora_rank: int = 16

    # Dropout
    dropout: float = 0.0

    # Layer grouping (derived, for clarity)
    @property
    def prelude_layer_range(self) -> range:
        return range(0, self.prelude_layers * 2)

    @property
    def recurrent_layer_range(self) -> range:
        start = self.prelude_layers * 2
        end = self.n_layers - self.coda_layers * 2
        return range(start, end)

    @property
    def coda_layer_range(self) -> range:
        start = self.n_layers - self.coda_layers * 2
        return range(start, self.n_layers)


class QwenAttention(nn.Module):
    """
    Qwen3 GQA attention — exact structure of a Qwen3 attention layer.

    Qwen3 uses head_dim=128 (not dim//n_heads=80), with per-head RMSNorm
    on Q and K before attention. Q projection is larger than the hidden dim
    (4096 vs 2560), and o_proj projects back.

    Preserves Qwen's weight layout so we can load checkpoint weights directly.
    """

    def __init__(self, cfg: QwenRecurrentConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.groups = cfg.n_heads // cfg.n_kv_heads
        self.dropout_p = cfg.dropout

        # Qwen naming convention (shapes match checkpoint exactly)
        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

        # Per-head RMSNorm on Q and K (Qwen3-specific)
        self.q_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Per-head RMSNorm before RoPE (Qwen3 order)
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if kv_cache is not None:
            if cache_key in kv_cache:
                k = torch.cat([kv_cache[cache_key]["k"], k], dim=1)
                v = torch.cat([kv_cache[cache_key]["v"], v], dim=1)
            kv_cache[cache_key] = {"k": k.detach(), "v": v.detach()}

        # GQA: expand KV heads to match Q heads
        k = k.repeat_interleave(self.groups, dim=2)
        v = v.repeat_interleave(self.groups, dim=2)

        orig_dtype = q.dtype
        q = q.transpose(1, 2)  # (B, H, T, head_dim)
        k = k.transpose(1, 2)  # (B, H, S, head_dim)
        v = v.transpose(1, 2)  # (B, H, S, head_dim)

        # FP32 for attention scores (numerical stability), cast back after
        scale = self.head_dim**-0.5
        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn + mask
        attn = torch.nn.functional.dropout(
            torch.nn.functional.softmax(attn, dim=-1),
            p=self.dropout_p,
            training=self.training,
        )
        out = torch.matmul(attn, v.float())
        out = out.to(orig_dtype)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class QwenFFN(nn.Module):
    """
    Qwen3 SwiGLU FFN — exact structure of a Qwen3 MLP layer.

    output = down(silu(gate(x)) * up(x))
    """

    def __init__(self, cfg: QwenRecurrentConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class QwenTransformerBlock(nn.Module):
    """
    A single Qwen3 transformer block (pre-norm residual).

    Structure:
        x = x + attn(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """

    def __init__(self, cfg: QwenRecurrentConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)
        self.self_attn = QwenAttention(cfg)
        self.mlp = QwenFFN(cfg)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.input_layernorm(x), freqs_cis, mask, kv_cache, cache_key
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class QwenRecurrentBlock(nn.Module):
    """
    The recurrent block — one QwenTransformerBlock looped T times with:
    - Loop-index embedding (positional signal for recurrence depth)
    - LTI-stable injection (A·h + B·e + transformer_out)
    - ACT halting (adaptive per-position early exit)
    - LoRA depth adapter (per-loop parameter variation)
    """

    def __init__(self, cfg: QwenRecurrentConfig):
        super().__init__()
        self.cfg = cfg
        self.block = QwenTransformerBlock(cfg)
        self.injection = LTIInjection(cfg.dim)
        self.act = ACTHalting(cfg.dim)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.norm = RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape
        loop_dim = self.cfg.loop_dim

        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)

        for t in range(n_loops):
            h_loop = loop_index_embedding(h, t, loop_dim)
            combined = self.norm(h_loop + e)
            cache_key = f"recurrent_loop_{t}"

            trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
            trans_out = trans_out + self.lora(trans_out, t)

            h = self.injection(h, e, trans_out)

            p = self.act(h)
            still_running = ~halted

            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= self.cfg.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.float()
            h_out = h_out + weight.unsqueeze(-1) * h

            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= self.cfg.act_threshold)

            if halted.all() and kv_cache is None:
                break

        return h_out


class QwenRecurrentModel(nn.Module):
    """
    Qwen3-4B restructured as a Recurrent-Depth Transformer.

    Topology:
        Input tokens
         ↓
        [Prelude]       — 2 Qwen transformer blocks (averaged from L0-L3)
         ↓
        [Recurrent]     — 1 Qwen transformer block looped T times
                          (averaged from L4-L31, with LTI/ACT/LoRA)
         ↑_______↓      h_{t+1} = A·h_t + B·e + Block(h_t, e)
         ↓
        [Coda]          — 2 Qwen transformer blocks (averaged from L32-L35)
         ↓
        Output logits

    Properties:
    - Keeps Qwen3-4B's dim (2560), GQA (32/8 heads, head_dim=128),
      SwiGLU FFN (9728), vocab (151k), q_norm/k_norm
    - 36 unique layers → 6 unique layers (6x weight reuse in loop)
    - Same parameters, more loops → deeper reasoning
    - ACT halting: variable compute per position
    - LTI-stable injection: spectral radius < 1 guaranteed
    """

    def __init__(self, cfg: Optional[QwenRecurrentConfig] = None):
        cfg = cfg or QwenRecurrentConfig()
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        # RoPE frequencies (per-head, head_dim=128)
        freqs = precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cis", freqs)

        # Prelude layers
        self.prelude = nn.ModuleList(
            [QwenTransformerBlock(cfg) for _ in range(cfg.prelude_layers)]
        )

        # Recurrent block
        self.recurrent = QwenRecurrentBlock(cfg)

        # Coda layers
        self.coda = nn.ModuleList(
            [QwenTransformerBlock(cfg) for _ in range(cfg.coda_layers)]
        )

        # Final norm + head (weight tied)
        self.norm = RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying (matches Qwen3)

    @staticmethod
    def _causal_mask(
        seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        mask = torch.full(
            (1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype
        )
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        T = input_ids.shape[1]
        device = input_ids.device

        x = self.embed(input_ids)
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]
        mask = self._causal_mask(T, device, x.dtype) if T > 1 else None

        for i, layer in enumerate(self.prelude):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"prelude_{i}")

        e = x  # frozen input for recurrent injection
        x = self.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)

        for i, layer in enumerate(self.coda):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"coda_{i}")

        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        n_loops: int = 8,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        kv_cache: dict = {}
        prompt_len = input_ids.shape[1]
        for step in range(max_new_tokens):
            if step == 0:
                cur_ids = input_ids
                start_pos = 0
            else:
                cur_ids = input_ids[:, -1:]
                start_pos = prompt_len + step - 1
            logits = self.forward(
                cur_ids, n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos
            )
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids

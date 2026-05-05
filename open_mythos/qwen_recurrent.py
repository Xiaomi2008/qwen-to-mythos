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
    LTIInjection,
    RMSNorm,
    apply_rope,
    loop_index_embedding,
    precompute_rope_freqs,
)


@dataclass
class QwenRecurrentConfig:
    """Configuration for the QwenRecurrentModel, mirroring Qwen3-4B.

    Layer assignment uses **verbatim copy** of contiguous Qwen layers:
        prelude  ← Qwen [0 .. prelude_layers - 1]                         (4 blocks: L0-L3)
        recurrent ← Qwen [prelude_layers .. prelude_layers + K - 1]        (4 blocks: L4-L7)
                   stacked into a K-block ModuleList, looped T times
        coda     ← Qwen [n_layers - coda_layers .. n_layers - 1]          (4 blocks: L32-L35)

    The alignment K · max_loop_iters = n_layers - prelude_layers - coda_layers
    (i.e. 4 · 7 = 28) makes a forward pass with n_loops=1 traverse exactly the
    contiguous Qwen subnet L0-L3 → L4-L7 → L32-L35 — coherent at init.
    """

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

    # Mythos structural choices — K=4 contiguous, T=7
    prelude_layers: int = 4
    coda_layers: int = 4
    n_recurrent_layers: int = 4

    # Recurrent loop
    max_loop_iters: int = 7
    act_threshold: float = 0.99
    loop_dim: int = 320  # dim // 8, channels for loop-index embedding

    # LoRA depth adaptation (one LoRA per recurrent block, indexed by loop)
    lora_rank: int = 16

    # Dropout
    dropout: float = 0.0

    # Layer mapping helpers — Qwen layers copied into each Mythos slot
    @property
    def prelude_qwen_layers(self) -> list[int]:
        return list(range(0, self.prelude_layers))

    @property
    def recurrent_base_qwen_layers(self) -> list[int]:
        start = self.prelude_layers
        return list(range(start, start + self.n_recurrent_layers))

    @property
    def coda_qwen_layers(self) -> list[int]:
        return list(range(self.n_layers - self.coda_layers, self.n_layers))

    def loop_target_qwen_layer(self, loop_t: int, block_k: int) -> int:
        """For loop iteration t and recurrent block k, the Qwen layer this
        position would correspond to in a fully unrolled forward pass."""
        return self.prelude_layers + loop_t * self.n_recurrent_layers + block_k


class LoopwiseLoRALinear(nn.Module):
    """
    Bias-free linear layer with optional per-loop low-rank weight deltas.

    The base weight is copied verbatim from one Qwen layer. For recurrent
    blocks, each loop index can add a rank-r delta initialized from the SVD of
    target_weight - base_weight, so loop t/block k approximates the matching
    original Qwen layer without giving up recurrent weight tying.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        max_loops: int,
        enable_lora: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.max_loops = max_loops
        self.enable_lora = enable_lora

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if enable_lora:
            self.lora_a = nn.Parameter(torch.zeros(max_loops, rank, in_features))
            self.lora_b = nn.Parameter(torch.zeros(max_loops, out_features, rank))
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    def forward(self, x: torch.Tensor, loop_t: Optional[int] = None) -> torch.Tensor:
        out = torch.nn.functional.linear(x, self.weight)
        if not self.enable_lora or loop_t is None:
            return out

        max_t = self.max_loops - 1
        t_idx = loop_t if loop_t <= max_t else max_t
        down = torch.nn.functional.linear(x, self.lora_a[t_idx])
        return out + torch.nn.functional.linear(down, self.lora_b[t_idx])


class QwenAttention(nn.Module):
    """
    Qwen3 GQA attention — exact structure of a Qwen3 attention layer.

    Qwen3 uses head_dim=128 (not dim//n_heads=80), with per-head RMSNorm
    on Q and K before attention. Q projection is larger than the hidden dim
    (4096 vs 2560), and o_proj projects back.

    Preserves Qwen's weight layout so we can load checkpoint weights directly.
    """

    def __init__(self, cfg: QwenRecurrentConfig, enable_loop_lora: bool = False):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.groups = cfg.n_heads // cfg.n_kv_heads
        self.dropout_p = cfg.dropout

        # Qwen naming convention (shapes match checkpoint exactly)
        self.q_proj = LoopwiseLoRALinear(
            cfg.dim,
            cfg.n_heads * cfg.head_dim,
            cfg.lora_rank,
            cfg.max_loop_iters,
            enable_loop_lora,
        )
        self.k_proj = LoopwiseLoRALinear(
            cfg.dim,
            cfg.n_kv_heads * cfg.head_dim,
            cfg.lora_rank,
            cfg.max_loop_iters,
            enable_loop_lora,
        )
        self.v_proj = LoopwiseLoRALinear(
            cfg.dim,
            cfg.n_kv_heads * cfg.head_dim,
            cfg.lora_rank,
            cfg.max_loop_iters,
            enable_loop_lora,
        )
        self.o_proj = LoopwiseLoRALinear(
            cfg.n_heads * cfg.head_dim,
            cfg.dim,
            cfg.lora_rank,
            cfg.max_loop_iters,
            enable_loop_lora,
        )

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
        loop_t: Optional[int] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x, loop_t).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x, loop_t).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x, loop_t).view(B, T, self.n_kv_heads, self.head_dim)

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
        return self.o_proj(out, loop_t)


class QwenFFN(nn.Module):
    """
    Qwen3 SwiGLU FFN — exact structure of a Qwen3 MLP layer.

    output = down(silu(gate(x)) * up(x))
    """

    def __init__(self, cfg: QwenRecurrentConfig, enable_loop_lora: bool = False):
        super().__init__()
        self.gate_proj = LoopwiseLoRALinear(
            cfg.dim,
            cfg.intermediate_size,
            cfg.lora_rank,
            cfg.max_loop_iters,
            enable_loop_lora,
        )
        self.up_proj = LoopwiseLoRALinear(
            cfg.dim,
            cfg.intermediate_size,
            cfg.lora_rank,
            cfg.max_loop_iters,
            enable_loop_lora,
        )
        self.down_proj = LoopwiseLoRALinear(
            cfg.intermediate_size,
            cfg.dim,
            cfg.lora_rank,
            cfg.max_loop_iters,
            enable_loop_lora,
        )

    def forward(self, x: torch.Tensor, loop_t: Optional[int] = None) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x, loop_t))
            * self.up_proj(x, loop_t),
            loop_t,
        )


class QwenTransformerBlock(nn.Module):
    """
    A single Qwen3 transformer block (pre-norm residual).

    Structure:
        x = x + attn(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """

    def __init__(self, cfg: QwenRecurrentConfig, enable_loop_lora: bool = False):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)
        self.self_attn = QwenAttention(cfg, enable_loop_lora)
        self.mlp = QwenFFN(cfg, enable_loop_lora)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
        loop_t: Optional[int] = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.input_layernorm(x), freqs_cis, mask, kv_cache, cache_key, loop_t
        )
        x = x + self.mlp(self.post_attention_layernorm(x), loop_t)
        return x


class QwenRecurrentBlock(nn.Module):
    """
    The recurrent block — a stack of K=cfg.n_recurrent_layers QwenTransformerBlocks
    looped T times with:
    - Loop-index embedding (positional signal for recurrence depth)
    - LTI-stable injection (A·h + B·e + transformer_out)
    - ACT halting (adaptive per-position early exit)
    - Per-loop low-rank deltas inside each block projection

    Per loop iteration t:
        h_loop = loop_index_embedding(h, t)
        out = norm(h_loop + e)
        for k in 0..K-1:
            out = blocks[k](out, loop_t=t)
        h = injection(h, e, out)
        ACT halting
    """

    def __init__(self, cfg: QwenRecurrentConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(
            [
                QwenTransformerBlock(cfg, enable_loop_lora=True)
                for _ in range(cfg.n_recurrent_layers)
            ]
        )
        self.injection = LTIInjection(cfg.dim)
        self.act = ACTHalting(cfg.dim)
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
            out = self.norm(h_loop + e)

            for k, block in enumerate(self.blocks):
                cache_key = f"recurrent_loop_{t}_block_{k}"
                out = block(out, freqs_cis, mask, kv_cache, cache_key, loop_t=t)

            h = self.injection(h, e, out)

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

    Topology (verbatim contiguous Qwen layers, no averaging):
        Input tokens
         ↓
        [Prelude]       — 4 Qwen transformer blocks copied from L0-L3
         ↓
        [Recurrent]     — K=4 stacked Qwen transformer blocks copied from
                          L4-L7, looped T=7 times (with LTI/ACT/projection LoRA)
         ↑_______↓      h_{t+1} = A·h_t + B·e + Block_K(...Block_1(h_t, e))
         ↓
        [Coda]          — 4 Qwen transformer blocks copied from L32-L35
         ↓
        Output logits

    Properties:
    - Keeps Qwen3-4B's dim (2560), GQA (32/8 heads, head_dim=128),
      SwiGLU FFN (9728), vocab (151k), q_norm/k_norm
    - 36 Qwen layers → 12 unique blocks (4 prelude + 4 recurrent + 4 coda)
      with 7x weight reuse on the recurrent block stack
    - K · T = 28 = (n_layers - prelude - coda), so forward(x, n_loops=1)
      traverses exactly the 12-layer Qwen subnet L0-L3 → L4-L7 → L32-L35,
      giving a Qwen-coherent starting point at init
    - Per-loop projection LoRAs are initialized to approximate the matching
      Qwen L4-L31 layers during a full T=7 recurrent pass
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

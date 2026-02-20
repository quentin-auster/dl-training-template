"""Simple transformer with TransformerLens HookPoints for interpretability.

This module demonstrates how to add HookPoints to a custom transformer,
giving you TransformerLens-style activation caching and hooking while
maintaining full control over the architecture.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformer_lens.hook_points import HookPoint, HookedRootModule


class Attention(nn.Module):
    """Multi-head self-attention with HookPoints."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # HookPoints for interpretability
        self.hook_q = HookPoint()  # [batch, seq, n_heads, d_head]
        self.hook_k = HookPoint()  # [batch, seq, n_heads, d_head]
        self.hook_v = HookPoint()  # [batch, seq, n_heads, d_head]
        self.hook_attn_scores = HookPoint()  # [batch, n_heads, seq, seq]
        self.hook_attn_pattern = HookPoint()  # [batch, n_heads, seq, seq]
        self.hook_z = HookPoint()  # [batch, seq, n_heads, d_head] (post-attention values)
        self.hook_result = HookPoint()  # [batch, seq, d_model] (output)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch, seq_len, _ = x.shape

        # Compute Q, K, V and reshape to [batch, seq, n_heads, d_head]
        q = self.W_Q(x).view(batch, seq_len, self.n_heads, self.d_head)
        k = self.W_K(x).view(batch, seq_len, self.n_heads, self.d_head)
        v = self.W_V(x).view(batch, seq_len, self.n_heads, self.d_head)

        q = self.hook_q(q)
        k = self.hook_k(k)
        v = self.hook_v(v)

        # Attention scores: [batch, n_heads, seq, seq]
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.d_head)
        scores = self.hook_attn_scores(scores)

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.bool(), float("-inf"))

        pattern = F.softmax(scores, dim=-1)
        pattern = self.hook_attn_pattern(pattern)
        pattern = self.dropout(pattern)

        # Apply attention to values: [batch, seq, n_heads, d_head]
        z = torch.einsum("bhqk,bkhd->bqhd", pattern, v)
        z = self.hook_z(z)

        # Combine heads and project
        z = z.reshape(batch, seq_len, self.d_model)
        out = self.W_O(z)
        out = self.hook_result(out)

        return out


class MLP(nn.Module):
    """Feed-forward network with HookPoints."""

    def __init__(
        self,
        d_model: int,
        d_mlp: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        d_mlp = d_mlp or 4 * d_model

        self.W_in = nn.Linear(d_model, d_mlp)
        self.W_out = nn.Linear(d_mlp, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.act_fn = F.gelu
        elif activation == "relu":
            self.act_fn = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # HookPoints
        self.hook_pre = HookPoint()  # [batch, seq, d_mlp] before activation
        self.hook_post = HookPoint()  # [batch, seq, d_mlp] after activation
        self.hook_result = HookPoint()  # [batch, seq, d_model] output

    def forward(self, x: Tensor) -> Tensor:
        pre = self.W_in(x)
        pre = self.hook_pre(pre)

        post = self.act_fn(pre)
        post = self.hook_post(post)
        post = self.dropout(post)

        out = self.W_out(post)
        out = self.hook_result(out)

        return out


class TransformerBlock(nn.Module):
    """Transformer block with optional pre-norm and HookPoints."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_ln: bool = True,
    ) -> None:
        super().__init__()
        self.use_ln = use_ln
        self.ln1 = nn.LayerNorm(d_model) if use_ln else nn.Identity()
        self.attn = Attention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model) if use_ln else nn.Identity()
        self.mlp = MLP(d_model, d_mlp, dropout, activation)

        # HookPoints for residual stream
        self.hook_resid_pre = HookPoint()  # [batch, seq, d_model] input
        self.hook_resid_mid = HookPoint()  # [batch, seq, d_model] after attn
        self.hook_resid_post = HookPoint()  # [batch, seq, d_model] output

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.hook_resid_pre(x)

        # Attention with residual
        x = x + self.attn(self.ln1(x), attention_mask)
        x = self.hook_resid_mid(x)

        # MLP with residual
        x = x + self.mlp(self.ln2(x))
        x = self.hook_resid_post(x)

        return x


class TinyTransformer(HookedRootModule):
    """Small transformer for mechanistic interpretability experiments.

    Inherits from HookedRootModule to get TransformerLens-style:
    - run_with_cache() for activation caching
    - run_with_hooks() for activation patching
    - Standardized hook naming

    Example:
        model = TinyTransformer(vocab_size=100, d_model=64, n_layers=2, n_heads=4)

        # Run with caching
        logits, cache = model.run_with_cache(input_ids)
        attn_pattern = cache["blocks.0.attn.hook_attn_pattern"]

        # Run with hooks (e.g., ablation)
        def zero_head_hook(tensor, hook):
            tensor[:, :, 0, :] = 0  # zero out head 0
            return tensor

        logits = model.run_with_hooks(
            input_ids,
            fwd_hooks=[("blocks.0.attn.hook_z", zero_head_hook)],
        )
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        d_mlp: int | None = None,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
        tie_embed: bool = True,
        use_ln: bool = True,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Embeddings (scaled init to match Nanda's grokking setup)
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.embed_pos = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.embed_tokens.weight, std=d_model ** -0.5)
        nn.init.normal_(self.embed_pos.weight, std=d_model ** -0.5)
        self.dropout = nn.Dropout(dropout)

        # HookPoints for embeddings
        self.hook_embed = HookPoint()  # [batch, seq, d_model]
        self.hook_pos_embed = HookPoint()  # [batch, seq, d_model]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_mlp, dropout, activation, use_ln)
            for _ in range(n_layers)
        ])

        # Output
        self.ln_final = nn.LayerNorm(d_model) if use_ln else nn.Identity()
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

        if tie_embed:
            self.unembed.weight = self.embed_tokens.weight

        # HookPoints for final output
        self.hook_resid_final = HookPoint()  # [batch, seq, d_model] after final ln

        # IMPORTANT: This sets up the hook name -> hook point mapping
        self.setup()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Embeddings
        tok_embed = self.embed_tokens(input_ids)
        tok_embed = self.hook_embed(tok_embed)

        pos_embed = self.embed_pos(positions)
        pos_embed = self.hook_pos_embed(pos_embed)

        x = self.dropout(tok_embed + pos_embed)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # Output
        x = self.ln_final(x)
        x = self.hook_resid_final(x)
        logits = self.unembed(x)

        return logits

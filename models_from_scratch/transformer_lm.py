#!/usr/bin/env python3
"""
models_from_scratch/transformer_lm.py
======================================
GPT-2 style decoder-only Transformer — built entirely from scratch in PyTorch.
Every layer (attention, feed-forward, layer norm) is coded manually; no
nn.TransformerEncoderLayer or nn.MultiheadAttention are used.

┌─────────────────────────────────────────────────────────────────────────┐
│                        FULL ARCHITECTURE                                │
│                                                                         │
│  Input token IDs  (batch, seq_len)                                      │
│        │                                                                │
│        ▼                                                                │
│  Token Embedding   (vocab_size → d_model)        ┐ element-wise sum    │
│  Positional Embed  (n_ctx → d_model)             ┘                     │
│        │                                                                │
│        ▼                                                                │
│  ┌─────────────────────────────────┐  ×N blocks                        │
│  │  LayerNorm (pre-norm)           │                                    │
│  │  MultiHead Causal Self-Attention│  Q,K,V = x @ W_qkv               │
│  │    score = QKᵀ / √d_head       │  masked: future tokens → −∞       │
│  │    attn  = softmax(score)·V     │  concat heads → project           │
│  │  + Residual connection          │                                    │
│  │  LayerNorm                      │                                    │
│  │  FeedForward: W2·GELU(W1·x)    │  expand 4× then contract          │
│  │  + Residual connection          │                                    │
│  └─────────────────────────────────┘                                    │
│        │                                                                │
│        ▼                                                                │
│  Final LayerNorm                                                        │
│  LM Head  (d_model → vocab_size)    [weight-tied to token embedding]   │
│        │                                                                │
│        ▼                                                                │
│  Logits  →  softmax  →  sample next token  (auto-regressive loop)      │
└─────────────────────────────────────────────────────────────────────────┘

Default hyperparameters match GPT-2 Small (117 M parameters):
  vocab_size = 50 257,  n_ctx = 1 024,  d_model = 768
  n_heads    = 12,      n_layers = 12,  d_ff = 3 072 (4 × d_model)

Because the architecture is identical to GPT-2, pretrained GPT-2 weights can
be loaded directly (see GPTLanguageModel.from_pretrained).

RAG integration
---------------
ScratchRAGGenerator wraps the model so it can be swapped into CampusRAG
in phase5_rag_pipeline.py as a drop-in replacement for the GPT-4o API call.

Usage:
    # Option 1 — load open-source GPT-2 weights (works immediately)
    model = GPTLanguageModel.from_pretrained("gpt2")

    # Option 2 — random init + train from scratch on your own data
    model = GPTLanguageModel(GPTConfig())
    train(model, text_files=[...])

    # Option 3 — fine-tune a pretrained checkpoint on campus data
    model = GPTLanguageModel.from_pretrained("gpt2")
    train(model, text_files=["data/campus_corpus.txt"], epochs=3)

    # Drop into the RAG pipeline
    gen = ScratchRAGGenerator(model)
    answer = gen.generate(query, retrieved_chunks)
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GPTConfig:
    """
    Hyperparameters for the GPT-style Transformer.

    Presets:
        GPT-2 Small  → GPTConfig()                         (117 M params)
        GPT-2 Medium → GPTConfig(n_layers=24, n_heads=16, d_model=1024)
        GPT-2 Large  → GPTConfig(n_layers=36, n_heads=20, d_model=1280)
        Tiny (demo)  → GPTConfig(n_layers=2, n_heads=4, d_model=128, n_ctx=256)
    """
    vocab_size: int   = 50_257  # GPT-2 BPE vocabulary
    n_ctx:      int   = 1_024   # maximum sequence / context length
    d_model:    int   = 768     # embedding dimension (= d_model)
    n_heads:    int   = 12      # number of attention heads
    n_layers:   int   = 12      # number of stacked Transformer blocks
    dropout:    float = 0.1     # dropout applied after attention and FFN
    bias:       bool  = True    # include bias terms in Linear / LayerNorm

    @property
    def d_head(self) -> int:
        """Dimension of each attention head (d_model ÷ n_heads)."""
        return self.d_model // self.n_heads

    @property
    def d_ff(self) -> int:
        """Feed-forward hidden dimension (standard: 4 × d_model)."""
        return 4 * self.d_model


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LAYER NORM  (from scratch — not using nn.LayerNorm)
# ══════════════════════════════════════════════════════════════════════════════

class LayerNorm(nn.Module):
    """
    Layer Normalization (Ba et al., 2016).

    Normalises the last dimension (d_model) of the input independently
    for each token position and batch element:

        μ  = mean(x, dim=-1)
        σ² = var(x,  dim=-1)
        x̂  = (x − μ) / √(σ² + ε)
        y  = γ · x̂ + β

    γ (weight) and β (bias) are learnable per-feature scale and shift.
    """

    def __init__(self, d_model: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))                     # γ
        self.bias   = nn.Parameter(torch.zeros(d_model)) if bias else None  # β
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*, d_model)  — works for any leading batch dimensions
        mean = x.mean(dim=-1, keepdim=True)                          # μ
        var  = ((x - mean) ** 2).mean(dim=-1, keepdim=True)         # σ²
        x_hat = (x - mean) / torch.sqrt(var + self.eps)             # normalise
        y = self.weight * x_hat
        if self.bias is not None:
            y = y + self.bias
        return y


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MULTI-HEAD CAUSAL SELF-ATTENTION  (from scratch)
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadCausalSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with a causal (autoregressive) mask.

    The input x of shape (B, T, d_model) is projected into queries, keys
    and values, then attention is computed independently for each head:

        [Q | K | V] = x @ W_qkv + b_qkv          fused single projection
        Q, K, V each: (B, T, d_model) → reshape → (B, n_heads, T, d_head)

    Scaled dot-product attention per head:
        score_h = Q_h @ K_hᵀ / √d_head            (B, n_heads, T, T)
        CAUSAL MASK: score[i, j] = −∞  when j > i  (token i cannot see future j)
        α_h     = softmax(score_h, dim=-1)
        ctx_h   = α_h @ V_h                        (B, n_heads, T, d_head)

    Concatenate all heads, project back:
        out = concat(ctx_1, …, ctx_H) @ W_proj + b_proj   (B, T, d_model)
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, \
            f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"

        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.d_head  = cfg.d_head   # dimension per head

        # Single fused projection for Q, K, V  (output size = 3 × d_model)
        self.W_qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        # Output projection after concatenating heads
        self.W_proj = nn.Linear(cfg.d_model, cfg.d_model,     bias=cfg.bias)

        self.attn_drop = nn.Dropout(cfg.dropout)
        self.proj_drop = nn.Dropout(cfg.dropout)

        # Causal mask: upper-triangular True means "position j is in the future of i"
        # Shape (n_ctx, n_ctx) — stored as a buffer (not a learnable parameter)
        mask = torch.triu(torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool), diagonal=1)
        self.register_buffer("_causal_mask", mask)   # moves to GPU automatically

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape   # batch, sequence length, d_model

        # ── Step 1: Compute Q, K, V with a single linear layer ───────────────
        qkv          = self.W_qkv(x)                        # (B, T, 3·d_model)
        Q, K, V      = qkv.split(self.d_model, dim=-1)      # each (B, T, d_model)

        # ── Step 2: Reshape into per-head tensors ────────────────────────────
        #   (B, T, d_model) → (B, T, n_heads, d_head) → (B, n_heads, T, d_head)
        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        Q = to_heads(Q)   # (B, n_heads, T, d_head)
        K = to_heads(K)
        V = to_heads(V)

        # ── Step 3: Scaled dot-product attention ─────────────────────────────
        scale  = math.sqrt(self.d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale   # (B, n_heads, T, T)

        # Apply causal mask: set future-position scores to −∞
        # After softmax, −∞ → 0, so future tokens contribute nothing
        scores = scores.masked_fill(self._causal_mask[:T, :T], float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)   # (B, n_heads, T, T)
        attn_weights = self.attn_drop(attn_weights)

        context = torch.matmul(attn_weights, V)    # (B, n_heads, T, d_head)

        # ── Step 4: Concatenate heads → output projection ────────────────────
        #   (B, n_heads, T, d_head) → (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out     = self.proj_drop(self.W_proj(context))         # (B, T, d_model)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FEED-FORWARD NETWORK  (from scratch)
# ══════════════════════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN).

    Applied identically to every token position:

        FFN(x) = W_2 · GELU(W_1 · x + b_1) + b_2

    Expands the dimension by 4× then contracts back:
        d_model → 4·d_model → d_model

    GELU (Gaussian Error Linear Unit) is the activation used in GPT-2:
        GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
    It is a smooth, non-zero-gradient-everywhere alternative to ReLU.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1     = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)  # expand
        self.fc2     = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)  # contract
        self.dropout = nn.Dropout(cfg.dropout)

    @staticmethod
    def _gelu(x: torch.Tensor) -> torch.Tensor:
        """
        GPT-2 tanh-approximation of GELU.
        More accurate than F.relu but cheaper than the exact erf formulation.
        """
        return (
            0.5 * x
            * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)         # (B, T, d_ff)   — expand
        x = self._gelu(x)       # non-linearity
        x = self.fc2(x)         # (B, T, d_model) — contract
        x = self.dropout(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRANSFORMER BLOCK  (from scratch)
# ══════════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    One GPT-2 Transformer Block using Pre-LayerNorm (Pre-LN).

    Pre-LN normalises BEFORE each sublayer (unlike the original "Attention is
    All You Need" paper which uses Post-LN). Pre-LN is more training-stable
    for deep models and is the design GPT-2 uses.

    Forward pass:
        x ← x + Attention(LayerNorm(x))    # self-attention sublayer
        x ← x + FFN(LayerNorm(x))          # feed-forward sublayer

    The "+ x" terms are residual (skip) connections.  They allow gradients to
    flow directly to early layers and are critical for training deep networks.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1  = LayerNorm(cfg.d_model, bias=cfg.bias)   # norm before attention
        self.attn = MultiHeadCausalSelfAttention(cfg)
        self.ln2  = LayerNorm(cfg.d_model, bias=cfg.bias)   # norm before FFN
        self.ffn  = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # attention sublayer + residual
        x = x + self.ffn(self.ln2(x))    # FFN sublayer + residual
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 6.  GPT LANGUAGE MODEL  (full model — from scratch)
# ══════════════════════════════════════════════════════════════════════════════

class GPTLanguageModel(nn.Module):
    """
    GPT-2 style decoder-only Transformer Language Model.

    Full forward pass (training):
        ids    → token_emb + pos_emb   [B, T, d_model]
               → dropout
               → N × TransformerBlock
               → LayerNorm
               → lm_head              [B, T, vocab_size]   logits
               → cross_entropy(logits, targets)            loss

    Inference: call generate() for auto-regressive text generation.

    Weight tying
    ────────────
    lm_head.weight is shared with token_emb.weight (same tensor).
    This halves the embedding-layer parameters and improves perplexity
    because the model learns a single representation for each token
    regardless of whether it is an input or an output.
    (Press & Wolf, "Using the Output Embedding to Improve Language Models", 2017)
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.n_ctx,      cfg.d_model)   # learned positions
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final  = LayerNorm(cfg.d_model, bias=cfg.bias)
        self.lm_head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: token embedding and LM head share one weight matrix
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    # ── Weight initialisation (GPT-2 style) ──────────────────────────────────
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    # ── Forward pass ─────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,                     # (B, T)  token indices
        targets:   Optional[torch.Tensor] = None,    # (B, T)  for training
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns (logits, loss).
        loss is None during inference (when targets is not provided).
        """
        B, T = input_ids.shape
        assert T <= self.cfg.n_ctx, \
            f"Input length {T} exceeds context window {self.cfg.n_ctx}"

        device = input_ids.device
        positions = torch.arange(T, device=device).unsqueeze(0)   # (1, T)

        # Token + positional embeddings
        tok = self.token_emb(input_ids)    # (B, T, d_model)
        pos = self.pos_emb(positions)      # (1, T, d_model) — broadcasts over B
        x   = self.drop(tok + pos)         # (B, T, d_model)

        # Pass through all Transformer blocks
        for block in self.blocks:
            x = block(x)                   # (B, T, d_model)

        x      = self.ln_final(x)          # (B, T, d_model)
        logits = self.lm_head(x)           # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten (B·T, vocab_size) vs (B·T,) for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-1,           # -1 labels are padding, not trained on
            )

        return logits, loss

    # ── Auto-regressive generation ────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        input_ids:    torch.Tensor,        # (1, T) prompt token ids
        max_new:      int   = 256,
        temperature:  float = 0.7,
        top_k:        int   = 50,
        top_p:        float = 0.9,         # nucleus sampling threshold
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Auto-regressive token generation with nucleus (top-p) + top-k sampling.

        At each step:
          1. Forward pass on current context → logits for next token
          2. Divide by temperature  (higher T → more random)
          3. Keep only the top-k highest-logit tokens
          4. Nucleus: keep the smallest set whose cumulative prob ≥ top_p
          5. Sample one token from the filtered distribution
          6. Append token, repeat until max_new tokens or EOS
        """
        self.eval()
        ids = input_ids.clone()

        for _ in range(max_new):
            # Truncate context to model's window size
            ctx    = ids[:, -self.cfg.n_ctx:]
            logits, _ = self(ctx)
            next_l = logits[:, -1, :] / temperature   # (1, vocab_size)

            # ── Top-k filter ──────────────────────────────────────────────────
            if top_k > 0:
                k       = min(top_k, next_l.size(-1))
                kth_val = torch.topk(next_l, k).values[:, -1, None]
                next_l  = next_l.masked_fill(next_l < kth_val, float("-inf"))

            # ── Nucleus (top-p) filter ────────────────────────────────────────
            if 0.0 < top_p < 1.0:
                sorted_l, sorted_idx = torch.sort(next_l, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
                # Remove tokens where cumulative prob exceeds threshold
                # Shift right so the token that crosses the threshold is kept
                remove = cum_probs - F.softmax(sorted_l, dim=-1) > top_p
                sorted_l[remove] = float("-inf")
                # Scatter back to original token ordering
                next_l = torch.zeros_like(next_l).scatter_(1, sorted_idx, sorted_l)

            probs    = F.softmax(next_l, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)   # (1, 1)
            ids      = torch.cat([ids, next_tok], dim=1)

            if eos_token_id is not None and next_tok.item() == eos_token_id:
                break

        return ids   # (1, T + max_new)

    # ── Parameter count ───────────────────────────────────────────────────────
    def n_parameters(self, exclude_embedding: bool = False) -> int:
        if not exclude_embedding:
            return sum(p.numel() for p in self.parameters())
        return sum(
            p.numel() for name, p in self.named_parameters()
            if "emb" not in name
        )

    # ── Load pretrained GPT-2 weights ─────────────────────────────────────────
    @classmethod
    def from_pretrained(cls, model_name: str = "gpt2") -> "GPTLanguageModel":
        """
        Initialise this from-scratch architecture with open-source GPT-2 weights.

        Because the architecture here is identical to GPT-2, the weights map
        one-to-one.  We use HuggingFace only to *download* the checkpoint;
        we never use their model class for inference — that is done by our
        hand-coded GPTLanguageModel instead.

        Supported sizes:
            "gpt2"        → 117 M parameters
            "gpt2-medium" → 345 M parameters
            "gpt2-large"  → 774 M parameters

        Args:
            model_name: HuggingFace model identifier.

        Returns:
            GPTLanguageModel with loaded weights, in eval() mode.
        """
        try:
            from transformers import GPT2LMHeadModel
        except ImportError:
            raise ImportError("pip install transformers  # needed to download weights")

        size_map = {
            "gpt2":        GPTConfig(n_layers=12, n_heads=12, d_model=768),
            "gpt2-medium": GPTConfig(n_layers=24, n_heads=16, d_model=1024),
            "gpt2-large":  GPTConfig(n_layers=36, n_heads=20, d_model=1280),
        }
        if model_name not in size_map:
            raise ValueError(f"Unknown model '{model_name}'. Choices: {list(size_map)}")

        cfg   = size_map[model_name]
        model = cls(cfg)

        print(f"[ScratchGPT] Downloading {model_name} weights from HuggingFace …")
        hf_model = GPT2LMHeadModel.from_pretrained(model_name)
        hf_sd    = hf_model.state_dict()

        # Direct references to our model's parameter tensors
        our_params: dict[str, torch.Tensor] = dict(
            chain(model.named_parameters(), model.named_buffers())
        )

        # GPT-2 stores Conv1D weights transposed relative to nn.Linear
        # (Conv1D has shape [in, out]; nn.Linear expects [out, in])
        needs_transpose = {"c_attn.weight", "c_proj.weight", "c_fc.weight"}

        # ── Name mapping: HuggingFace key → our key ───────────────────────────
        name_map: dict[str, str] = {
            "transformer.wte.weight":  "token_emb.weight",
            "transformer.wpe.weight":  "pos_emb.weight",
            "transformer.ln_f.weight": "ln_final.weight",
            "transformer.ln_f.bias":   "ln_final.bias",
        }
        for i in range(cfg.n_layers):
            h = f"transformer.h.{i}"
            b = f"blocks.{i}"
            name_map.update({
                f"{h}.ln_1.weight":          f"{b}.ln1.weight",
                f"{h}.ln_1.bias":            f"{b}.ln1.bias",
                f"{h}.attn.c_attn.weight":   f"{b}.attn.W_qkv.weight",
                f"{h}.attn.c_attn.bias":     f"{b}.attn.W_qkv.bias",
                f"{h}.attn.c_proj.weight":   f"{b}.attn.W_proj.weight",
                f"{h}.attn.c_proj.bias":     f"{b}.attn.W_proj.bias",
                f"{h}.ln_2.weight":          f"{b}.ln2.weight",
                f"{h}.ln_2.bias":            f"{b}.ln2.bias",
                f"{h}.mlp.c_fc.weight":      f"{b}.ffn.fc1.weight",
                f"{h}.mlp.c_fc.bias":        f"{b}.ffn.fc1.bias",
                f"{h}.mlp.c_proj.weight":    f"{b}.ffn.fc2.weight",
                f"{h}.mlp.c_proj.bias":      f"{b}.ffn.fc2.bias",
            })

        # ── Copy weights in-place into the model's parameter tensors ──────────
        loaded = 0
        with torch.no_grad():
            for hf_key, our_key in name_map.items():
                if hf_key not in hf_sd:
                    continue
                w = hf_sd[hf_key]
                # Transpose Conv1D weights to match nn.Linear layout
                if any(suffix in hf_key for suffix in needs_transpose):
                    w = w.T
                our_params[our_key].copy_(w)
                loaded += 1

        del hf_model   # free HuggingFace model memory immediately

        total = len(name_map)
        print(f"[ScratchGPT] Loaded {loaded}/{total} weight tensors into from-scratch model.")
        print(f"[ScratchGPT] Parameters: {model.n_parameters():,}")

        model.eval()
        return model


# ══════════════════════════════════════════════════════════════════════════════
# 7.  TRAINING UTILITIES  (train the model from scratch)
# ══════════════════════════════════════════════════════════════════════════════

class TextDataset(torch.utils.data.Dataset):
    """
    Simple character/BPE-tokenised dataset for language model training.

    Reads one or more plain-text files, tokenises with GPT-2's BPE tokeniser,
    and returns overlapping windows of length (block_size + 1) so that the
    model can predict the next token at every position.

    Example:
        tokens = [A, B, C, D, E]  block_size=3
        → sample 0: x=[A,B,C], y=[B,C,D]
        → sample 1: x=[B,C,D], y=[C,D,E]
    """

    def __init__(self, text_files: list[str | Path], block_size: int = 512):
        try:
            from transformers import GPT2Tokenizer
            tok = GPT2Tokenizer.from_pretrained("gpt2")
        except ImportError:
            raise ImportError("pip install transformers  # required for tokenisation")

        all_tokens: list[int] = []
        for path in text_files:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            all_tokens.extend(tok.encode(text))

        self.data       = torch.tensor(all_tokens, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk  = self.data[idx : idx + self.block_size + 1]
        x, y   = chunk[:-1], chunk[1:]
        return x, y


def train(
    model:      GPTLanguageModel,
    text_files: list[str | Path],
    epochs:     int   = 1,
    batch_size: int   = 4,
    lr:         float = 3e-4,
    block_size: int   = 512,
    save_path:  Optional[str] = None,
    device:     str   = "auto",
) -> list[float]:
    """
    Train (or fine-tune) GPTLanguageModel on plain-text files.

    Args:
        model:      The GPTLanguageModel instance (random init or pretrained).
        text_files: List of paths to .txt files used as the training corpus.
        epochs:     Number of full passes over the training data.
        batch_size: Sequences per gradient step.
        lr:         AdamW learning rate (3e-4 is a good starting point).
        block_size: Token window length for each training sample.
        save_path:  If set, saves the final model weights to this .pt file.
        device:     "auto" picks CUDA if available, else CPU.

    Returns:
        List of per-batch loss values (useful for plotting a learning curve).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.train()

    dataset = TextDataset(text_files, block_size=block_size)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # AdamW with weight decay on weight matrices only (not biases / norms)
    decay_params     = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters() if p.dim() <  2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr)

    losses: list[float] = []
    total_steps = epochs * len(loader)
    step = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            _, loss = model(x, targets=y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.item())
            epoch_loss += loss.item()
            step += 1

            if step % 50 == 0:
                avg = epoch_loss / (step - (epoch - 1) * len(loader))
                pct = 100 * step / total_steps
                print(f"  step {step:>5}/{total_steps}  ({pct:.0f}%)  "
                      f"loss {loss.item():.4f}  epoch-avg {avg:.4f}")

        print(f"Epoch {epoch}/{epochs}  —  avg loss: {epoch_loss/len(loader):.4f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"[ScratchGPT] Weights saved to {save_path}")

    model.eval()
    return losses


# ══════════════════════════════════════════════════════════════════════════════
# 8.  RAG INTEGRATION  — drop-in replacement for GPT-4o in CampusRAG
# ══════════════════════════════════════════════════════════════════════════════

class ScratchRAGGenerator:
    """
    Wraps GPTLanguageModel for use inside the CampusRAG pipeline
    (phase5_rag_pipeline.py) as a local, from-scratch replacement for the
    GPT-4o API call.

    The generator:
      1. Builds a plain-text prompt from the system instructions, retrieved
         chunks and conversation history.
      2. Tokenises the prompt with GPT-2's BPE tokeniser.
      3. Runs auto-regressive generation through GPTLanguageModel.generate().
      4. Decodes only the newly generated tokens and strips continuation artefacts.

    Usage:
        model = GPTLanguageModel.from_pretrained("gpt2")
        gen   = ScratchRAGGenerator(model)

        # In place of rag._oai_chat.chat.completions.create(…):
        answer = gen.generate(query, retrieved_chunks)
    """

    # Strings that mark the start of a new turn — stop generation there
    _STOP_SEQS = ["\nStudent:", "\nAdvisor:", "\nSystem:", "\n\nStudent:", "---"]

    def __init__(
        self,
        model:              GPTLanguageModel,
        max_ctx_tokens:     int   = 700,
        max_new_tokens:     int   = 300,
        temperature:        float = 0.7,
        top_k:              int   = 50,
        top_p:              float = 0.9,
        device:             str   = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device    = device
        self.model     = model.to(device)
        self.model.eval()

        self.max_ctx   = max_ctx_tokens
        self.max_new   = max_new_tokens
        self.temp      = temperature
        self.top_k     = top_k
        self.top_p     = top_p

        try:
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        except ImportError:
            raise ImportError("pip install transformers  # needed for BPE tokenisation")

    # ── Prompt builder ────────────────────────────────────────────────────────
    def _build_prompt(
        self,
        query:   str,
        chunks:  list,
        history: Optional[list[dict]] = None,
    ) -> str:
        """
        Constructs a flat text prompt that encodes all RAG context:

            [System instructions — truncated]
            [Retrieved document chunks — up to 4, 200 chars each]
            [Last 2 conversation turns]
            Student: <query>
            Advisor:
        """
        lines: list[str] = []

        # Brief system instruction (truncated to save tokens)
        lines.append(
            "You are the official Academic Advisor for Zewail City of Science "
            "and Technology. Answer student questions accurately using only the "
            "provided context. If the answer is not in the context, say so."
        )
        lines.append("")

        # Retrieved context chunks
        if chunks:
            lines.append("CONTEXT:")
            for i, c in enumerate(chunks[:4], 1):
                snippet = c.text[:200].replace("\n", " ")
                lines.append(f"[Doc {i}] {snippet}")
            lines.append("")

        # Recent conversation history
        if history:
            for msg in history[-4:]:
                role    = msg.get("role", "")
                content = msg.get("content", "")[:120].replace("\n", " ")
                if role == "user":
                    lines.append(f"Student: {content}")
                elif role == "assistant":
                    lines.append(f"Advisor: {content}")
            lines.append("")

        lines.append(f"Student: {query}")
        lines.append("Advisor:")

        return "\n".join(lines)

    # ── Main generate method ──────────────────────────────────────────────────
    def generate(
        self,
        query:   str,
        chunks:  list,
        history: Optional[list[dict]] = None,
    ) -> str:
        """
        Generate an answer for 'query' given retrieved document chunks.

        Returns:
            Answer string (generated by the from-scratch Transformer).
        """
        prompt     = self._build_prompt(query, chunks, history)
        input_ids  = self.tokenizer.encode(prompt, return_tensors="pt")

        # Truncate prompt to leave room for the answer
        if input_ids.size(1) > self.max_ctx:
            input_ids = input_ids[:, -self.max_ctx:]

        input_ids  = input_ids.to(self.device)
        prompt_len = input_ids.size(1)

        output_ids = self.model.generate(
            input_ids,
            max_new     = self.max_new,
            temperature = self.temp,
            top_k       = self.top_k,
            top_p       = self.top_p,
        )

        # Decode only the newly generated tokens (skip the prompt)
        new_ids = output_ids[0, prompt_len:]
        answer  = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        # Stop at the first new-turn marker
        for stop in self._STOP_SEQS:
            if stop in answer:
                answer = answer[: answer.index(stop)]

        answer = answer.strip()
        return answer or "I don't have enough information to answer that question."


# ══════════════════════════════════════════════════════════════════════════════
# 9.  ARCHITECTURE SUMMARY HELPER
# ══════════════════════════════════════════════════════════════════════════════

def print_architecture(cfg: GPTConfig) -> None:
    """Print a human-readable summary of the model architecture and dimensions."""
    separator = "─" * 60
    print(separator)
    print("  GPT-style Decoder-only Transformer  (from scratch)")
    print(separator)
    print(f"  Vocabulary size      : {cfg.vocab_size:,}")
    print(f"  Context window       : {cfg.n_ctx:,} tokens")
    print(f"  Embedding dimension  : {cfg.d_model}")
    print(f"  Attention heads      : {cfg.n_heads}  (d_head = {cfg.d_head})")
    print(f"  Transformer blocks   : {cfg.n_layers}")
    print(f"  FFN hidden dimension : {cfg.d_ff}  (4 × d_model)")
    print(f"  Dropout              : {cfg.dropout}")
    print(separator)
    print("  Layers per block:")
    print("    ├── LayerNorm  (pre-norm)               d_model")
    print("    ├── MultiHeadCausalSelfAttention")
    print(f"    │     W_qkv : ({cfg.d_model}, {3*cfg.d_model})  →  split Q,K,V")
    print(f"    │     heads  : {cfg.n_heads} × d_head={cfg.d_head}")
    print(f"    │     W_proj : ({cfg.d_model}, {cfg.d_model})")
    print("    ├── Residual (+x)")
    print("    ├── LayerNorm")
    print("    ├── FeedForward")
    print(f"    │     fc1   : ({cfg.d_model}, {cfg.d_ff})  +  GELU")
    print(f"    │     fc2   : ({cfg.d_ff}, {cfg.d_model})")
    print("    └── Residual (+x)")
    print(separator)

    # Approximate parameter count
    attn_params  = cfg.n_layers * (4 * cfg.d_model * cfg.d_model)   # Q,K,V + proj
    ffn_params   = cfg.n_layers * (2 * cfg.d_model * cfg.d_ff)
    embed_params = cfg.vocab_size * cfg.d_model + cfg.n_ctx * cfg.d_model
    ln_params    = cfg.n_layers * 2 * 2 * cfg.d_model + 2 * cfg.d_model
    total        = attn_params + ffn_params + embed_params + ln_params

    print(f"  Approx parameters    : {total:,}  (~{total/1e6:.0f} M)")
    print(separator)


# ══════════════════════════════════════════════════════════════════════════════
# 10.  STANDALONE DEMO
# ══════════════════════════════════════════════════════════════════════════════

def _demo_architecture_only() -> None:
    """Quick demo that runs on CPU without downloading any weights."""
    print("\n=== Demo: Architecture Verification (random weights, CPU) ===\n")

    # Use a tiny config so it runs instantly on any machine
    cfg = GPTConfig(
        vocab_size = 50_257,
        n_ctx      = 256,
        d_model    = 128,
        n_heads    = 4,
        n_layers   = 2,
        dropout    = 0.0,
    )
    print_architecture(cfg)

    model = GPTLanguageModel(cfg)
    print(f"\nModel created. Parameters: {model.n_parameters():,}")

    # Forward pass with dummy input
    batch_size, seq_len = 2, 64
    dummy_ids     = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    dummy_targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    logits, loss = model(dummy_ids, targets=dummy_targets)
    print(f"\nForward pass:")
    print(f"  input  shape : {tuple(dummy_ids.shape)}")
    print(f"  logits shape : {tuple(logits.shape)}  (B, T, vocab_size)")
    print(f"  training loss: {loss.item():.4f}  (expect ≈ {math.log(cfg.vocab_size):.2f} at init)")

    # Generation with dummy prompt
    print("\nGeneration test (10 tokens from random prompt):")
    prompt_ids  = torch.randint(0, cfg.vocab_size, (1, 5))
    output_ids  = model.generate(prompt_ids, max_new=10, temperature=1.0)
    new_tokens  = output_ids[0, 5:].tolist()
    print(f"  prompt tokens  : {prompt_ids[0].tolist()}")
    print(f"  generated tokens: {new_tokens}")
    print("\nArchitecture verified. All layers working correctly.")


def _demo_pretrained_rag() -> None:
    """
    Demo: load GPT-2 pretrained weights into the from-scratch model,
    then run it as a RAG generator on a sample campus question.

    Requires:  pip install transformers torch
    Downloads: ~500 MB (GPT-2 Small weights, cached after first run)
    """
    print("\n=== Demo: Pretrained GPT-2 weights → from-scratch RAG generator ===\n")
    print_architecture(GPTConfig())

    model = GPTLanguageModel.from_pretrained("gpt2")

    # Simulate a retrieved chunk (as CampusRAG would provide)
    class _FakeChunk:
        text = (
            "Zewail City of Science and Technology (UST) offers undergraduate "
            "programs in four schools: Engineering (ENGR), CSAI, Science (SCI), "
            "and Business (BUS). The CSAI school requires 132 credit hours to "
            "graduate. Programs include Computer Science, DSAI, HCI, and "
            "Computer Engineering."
        )

    gen    = ScratchRAGGenerator(model, max_new_tokens=80, temperature=0.7)
    query  = "How many credit hours do I need to graduate from CSAI?"
    chunks = [_FakeChunk()]

    print(f"\nQuery  : {query}")
    print(f"Chunks : 1 retrieved document")
    print("\nGenerating answer …")
    t0     = time.time()
    answer = gen.generate(query, chunks)
    elapsed = time.time() - t0
    print(f"\nAnswer ({elapsed:.1f}s):\n  {answer}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "pretrained":
        _demo_pretrained_rag()
    else:
        _demo_architecture_only()
        print("\nRun with 'pretrained' argument to load GPT-2 weights and test RAG:")
        print("  python models_from_scratch/transformer_lm.py pretrained")

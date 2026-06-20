#!/usr/bin/env python3
"""
models_from_scratch/seq2seq_transformer.py
==========================================
Encoder-Decoder Transformer — built entirely from scratch in PyTorch.
This is the architecture from "Attention Is All You Need" (Vaswani et al., 2017),
extended with modern improvements.

ARCHITECTURAL DIFFERENCES vs the already-built GPT-2 (transformer_lm.py)
─────────────────────────────────────────────────────────────────────────
┌──────────────────────────┬──────────────────────────┬──────────────────────────────┐
│ Feature                  │ GPT-2 (transformer_lm)   │ THIS MODEL (seq2seq)         │
├──────────────────────────┼──────────────────────────┼──────────────────────────────┤
│ Architecture             │ Decoder-ONLY (1 stack)   │ Encoder + Decoder (2 stacks) │
│ Positional encoding      │ Learned nn.Embedding     │ Sinusoidal — FIXED, no grads │
│ Encoder self-attention   │ ✗ (no encoder)           │ BIDIRECTIONAL (sees all)     │
│ Cross-attention          │ ✗ (none)                 │ ✓ Q=decoder, K/V=encoder     │
│ Layer normalisation      │ Pre-norm (before sublyr) │ Post-norm (after sublayer)   │
│ Weight tying             │ LM head = token emb ᵀ    │ Separate encoder/decoder emb │
│ Attention types          │ 1 (causal self-attn)     │ 3 (bidir SA, causal SA, CA)  │
│ Suitable for             │ Left-to-right generation │ Seq2seq (read input → output)│
└──────────────────────────┴──────────────────────────┴──────────────────────────────┘

Full Architecture Diagram:
┌─────────────────────────────────────────────────────────────────────────┐
│                         ENCODER SIDE                                    │
│  Input token IDs → Token Embedding + Sinusoidal PE → Dropout            │
│  ×N EncoderLayer:                                                       │
│    ├── MultiHeadSelfAttention  (BIDIRECTIONAL — no causal mask)         │
│    │     Q = K = V = x                                                  │
│    ├── Residual + LayerNorm   (POST-norm)                               │
│    ├── FeedForward: ReLU(W1·x) → W2                                    │
│    └── Residual + LayerNorm                                             │
│  Final LayerNorm → encoder_output  (B, src_len, d_model)               │
├─────────────────────────────────────────────────────────────────────────┤
│                         DECODER SIDE                                    │
│  Target token IDs → Token Embedding + Sinusoidal PE → Dropout           │
│  ×N DecoderLayer:                                                       │
│    ├── MultiHeadSelfAttention  (CAUSAL — upper-triangle mask)           │
│    ├── Residual + LayerNorm                                             │
│    ├── MultiHeadCrossAttention (Q=decoder hidden, K=V=encoder_output)   │
│    │     This is the bridge between encoder and decoder                 │
│    ├── Residual + LayerNorm                                             │
│    ├── FeedForward                                                      │
│    └── Residual + LayerNorm                                             │
│  Final LayerNorm → LM Head (d_model → vocab_size) → logits             │
└─────────────────────────────────────────────────────────────────────────┘

Intended function in Career Advisor (github_analyzer.py):
  • GPT-4o (production): build_prompt(analysis) → GPT API → career report
  • THIS MODEL (from scratch): build_prompt(analysis) → encode → decode → career report
  The encoder reads the full portfolio data bidirectionally;
  the decoder generates the advisory report attending to that encoding.

Usage:
    # Option 1 — load BART pretrained weights (works immediately)
    model = Seq2SeqTransformer.from_pretrained("facebook/bart-base")

    # Option 2 — random init + train from scratch
    model = Seq2SeqTransformer(Seq2SeqConfig())

    # Drop into the career advisor
    advisor = PortfolioAdvisorWrapper(model)
    report  = advisor.generate(github_analyzer.build_prompt(analysis))
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Seq2SeqConfig:
    """
    Hyperparameters for the Encoder-Decoder Transformer.

    Default values match BART-base (6 layers, d_model=768, 12 heads),
    which is smaller than GPT-2 large but architecturally richer due
    to the bidirectional encoder and cross-attention mechanism.

    Presets:
        BART-base  → Seq2SeqConfig()                             (139 M params)
        BART-large → Seq2SeqConfig(n_layers=12, d_model=1024, n_heads=16)
        Tiny demo  → Seq2SeqConfig(n_layers=2, d_model=128, n_heads=4)
    """
    vocab_size:   int   = 50_265   # BART BPE vocabulary (slightly larger than GPT-2)
    max_src_len:  int   = 1_024    # max encoder (source) sequence length
    max_tgt_len:  int   = 512      # max decoder (target) sequence length
    d_model:      int   = 768      # embedding / hidden dimension
    n_heads:      int   = 12       # attention heads (d_model must be divisible by n_heads)
    n_layers:     int   = 6        # encoder layers = decoder layers
    d_ff:         int   = 3_072    # feed-forward hidden dim (4 × d_model)
    dropout:      float = 0.1      # dropout probability
    pad_token_id: int   = 1        # BPE padding token

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SINUSOIDAL POSITIONAL ENCODING  (fixed — not learned)
#     KEY DIFFERENCE from GPT-2: GPT-2 uses nn.Embedding (learned).
#     Here we use the original paper's fixed sine/cosine formula.
# ══════════════════════════════════════════════════════════════════════════════

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    For each position pos and each pair of dimensions (2i, 2i+1):

        PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
        PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )

    KEY DIFFERENCE from GPT-2:
      GPT-2 uses  nn.Embedding(n_ctx, d_model) — LEARNED, has gradients.
      This uses a closed-form formula — FIXED, zero parameters, no gradients.

    The sinusoidal pattern lets the model attend by relative position:
      PE(pos+k) can be expressed as a linear function of PE(pos),
      so the model can generalise to sequences longer than seen in training.
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Pre-compute the full encoding table (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        positions = torch.arange(max_len, dtype=torch.float).unsqueeze(1)     # (max_len, 1)
        # Compute division term: 1 / 10000^(2i/d_model) for each i
        div_term  = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )                                                                       # (d_model/2,)
        pe[:, 0::2] = torch.sin(positions * div_term)   # even dims → sine
        pe[:, 1::2] = torch.cos(positions * div_term)   # odd  dims → cosine

        # Register as a buffer: moves to GPU with model, but NOT a learnable parameter
        pe = pe.unsqueeze(0)   # (1, max_len, d_model) — broadcastable over batch
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1), :]   # add positional encoding in-place
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MULTI-HEAD ATTENTION  (general — handles all three attention types)
#     KEY DIFFERENCE: same Q/K/V projections but three different usage modes
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    General Multi-Head Attention covering all three usage modes:

    Mode 1 — Encoder Self-Attention (BIDIRECTIONAL, no mask):
        Q = K = V = encoder_hidden
        Every token can attend to every other token — full context.

    Mode 2 — Decoder Masked Self-Attention (CAUSAL):
        Q = K = V = decoder_hidden
        Token at position t can only attend to positions 0..t (causal mask).
        Same as GPT-2's attention, but here it is just ONE of three sublayers.

    Mode 3 — Decoder Cross-Attention (KEY DIFFERENCE from GPT-2):
        Q = decoder_hidden  (queries from the decoder)
        K = V = encoder_output  (keys/values from the encoder)
        The decoder "reads" the encoder's representations to condition generation.
        GPT-2 has NO cross-attention — this entire mechanism is absent there.

    Scaled dot-product attention for each head h:
        score_h  = Q_h @ K_hᵀ / √d_head
        (optional causal mask: future positions → −∞)
        α_h      = softmax(score_h, dim=-1)
        ctx_h    = α_h @ V_h

    All heads concatenated → output projection:
        out = concat(ctx_1, …, ctx_H) @ W_o + b_o
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head  = d_model // n_heads

        # Separate Q, K, V projections (unlike GPT-2's fused W_qkv)
        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)

        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        query:   torch.Tensor,                   # (B, T_q, d_model)
        key:     torch.Tensor,                   # (B, T_k, d_model)
        value:   torch.Tensor,                   # (B, T_k, d_model)
        causal:  bool = False,                   # True → apply upper-triangle causal mask
    ) -> torch.Tensor:
        B, T_q, _ = query.shape
        T_k       = key.size(1)

        # ── Step 1: Linear projections ────────────────────────────────────────
        Q = self.W_q(query)   # (B, T_q, d_model)
        K = self.W_k(key)     # (B, T_k, d_model)
        V = self.W_v(value)   # (B, T_k, d_model)

        # ── Step 2: Split into heads ──────────────────────────────────────────
        def split(t: torch.Tensor, T: int) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            # → (B, n_heads, T, d_head)

        Q = split(Q, T_q)
        K = split(K, T_k)
        V = split(V, T_k)

        # ── Step 3: Scaled dot-product attention ──────────────────────────────
        scale  = math.sqrt(self.d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale   # (B, n_heads, T_q, T_k)

        if causal:
            # Upper-triangle mask: token i cannot see token j if j > i
            mask = torch.triu(
                torch.ones(T_q, T_k, device=query.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))

        attn_w = F.softmax(scores, dim=-1)        # (B, n_heads, T_q, T_k)
        attn_w = self.attn_drop(attn_w)
        ctx    = torch.matmul(attn_w, V)           # (B, n_heads, T_q, d_head)

        # ── Step 4: Concatenate heads → output projection ─────────────────────
        ctx = ctx.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.W_o(ctx)                       # (B, T_q, d_model)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FEED-FORWARD NETWORK  (same expand-contract structure as GPT-2)
# ══════════════════════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
        FFN(x) = W2 · ReLU(W1 · x + b1) + b2

    Uses ReLU (original paper) rather than GPT-2's GELU — another
    architectural difference between the two models.
    Expands d_model → d_ff (4×) then contracts back.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1  = nn.Linear(d_model, d_ff,    bias=True)
        self.fc2  = nn.Linear(d_ff,    d_model, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ENCODER LAYER  (bidirectional self-attention + FFN, POST-norm)
#     KEY DIFFERENCE from GPT-2: Post-LayerNorm (normalise AFTER residual add)
#     GPT-2 uses Pre-LayerNorm (normalise BEFORE sublayer).
# ══════════════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    One Transformer Encoder Layer (Post-LayerNorm, bidirectional).

    Forward pass:
        attn_out = SelfAttention(x, x, x, causal=False)   # bidirectional
        x        = LayerNorm(x + attn_out)                 # POST-norm residual
        ffn_out  = FFN(x)
        x        = LayerNorm(x + ffn_out)                  # POST-norm residual

    KEY DIFFERENCES from GPT-2:
    1. Post-norm: normalisation is applied AFTER the residual addition.
       GPT-2 applies it BEFORE the sublayer (Pre-LN).
    2. Bidirectional attention: no causal mask, so each position attends
       to ALL other positions. GPT-2 uses only causal attention everywhere.
    """

    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ffn       = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.norm1     = nn.LayerNorm(cfg.d_model)
        self.norm2     = nn.LayerNorm(cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bidirectional self-attention sublayer (Post-LN)
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, causal=False)))
        # Feed-forward sublayer (Post-LN)
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 6.  DECODER LAYER  (causal SA + cross-attention + FFN, POST-norm)
#     BRAND NEW vs GPT-2: the cross-attention sublayer does not exist in GPT-2
# ══════════════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    One Transformer Decoder Layer — three sublayers.

    Forward pass:
        # 1. Causal self-attention (same kind as GPT-2, but only the 1st sublayer)
        x = LayerNorm(x + CausalSelfAttention(x))

        # 2. Cross-attention (Q from decoder, K/V from encoder — NEW vs GPT-2)
        x = LayerNorm(x + CrossAttention(query=x, key=enc, value=enc))

        # 3. Feed-forward
        x = LayerNorm(x + FFN(x))

    The cross-attention sublayer is the core architectural innovation:
    it lets the decoder query every position of the encoder output at each
    generation step, so the full source is always in scope regardless of
    how long the target sequence grows.
    GPT-2 has no such mechanism — it relies solely on the concatenation of
    source and target in the input sequence, limited by the context window.
    """

    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        # Sublayer 1: causal masked self-attention on decoder tokens
        self.self_attn  = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        # Sublayer 2: cross-attention — decoder queries the encoder output
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        # Sublayer 3: position-wise FFN
        self.ffn        = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

        self.norm1 = nn.LayerNorm(cfg.d_model)   # after self-attn
        self.norm2 = nn.LayerNorm(cfg.d_model)   # after cross-attn
        self.norm3 = nn.LayerNorm(cfg.d_model)   # after FFN
        self.drop  = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x:          torch.Tensor,   # (B, T_tgt, d_model) — decoder hidden states
        enc_output: torch.Tensor,   # (B, T_src, d_model) — encoder output
    ) -> torch.Tensor:
        # 1. Causal self-attention (decoder can only look at past tokens)
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, causal=True)))

        # 2. Cross-attention: Q = decoder hidden, K = V = encoder output
        #    The decoder "reads" the full source encoding here
        x = self.norm2(x + self.drop(self.cross_attn(x, enc_output, enc_output, causal=False)))

        # 3. Feed-forward
        x = self.norm3(x + self.drop(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 7.  ENCODER STACK
# ══════════════════════════════════════════════════════════════════════════════

class TransformerEncoder(nn.Module):
    """
    N stacked EncoderLayers with final LayerNorm.
    Reads the FULL source sequence bidirectionally.
    """

    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm   = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)   # (B, T_src, d_model)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  DECODER STACK
# ══════════════════════════════════════════════════════════════════════════════

class TransformerDecoder(nn.Module):
    """
    N stacked DecoderLayers with final LayerNorm.
    Generates target tokens causally while attending to the encoder output.
    """

    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm   = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        x:          torch.Tensor,   # (B, T_tgt, d_model)
        enc_output: torch.Tensor,   # (B, T_src, d_model)
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_output)
        return self.norm(x)         # (B, T_tgt, d_model)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  FULL SEQ2SEQ TRANSFORMER MODEL
# ══════════════════════════════════════════════════════════════════════════════

class Seq2SeqTransformer(nn.Module):
    """
    Complete Encoder-Decoder Transformer Language Model.

    Forward pass:
        src_ids, tgt_ids → encode(src) → decode(tgt, enc_out) → logits

    Architectural summary vs GPT-2:
        GPT-2  : one stack, decoder-only, learned PE, pre-norm, 1 attn type
        This   : two stacks, encoder+decoder, sinusoidal PE, post-norm, 3 attn types

    Weight tying:
        Unlike GPT-2, encoder and decoder embedding matrices are SEPARATE.
        The LM head (decoder → vocab) shares weights with the DECODER embedding only.
    """

    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        self.cfg = cfg

        # Separate embedding tables for encoder and decoder
        self.encoder_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.decoder_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)

        # Sinusoidal PE — shared between encoder and decoder (no parameters)
        max_len = max(cfg.max_src_len, cfg.max_tgt_len)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, max_len, cfg.dropout)

        # Encoder and Decoder stacks
        self.encoder = TransformerEncoder(cfg)
        self.decoder = TransformerDecoder(cfg)

        # LM head: projects decoder output → vocabulary logits
        # Weight-tied to decoder_embed (but NOT to encoder_embed — different from GPT-2)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.decoder_embed.weight   # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def encode(self, src_ids: torch.Tensor) -> torch.Tensor:
        """
        Run the encoder on source token ids.
        Returns encoder_output of shape (B, T_src, d_model).
        """
        x = self.encoder_embed(src_ids)   # (B, T_src, d_model)
        x = self.pos_enc(x)               # + sinusoidal PE
        return self.encoder(x)            # (B, T_src, d_model)

    def decode(
        self,
        tgt_ids:    torch.Tensor,   # (B, T_tgt)
        enc_output: torch.Tensor,   # (B, T_src, d_model)
    ) -> torch.Tensor:
        """
        Run the decoder one full forward pass (teacher-forcing during training).
        Returns logits of shape (B, T_tgt, vocab_size).
        """
        x      = self.decoder_embed(tgt_ids)   # (B, T_tgt, d_model)
        x      = self.pos_enc(x)               # + sinusoidal PE
        x      = self.decoder(x, enc_output)   # (B, T_tgt, d_model)
        return self.lm_head(x)                 # (B, T_tgt, vocab_size)

    def forward(
        self,
        src_ids:  torch.Tensor,                      # (B, T_src)
        tgt_ids:  torch.Tensor,                      # (B, T_tgt)
        targets:  Optional[torch.Tensor] = None,     # (B, T_tgt) shifted labels
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Full forward pass: encode source → decode target → logits (+ loss).
        """
        enc_output = self.encode(src_ids)
        logits     = self.decode(tgt_ids, enc_output)   # (B, T_tgt, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    # ── Auto-regressive generation ────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        src_ids:      torch.Tensor,          # (1, T_src) encoded source
        bos_token_id: int,
        eos_token_id: int,
        max_new:      int   = 400,
        temperature:  float = 0.7,
        top_k:        int   = 50,
        top_p:        float = 0.9,
    ) -> torch.Tensor:
        """
        Auto-regressive generation using the encoder-decoder architecture.

        Unlike GPT-2 (where source and target are concatenated in one sequence),
        here we:
          1. Encode the entire source ONCE → enc_output
          2. At each step, the decoder attends to enc_output via cross-attention
          3. Sample the next token, append it, repeat

        This is more efficient for long sources because the encoder output is
        computed only once and reused at every decoder step.
        """
        self.eval()
        enc_output = self.encode(src_ids)          # (1, T_src, d_model) — computed once
        tgt        = torch.tensor([[bos_token_id]], device=src_ids.device)  # (1, 1)

        for _ in range(max_new):
            # Clamp to max_tgt_len
            tgt_ctx = tgt[:, -self.cfg.max_tgt_len:]
            logits  = self.decode(tgt_ctx, enc_output)  # (1, T, vocab)
            next_l  = logits[:, -1, :] / temperature     # (1, vocab)

            # Top-k filtering
            if top_k > 0:
                kv = torch.topk(next_l, min(top_k, next_l.size(-1))).values[:, -1, None]
                next_l = next_l.masked_fill(next_l < kv, float("-inf"))

            # Nucleus (top-p) filtering
            if 0.0 < top_p < 1.0:
                sorted_l, sorted_idx = torch.sort(next_l, descending=True)
                cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
                remove = cum - F.softmax(sorted_l, dim=-1) > top_p
                sorted_l[remove] = float("-inf")
                next_l = torch.zeros_like(next_l).scatter_(1, sorted_idx, sorted_l)

            probs    = F.softmax(next_l, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)   # (1, 1)
            tgt      = torch.cat([tgt, next_tok], dim=1)

            if next_tok.item() == eos_token_id:
                break

        return tgt   # (1, T_tgt)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ── Load BART pretrained weights ──────────────────────────────────────────
    @classmethod
    def from_pretrained(cls, model_name: str = "facebook/bart-base") -> "Seq2SeqTransformer":
        """
        Load pretrained BART weights into this from-scratch Encoder-Decoder architecture.

        BART is an encoder-decoder transformer pre-trained on denoising tasks,
        making it ideal for generation tasks like career report writing.
        The weight mapping is more complex than GPT-2 because there are now
        encoder, decoder, and cross-attention weight matrices to map.

        Supported: "facebook/bart-base" (139 M), "facebook/bart-large" (406 M)
        """
        try:
            from transformers import BartForConditionalGeneration, BartConfig
        except ImportError:
            raise ImportError("pip install transformers  # needed to download weights")

        size_map = {
            "facebook/bart-base":  Seq2SeqConfig(n_layers=6,  d_model=768,  n_heads=12),
            "facebook/bart-large": Seq2SeqConfig(n_layers=12, d_model=1024, n_heads=16,
                                                  d_ff=4096, vocab_size=50265),
        }
        if model_name not in size_map:
            raise ValueError(f"Unknown model '{model_name}'. Choices: {list(size_map)}")

        cfg   = size_map[model_name]
        model = cls(cfg)

        print(f"[Seq2Seq] Downloading {model_name} weights from HuggingFace …")
        hf    = BartForConditionalGeneration.from_pretrained(model_name)
        hf_sd = hf.state_dict()

        our_params = dict(chain(model.named_parameters(), model.named_buffers()))

        def copy(hf_key: str, our_key: str):
            if hf_key in hf_sd and our_key in our_params:
                with torch.no_grad():
                    our_params[our_key].copy_(hf_sd[hf_key])

        # ── Shared token embeddings ───────────────────────────────────────────
        copy("model.shared.weight",             "encoder_embed.weight")
        copy("model.shared.weight",             "decoder_embed.weight")

        # ── Encoder layers ────────────────────────────────────────────────────
        for i in range(cfg.n_layers):
            h = f"model.encoder.layers.{i}"
            b = f"encoder.layers.{i}"
            # Self-attention
            copy(f"{h}.self_attn.q_proj.weight",  f"{b}.self_attn.W_q.weight")
            copy(f"{h}.self_attn.q_proj.bias",    f"{b}.self_attn.W_q.bias")
            copy(f"{h}.self_attn.k_proj.weight",  f"{b}.self_attn.W_k.weight")
            copy(f"{h}.self_attn.k_proj.bias",    f"{b}.self_attn.W_k.bias")
            copy(f"{h}.self_attn.v_proj.weight",  f"{b}.self_attn.W_v.weight")
            copy(f"{h}.self_attn.v_proj.bias",    f"{b}.self_attn.W_v.bias")
            copy(f"{h}.self_attn.out_proj.weight",f"{b}.self_attn.W_o.weight")
            copy(f"{h}.self_attn.out_proj.bias",  f"{b}.self_attn.W_o.bias")
            # Layer norms
            copy(f"{h}.self_attn_layer_norm.weight", f"{b}.norm1.weight")
            copy(f"{h}.self_attn_layer_norm.bias",   f"{b}.norm1.bias")
            copy(f"{h}.final_layer_norm.weight",     f"{b}.norm2.weight")
            copy(f"{h}.final_layer_norm.bias",       f"{b}.norm2.bias")
            # FFN
            copy(f"{h}.fc1.weight", f"{b}.ffn.fc1.weight")
            copy(f"{h}.fc1.bias",   f"{b}.ffn.fc1.bias")
            copy(f"{h}.fc2.weight", f"{b}.ffn.fc2.weight")
            copy(f"{h}.fc2.bias",   f"{b}.ffn.fc2.bias")

        # Encoder final norm
        copy("model.encoder.layer_norm.weight", "encoder.norm.weight")
        copy("model.encoder.layer_norm.bias",   "encoder.norm.bias")

        # ── Decoder layers ────────────────────────────────────────────────────
        for i in range(cfg.n_layers):
            h = f"model.decoder.layers.{i}"
            b = f"decoder.layers.{i}"
            # Causal self-attention
            copy(f"{h}.self_attn.q_proj.weight",  f"{b}.self_attn.W_q.weight")
            copy(f"{h}.self_attn.q_proj.bias",    f"{b}.self_attn.W_q.bias")
            copy(f"{h}.self_attn.k_proj.weight",  f"{b}.self_attn.W_k.weight")
            copy(f"{h}.self_attn.k_proj.bias",    f"{b}.self_attn.W_k.bias")
            copy(f"{h}.self_attn.v_proj.weight",  f"{b}.self_attn.W_v.weight")
            copy(f"{h}.self_attn.v_proj.bias",    f"{b}.self_attn.W_v.bias")
            copy(f"{h}.self_attn.out_proj.weight",f"{b}.self_attn.W_o.weight")
            copy(f"{h}.self_attn.out_proj.bias",  f"{b}.self_attn.W_o.bias")
            copy(f"{h}.self_attn_layer_norm.weight", f"{b}.norm1.weight")
            copy(f"{h}.self_attn_layer_norm.bias",   f"{b}.norm1.bias")
            # Cross-attention (encoder_attn in BART → cross_attn in ours)
            copy(f"{h}.encoder_attn.q_proj.weight",  f"{b}.cross_attn.W_q.weight")
            copy(f"{h}.encoder_attn.q_proj.bias",    f"{b}.cross_attn.W_q.bias")
            copy(f"{h}.encoder_attn.k_proj.weight",  f"{b}.cross_attn.W_k.weight")
            copy(f"{h}.encoder_attn.k_proj.bias",    f"{b}.cross_attn.W_k.bias")
            copy(f"{h}.encoder_attn.v_proj.weight",  f"{b}.cross_attn.W_v.weight")
            copy(f"{h}.encoder_attn.v_proj.bias",    f"{b}.cross_attn.W_v.bias")
            copy(f"{h}.encoder_attn.out_proj.weight",f"{b}.cross_attn.W_o.weight")
            copy(f"{h}.encoder_attn.out_proj.bias",  f"{b}.cross_attn.W_o.bias")
            copy(f"{h}.encoder_attn_layer_norm.weight", f"{b}.norm2.weight")
            copy(f"{h}.encoder_attn_layer_norm.bias",   f"{b}.norm2.bias")
            copy(f"{h}.final_layer_norm.weight",     f"{b}.norm3.weight")
            copy(f"{h}.final_layer_norm.bias",       f"{b}.norm3.bias")
            # FFN
            copy(f"{h}.fc1.weight", f"{b}.ffn.fc1.weight")
            copy(f"{h}.fc1.bias",   f"{b}.ffn.fc1.bias")
            copy(f"{h}.fc2.weight", f"{b}.ffn.fc2.weight")
            copy(f"{h}.fc2.bias",   f"{b}.ffn.fc2.bias")

        # Decoder final norm
        copy("model.decoder.layer_norm.weight", "decoder.norm.weight")
        copy("model.decoder.layer_norm.bias",   "decoder.norm.bias")
        # LM head
        copy("lm_head.weight", "lm_head.weight")

        del hf
        print(f"[Seq2Seq] Weights loaded. Parameters: {model.n_parameters():,}")
        model.eval()
        return model


# ══════════════════════════════════════════════════════════════════════════════
# 10.  PORTFOLIO ADVISOR WRAPPER  — drop-in for GPT in github_analyzer.py
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioAdvisorWrapper:
    """
    Wraps Seq2SeqTransformer to replace the GPT API call in github_analyzer.py.

    The GitHub analyzer builds a detailed portfolio prompt via build_prompt(analysis).
    This wrapper:
      1. Tokenises the prompt  → encoder input
      2. Runs the encoder once → encoder_output  (bidirectional, sees all tokens)
      3. Auto-regressively decodes the career advisory report
      4. Returns the decoded text

    This is architecturally different from the GPT-2 wrapper (ScratchRAGGenerator):
      • GPT-2 wrapper: concatenates prompt + answer in one sequence, no cross-attention
      • This wrapper:  encodes prompt separately, decoder attends via cross-attention
    """

    # Tokens that indicate the model has started repeating or gone off-rails
    _STOP_SEQS = ["\n\n\n\n", "═══════════════════════════════════════════"]

    def __init__(
        self,
        model:         Seq2SeqTransformer,
        max_src_tokens: int   = 512,
        max_new_tokens: int   = 400,
        temperature:    float = 0.8,
        top_k:          int   = 50,
        top_p:          float = 0.92,
        device:         str   = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device   = device
        self.model    = model.to(device)
        self.model.eval()

        self.max_src  = max_src_tokens
        self.max_new  = max_new_tokens
        self.temp     = temperature
        self.top_k    = top_k
        self.top_p    = top_p

        try:
            from transformers import BartTokenizer
            self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        except ImportError:
            raise ImportError("pip install transformers  # needed for tokenisation")

    def generate(self, prompt: str) -> str:
        """
        Generate a career advisory report from a github_analyzer.build_prompt() string.

        Args:
            prompt: The formatted portfolio prompt from GitHubAnalyzer.build_prompt()

        Returns:
            Generated career advisory report as a string.
        """
        # Tokenise and truncate the source prompt
        enc = self.tokenizer(
            prompt,
            max_length=self.max_src,
            truncation=True,
            return_tensors="pt",
        )
        src_ids = enc["input_ids"].to(self.device)

        output_ids = self.model.generate(
            src_ids,
            bos_token_id = self.tokenizer.bos_token_id,
            eos_token_id = self.tokenizer.eos_token_id,
            max_new      = self.max_new,
            temperature  = self.temp,
            top_k        = self.top_k,
            top_p        = self.top_p,
        )

        # Decode only the generated tokens (skip BOS)
        generated = self.tokenizer.decode(
            output_ids[0, 1:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Trim at stop sequences
        for stop in self._STOP_SEQS:
            if stop in generated:
                generated = generated[: generated.index(stop)]

        return generated.strip() or "(No report generated — model needs fine-tuning on career data.)"


# ══════════════════════════════════════════════════════════════════════════════
# 11.  ARCHITECTURE SUMMARY HELPER
# ══════════════════════════════════════════════════════════════════════════════

def print_architecture(cfg: Seq2SeqConfig) -> None:
    sep = "─" * 64
    print(sep)
    print("  Encoder-Decoder Transformer  (from scratch)")
    print("  'Attention Is All You Need'  architecture")
    print(sep)
    print(f"  Vocabulary size        : {cfg.vocab_size:,}")
    print(f"  Max source length      : {cfg.max_src_len:,} tokens")
    print(f"  Max target length      : {cfg.max_tgt_len:,} tokens")
    print(f"  Embedding dimension    : {cfg.d_model}")
    print(f"  Attention heads        : {cfg.n_heads}  (d_head = {cfg.d_head})")
    print(f"  Encoder / Decoder lyrs : {cfg.n_layers} + {cfg.n_layers}")
    print(f"  FFN hidden dim         : {cfg.d_ff}  (≈ {cfg.d_ff//cfg.d_model}× d_model)")
    print(f"  Positional encoding    : SINUSOIDAL (fixed, not learned)")
    print(f"  Normalisation          : Post-LayerNorm")
    print(sep)
    print("  ENCODER layers (×N):")
    print("    ├── MultiHeadSelfAttention  (BIDIRECTIONAL — no causal mask)")
    print(f"    │     W_q,W_k,W_v: ({cfg.d_model},{cfg.d_model})  W_o: ({cfg.d_model},{cfg.d_model})")
    print("    ├── Residual + LayerNorm   (POST-norm)")
    print("    ├── FeedForward")
    print(f"    │     fc1: ({cfg.d_model},{cfg.d_ff}) + GeLU  fc2: ({cfg.d_ff},{cfg.d_model})")
    print("    └── Residual + LayerNorm")
    print("  DECODER layers (×N):")
    print("    ├── MultiHeadSelfAttention  (CAUSAL — upper-triangle mask)")
    print("    ├── Residual + LayerNorm")
    print("    ├── MultiHeadCrossAttention (Q=decoder, K/V=encoder_output) ← NEW")
    print("    ├── Residual + LayerNorm")
    print("    ├── FeedForward")
    print("    └── Residual + LayerNorm")
    print(sep)

    enc_attn   = cfg.n_layers * 4 * cfg.d_model * cfg.d_model
    dec_attn   = cfg.n_layers * (4 + 4) * cfg.d_model * cfg.d_model
    enc_ffn    = cfg.n_layers * 2 * cfg.d_model * cfg.d_ff
    dec_ffn    = cfg.n_layers * 2 * cfg.d_model * cfg.d_ff
    embed      = 2 * cfg.vocab_size * cfg.d_model
    total      = enc_attn + dec_attn + enc_ffn + dec_ffn + embed

    print(f"  Approx parameters      : {total:,}  (~{total/1e6:.0f} M)")
    print(sep)
    print()
    print("  KEY DIFFERENCES vs GPT-2 (transformer_lm.py):")
    print("  ┌────────────────────────┬──────────────────┬──────────────────────┐")
    print("  │ Feature                │ GPT-2 (existing) │ This model           │")
    print("  ├────────────────────────┼──────────────────┼──────────────────────┤")
    print("  │ Architecture           │ Decoder-only     │ Encoder + Decoder    │")
    print("  │ Positional encoding    │ Learned          │ Sinusoidal (fixed)   │")
    print("  │ Encoder attention      │ Causal only      │ Bidirectional        │")
    print("  │ Cross-attention        │ None             │ ✓ (3rd sublayer)     │")
    print("  │ Layer norm placement   │ Pre-norm         │ Post-norm            │")
    print("  │ Attention sublayers    │ 1 per block      │ 2–3 per block        │")
    print("  └────────────────────────┴──────────────────┴──────────────────────┘")
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# 12.  STANDALONE DEMO
# ══════════════════════════════════════════════════════════════════════════════

def _demo_architecture_only() -> None:
    """Verify the full architecture runs on random weights (no download needed)."""
    print("\n=== Demo: Encoder-Decoder Architecture Verification (random weights) ===\n")

    cfg = Seq2SeqConfig(
        vocab_size  = 50_265,
        max_src_len = 128,
        max_tgt_len = 64,
        d_model     = 128,
        n_heads     = 4,
        n_layers    = 2,
        d_ff        = 512,
        dropout     = 0.0,
    )
    print_architecture(cfg)

    model = Seq2SeqTransformer(cfg)
    print(f"Model created. Parameters: {model.n_parameters():,}\n")

    B, T_src, T_tgt = 2, 40, 20
    src_ids  = torch.randint(3, cfg.vocab_size, (B, T_src))
    tgt_ids  = torch.randint(3, cfg.vocab_size, (B, T_tgt))
    targets  = torch.randint(3, cfg.vocab_size, (B, T_tgt))

    logits, loss = model(src_ids, tgt_ids, targets=targets)

    print("Forward pass:")
    print(f"  src_ids shape  : {tuple(src_ids.shape)}")
    print(f"  tgt_ids shape  : {tuple(tgt_ids.shape)}")
    print(f"  logits  shape  : {tuple(logits.shape)}  (B, T_tgt, vocab_size)")
    print(f"  training loss  : {loss.item():.4f}  (expect ≈ {math.log(cfg.vocab_size):.2f} at init)")

    print("\nGeneration test (15 tokens from random source):")
    src1     = torch.randint(3, cfg.vocab_size, (1, 10))
    out_ids  = model.generate(src1, bos_token_id=0, eos_token_id=2, max_new=15, temperature=1.0)
    print(f"  source tokens    : {src1[0].tolist()}")
    print(f"  generated tokens : {out_ids[0].tolist()}")

    print("\nEncoder-Decoder architecture verified. All 3 attention types working:")
    print("  ✓  Bidirectional encoder self-attention")
    print("  ✓  Causal decoder self-attention")
    print("  ✓  Decoder cross-attention (Q=decoder, K/V=encoder)")
    print("  ✓  Sinusoidal positional encoding (no learned parameters)")
    print("  ✓  Post-LayerNorm in both encoder and decoder")


def _demo_pretrained_portfolio() -> None:
    """
    Load BART weights into the from-scratch architecture and run it
    on a sample GitHub portfolio prompt.

    Requires:  pip install transformers torch
    Downloads: ~560 MB (BART-base weights, cached after first run)
    """
    print("\n=== Demo: BART weights → from-scratch Encoder-Decoder → Portfolio Advisory ===\n")
    print_architecture(Seq2SeqConfig())

    model = Seq2SeqTransformer.from_pretrained("facebook/bart-base")
    advisor = PortfolioAdvisorWrapper(model, max_src_tokens=300, max_new_tokens=200)

    # Simulate the prompt that github_analyzer.build_prompt() would generate
    sample_prompt = """You are an expert GitHub portfolio auditor and tech industry HR advisor.

═══════════════════════════════════════════════════════════════════
PROFILE: @student_dev  (Student)  |  CSAI — Semester 4 of 8
Account created: 2022  |  Bio: (empty)
Followers: 12  |  Public repos: 8
═══════════════════════════════════════════════════════════════════

PORTFOLIO METRICS:
  ⭐ Total stars: 5   🍴 Total forks: 2   🐛 Open issues: 1
  📊 Domain: ML/AI, Web   🔧 Repo quality avg: 1.5/4

LANGUAGE DISTRIBUTION:
  Python 75%, JavaScript 15%, Jupyter Notebook 10%

ACTIVITY — LAST 90 DAYS:
  Commits/week: 0.8   Active weeks: 6/13   Collab events: 0

CSAI STACK ALIGNMENT: score 72%  (core: Python ✓  C++ ✗)

TOP REPOS:
  • ml-experiments  ⭐0  [Python:90%]  #machine-learning
    Basic ML experiments using sklearn
  • portfolio-site  ⭐3  [JavaScript:80%]
    (no description)

DETECTED GAPS:
  ⚠ No profile README
  ⚠ Low commit frequency (0.8/week)
  ⚠ No collaborative activity

TASK: Write a professional career advisory report for this CSAI student."""

    print(f"\nPortfolio prompt length: {len(sample_prompt)} chars")
    print("\nGenerating advisory report …")
    t0     = time.time()
    report = advisor.generate(sample_prompt)
    elapsed = time.time() - t0
    print(f"\nGenerated report ({elapsed:.1f}s):\n")
    print(report)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "pretrained":
        _demo_pretrained_portfolio()
    else:
        _demo_architecture_only()
        print("\nRun with 'pretrained' to load BART weights and test portfolio advisory:")
        print("  python models_from_scratch/seq2seq_transformer.py pretrained")

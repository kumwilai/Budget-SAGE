"""Compositional UPC Sampling (CUPS) — a non-retrieving SLP generation primitive.

For each gloss g and a target span length T, CUPS samples a per-gloss UPC
sub-sequence

    u ~ q_phi(. | g, T)        autoregressive transformer over the UPC alphabet

and decodes it through the existing UPC-FM pose generator. No train clip
is ever copied; per-clip frame-source concentration (the share of frames
attributable to a single train source) is identically zero by construction.

The training target distribution is the empirical one. For each gloss g
and each train-clip occurrence of g (a span (sid, s, e) in the exemplar
bank), we extract the UPC sub-sequence the bank already stored for that
occurrence; the (g, u, T) triple becomes one MLE training example. No new
annotation is required.

Module contents:
  - PhoenixExemplarTriples: extracts (gloss_id, upc_seq, length) from
    `outputs/sota_chase/phase40_slrtp178/exemplars_slrtp178_upc.pt`.
  - CUPSSampler: small autoregressive transformer over the UPC alphabet
    conditioned on (gloss embedding, target length embedding). Shared
    across glosses, so the parameter count does not grow with vocab size.
  - sample(g, T): single-pass top-k autoregressive sampling.

The module is intentionally CPU-importable; only training and sampling
need a GPU.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# 1. Data extraction
# ---------------------------------------------------------------------------

@dataclass
class CUPSTriple:
    """One training example: a per-gloss UPC sub-sequence."""
    gloss: str
    upc: list[int]
    length: int


def extract_phoenix_triples(
    bank_path: Path,
    upc_source: str = "combined",
    min_length: int = 4,
    max_length: int = 64,
) -> list[CUPSTriple]:
    """Extract (gloss, UPC subseq, length) triples from a Phoenix exemplar bank.

    Args:
        bank_path: path to e.g. `outputs/sota_chase/phase40_slrtp178/exemplars_slrtp178_upc.pt`
        upc_source: 'lh' uses lh_codes only, 'rh' uses rh_codes, 'combined' interleaves frame-by-frame
        min_length / max_length: drop spans outside this range
    """
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    g2e: dict[str, list[tuple[str, int, int]]] = bank["gloss_to_exemplars"]
    eupc: dict[tuple, dict] = bank["exemplar_upc"]

    triples: list[CUPSTriple] = []
    for gloss, exemplars in g2e.items():
        for sid, s, e in exemplars:
            key = (sid, s, e)
            if key not in eupc:
                continue
            d = eupc[key]
            lh = d.get("lh_codes", [])
            rh = d.get("rh_codes", [])
            if hasattr(lh, "tolist"):
                lh = lh.tolist()
            if hasattr(rh, "tolist"):
                rh = rh.tolist()
            if upc_source == "lh":
                seq = list(lh)
            elif upc_source == "rh":
                seq = list(rh)
            else:  # combined
                # interleave per-frame; if lengths differ, pad to max with the last token
                T = max(len(lh), len(rh))
                lh += [lh[-1]] * (T - len(lh)) if lh else [0] * T
                rh += [rh[-1]] * (T - len(rh)) if rh else [0] * T
                seq = []
                for t in range(T):
                    seq.append(int(lh[t]))
                    seq.append(int(rh[t]))
            seq = [int(c) for c in seq if c is not None]
            if len(seq) < min_length or len(seq) > max_length:
                continue
            triples.append(CUPSTriple(gloss=gloss, upc=seq, length=len(seq)))
    return triples


def build_gloss_vocab(triples: Iterable[CUPSTriple]) -> dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for t in triples:
        if t.gloss not in vocab:
            vocab[t.gloss] = len(vocab)
    return vocab


# ---------------------------------------------------------------------------
# 2. Dataset / collation
# ---------------------------------------------------------------------------

# Special UPC tokens. The VQ alphabet is K codes; we shift codes by 3.
PAD, BOS, EOS = 0, 1, 2
UPC_OFFSET = 3


class CUPSDataset(Dataset):
    def __init__(self, triples: list[CUPSTriple], gloss_vocab: dict[str, int],
                 max_T: int = 64) -> None:
        self.triples = triples
        self.gloss_vocab = gloss_vocab
        self.max_T = max_T

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int):
        t = self.triples[idx]
        codes = [int(c) + UPC_OFFSET for c in t.upc[: self.max_T - 2]]
        seq = [BOS] + codes + [EOS]
        seq = seq + [PAD] * (self.max_T - len(seq))
        seq = torch.tensor(seq, dtype=torch.long)
        gid = self.gloss_vocab.get(t.gloss, self.gloss_vocab["<unk>"])
        T = min(t.length, self.max_T - 2)
        return {
            "gloss_id": torch.tensor(gid, dtype=torch.long),
            "length":   torch.tensor(T,  dtype=torch.long),
            "upc":      seq,
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ---------------------------------------------------------------------------
# 3. Model: CUPS sampler
# ---------------------------------------------------------------------------

def sinusoidal_pe(L: int, D: int) -> torch.Tensor:
    pe = torch.zeros(L, D)
    pos = torch.arange(0, L).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class CUPSSampler(nn.Module):
    """Small AR transformer over UPC tokens conditioned on (gloss, length).

    Vocabulary: K_upc + 3 specials (PAD, BOS, EOS). Effective vocab K_upc+3.
    """

    def __init__(
        self,
        n_gloss: int,
        n_upc_codes: int = 512,
        max_T: int = 64,
        max_length_token: int = 64,
        d_model: int = 256,
        n_heads: int = 4,
        ff: int = 1024,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cfg = dict(
            n_gloss=n_gloss, n_upc_codes=n_upc_codes,
            max_T=max_T, max_length_token=max_length_token,
            d_model=d_model, n_heads=n_heads, ff=ff,
            n_layers=n_layers, dropout=dropout,
        )
        self.vocab = n_upc_codes + 3
        self.tok_embed = nn.Embedding(self.vocab, d_model, padding_idx=PAD)
        self.gloss_embed = nn.Embedding(n_gloss, d_model, padding_idx=0)
        self.length_embed = nn.Embedding(max_length_token + 2, d_model)
        self.register_buffer("pe", sinusoidal_pe(max_T, d_model))
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        # Use a transformer encoder with causal mask (pure AR), conditioning is
        # injected by adding gloss/length embeddings to every position.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.vocab)
        self.max_T = max_T
        self.max_length_token = max_length_token
        self.n_upc_codes = n_upc_codes

    @staticmethod
    def _causal_mask(T: int, device) -> torch.Tensor:
        return torch.triu(torch.full((T, T), float("-inf"), device=device),
                           diagonal=1)

    def forward(self, gloss_id: torch.Tensor, length: torch.Tensor,
                upc: torch.Tensor) -> torch.Tensor:
        """Return logits [B, T, V] for next-token prediction (last position
        omitted to align with shifted targets)."""
        B, T = upc.shape
        x = self.tok_embed(upc) + self.pe[:T].unsqueeze(0)
        cond = (self.gloss_embed(gloss_id)
                + self.length_embed(torch.clamp(length, 0, self.max_length_token + 1)))
        x = x + cond.unsqueeze(1)
        m = self._causal_mask(T, x.device)
        kpm = upc == PAD
        h = self.transformer(x, mask=m, src_key_padding_mask=kpm)
        h = self.norm(h)
        return self.head(h)

    @torch.no_grad()
    def sample(self, gloss_id: int, length: int,
               temperature: float = 1.0, top_k: int = 0,
               top_p: float = 0.0,
               device: Optional[str] = None) -> list[int]:
        """Autoregressive sampling. Returns a list of UPC codes (offset removed)."""
        device = device or next(self.parameters()).device
        gid = torch.tensor([gloss_id], dtype=torch.long, device=device)
        lt = torch.tensor([min(length, self.max_length_token + 1)],
                           dtype=torch.long, device=device)
        T_max = min(length + 2, self.max_T)
        seq = torch.full((1, T_max), PAD, dtype=torch.long, device=device)
        seq[0, 0] = BOS
        out_codes: list[int] = []
        for t in range(1, T_max):
            logits = self.forward(gid, lt, seq[:, :t])[:, -1, :]
            logits = logits / max(1e-6, temperature)
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p > 0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                mask = cum > top_p
                mask[:, 0] = False
                sorted_logits[mask] = float("-inf")
                logits = torch.zeros_like(logits).scatter(
                    -1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            tok = int(torch.multinomial(probs, num_samples=1).item())
            seq[0, t] = tok
            if tok == EOS or tok == PAD:
                break
            if tok >= UPC_OFFSET and (tok - UPC_OFFSET) < self.n_upc_codes:
                out_codes.append(tok - UPC_OFFSET)
                if len(out_codes) >= length:
                    break
        return out_codes


# ---------------------------------------------------------------------------
# 4. Training loss + step
# ---------------------------------------------------------------------------

def cups_loss(model: CUPSSampler, batch: dict) -> torch.Tensor:
    """Standard next-token cross-entropy (teacher forcing), ignoring PAD."""
    upc = batch["upc"]
    inp = upc[:, :-1]
    tgt = upc[:, 1:]
    logits = model(batch["gloss_id"], batch["length"], inp)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt.reshape(-1),
        ignore_index=PAD,
    )
    return loss


__all__ = [
    "CUPSTriple",
    "extract_phoenix_triples",
    "build_gloss_vocab",
    "CUPSDataset",
    "collate",
    "CUPSSampler",
    "cups_loss",
    "PAD", "BOS", "EOS", "UPC_OFFSET",
]

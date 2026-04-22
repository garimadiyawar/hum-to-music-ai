"""
models/harmony_generator.py
────────────────────────────
Transformer-based harmony generator.

Given a melody token sequence it predicts a chord progression token sequence
(one chord per bar / sub-phrase).

Vocabulary
──────────
  Melody tokens  : 0-127 MIDI pitch + 128 SOS + 129 EOS + 130 PAD
  Chord tokens   : 0 … chord_vocab_size()-1   (root × type)   + NO_CHORD

Architecture
────────────
Melody embedding → Transformer Encoder → per-position Linear → chord logits

Because chord prediction is a classification per melody segment (not a
seq-to-seq task), the decoder is simply a linear head applied to each
encoder output position.  For full chord sequence generation, an
autoregressive decoder variant is also provided.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.audio_encoder import PositionalEncoding
from utils.music_theory import (
    chord_vocab_size,
    NO_CHORD_TOKEN,
    token_to_chord,
    chord_name,
    PITCH_CLASSES,
    SOS_TOKEN,
    EOS_TOKEN,
)

CHORD_VOCAB  = chord_vocab_size()     # 217
MEL_VOCAB    = 131                    # 0-127 MIDI + SOS + EOS + PAD
MEL_PAD      = 130


# ─── Harmony Generator ───────────────────────────────────────────────────────

class HarmonyGenerator(nn.Module):
    """
    Encoder-only chord predictor.

    For each position in the melody token sequence the model outputs
    a probability distribution over all chord tokens.

    Parameters
    ----------
    mel_vocab_size  : melody vocabulary size (131)
    chord_vocab     : chord vocabulary size  (217)
    d_model         : hidden dimension
    nhead           : attention heads
    num_layers      : encoder depth
    dim_feedforward : FFN dimension
    dropout         : dropout
    max_seq_len     : max melody length
    """

    def __init__(
        self,
        mel_vocab_size:  int = MEL_VOCAB,
        chord_vocab:     int = CHORD_VOCAB,
        d_model:         int = 256,
        nhead:           int = 4,
        num_layers:      int = 4,
        dim_feedforward: int = 1024,
        dropout:         float = 0.1,
        max_seq_len:     int = 256,
    ):
        super().__init__()

        self.d_model     = d_model
        self.chord_vocab = chord_vocab

        # ── Melody embedding ─────────────────────────────────────────────────
        self.mel_embedding = nn.Embedding(mel_vocab_size, d_model, padding_idx=MEL_PAD)
        self.pos_enc       = PositionalEncoding(d_model=d_model, dropout=dropout,
                                                max_len=max_seq_len + 2)

        # ── Transformer encoder ───────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── Chord classification head ─────────────────────────────────────────
        # Applied to every position → predict chord at that position
        self.chord_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, chord_vocab),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mel_embedding.weight, mean=0, std=self.d_model ** -0.5)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        melody:   torch.Tensor,                     # (B, N) melody tokens
        src_mask: Optional[torch.Tensor] = None,    # (B, N) True = ignore
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        melody   : (B, N) long
        src_mask : (B, N) bool, True at padding positions

        Returns
        -------
        logits : (B, N, chord_vocab)
        """
        emb = self.mel_embedding(melody) * math.sqrt(self.d_model)  # (B, N, d)
        emb = self.pos_enc(emb)

        enc = self.encoder(emb, src_key_padding_mask=src_mask)       # (B, N, d)
        logits = self.chord_head(enc)                                 # (B, N, chord_vocab)
        return logits

    # ── Inference helpers ────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_chords(
        self,
        melody: torch.Tensor,                    # (B, N) or (N,)
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict chord tokens for each melody position.

        Returns
        -------
        chord_tokens : (B, N) long  – best chord per position
        chord_probs  : (B, N, chord_vocab) float – probability distributions
        """
        self.eval()
        if melody.dim() == 1:
            melody = melody.unsqueeze(0)         # → (1, N)
        src_mask = (melody == MEL_PAD)

        logits = self.forward(melody, src_mask)  # (B, N, chord_vocab)

        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)

        if top_k > 0:
            # Zero out all but top-k
            thresh = logits.topk(top_k, dim=-1).values[:, :, -1:]
            logits = logits.masked_fill(logits < thresh, float("-inf"))

        probs  = F.softmax(logits, dim=-1)       # (B, N, chord_vocab)
        tokens = probs.argmax(dim=-1)            # (B, N)
        return tokens, probs

    @torch.no_grad()
    def melody_to_chord_sequence(
        self,
        melody_tokens: List[int],
        merge_consecutive: bool = True,
        temperature: float = 1.0,
    ) -> List[dict]:
        """
        High-level API: MIDI token list → list of chord dicts.

        Each chord dict:
            {"token": int, "root_pc": int|None, "chord_type": str|None,
             "name": str, "position": int}

        Parameters
        ----------
        merge_consecutive : collapse adjacent identical chords into one
        """
        device = next(self.parameters()).device
        mel_t = torch.tensor(melody_tokens, dtype=torch.long, device=device).unsqueeze(0)

        chord_tokens, _ = self.predict_chords(mel_t, temperature=temperature)
        chord_tokens = chord_tokens.squeeze(0).tolist()   # (N,)

        # Decode
        result = []
        for pos, tok in enumerate(chord_tokens):
            root_pc, ctype = token_to_chord(tok)
            name = chord_name(root_pc, ctype) if root_pc is not None else "N.C."
            result.append(
                {"token": tok, "root_pc": root_pc, "chord_type": ctype,
                 "name": name, "position": pos}
            )

        if merge_consecutive:
            merged = []
            for chord in result:
                if merged and merged[-1]["token"] == chord["token"]:
                    continue
                merged.append(chord)
            return merged

        return result


# ─── Autoregressive chord sequence generator (bonus) ─────────────────────────

class AutoregressiveChordGenerator(nn.Module):
    """
    Encoder-decoder variant: generates a full chord token sequence
    autoregressively given the full melody encoding.

    Useful for longer-range harmonic coherence.
    """

    def __init__(
        self,
        mel_vocab_size:  int = MEL_VOCAB,
        chord_vocab:     int = CHORD_VOCAB,
        d_model:         int = 256,
        nhead:           int = 4,
        num_enc_layers:  int = 4,
        num_dec_layers:  int = 4,
        dim_feedforward: int = 1024,
        dropout:         float = 0.1,
        max_seq_len:     int = 256,
    ):
        super().__init__()
        self.d_model     = d_model
        self.chord_vocab = chord_vocab

        # Shared melody encoder (same as HarmonyGenerator)
        self.mel_embedding = nn.Embedding(mel_vocab_size, d_model, padding_idx=MEL_PAD)
        self.mel_pos_enc   = PositionalEncoding(d_model, dropout, max_seq_len + 2)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                               dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_enc_layers, nn.LayerNorm(d_model))

        # Chord decoder
        self.chord_embedding = nn.Embedding(chord_vocab + 2, d_model)  # +2 for SOS/EOS
        self.chord_pos_enc   = PositionalEncoding(d_model, dropout, max_seq_len + 2)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                               dropout=dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_dec_layers, nn.LayerNorm(d_model))
        self.proj = nn.Linear(d_model, chord_vocab)

        self.CHORD_SOS = chord_vocab
        self.CHORD_EOS = chord_vocab + 1

    def _causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(
        self,
        melody:     torch.Tensor,   # (B, N)
        tgt_chords: torch.Tensor,   # (B, M)
        mel_mask:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:              # → (B, M-1, chord_vocab)
        B, N = melody.shape
        device = melody.device

        mel_emb = self.mel_embedding(melody) * math.sqrt(self.d_model)
        mel_emb = self.mel_pos_enc(mel_emb)
        mem     = self.encoder(mel_emb, src_key_padding_mask=mel_mask)

        tgt_in  = tgt_chords[:, :-1]
        tgt_emb = self.chord_embedding(tgt_in) * math.sqrt(self.d_model)
        tgt_emb = self.chord_pos_enc(tgt_emb)
        M1 = tgt_in.size(1)
        causal = self._causal_mask(M1, device)
        out = self.decoder(tgt_emb, mem, tgt_mask=causal, memory_key_padding_mask=mel_mask)
        return self.proj(out)       # (B, M-1, chord_vocab)

    @torch.no_grad()
    def generate(
        self,
        melody: torch.Tensor,       # (1, N)
        max_len: int = 64,
        temperature: float = 0.9,
    ) -> List[int]:
        self.eval()
        device = melody.device
        mel_emb = self.mel_embedding(melody) * math.sqrt(self.d_model)
        mel_emb = self.mel_pos_enc(mel_emb)
        mem = self.encoder(mel_emb)

        tgt = torch.tensor([[self.CHORD_SOS]], dtype=torch.long, device=device)
        chords = []

        for _ in range(max_len):
            tgt_emb = self.chord_embedding(tgt) * math.sqrt(self.d_model)
            tgt_emb = self.chord_pos_enc(tgt_emb)
            causal = self._causal_mask(tgt.size(1), device)
            out = self.decoder(tgt_emb, mem.expand(1, -1, -1), tgt_mask=causal)
            logits = self.proj(out[:, -1, :]) / max(temperature, 1e-6)
            probs  = F.softmax(logits, dim=-1)
            tok    = torch.multinomial(probs, num_samples=1).squeeze()
            if tok.item() == self.CHORD_EOS:
                break
            chords.append(tok.item())
            tgt = torch.cat([tgt, tok.view(1, 1)], dim=1)

        return chords


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, N = 4, 32

    model = HarmonyGenerator(d_model=128, nhead=4, num_layers=2, dim_feedforward=256)
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward
    melody = torch.randint(0, 128, (B, N))
    melody[:, 0] = SOS_TOKEN
    logits = model(melody)
    print(f"\nForward: melody {melody.shape} → chord logits {logits.shape}")

    # Loss
    targets = torch.randint(0, CHORD_VOCAB, (B, N))
    loss = F.cross_entropy(logits.reshape(-1, CHORD_VOCAB), targets.reshape(-1))
    print(f"CE loss: {loss.item():.4f}")

    # Predict
    toks, probs = model.predict_chords(melody[:1], temperature=1.0)
    print(f"Predicted chord tokens (first 8): {toks[0, :8].tolist()}")

    # High-level API
    mel_sequence = [SOS_TOKEN] + [60, 62, 64, 65, 67] + [EOS_TOKEN]
    chord_dicts = model.melody_to_chord_sequence(mel_sequence)
    print(f"\nChord sequence for C-D-E-F-G melody:")
    for c in chord_dicts[:5]:
        print(f"  pos={c['position']}  {c['name']}")

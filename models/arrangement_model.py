"""
models/arrangement_model.py
────────────────────────────
Multi-instrument arrangement generator.

Given:
  • melody token sequence
  • chord token sequence

Generates a flat MIDI-like event token sequence that can be decoded into
separate instrument tracks (bass, drums, piano, strings, synth pad).

Architecture
────────────
Melody + Chord → Joint Encoder → Transformer Decoder → Event token sequence

The event token sequence uses a compound-token vocabulary:
  • NOTE_ON   tokens  (0–127)
  • NOTE_OFF  tokens  (128–255)
  • TIME_SHIFT tokens (256–511)  = 256 bins × 10 ms
  • VELOCITY  tokens  (512–543)  = 32 velocity bins
  • TRACK     tokens  (544–548)  = 5 instrument roles
  • SOS  = 549
  • EOS  = 550
  • PAD  = 551
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.audio_encoder import PositionalEncoding

# ─── Token vocabulary ────────────────────────────────────────────────────────

NOTE_ON_OFFSET  = 0
NOTE_OFF_OFFSET = 128
TIME_OFFSET     = 256        # 256 bins of 10 ms
VEL_OFFSET      = 512        # 32 velocity bins
TRACK_OFFSET    = 544        # 5 instrument track tokens
ARR_SOS = 549
ARR_EOS = 550
ARR_PAD = 551
ARR_VOCAB_SIZE = 552

TRACK_NAMES = ["melody", "bass", "piano", "strings", "pad", "drums"]
TRACK_TOKENS = {name: TRACK_OFFSET + i for i, name in enumerate(TRACK_NAMES)}


MEL_VOCAB   = 131
CHORD_VOCAB = 217            # from music_theory
SRC_PAD     = 130            # melody PAD token


# ─── Arrangement Model ───────────────────────────────────────────────────────

class ArrangementModel(nn.Module):
    """
    Encoder–decoder model for multi-track MIDI arrangement generation.

    Parameters
    ----------
    mel_vocab    : melody vocabulary (131)
    chord_vocab  : chord vocabulary (217)
    arr_vocab    : arrangement event vocabulary (552)
    d_model      : transformer hidden dimension
    nhead        : attention heads
    num_enc_layers, num_dec_layers : encoder / decoder depth
    dim_feedforward: FFN inner dimension
    dropout      : dropout
    max_src_len  : max melody + chord source length
    max_tgt_len  : max arrangement token length
    """

    def __init__(
        self,
        mel_vocab:       int = MEL_VOCAB,
        chord_vocab:     int = CHORD_VOCAB,
        arr_vocab:       int = ARR_VOCAB_SIZE,
        d_model:         int = 512,
        nhead:           int = 8,
        num_enc_layers:  int = 6,
        num_dec_layers:  int = 6,
        dim_feedforward: int = 2048,
        dropout:         float = 0.1,
        max_src_len:     int = 256,
        max_tgt_len:     int = 1024,
    ):
        super().__init__()
        self.d_model   = d_model
        self.arr_vocab = arr_vocab

        # ── Source embeddings ────────────────────────────────────────────────
        self.mel_emb   = nn.Embedding(mel_vocab,   d_model, padding_idx=SRC_PAD)
        self.chord_emb = nn.Embedding(chord_vocab + 1, d_model)   # +1 for no-chord
        self.src_pos   = PositionalEncoding(d_model, dropout, max_src_len * 2 + 4)

        # Segment embedding: 0 = melody, 1 = chords
        self.seg_emb = nn.Embedding(2, d_model)

        # ── Encoder ──────────────────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_enc_layers, nn.LayerNorm(d_model))

        # ── Target embedding ─────────────────────────────────────────────────
        self.arr_emb = nn.Embedding(arr_vocab, d_model, padding_idx=ARR_PAD)
        self.tgt_pos = PositionalEncoding(d_model, dropout, max_tgt_len + 4)

        # ── Decoder ──────────────────────────────────────────────────────────
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_dec_layers, nn.LayerNorm(d_model))

        # ── Output head ──────────────────────────────────────────────────────
        self.output_proj = nn.Linear(d_model, arr_vocab)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for emb in (self.mel_emb, self.chord_emb, self.arr_emb):
            nn.init.normal_(emb.weight, std=self.d_model ** -0.5)

    # ── Joint source encoding ────────────────────────────────────────────────

    def encode(
        self,
        melody: torch.Tensor,           # (B, N_mel)
        chords: torch.Tensor,           # (B, N_chord)
        mel_pad_mask: Optional[torch.Tensor] = None,    # (B, N_mel)
        chord_pad_mask: Optional[torch.Tensor] = None,  # (B, N_chord)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        mem       : (B, N_mel + N_chord, d_model)
        mem_mask  : (B, N_mel + N_chord)  bool – True = ignore
        """
        B = melody.size(0)
        device = melody.device

        m_emb = self.mel_emb(melody)   * math.sqrt(self.d_model)   # (B, N_mel, d)
        c_emb = self.chord_emb(chords) * math.sqrt(self.d_model)   # (B, N_crd, d)

        # Segment embeddings
        seg0 = self.seg_emb(torch.zeros(B, melody.size(1), dtype=torch.long, device=device))
        seg1 = self.seg_emb(torch.ones(B,  chords.size(1), dtype=torch.long, device=device))
        m_emb = m_emb + seg0
        c_emb = c_emb + seg1

        # Concatenate along sequence dimension
        src = torch.cat([m_emb, c_emb], dim=1)   # (B, N_mel+N_crd, d)
        src = self.src_pos(src)

        # Build combined padding mask
        if mel_pad_mask is not None and chord_pad_mask is not None:
            mem_mask = torch.cat([mel_pad_mask, chord_pad_mask], dim=1)
        else:
            mem_mask = None

        mem = self.encoder(src, src_key_padding_mask=mem_mask)
        return mem, mem_mask

    # ── Forward (training, teacher-forced) ───────────────────────────────────

    def forward(
        self,
        melody:   torch.Tensor,         # (B, N_mel)
        chords:   torch.Tensor,         # (B, N_crd)
        tgt:      torch.Tensor,         # (B, T_arr)
        mel_pad_mask:   Optional[torch.Tensor] = None,
        chord_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:                  # → logits (B, T_arr-1, arr_vocab)
        device = melody.device

        mem, mem_mask = self.encode(melody, chords, mel_pad_mask, chord_pad_mask)

        tgt_in = tgt[:, :-1]            # teacher-forced input
        t_emb  = self.arr_emb(tgt_in) * math.sqrt(self.d_model)
        t_emb  = self.tgt_pos(t_emb)

        T = tgt_in.size(1)
        causal = self._causal_mask(T, device)
        tgt_pad = (tgt_in == ARR_PAD)

        out    = self.decoder(t_emb, mem, tgt_mask=causal,
                              tgt_key_padding_mask=tgt_pad,
                              memory_key_padding_mask=mem_mask)
        logits = self.output_proj(out)  # (B, T-1, arr_vocab)
        return logits

    def _causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    # ── Greedy / sampling generation ─────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        melody:      torch.Tensor,      # (1, N_mel)
        chords:      torch.Tensor,      # (1, N_crd)
        max_len:     int = 1024,
        temperature: float = 0.9,
        top_k:       int = 50,
        top_p:       float = 0.95,
    ) -> List[int]:
        """
        Autoregressive generation with temperature + nucleus (top-p) sampling.

        Returns
        -------
        list of arrangement event tokens
        """
        self.eval()
        device = melody.device

        mem, mem_mask = self.encode(melody, chords)

        tgt = torch.tensor([[ARR_SOS]], dtype=torch.long, device=device)
        generated = []

        for step in range(max_len):
            t_emb = self.arr_emb(tgt) * math.sqrt(self.d_model)
            t_emb = self.tgt_pos(t_emb)
            causal = self._causal_mask(tgt.size(1), device)
            out = self.decoder(t_emb, mem.expand(1, -1, -1),
                               tgt_mask=causal, memory_key_padding_mask=mem_mask)
            logits = self.output_proj(out[:, -1, :]) / max(temperature, 1e-6)  # (1, vocab)

            # Top-k filter
            if top_k > 0:
                thresh = torch.topk(logits, top_k).values[:, -1:]
                logits = logits.masked_fill(logits < thresh, float("-inf"))

            # Top-p nucleus filter
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits[remove] = float("-inf")
                logits.scatter_(-1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            tok   = torch.multinomial(probs, num_samples=1).item()

            if tok == ARR_EOS:
                break
            if tok != ARR_PAD:
                generated.append(tok)
            tgt = torch.cat([tgt, torch.tensor([[tok]], device=device)], dim=1)

        return generated


# ─── Arrangement token decoder ───────────────────────────────────────────────

def decode_arrangement_tokens(
    tokens: List[int],
    default_tempo: float = 120.0,
    bin_size_sec: float = 0.01,
) -> Dict[str, List[dict]]:
    """
    Convert a flat arrangement token sequence back to per-track note events.

    Returns
    -------
    dict mapping track_name → list of note_event dicts
        {"pitch_midi", "start", "end", "velocity"}
    """
    tracks: Dict[str, List[dict]] = {name: [] for name in TRACK_NAMES}
    current_time  = 0.0
    current_vel   = 64
    current_track = TRACK_NAMES[0]
    open_notes: Dict[Tuple[str, int], float] = {}  # (track, pitch) → onset time

    for tok in tokens:
        if ARR_SOS <= tok <= ARR_EOS or tok == ARR_PAD:
            continue

        if 0 <= tok < 128:
            # NOTE_ON
            pitch = tok
            open_notes[(current_track, pitch)] = current_time
            tracks[current_track].append(
                {"pitch_midi": pitch, "start": current_time,
                 "end": current_time + 0.5,    # placeholder; updated on NOTE_OFF
                 "velocity": current_vel}
            )

        elif 128 <= tok < 256:
            # NOTE_OFF
            pitch = tok - 128
            key   = (current_track, pitch)
            if key in open_notes:
                onset = open_notes.pop(key)
                # Update the matching open note's end time
                for ev in reversed(tracks[current_track]):
                    if ev["pitch_midi"] == pitch and ev["start"] == onset:
                        ev["end"] = current_time
                        break

        elif 256 <= tok < 512:
            # TIME_SHIFT
            bins = tok - 256
            current_time += bins * bin_size_sec

        elif 512 <= tok < 544:
            # VELOCITY
            vel_bin = tok - 512
            current_vel = int((vel_bin + 0.5) * 128 / 32)

        elif 544 <= tok < 549:
            # TRACK CHANGE
            current_track = TRACK_NAMES[tok - TRACK_OFFSET]

    return tracks


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, N_mel, N_crd, T_arr = 2, 32, 16, 64

    model = ArrangementModel(
        d_model=256, nhead=4,
        num_enc_layers=2, num_dec_layers=2,
        dim_feedforward=512,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    melody = torch.randint(0, 128, (B, N_mel))
    chords = torch.randint(0, 216, (B, N_crd))
    tgt    = torch.randint(0, ARR_VOCAB_SIZE, (B, T_arr))
    tgt[:, 0] = ARR_SOS

    logits = model(melody, chords, tgt)
    print(f"Logits: {logits.shape}")

    loss = F.cross_entropy(
        logits.reshape(-1, ARR_VOCAB_SIZE),
        tgt[:, 1:].reshape(-1),
        ignore_index=ARR_PAD,
    )
    print(f"Loss: {loss.item():.4f}")

    # Generation
    gen_tokens = model.generate(melody[:1], chords[:1], max_len=64)
    print(f"Generated {len(gen_tokens)} tokens")

    # Decode
    track_events = decode_arrangement_tokens(gen_tokens)
    for tname, evs in track_events.items():
        print(f"  {tname}: {len(evs)} notes")

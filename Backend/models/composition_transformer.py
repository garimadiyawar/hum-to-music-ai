"""
models/composition_transformer.py
───────────────────────────────────
Composition engine that:
  1. Extends a short hummed melody into a full song structure.
  2. Applies motivic development, phrase repetition, and structural sections
     (intro / verse / chorus / bridge / outro).

Architecture
────────────
Melody Encoder (Transformer) → Structure-conditioned Decoder
                             → extended melody token sequence

The section conditioning uses learned section embeddings added to the
decoder's positional encoding, so the model learns different "styles"
for each structural section.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.audio_encoder import PositionalEncoding

# ─── Song structure constants ────────────────────────────────────────────────

SECTION_NAMES = ["intro", "verse", "chorus", "bridge", "outro", "fill"]
SECTION_TO_IDX: Dict[str, int] = {s: i for i, s in enumerate(SECTION_NAMES)}
NUM_SECTIONS = len(SECTION_NAMES)

MEL_VOCAB = 131
MEL_PAD   = 130
MEL_SOS   = 128
MEL_EOS   = 129


# ─── Default song templates ──────────────────────────────────────────────────

DEFAULT_SONG_STRUCTURES: Dict[str, List[str]] = {
    "pop":      ["intro", "verse", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"],
    "ballad":   ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"],
    "minimal":  ["intro", "verse", "chorus", "chorus", "outro"],
    "jazz":     ["intro", "verse", "verse", "chorus", "verse", "outro"],
}

DEFAULT_SECTION_BARS: Dict[str, int] = {
    "intro":  4,
    "verse":  8,
    "chorus": 8,
    "bridge": 4,
    "outro":  4,
    "fill":   2,
}


# ─── Composition config ───────────────────────────────────────────────────────

@dataclass
class CompositionConfig:
    structure_template: str = "pop"           # key into DEFAULT_SONG_STRUCTURES
    beats_per_bar:       int = 4
    default_tempo:       float = 120.0
    motif_repetitions:   int = 2             # how many times to repeat motif in verse
    transpose_chorus:    int = 5             # semitones up for chorus (0 = same)
    vary_intensity:      bool = True         # velocity modulation per section
    section_bars:        Dict[str, int] = field(default_factory=lambda: DEFAULT_SECTION_BARS.copy())


# ─── Composition Transformer ─────────────────────────────────────────────────

class CompositionTransformer(nn.Module):
    """
    Extends a seed melody into a full song structure.

    Parameters
    ----------
    d_model          : hidden dimension
    nhead            : attention heads
    num_enc_layers   : encoder depth
    num_dec_layers   : decoder depth
    dim_feedforward  : FFN inner dim
    dropout          : dropout
    max_src_len      : max seed melody length
    max_tgt_len      : max extended melody length
    """

    def __init__(
        self,
        d_model:         int = 256,
        nhead:           int = 4,
        num_enc_layers:  int = 4,
        num_dec_layers:  int = 4,
        dim_feedforward: int = 1024,
        dropout:         float = 0.1,
        max_src_len:     int = 64,
        max_tgt_len:     int = 512,
    ):
        super().__init__()
        self.d_model   = d_model
        self.max_tgt_len = max_tgt_len

        # ── Source (seed melody) embedding ────────────────────────────────────
        self.mel_emb = nn.Embedding(MEL_VOCAB, d_model, padding_idx=MEL_PAD)
        self.src_pos = PositionalEncoding(d_model, dropout, max_src_len + 4)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_enc_layers, nn.LayerNorm(d_model))

        # ── Target (extended melody) embedding ────────────────────────────────
        self.tgt_emb = nn.Embedding(MEL_VOCAB, d_model, padding_idx=MEL_PAD)
        self.tgt_pos = PositionalEncoding(d_model, dropout, max_tgt_len + 4)

        # ── Section conditioning embedding ────────────────────────────────────
        self.section_emb = nn.Embedding(NUM_SECTIONS, d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_dec_layers, nn.LayerNorm(d_model))

        # ── Output head ───────────────────────────────────────────────────────
        self.proj = nn.Linear(d_model, MEL_VOCAB)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for emb in (self.mel_emb, self.tgt_emb, self.section_emb):
            nn.init.normal_(emb.weight, std=self.d_model ** -0.5)

    def _causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(
        self,
        src:          torch.Tensor,          # (B, N_src)
        tgt:          torch.Tensor,          # (B, T_tgt)
        section_ids:  torch.Tensor,          # (B, T_tgt) – section index per position
        src_pad_mask: Optional[torch.Tensor] = None,   # (B, N_src)
    ) -> torch.Tensor:                       # → logits (B, T_tgt-1, MEL_VOCAB)
        device = src.device

        # Encode seed melody
        s_emb = self.mel_emb(src) * math.sqrt(self.d_model)
        s_emb = self.src_pos(s_emb)
        mem   = self.encoder(s_emb, src_key_padding_mask=src_pad_mask)

        # Decode extended melody (teacher-forced)
        tgt_in  = tgt[:, :-1]
        sec_in  = section_ids[:, :-1]

        t_emb   = self.tgt_emb(tgt_in) * math.sqrt(self.d_model)
        t_emb   = self.tgt_pos(t_emb)
        t_emb   = t_emb + self.section_emb(sec_in)         # add section conditioning

        T = tgt_in.size(1)
        causal = self._causal_mask(T, device)
        tgt_pad = (tgt_in == MEL_PAD)

        out    = self.decoder(t_emb, mem, tgt_mask=causal,
                              tgt_key_padding_mask=tgt_pad,
                              memory_key_padding_mask=src_pad_mask)
        logits = self.proj(out)             # (B, T-1, MEL_VOCAB)
        return logits

    @torch.no_grad()
    def generate(
        self,
        seed_tokens:  List[int],
        section_plan: List[Tuple[str, int]],    # [(section_name, length_tokens), …]
        temperature:  float = 0.9,
        top_k:        int = 40,
    ) -> Tuple[List[int], List[str]]:
        """
        Generate an extended melody following section_plan.

        Parameters
        ----------
        seed_tokens  : MIDI pitch tokens (seed melody)
        section_plan : ordered list of (section_name, n_tokens) pairs
        temperature  : sampling temperature
        top_k        : top-k token filter

        Returns
        -------
        (generated_tokens, section_labels)
            generated_tokens : list of MIDI pitch tokens (full song)
            section_labels   : section name for each generated token position
        """
        self.eval()
        device = next(self.parameters()).device

        src = torch.tensor([MEL_SOS] + seed_tokens + [MEL_EOS],
                           dtype=torch.long, device=device).unsqueeze(0)  # (1, N)
        src_pad = (src == MEL_PAD)
        s_emb = self.mel_emb(src) * math.sqrt(self.d_model)
        s_emb = self.src_pos(s_emb)
        mem   = self.encoder(s_emb, src_key_padding_mask=src_pad)

        tgt = torch.tensor([[MEL_SOS]], dtype=torch.long, device=device)
        generated_tokens: List[int] = []
        section_labels:   List[str] = []

        for section_name, n_tokens in section_plan:
            sec_idx = SECTION_TO_IDX.get(section_name, 0)

            for _ in range(n_tokens):
                # Build section id sequence for current tgt
                sec_ids = torch.tensor(
                    [sec_idx] * tgt.size(1),
                    dtype=torch.long, device=device,
                ).unsqueeze(0)

                t_emb = self.tgt_emb(tgt) * math.sqrt(self.d_model)
                t_emb = self.tgt_pos(t_emb)
                t_emb = t_emb + self.section_emb(sec_ids)

                causal = self._causal_mask(tgt.size(1), device)
                out    = self.decoder(t_emb, mem.expand(1, -1, -1), tgt_mask=causal)
                logits = self.proj(out[:, -1, :]) / max(temperature, 1e-6)

                if top_k > 0:
                    thresh = logits.topk(top_k).values[:, -1:]
                    logits = logits.masked_fill(logits < thresh, float("-inf"))

                # Prevent SOS / PAD
                logits[:, MEL_SOS] = float("-inf")
                logits[:, MEL_PAD] = float("-inf")

                probs  = F.softmax(logits, dim=-1)
                tok    = torch.multinomial(probs, 1).item()

                if tok == MEL_EOS:
                    # Don't stop early; use previous note instead
                    tok = generated_tokens[-1] if generated_tokens else 60

                generated_tokens.append(tok)
                section_labels.append(section_name)
                tgt = torch.cat([tgt, torch.tensor([[tok]], device=device)], dim=1)

        return generated_tokens, section_labels


# ─── Rule-based composition engine (no model required) ───────────────────────

class RuleBasedComposer:
    """
    Extends a short melody without a trained model using:
      • Motif repetition
      • Transposition for chorus (by a 4th / 5th)
      • Mirror / retrograde for variation
      • Random pitch variation within key

    Useful as a fallback when model checkpoints are not available.
    """

    def __init__(self, config: Optional[CompositionConfig] = None):
        self.cfg = config or CompositionConfig()

    def compose(
        self,
        seed_notes: List[dict],             # note event dicts (pitch_midi, start, end, …)
        key_root_pc: int = 0,
        key_scale: str = "major",
    ) -> Tuple[List[dict], List[str]]:
        """
        Extend seed notes into a full song structure.

        Returns
        -------
        (all_notes, section_labels)
            all_notes      : list of note event dicts covering the full arrangement
            section_labels : section label for each note
        """
        from utils.music_theory import get_scale, SCALE_INTERVALS

        structure = DEFAULT_SONG_STRUCTURES[self.cfg.structure_template]
        bars = self.cfg.section_bars
        beats_per_bar = self.cfg.beats_per_bar
        sec_per_beat = 60.0 / self.cfg.default_tempo

        # Build scale for key-constrained variation
        scale_pcs = set(SCALE_INTERVALS.get(key_scale, SCALE_INTERVALS["major"]))

        # Helper: shift all notes by `dt` seconds
        def shift_time(notes, dt):
            return [
                {**n, "start": n["start"] + dt, "end": n["end"] + dt}
                for n in notes
            ]

        # Helper: transpose notes by `semitones`
        def transpose(notes, semitones):
            return [
                {**n, "pitch_midi": max(21, min(108, n["pitch_midi"] + semitones))}
                for n in notes
            ]

        # Helper: vary melody slightly (chromatic walk within scale)
        def vary(notes, amount=1):
            varied = []
            for n in notes:
                shift = random.choice([-amount, 0, amount])
                new_pitch = n["pitch_midi"] + shift
                # Snap to scale
                pc = new_pitch % 12
                if pc not in scale_pcs:
                    new_pitch = n["pitch_midi"]     # revert if out of scale
                varied.append({**n, "pitch_midi": max(21, min(108, new_pitch))})
            return varied

        # Helper: retro (reverse) motif
        def retrograde(notes):
            if not notes:
                return notes
            total_dur = notes[-1]["end"] - notes[0]["start"]
            reversed_notes = []
            for n in reversed(notes):
                dur = n["end"] - n["start"]
                new_start = total_dur - (n["end"] - notes[0]["start"])
                reversed_notes.append({**n, "start": new_start, "end": new_start + dur})
            return reversed_notes

        # Normalise seed to start at 0
        if seed_notes:
            t0 = seed_notes[0]["start"]
            seed = shift_time(seed_notes, -t0)
            seed_dur = seed[-1]["end"] if seed else 4.0 * sec_per_beat
        else:
            seed = []
            seed_dur = 4.0 * sec_per_beat

        # Section velocity profiles
        section_velocity: Dict[str, float] = {
            "intro":  0.6,
            "verse":  0.75,
            "chorus": 1.0,
            "bridge": 0.8,
            "outro":  0.55,
            "fill":   0.65,
        }

        all_notes: List[dict] = []
        all_labels: List[str] = []
        cursor = 0.0   # current time position

        for section in structure:
            n_bars  = bars.get(section, 4)
            sec_dur = n_bars * beats_per_bar * sec_per_beat
            vel_scale = section_velocity.get(section, 0.8) if self.cfg.vary_intensity else 1.0

            # Generate section content
            if section == "intro":
                # Simplified seed (first half)
                half = [n for n in seed if n["start"] < seed_dur / 2]
                content = half or seed

            elif section == "verse":
                # Repeat motif with slight variation
                content = []
                t = 0.0
                for rep in range(self.cfg.motif_repetitions):
                    m = vary(seed, amount=1) if rep > 0 else seed
                    for n in m:
                        if n["start"] < seed_dur:
                            content.append({**n, "start": n["start"] + t,
                                            "end": n["end"] + t})
                    t += seed_dur

            elif section == "chorus":
                # Transpose up
                semitones = self.cfg.transpose_chorus
                content = transpose(seed, semitones)
                # Extend to fill section
                content_dur = content[-1]["end"] if content else seed_dur
                if content_dur < sec_dur:
                    extra = []
                    t = content_dur
                    while t < sec_dur:
                        extra.extend(shift_time(vary(content, 2), t - 0.0))
                        t += content_dur
                    content = content + extra

            elif section == "bridge":
                content = retrograde(vary(seed, 2))

            elif section == "outro":
                # Quieter descending motif
                content = transpose(seed, -12)

            else:
                content = seed

            # Scale velocity, shift time, add
            for n in content:
                if n["start"] < sec_dur:
                    new_vel = max(10, min(127, int(n.get("velocity", 80) * vel_scale)))
                    all_notes.append(
                        {**n,
                         "start": cursor + n["start"],
                         "end":   cursor + min(n["end"], n["start"] + sec_dur - 0.01),
                         "velocity": new_vel}
                    )
                    all_labels.append(section)

            cursor += sec_dur

        return all_notes, all_labels


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Rule-based composer (no model) ──────────────────────────────────────
    seed = [
        {"pitch_midi": 60, "start": 0.0, "end": 0.5, "velocity": 80},
        {"pitch_midi": 62, "start": 0.5, "end": 1.0, "velocity": 80},
        {"pitch_midi": 64, "start": 1.0, "end": 1.5, "velocity": 80},
        {"pitch_midi": 65, "start": 1.5, "end": 2.0, "velocity": 80},
    ]
    composer = RuleBasedComposer()
    notes, labels = composer.compose(seed, key_root_pc=0, key_scale="major")
    print(f"Rule-based composition: {len(notes)} notes across {len(set(labels))} sections")
    for section in dict.fromkeys(labels):
        count = labels.count(section)
        print(f"  {section}: {count} notes")

    # ── Neural model smoke test ──────────────────────────────────────────────
    model = CompositionTransformer(d_model=128, nhead=4, num_enc_layers=2, num_dec_layers=2,
                                    dim_feedforward=256)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    B, N_src, T_tgt = 2, 16, 32
    src = torch.randint(0, 128, (B, N_src))
    tgt = torch.randint(0, 128, (B, T_tgt))
    sec = torch.randint(0, NUM_SECTIONS, (B, T_tgt))
    logits = model(src, tgt, sec)
    print(f"Forward: src {src.shape}, tgt {tgt.shape} → logits {logits.shape}")

    seed_tokens = [60, 62, 64, 65, 67]
    section_plan = [("verse", 8), ("chorus", 8), ("verse", 8), ("chorus", 8)]
    gen, labels_gen = model.generate(seed_tokens, section_plan, temperature=0.9)
    print(f"Generated {len(gen)} tokens: {gen[:16]}…")

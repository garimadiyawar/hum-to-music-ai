"""
models/melody_transcriber.py
─────────────────────────────
Sequence-to-sequence model that converts an audio encoding (from
AudioEncoder) into a MIDI melody token sequence.

Architecture
────────────
AudioEncoder → Transformer Encoder → Transformer Decoder → Linear → MIDI tokens

The decoder is autoregressive: at inference time it runs step by step
(greedy or beam search).  At training time, teacher forcing is used.

Token vocabulary (131 tokens)
───────────────────────────────
  0–127 : MIDI pitch
  128   : SOS
  129   : EOS
  130   : PAD
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.audio_encoder import AudioEncoder, PositionalEncoding


PAD_TOKEN = 130
SOS_TOKEN = 128
EOS_TOKEN = 129
VOCAB_SIZE = 131


# ─── Melody Transcriber ──────────────────────────────────────────────────────

class MelodyTranscriber(nn.Module):
    """
    Encoder-decoder Transformer for audio → MIDI melody.

    Parameters
    ----------
    n_mels              : mel bins (height of input spectrogram)
    vocab_size          : melody token vocabulary size
    d_model             : transformer hidden dim
    nhead               : number of attention heads
    num_encoder_layers  : encoder depth
    num_decoder_layers  : decoder depth
    dim_feedforward     : FFN inner dim
    dropout             : dropout probability
    cnn_channels        : channel progression for AudioEncoder CNN
    max_mel_len         : max spectrogram time frames
    max_seq_len         : max output token sequence length
    """

    def __init__(
        self,
        n_mels:             int = 128,
        vocab_size:         int = VOCAB_SIZE,
        d_model:            int = 512,
        nhead:              int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward:    int = 2048,
        dropout:            float = 0.1,
        cnn_channels:       List[int] = None,
        max_mel_len:        int = 2048,
        max_seq_len:        int = 512,
    ):
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [32, 64, 128, 256]

        self.d_model    = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # ── Audio encoder (CNN + positional encoding) ──────────────────────
        self.audio_encoder = AudioEncoder(
            n_mels=n_mels,
            d_model=d_model,
            cnn_channels=cnn_channels,
            dropout=dropout,
            max_seq_len=max_mel_len,
        )

        # ── Transformer encoder layers ────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LN for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── Token embedding + positional encoding for decoder ─────────────
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos_enc = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_seq_len + 2)

        # ── Transformer decoder layers ────────────────────────────────────
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            dec_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── Output projection ─────────────────────────────────────────────
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Embedding scale (standard Transformer practice)
        nn.init.normal_(self.token_embedding.weight, mean=0, std=self.d_model ** -0.5)

    # ── Causal mask ──────────────────────────────────────────────────────────

    def _causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask (additive) for autoregressive decoding."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    # ── Forward (teacher forcing) ─────────────────────────────────────────────

    def forward(
        self,
        mel:            torch.Tensor,          # (B, 1, n_mels, T)
        tgt_tokens:     torch.Tensor,          # (B, N) – teacher-forced targets
        mel_lengths:    Optional[torch.Tensor] = None,   # (B,)
        tgt_lengths:    Optional[torch.Tensor] = None,   # (B,)
    ) -> torch.Tensor:
        """
        Returns
        -------
        logits : (B, N, vocab_size)
        """
        B, N = tgt_tokens.shape
        device = mel.device

        # ── Encode audio ─────────────────────────────────────────────────────
        mem, mem_lens = self.audio_encoder(mel, mel_lengths)       # (B, T', d_model)

        # Build memory key padding mask
        if mem_lens is not None:
            mem_mask = self.audio_encoder.make_padding_mask(mem_lens, mem.size(1))
        else:
            mem_mask = None

        # ── Transformer encoder ──────────────────────────────────────────────
        mem = self.transformer_encoder(mem, src_key_padding_mask=mem_mask)

        # ── Decode target tokens ─────────────────────────────────────────────
        # Shift right: input is tgt[:-1], label is tgt[1:]
        tgt_in  = tgt_tokens[:, :-1]          # (B, N-1)
        tgt_emb = self.token_embedding(tgt_in) * math.sqrt(self.d_model)
        tgt_emb = self.pos_enc(tgt_emb)        # (B, N-1, d_model)

        tgt_len = tgt_in.size(1)
        causal_mask = self._causal_mask(tgt_len, device)

        # Target padding mask
        if tgt_lengths is not None:
            tgt_pad_mask = torch.arange(tgt_len, device=device).unsqueeze(0) >= (tgt_lengths - 1).unsqueeze(1)
        else:
            tgt_pad_mask = (tgt_in == PAD_TOKEN)

        out = self.transformer_decoder(
            tgt_emb,
            mem,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=mem_mask,
        )                                       # (B, N-1, d_model)

        logits = self.output_proj(out)          # (B, N-1, vocab_size)
        return logits

    # ── Greedy decoding ───────────────────────────────────────────────────────

    @torch.no_grad()
    def greedy_decode(
        self,
        mel:         torch.Tensor,          # (1, 1, n_mels, T)  or (B, 1, n_mels, T)
        max_len:     int = 256,
        mel_lengths: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """
        Greedy autoregressive decoding.

        Returns
        -------
        list of token sequences (one per batch item), stripped of SOS/EOS/PAD
        """
        self.eval()
        device = mel.device
        B = mel.size(0)

        # Encode
        mem, mem_lens = self.audio_encoder(mel, mel_lengths)
        if mem_lens is not None:
            mem_mask = self.audio_encoder.make_padding_mask(mem_lens, mem.size(1))
        else:
            mem_mask = None
        mem = self.transformer_encoder(mem, src_key_padding_mask=mem_mask)

        # Start token
        tgt = torch.full((B, 1), SOS_TOKEN, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        sequences = [[] for _ in range(B)]

        for _ in range(max_len):
            tgt_emb = self.token_embedding(tgt) * math.sqrt(self.d_model)
            tgt_emb = self.pos_enc(tgt_emb)
            T_dec = tgt.size(1)
            causal = self._causal_mask(T_dec, device)

            out = self.transformer_decoder(tgt_emb, mem,
                                            tgt_mask=causal,
                                            memory_key_padding_mask=mem_mask)
            logits = self.output_proj(out[:, -1, :])    # (B, vocab_size)
            next_tok = logits.argmax(dim=-1)             # (B,)

            for i in range(B):
                if not finished[i]:
                    tok = next_tok[i].item()
                    if tok == EOS_TOKEN:
                        finished[i] = True
                    elif tok != PAD_TOKEN:
                        sequences[i].append(tok)

            if finished.all():
                break
            tgt = torch.cat([tgt, next_tok.unsqueeze(1)], dim=1)

        return sequences

    # ── Beam search decoding ─────────────────────────────────────────────────

    @torch.no_grad()
    def beam_search(
        self,
        mel:         torch.Tensor,          # (1, 1, n_mels, T)
        beam_size:   int = 5,
        max_len:     int = 256,
        mel_lengths: Optional[torch.Tensor] = None,
        alpha:       float = 0.6,           # length penalty exponent
    ) -> List[int]:
        """
        Beam-search decoding for a single example.

        Returns the best token sequence (MIDI pitches, no special tokens).
        """
        self.eval()
        assert mel.size(0) == 1, "beam_search handles one example at a time"
        device = mel.device

        # Encode memory once
        mem, mem_lens = self.audio_encoder(mel, mel_lengths)
        if mem_lens is not None:
            mem_mask = self.audio_encoder.make_padding_mask(mem_lens, mem.size(1))
        else:
            mem_mask = None
        mem = self.transformer_encoder(mem, src_key_padding_mask=mem_mask)

        # Expand for beam
        mem = mem.expand(beam_size, -1, -1)
        if mem_mask is not None:
            mem_mask = mem_mask.expand(beam_size, -1)

        # Beams: list of (score, token_sequence)
        beams = [(0.0, [SOS_TOKEN])]
        completed = []

        for step in range(max_len):
            new_beams = []
            for score, seq in beams:
                tgt = torch.tensor([seq], dtype=torch.long, device=device)
                tgt = tgt.expand(beam_size, -1)
                tgt_emb = self.token_embedding(tgt) * math.sqrt(self.d_model)
                tgt_emb = self.pos_enc(tgt_emb)
                causal = self._causal_mask(tgt.size(1), device)
                out = self.transformer_decoder(tgt_emb, mem,
                                               tgt_mask=causal,
                                               memory_key_padding_mask=mem_mask)
                logits = self.output_proj(out[0, -1, :])   # single beam expansion
                log_probs = F.log_softmax(logits, dim=-1)

                top_probs, top_toks = log_probs.topk(beam_size)
                for prob, tok in zip(top_probs.tolist(), top_toks.tolist()):
                    new_score = score + prob
                    new_seq = seq + [tok]
                    if tok == EOS_TOKEN:
                        # Length penalty
                        lp = ((5.0 + len(new_seq)) / 6.0) ** alpha
                        completed.append((new_score / lp, new_seq))
                    else:
                        new_beams.append((new_score, new_seq))

            if not new_beams:
                break
            # Keep top beam_size beams
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_size]

        if completed:
            completed.sort(key=lambda x: x[0], reverse=True)
            best_seq = completed[0][1]
        else:
            best_seq = beams[0][1]

        # Strip SOS/EOS/PAD
        return [t for t in best_seq if t not in (SOS_TOKEN, EOS_TOKEN, PAD_TOKEN)]


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, n_mels, T = 2, 128, 256
    N = 32

    mel    = torch.randn(B, 1, n_mels, T)
    tokens = torch.randint(0, 128, (B, N))
    tokens[:, 0] = SOS_TOKEN
    tokens[:, -1] = EOS_TOKEN

    model = MelodyTranscriber(
        n_mels=n_mels,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
    )
    print(model)

    # Forward
    logits = model(mel, tokens)
    print(f"\nForward pass: mel {mel.shape} → logits {logits.shape}")

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    labels = tokens[:, 1:]                      # shift
    loss = criterion(
        logits.reshape(-1, VOCAB_SIZE),
        labels.reshape(-1),
    )
    print(f"CE Loss: {loss.item():.4f}")

    # Greedy decode
    mel1 = mel[:1]
    pred = model.greedy_decode(mel1, max_len=20)
    print(f"Greedy decode: {pred}")

    # Param count
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

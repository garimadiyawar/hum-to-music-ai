"""
models/audio_encoder.py
───────────────────────
CNN audio encoder that maps a log-mel spectrogram (B, 1, n_mels, T)
to a sequence of dense feature vectors (B, T', d_model) for downstream
Transformer processing.

Architecture
────────────
  • 4 × [Conv2d → BatchNorm → ReLU → MaxPool] blocks
  • Frequency dimension collapsed via pooling
  • Linear projection to d_model
  • Positional encoding appended
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Sinusoidal positional encoding ──────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al., 2017).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)           # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─── CNN block ───────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Single Conv → BN → ReLU → (optional) MaxPool block.
    Preserves time dimension unless pool_time=True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_freq: bool = True,
        pool_time: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(kernel_size // 2, kernel_size // 2),
        )
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=dropout)

        freq_pool = 2 if pool_freq else 1
        time_pool = 2 if pool_time else 1
        self.pool = nn.MaxPool2d(kernel_size=(freq_pool, time_pool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, freq, time)"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.pool(x)
        return x


# ─── Main encoder ────────────────────────────────────────────────────────────

class AudioEncoder(nn.Module):
    """
    Mel-spectrogram → sequence of context vectors.

    Parameters
    ----------
    n_mels      : number of mel bins (height of input)
    d_model     : output feature dimension per time step
    cnn_channels: list of channel counts for successive ConvBlocks
    kernel_size : convolutional kernel size
    dropout     : dropout probability throughout
    max_seq_len : for positional encoding

    Input
    -----
    mel : (B, 1, n_mels, T)    – padded log-mel spectrogram in [0, 1]
    lengths : (B,) optional     – actual time lengths before padding (frames)

    Output
    ------
    out      : (B, T', d_model)  – encoded sequence
    out_lens : (B,)              – actual lengths in the encoded time axis
    """

    def __init__(
        self,
        n_mels:       int = 128,
        d_model:      int = 512,
        cnn_channels: List[int] = None,
        kernel_size:  int = 3,
        dropout:      float = 0.1,
        max_seq_len:  int = 2048,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 128, 256]

        # ── CNN backbone ─────────────────────────────────────────────────────
        layers = []
        in_ch = 1
        for i, out_ch in enumerate(cnn_channels):
            # Pool frequency at every layer, pool time only at first 2 layers
            layers.append(
                ConvBlock(
                    in_ch, out_ch,
                    kernel_size=kernel_size,
                    pool_freq=True,
                    pool_time=(i < 2),
                    dropout=dropout,
                )
            )
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # Compute collapsed frequency dimension after pooling
        freq_after = n_mels
        for i in range(len(cnn_channels)):
            freq_after = freq_after // 2       # pool_freq=True every layer
        # freq_after could be 0 if n_mels is very small
        freq_after = max(1, freq_after)

        self.freq_out = freq_after
        self.cnn_out_ch = cnn_channels[-1]

        # ── Temporal pooling: flatten freq ────────────────────────────────────
        # After CNN: (B, C, freq_after, T')
        # → (B, T', C * freq_after)
        flat_dim = self.cnn_out_ch * freq_after

        # ── Projection to d_model ─────────────────────────────────────────────
        self.proj = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # ── Positional encoding ───────────────────────────────────────────────
        self.pos_enc = PositionalEncoding(d_model=d_model, dropout=dropout,
                                          max_len=max_seq_len)

        # Track time downsampling factor (product of time pool strides)
        self._time_downsample = 4   # 2 time-pool layers → stride 4

    def forward(
        self,
        mel:     torch.Tensor,               # (B, 1, n_mels, T)
        lengths: Optional[torch.Tensor] = None,  # (B,) – unpadded frame counts
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # CNN
        x = self.cnn(mel)                    # (B, C, freq', T')

        B, C, F, T_prime = x.shape
        # Flatten freq dimension
        x = x.permute(0, 3, 1, 2)           # (B, T', C, freq')
        x = x.reshape(B, T_prime, C * F)    # (B, T', C*freq')

        # Project
        x = self.proj(x)                     # (B, T', d_model)

        # Positional encoding
        x = self.pos_enc(x)                  # (B, T', d_model)

        # Adjust lengths
        out_lengths = None
        if lengths is not None:
            out_lengths = (lengths / self._time_downsample).long().clamp(min=1)

        return x, out_lengths

    def make_padding_mask(
        self,
        lengths: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        """
        Build a boolean padding mask (B, T') where True = ignore.

        Parameters
        ----------
        lengths : (B,) actual lengths in the *encoded* time axis
        max_len : T' (encoded time dimension)
        """
        ids = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # (1, T')
        mask = ids >= lengths.unsqueeze(1)                                # (B, T')
        return mask


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, n_mels, T = 4, 128, 512
    mel = torch.randn(B, 1, n_mels, T)
    lengths = torch.tensor([512, 400, 350, 200])

    encoder = AudioEncoder(n_mels=n_mels, d_model=512)
    print(encoder)

    out, out_lens = encoder(mel, lengths)
    print(f"\nInput  : {mel.shape}")
    print(f"Output : {out.shape}   (B, T', d_model)")
    print(f"Lengths: {out_lens}")

    # Padding mask
    mask = encoder.make_padding_mask(out_lens, max_len=out.shape[1])
    print(f"Padding mask shape: {mask.shape},  masked positions: {mask.sum().item()}")

    # Parameter count
    total = sum(p.numel() for p in encoder.parameters())
    print(f"Parameters: {total:,}")

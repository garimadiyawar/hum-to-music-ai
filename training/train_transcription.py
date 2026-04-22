"""
training/train_transcription.py
────────────────────────────────
Training script for MelodyTranscriber (audio → MIDI melody).

Usage
-----
python -m training.train_transcription \
    --maestro_root /data/maestro \
    --checkpoint_dir checkpoints/transcription \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4

Or edit configs/model_config.yaml and let this script read it:
python -m training.train_transcription --config configs/model_config.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_loader import HummingTranscriptionDataset, build_dataloaders
from models.melody_transcriber import MelodyTranscriber, PAD_TOKEN, VOCAB_SIZE


# ─── Warmup Cosine Scheduler ─────────────────────────────────────────────────

class WarmupCosineScheduler:
    """Linear warm-up followed by cosine decay."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer     = optimizer
        self.warmup_steps  = warmup_steps
        self.total_steps   = total_steps
        self.min_lr        = min_lr
        self._step = 0
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self):
        self._step += 1
        s = self._step
        for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            if s <= self.warmup_steps:
                lr = base_lr * s / self.warmup_steps
            else:
                progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            pg["lr"] = lr

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ─── Metrics ─────────────────────────────────────────────────────────────────

def token_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad: int = PAD_TOKEN) -> float:
    """Accuracy excluding PAD positions."""
    preds   = logits.argmax(dim=-1)        # (B, N-1)
    mask    = targets != pad               # (B, N-1) True = valid
    correct = (preds == targets) & mask
    return correct.sum().item() / max(mask.sum().item(), 1)


# ─── Training loop ───────────────────────────────────────────────────────────

def train_one_epoch(
    model:     MelodyTranscriber,
    loader:    torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scheduler,
    device:    torch.device,
    grad_clip: float = 1.0,
    log_interval: int = 50,
    writer:    SummaryWriter = None,
    epoch:     int = 0,
    global_step: int = 0,
) -> Tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Train E{epoch}", leave=False)):
        mel    = batch["mel"].to(device)                    # (B, 1, n_mels, T)
        tokens = batch["tokens"].to(device)                 # (B, N)
        mel_lens = batch["mel_length"].to(device)
        tok_lens = batch["token_length"].to(device)

        # Forward
        logits = model(mel, tokens, mel_lengths=mel_lens, tgt_lengths=tok_lens)
        # logits: (B, N-1, vocab_size)   targets: tokens[:, 1:]
        labels = tokens[:, 1:]
        B, N1 = labels.shape

        loss = criterion(
            logits.reshape(B * N1, VOCAB_SIZE),
            labels.reshape(B * N1),
        )
        acc = token_accuracy(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc  += acc
        n_batches  += 1
        global_step += 1

        if log_interval > 0 and batch_idx % log_interval == 0:
            avg_loss = total_loss / n_batches
            avg_acc  = total_acc  / n_batches
            lr = scheduler.current_lr
            tqdm.write(
                f"  Step {global_step:6d} | Loss {avg_loss:.4f} | Acc {avg_acc:.3f} | LR {lr:.2e}"
            )
            if writer:
                writer.add_scalar("train/loss",     avg_loss,  global_step)
                writer.add_scalar("train/accuracy",  avg_acc,   global_step)
                writer.add_scalar("train/lr",        lr,        global_step)

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1), global_step


@torch.no_grad()
def evaluate(
    model:     MelodyTranscriber,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    epoch:     int = 0,
    writer:    SummaryWriter = None,
    global_step: int = 0,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    for batch in tqdm(loader, desc=f"Val   E{epoch}", leave=False):
        mel    = batch["mel"].to(device)
        tokens = batch["tokens"].to(device)
        mel_lens = batch["mel_length"].to(device)
        tok_lens = batch["token_length"].to(device)

        logits = model(mel, tokens, mel_lengths=mel_lens, tgt_lengths=tok_lens)
        labels = tokens[:, 1:]
        B, N1 = labels.shape

        loss = criterion(logits.reshape(B * N1, VOCAB_SIZE), labels.reshape(B * N1))
        acc  = token_accuracy(logits, labels)

        total_loss += loss.item()
        total_acc  += acc
        n_batches  += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc  = total_acc  / max(n_batches, 1)
    if writer:
        writer.add_scalar("val/loss",    avg_loss,  global_step)
        writer.add_scalar("val/accuracy", avg_acc,  global_step)
    return avg_loss, avg_acc


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    from typing import Tuple   # local import for annotation only

    parser = argparse.ArgumentParser(description="Train MelodyTranscriber")
    parser.add_argument("--maestro_root",   default="data/raw/maestro")
    parser.add_argument("--checkpoint_dir", default="checkpoints/transcription")
    parser.add_argument("--log_dir",        default="logs/transcription")
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--d_model",        type=int,   default=512)
    parser.add_argument("--nhead",          type=int,   default=8)
    parser.add_argument("--enc_layers",     type=int,   default=6)
    parser.add_argument("--dec_layers",     type=int,   default=6)
    parser.add_argument("--warmup_steps",   type=int,   default=1000)
    parser.add_argument("--grad_clip",      type=float, default=1.0)
    parser.add_argument("--val_split",      type=float, default=0.1)
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--max_mel_frames", type=int,   default=1024)
    parser.add_argument("--max_notes",      type=int,   default=256)
    parser.add_argument("--resume",         default=None, help="path to checkpoint to resume")
    parser.add_argument("--device",         default="auto")
    args = parser.parse_args()

    # ── Device ──────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps"  if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # ── Paths ────────────────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir  = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer   = SummaryWriter(log_dir=str(log_dir))

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("Building dataset …")
    if Path(args.maestro_root).exists():
        dataset = HummingTranscriptionDataset.from_maestro(
            maestro_root=args.maestro_root,
            max_mel_frames=args.max_mel_frames,
            max_notes=args.max_notes,
            cache_dir=Path("data/cache/transcription"),
            augment=True,
        )
    else:
        print(f"WARNING: {args.maestro_root} not found. Creating synthetic dataset.")
        # Minimal synthetic dataset for demo
        from data.dataset_loader import HummingTranscriptionDataset
        import numpy as np
        from utils.midi_utils import notes_to_midi, save_midi
        from utils.audio_utils import save_audio
        import tempfile, os
        tmpdir = Path(tempfile.mkdtemp())
        audio_paths, midi_paths = [], []
        for i in range(32):
            # Synthetic audio
            sr = 22050
            pitches = [60, 62, 64, 65, 67, 69, 71, 72]
            t = np.linspace(0, 4, 4 * sr)
            y = np.zeros(len(t), dtype=np.float32)
            for j, p in enumerate(pitches):
                hz = 440 * 2 ** ((p - 69) / 12)
                s, e = j * sr // 2, (j + 1) * sr // 2
                y[s:e] += np.sin(2 * np.pi * hz * t[s:e]) * 0.3
            wav_path = tmpdir / f"sample_{i}.wav"
            save_audio(y, wav_path, sr=sr)
            # Synthetic MIDI
            note_evs = [{"pitch_midi": p, "start": j * 0.5, "end": j * 0.5 + 0.4}
                        for j, p in enumerate(pitches)]
            pm = notes_to_midi(note_evs, tempo=120)
            mid_path = tmpdir / f"sample_{i}.mid"
            save_midi(pm, mid_path)
            audio_paths.append(wav_path)
            midi_paths.append(mid_path)
        dataset = HummingTranscriptionDataset(
            audio_paths, midi_paths,
            max_mel_frames=args.max_mel_frames,
            max_notes=args.max_notes,
        )

    train_loader, val_loader = build_dataloaders(
        dataset,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)},  Val batches: {len(val_loader)}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = MelodyTranscriber(
        n_mels=128,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss / Optimizer ─────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    scheduler   = WarmupCosineScheduler(optimizer, args.warmup_steps, total_steps)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    global_step  = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        global_step   = ckpt.get("global_step", 0)
        print(f"Resumed from epoch {start_epoch - 1},  best_val_loss={best_val_loss:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler,
            device, args.grad_clip, log_interval=50,
            writer=writer, epoch=epoch, global_step=global_step,
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device,
            epoch=epoch, writer=writer, global_step=global_step,
        )

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"Val loss {val_loss:.4f} acc {val_acc:.3f} | "
            f"{elapsed:.0f}s"
        )

        # Save checkpoint
        ckpt = {
            "epoch":       epoch,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "val_loss":    val_loss,
            "global_step": global_step,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  ✓ New best val_loss: {best_val_loss:.4f}")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    # Python 3.9+ compatible Tuple import
    from typing import Tuple
    main()

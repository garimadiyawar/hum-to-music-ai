"""
training/train_arrangement_model.py
─────────────────────────────────────
Training script for ArrangementModel (melody + chords → multi-track MIDI).

Usage
-----
python -m training.train_arrangement_model \
    --lakh_root /data/lakh \
    --checkpoint_dir checkpoints/arrangement \
    --epochs 120 \
    --batch_size 8
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_loader import ArrangementDataset, build_dataloaders
from data.midi_processing import scan_midi_directory
from models.arrangement_model import (
    ArrangementModel,
    ARR_VOCAB_SIZE,
    ARR_PAD,
    ARR_SOS,
    ARR_EOS,
)
from training.train_transcription import WarmupCosineScheduler


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_perplexity(loss: float) -> float:
    return math.exp(min(loss, 100.0))


def sequence_accuracy(
    logits:  torch.Tensor,     # (B, T-1, vocab)
    targets: torch.Tensor,     # (B, T-1)
    pad:     int = ARR_PAD,
) -> float:
    preds  = logits.argmax(dim=-1)
    mask   = targets != pad
    correct = (preds == targets) & mask
    return correct.sum().item() / max(mask.sum().item(), 1)


# ─── Train / Eval ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model:       ArrangementModel,
    loader:      torch.utils.data.DataLoader,
    optimizer:   optim.Optimizer,
    criterion:   nn.Module,
    scheduler,
    device:      torch.device,
    grad_clip:   float = 1.0,
    log_interval: int  = 50,
    writer:      SummaryWriter = None,
    epoch:       int = 0,
    global_step: int = 0,
) -> Tuple[float, float, float, int]:
    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    total_ppl  = 0.0
    n_batches  = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Train E{epoch}", leave=False)):
        melody = batch["melody"].to(device)           # (B, N_mel)
        chords = batch["chords"].to(device)           # (B, N_crd)
        arr    = batch["arrangement"].to(device)      # (B, T_arr)

        mel_pad   = (melody == 130)
        chord_pad = torch.zeros_like(chords, dtype=torch.bool)   # no pad token for chords in this impl

        # Forward (teacher-forced)
        logits = model(melody, chords, arr,
                       mel_pad_mask=mel_pad,
                       chord_pad_mask=chord_pad)     # (B, T-1, vocab)
        targets = arr[:, 1:]                          # (B, T-1)
        B, T1 = targets.shape

        loss = criterion(
            logits.reshape(B * T1, ARR_VOCAB_SIZE),
            targets.reshape(B * T1),
        )
        acc = sequence_accuracy(logits, targets)
        ppl = compute_perplexity(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc  += acc
        total_ppl  += ppl
        n_batches  += 1
        global_step += 1

        if log_interval > 0 and batch_idx % log_interval == 0:
            n = max(n_batches, 1)
            tqdm.write(
                f"  Step {global_step:6d} | Loss {total_loss/n:.4f} | "
                f"Acc {total_acc/n:.3f} | PPL {total_ppl/n:.1f} | "
                f"LR {scheduler.current_lr:.2e}"
            )
            if writer:
                writer.add_scalar("train/loss",        total_loss / n, global_step)
                writer.add_scalar("train/accuracy",    total_acc  / n, global_step)
                writer.add_scalar("train/perplexity",  total_ppl  / n, global_step)
                writer.add_scalar("train/lr",          scheduler.current_lr, global_step)

    n = max(n_batches, 1)
    return total_loss / n, total_acc / n, total_ppl / n, global_step


@torch.no_grad()
def evaluate(
    model:     ArrangementModel,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    epoch:     int = 0,
    writer:    SummaryWriter = None,
    global_step: int = 0,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    total_ppl  = 0.0
    n_batches  = 0

    for batch in tqdm(loader, desc=f"Val   E{epoch}", leave=False):
        melody = batch["melody"].to(device)
        chords = batch["chords"].to(device)
        arr    = batch["arrangement"].to(device)

        mel_pad = (melody == 130)
        chord_pad = torch.zeros_like(chords, dtype=torch.bool)

        logits  = model(melody, chords, arr, mel_pad_mask=mel_pad, chord_pad_mask=chord_pad)
        targets = arr[:, 1:]
        B, T1   = targets.shape

        loss = criterion(logits.reshape(B * T1, ARR_VOCAB_SIZE), targets.reshape(B * T1))
        acc  = sequence_accuracy(logits, targets)
        ppl  = compute_perplexity(loss.item())

        total_loss += loss.item()
        total_acc  += acc
        total_ppl  += ppl
        n_batches  += 1

    n = max(n_batches, 1)
    vl, va, vp = total_loss / n, total_acc / n, total_ppl / n
    if writer:
        writer.add_scalar("val/loss",       vl, global_step)
        writer.add_scalar("val/accuracy",   va, global_step)
        writer.add_scalar("val/perplexity", vp, global_step)
    return vl, va, vp


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ArrangementModel")
    parser.add_argument("--lakh_root",      default="data/raw/lakh")
    parser.add_argument("--checkpoint_dir", default="checkpoints/arrangement")
    parser.add_argument("--log_dir",        default="logs/arrangement")
    parser.add_argument("--epochs",         type=int,   default=120)
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--d_model",        type=int,   default=512)
    parser.add_argument("--nhead",          type=int,   default=8)
    parser.add_argument("--enc_layers",     type=int,   default=6)
    parser.add_argument("--dec_layers",     type=int,   default=6)
    parser.add_argument("--warmup_steps",   type=int,   default=2000)
    parser.add_argument("--grad_clip",      type=float, default=1.0)
    parser.add_argument("--max_files",      type=int,   default=None)
    parser.add_argument("--max_src_len",    type=int,   default=128)
    parser.add_argument("--max_tgt_len",    type=int,   default=512)
    parser.add_argument("--resume",         default=None)
    parser.add_argument("--device",         default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps"  if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    ckpt_dir = Path(args.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir  = Path(args.log_dir);        log_dir.mkdir(parents=True, exist_ok=True)
    writer   = SummaryWriter(log_dir=str(log_dir))

    # ── Dataset ──────────────────────────────────────────────────────────────
    lakh_root = Path(args.lakh_root)
    if lakh_root.exists():
        midi_paths = scan_midi_directory(lakh_root, max_files=args.max_files)
    else:
        print(f"WARNING: {lakh_root} not found. Using synthetic MIDI.")
        import tempfile, pretty_midi as pm_lib
        tmpdir = Path(tempfile.mkdtemp())
        midi_paths = []
        for i in range(32):
            pm_ = pm_lib.PrettyMIDI(initial_tempo=120)
            for prog, pitches, vels in [
                (0,  [60, 64, 67, 69, 71, 72], 80),   # melody
                (32, [36, 38, 40, 41],          70),   # bass
                (48, [60, 64, 67, 71],          65),   # strings
            ]:
                inst = pm_lib.Instrument(program=prog)
                for j, p in enumerate(pitches):
                    inst.notes.append(pm_lib.Note(velocity=vels, pitch=p,
                                                  start=j * 0.5, end=j * 0.5 + 0.4))
                pm_.instruments.append(inst)
            p = tmpdir / f"arr_{i}.mid"
            pm_.write(str(p))
            midi_paths.append(p)

    print(f"Building ArrangementDataset from {len(midi_paths)} files …")
    dataset = ArrangementDataset(midi_paths,
                                  max_src_len=args.max_src_len,
                                  max_tgt_len=args.max_tgt_len)
    if len(dataset) == 0:
        print("ERROR: Empty dataset. Check MIDI data.")
        return
    print(f"Dataset: {len(dataset)} examples")

    train_loader, val_loader = build_dataloaders(dataset, batch_size=args.batch_size, num_workers=2)

    # ── Model ────────────────────────────────────────────────────────────────
    model = ArrangementModel(
        d_model=args.d_model,
        nhead=args.nhead,
        num_enc_layers=args.enc_layers,
        num_dec_layers=args.dec_layers,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion   = nn.CrossEntropyLoss(ignore_index=ARR_PAD, label_smoothing=0.1)
    optimizer   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * max(len(train_loader), 1)
    scheduler   = WarmupCosineScheduler(optimizer, args.warmup_steps, total_steps)

    start_epoch   = 0
    best_val_loss = float("inf")
    global_step   = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        global_step   = ckpt.get("global_step", 0)
        print(f"Resumed from epoch {start_epoch - 1}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tl, ta, tp, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler,
            device, args.grad_clip, 50, writer, epoch, global_step,
        )
        vl, va, vp = evaluate(
            model, val_loader, criterion, device, epoch, writer, global_step,
        )
        print(
            f"Epoch {epoch:3d} | "
            f"Train L={tl:.4f} Acc={ta:.3f} PPL={tp:.1f} | "
            f"Val   L={vl:.4f} Acc={va:.3f} PPL={vp:.1f} | "
            f"{time.time()-t0:.0f}s"
        )

        ckpt_d = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": vl, "global_step": global_step,
        }
        torch.save(ckpt_d, ckpt_dir / "last.pt")
        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(ckpt_d, ckpt_dir / "best.pt")
            print(f"  ✓ Best val_loss: {best_val_loss:.4f}")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()

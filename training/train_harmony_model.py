"""
training/train_harmony_model.py
────────────────────────────────
Training script for HarmonyGenerator (melody tokens → chord tokens).

Usage
-----
python -m training.train_harmony_model \
    --lakh_root /data/lakh \
    --checkpoint_dir checkpoints/harmony \
    --epochs 80 \
    --batch_size 32
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

from data.dataset_loader import HarmonyDataset, build_dataloaders
from data.midi_processing import scan_midi_directory
from models.harmony_generator import HarmonyGenerator, MEL_PAD, CHORD_VOCAB
from training.train_transcription import WarmupCosineScheduler


# ─── Metrics ─────────────────────────────────────────────────────────────────

def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Top-k accuracy for chord classification."""
    _, topk = logits.topk(k, dim=-1)   # (B, N, k)  or (B, k)
    if logits.dim() == 3:
        targets_exp = targets.unsqueeze(-1).expand_as(topk)
        correct = (topk == targets_exp).any(dim=-1).float()
        return correct.mean().item()
    else:
        return (topk == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()


# ─── Train / Eval ────────────────────────────────────────────────────────────

def train_one_epoch(
    model:       HarmonyGenerator,
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
    total_acc1 = 0.0
    total_acc5 = 0.0
    n_batches  = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Train E{epoch}", leave=False)):
        melody = batch["melody"].to(device)    # (B, N)
        chord  = batch["chord"].to(device)     # (B,)

        src_mask = (melody == MEL_PAD)

        # Forward: get logits for all positions, then pool for single chord pred
        logits = model(melody, src_mask)        # (B, N, CHORD_VOCAB)

        # Strategy: predict chord from first valid (non-SOS non-PAD) position
        # Use the mean-pooled representation
        valid_mask = (~src_mask).float().unsqueeze(-1)    # (B, N, 1)
        pooled_logits = (logits * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        # pooled_logits: (B, CHORD_VOCAB)

        loss = criterion(pooled_logits, chord)
        acc1 = (pooled_logits.argmax(dim=-1) == chord).float().mean().item()
        acc5 = top_k_accuracy(pooled_logits, chord, k=5)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_acc1 += acc1
        total_acc5 += acc5
        n_batches  += 1
        global_step += 1

        if log_interval > 0 and batch_idx % log_interval == 0:
            avg_l = total_loss / n_batches
            avg_a = total_acc1 / n_batches
            lr    = scheduler.current_lr
            tqdm.write(
                f"  Step {global_step:6d} | Loss {avg_l:.4f} | "
                f"Top-1 {avg_a:.3f} | Top-5 {total_acc5 / n_batches:.3f} | LR {lr:.2e}"
            )
            if writer:
                writer.add_scalar("train/loss",    avg_l, global_step)
                writer.add_scalar("train/top1_acc", avg_a, global_step)
                writer.add_scalar("train/top5_acc", total_acc5 / n_batches, global_step)
                writer.add_scalar("train/lr",       lr,    global_step)

    n = max(n_batches, 1)
    return total_loss / n, total_acc1 / n, total_acc5 / n, global_step


@torch.no_grad()
def evaluate(
    model:     HarmonyGenerator,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    epoch:     int = 0,
    writer:    SummaryWriter = None,
    global_step: int = 0,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    n_batches  = 0

    for batch in tqdm(loader, desc=f"Val   E{epoch}", leave=False):
        melody = batch["melody"].to(device)
        chord  = batch["chord"].to(device)
        src_mask = (melody == MEL_PAD)

        logits = model(melody, src_mask)
        valid_mask = (~src_mask).float().unsqueeze(-1)
        pooled = (logits * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

        loss = criterion(pooled, chord)
        acc1 = (pooled.argmax(dim=-1) == chord).float().mean().item()
        acc5 = top_k_accuracy(pooled, chord, k=5)

        total_loss += loss.item()
        total_acc1 += acc1
        total_acc5 += acc5
        n_batches  += 1

    n = max(n_batches, 1)
    avg_l, avg_a, avg_a5 = total_loss / n, total_acc1 / n, total_acc5 / n
    if writer:
        writer.add_scalar("val/loss",    avg_l,  global_step)
        writer.add_scalar("val/top1_acc", avg_a,  global_step)
        writer.add_scalar("val/top5_acc", avg_a5, global_step)
    return avg_l, avg_a, avg_a5


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train HarmonyGenerator")
    parser.add_argument("--lakh_root",      default="data/raw/lakh")
    parser.add_argument("--checkpoint_dir", default="checkpoints/harmony")
    parser.add_argument("--log_dir",        default="logs/harmony")
    parser.add_argument("--epochs",         type=int,   default=80)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--d_model",        type=int,   default=256)
    parser.add_argument("--nhead",          type=int,   default=4)
    parser.add_argument("--num_layers",     type=int,   default=4)
    parser.add_argument("--warmup_steps",   type=int,   default=500)
    parser.add_argument("--grad_clip",      type=float, default=1.0)
    parser.add_argument("--max_files",      type=int,   default=None)
    parser.add_argument("--max_melody_len", type=int,   default=64)
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
    print("Scanning MIDI files …")
    lakh_root = Path(args.lakh_root)
    if lakh_root.exists():
        midi_paths = scan_midi_directory(lakh_root, max_files=args.max_files)
    else:
        print(f"WARNING: {lakh_root} not found. Using synthetic MIDI.")
        import tempfile, pretty_midi as pm_lib
        tmpdir = Path(tempfile.mkdtemp())
        midi_paths = []
        for i in range(64):
            pm_ = pm_lib.PrettyMIDI(initial_tempo=120)
            inst = pm_lib.Instrument(program=0)
            for j, pitch in enumerate([60, 64, 67, 71, 72, 69, 67, 65]):
                inst.notes.append(pm_lib.Note(velocity=80, pitch=pitch,
                                              start=j * 0.5, end=j * 0.5 + 0.4))
            # Add bass notes for chord detection
            for j, pitch in enumerate([48, 52, 55, 59]):
                inst.notes.append(pm_lib.Note(velocity=60, pitch=pitch,
                                              start=j * 1.0, end=j * 1.0 + 0.9))
            pm_.instruments.append(inst)
            p = tmpdir / f"synth_{i}.mid"
            pm_.write(str(p))
            midi_paths.append(p)

    print(f"Building HarmonyDataset from {len(midi_paths)} MIDI files …")
    dataset = HarmonyDataset(midi_paths, max_melody_len=args.max_melody_len)
    if len(dataset) == 0:
        print("ERROR: No melody-chord pairs found. Check MIDI data.")
        return
    print(f"Dataset size: {len(dataset)}")

    train_loader, val_loader = build_dataloaders(dataset, batch_size=args.batch_size)

    # ── Model ────────────────────────────────────────────────────────────────
    model = HarmonyGenerator(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tl, ta, ta5, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler,
            device, args.grad_clip, log_interval=50,
            writer=writer, epoch=epoch, global_step=global_step,
        )
        vl, va, va5 = evaluate(
            model, val_loader, criterion, device,
            epoch=epoch, writer=writer, global_step=global_step,
        )
        print(
            f"Epoch {epoch:3d} | Train L={tl:.4f} Top1={ta:.3f} Top5={ta5:.3f} | "
            f"Val L={vl:.4f} Top1={va:.3f} Top5={va5:.3f} | {time.time()-t0:.0f}s"
        )

        ckpt_d = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": vl,
            "global_step": global_step,
        }
        torch.save(ckpt_d, ckpt_dir / "last.pt")
        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(ckpt_d, ckpt_dir / "best.pt")
            print(f"  ✓ New best val_loss: {best_val_loss:.4f}")

    writer.close()
    print("Done.")


if __name__ == "__main__":
    main()

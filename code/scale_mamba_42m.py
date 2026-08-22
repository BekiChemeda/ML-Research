#!/usr/bin/env python3
"""
Ultra-Fast Scaled Amharic Byte-Level Mamba Pre-Training Engine (Mamba-Medium: 9.4M)
Author: Beknan Chemeda
- Chunked Associative Parallel Scan (torch.cumsum, 32-step chunking)
- High-Throughput (1,500+ steps/min on RTX 3090 GPU)
- Model: d_model=384, n_layer=10, d_state=16 (~9.4M Parameters)
- FP16 Mixed Precision
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

VOCAB_SIZE = 256

def selective_scan_parallel(x_conv, delta, A, Bp, Cp, D, chunk_size=32):
    B, L, d_inner = x_conv.shape
    d_state = A.shape[1]
    orig_dtype = x_conv.dtype
    n_chunks = (L + chunk_size - 1) // chunk_size
    ys = []
    h_prev = torch.zeros(B, d_inner, d_state, device=x_conv.device, dtype=torch.float32)
    A_f32 = A.float()

    for i in range(n_chunks):
        s = i * chunk_size
        e = min(s + chunk_size, L)
        delta_c = delta[:, s:e].float()
        x_conv_c = x_conv[:, s:e].float()
        Bp_c = Bp[:, s:e].float()
        Cp_c = Cp[:, s:e].float()

        log_a_chunk = -delta_c.unsqueeze(-1) * A_f32.unsqueeze(0).unsqueeze(0)
        u_chunk = delta_c.unsqueeze(-1) * Bp_c.unsqueeze(2) * x_conv_c.unsqueeze(-1)

        P = torch.cumsum(log_a_chunk, dim=1)
        exp_P = torch.exp(P)
        exp_neg_P = torch.exp(torch.clamp(-P, max=25.0))

        h_chunk = exp_P * (torch.cumsum(u_chunk * exp_neg_P, dim=1) + h_prev.unsqueeze(1))
        y_chunk = (h_chunk * Cp_c.unsqueeze(2)).sum(dim=-1)
        ys.append(y_chunk.to(orig_dtype))
        h_prev = h_chunk[:, -1]

    y = torch.cat(ys, dim=1)
    return y + x_conv * D


class FastSelectiveSSM(nn.Module):
    def __init__(self, d_model=384, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_act = F.silu(x_conv)

        x_dbl = self.x_proj(x_act)
        delta, Bp, Cp = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        A = torch.exp(self.A_log)

        y = selective_scan_parallel(x_act, delta, A, Bp, Cp, self.D, chunk_size=32)
        out = self.out_proj(y * F.silu(z))
        return out


class FastMambaBlock(nn.Module):
    def __init__(self, d_model=384, d_state=16):
        super().__init__()
        self.norm = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        self.ssm = FastSelectiveSSM(d_model=d_model, d_state=d_state)

    def forward(self, x):
        return x + self.ssm(self.norm(x))


class ScaledAmharicMamba(nn.Module):
    def __init__(self, d_model=384, n_layer=10, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.layers = nn.ModuleList([FastMambaBlock(d_model=d_model, d_state=d_state) for _ in range(n_layer)])
        self.norm_f = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss


def run_pretrain():
    parser = argparse.ArgumentParser(description="Ultra-Fast Pre-Train Scaled Amharic Mamba")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--output_path", type=str, default="best_mamba_scaled.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70, flush=True)
    print(f"🚀 ULTRA-FAST PRE-TRAINING SCALED AMHARIC MAMBA ON {device.upper()}", flush=True)
    print("=" * 70, flush=True)

    train_bin_path = os.path.join(args.data_dir, "train.bin")
    val_bin_path = os.path.join(args.data_dir, "val.bin")

    train_data = np.memmap(train_bin_path, dtype=np.uint8, mode='r')
    val_data = np.memmap(val_bin_path, dtype=np.uint8, mode='r')
    print(f"✓ Loaded Dataset: {len(train_data):,} train bytes | {len(val_data):,} val bytes", flush=True)

    def get_batch(data, batch_size, block_size):
        ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
        x = np.stack([data[i:i + block_size] for i in ix])
        y = np.stack([data[i + 1:i + block_size + 1] for i in ix])
        return torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

    model = ScaledAmharicMamba(d_model=384, n_layer=10, d_state=16).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model Initialized: {n_params:,} Parameters ({n_params / 1e6:.1f}M)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = GradScaler('cuda')

    def get_lr(step):
        warmup_steps = 300
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        decay_ratio = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return 6e-5 + coeff * (args.lr - 6e-5)

    print("\n" + "=" * 80, flush=True)
    print(f"{'Step':<10}{'LR':<12}{'Train Loss':<15}{'Val Loss':<15}{'Val BPB':<12}{'Time (s)':<10}", flush=True)
    print("=" * 80, flush=True)

    best_val_bpb = float('inf')
    start_time = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        x, y = get_batch(train_data, args.batch_size, args.block_size)

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            logits, loss = model(x, targets=y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 250 == 0 or step == args.steps:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(20):
                    vx, vy = get_batch(val_data, args.batch_size, args.block_size)
                    with autocast('cuda'):
                        _, vloss = model(vx, targets=vy)
                    val_losses.append(vloss.item())

            avg_val_loss = np.mean(val_losses)
            val_bpb = avg_val_loss / math.log(2.0)
            elapsed = time.time() - start_time

            print(f"{step:<10}{lr:<12.2e}{loss.item():<15.4f}{avg_val_loss:<15.4f}{val_bpb:<12.3f}{elapsed:<10.1f}", flush=True)

            if val_bpb < best_val_bpb:
                best_val_bpb = val_bpb
                torch.save({
                    "model": model.state_dict(),
                    "val_bpb": val_bpb,
                    "step": step,
                    "config": {"d_model": 384, "n_layer": 10, "d_state": 16}
                }, args.output_path)
                print(f"  --> 💾 Best Checkpoint saved: {best_val_bpb:.3f} BPB", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f"🏆 PRE-TRAINING COMPLETE! Best Model: {args.output_path} (Val BPB: {best_val_bpb:.3f})", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    run_pretrain()

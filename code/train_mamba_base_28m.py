#!/usr/bin/env python3
"""
Pre-Training Script for Amharic Mamba-Base (28.5M Parameters, 10,000 Steps)
==========================================================================
Architecture:
- d_model: 512
- n_layer: 16
- d_state: 16
- d_conv: 4
- expand: 2 (d_inner = 1024)
- Chunked Associative Parallel Scan (chunk_size=32)
- Mixed Precision (AMP FP16 / BF16)
- Gradient Clipping: 1.0
- Cosine LR Warmup & Decay (6e-4 -> 3e-5)

Author: Beknan Chemeda / AI Research Team
"""

import os
import sys
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# FAST CHUNKED PARALLEL SELECTIVE SCAN
# ==============================================================================
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


# ==============================================================================
# MAMBA-BASE BLOCK & NEURAL NETWORK
# ==============================================================================
class MambaBaseBlock(nn.Module):
    def __init__(self, d_model=512, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)  # 1024
        self.dt_rank = math.ceil(d_model / 16)  # 32

        self.norm = nn.RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A_init = torch.repeat_interleave(torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0), self.d_inner, dim=0)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        
        # in_proj -> split into branch and gate
        xz = self.in_proj(x_norm)
        x_branch, z = xz.chunk(2, dim=-1)

        # 1D causal convolution
        L = x.shape[1]
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_act = F.silu(x_conv)

        # SSM projections
        ssm_p = self.x_proj(x_act)
        delta_raw, Bp, Cp = torch.split(ssm_p, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta_raw))
        A = torch.exp(self.A_log)

        # Parallel chunked scan
        y_ssm = selective_scan_parallel(x_act, delta, A, Bp, Cp, self.D, chunk_size=32)

        # Gated output projection
        y_gated = y_ssm * F.silu(z)
        out = self.out_proj(y_gated)
        return residual + out


class AmharicMambaBase(nn.Module):
    def __init__(self, vocab_size=256, d_model=512, n_layer=16, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaBaseBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layer)
        ])
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # Weight tying

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids, targets=None, memory_module=None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)

        if memory_module is not None:
            mem_bias = memory_module(x)
            x = x + mem_bias

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-100)

        return logits, loss

    @torch.no_grad()
    def generate(self, prompt_bytes: bytes, max_new_tokens=150, temperature=0.7, top_p=0.9, memory_module=None, device="cuda"):
        self.eval()
        tokens = list(prompt_bytes)
        for _ in range(max_new_tokens):
            x = torch.tensor([tokens[-384:]], dtype=torch.long, device=device)
            with torch.amp.autocast('cuda'):
                logits, _ = self(x, memory_module=memory_module)
            next_logits = logits[0, -1, :] / max(temperature, 1e-5)

            probs = F.softmax(next_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).item()
            tokens.append(next_tok)

            if len(tokens) >= 4 and tokens[-4:] == list(b"</s>"):
                break

        return bytes(tokens)


# ==============================================================================
# DATASET LOADER (STREAMING BYTE MEMORY-MAP)
# ==============================================================================
class AmharicByteCorpus:
    def __init__(self, data_dir="/workspace/ML-Research/code/data", seq_len=384):
        self.data_dir = data_dir
        self.seq_len = seq_len

        train_path = os.path.join(data_dir, "train.bin") if os.path.isdir(data_dir) else data_dir
        val_path = os.path.join(data_dir, "val.bin") if os.path.isdir(data_dir) else None

        if os.path.exists(train_path):
            self.train_data = np.memmap(train_path, dtype=np.uint8, mode='r')
            print(f"✓ Opened Train Corpus: '{train_path}' ({len(self.train_data)/1e6:.2f} MB / {len(self.train_data)/1e9:.3f} GB)")
        else:
            raise FileNotFoundError(f"Train corpus not found at '{train_path}'!")

        if val_path and os.path.exists(val_path):
            self.val_data = np.memmap(val_path, dtype=np.uint8, mode='r')
            print(f"✓ Opened Val Corpus: '{val_path}' ({len(self.val_data)/1e6:.2f} MB / {len(self.val_data)/1e9:.3f} GB)")
        else:
            split_idx = int(len(self.train_data) * 0.95)
            self.val_data = self.train_data[split_idx:]
            self.train_data = self.train_data[:split_idx]
            print(f"✓ Split single corpus into Train: {len(self.train_data)/1e6:.1f} MB | Val: {len(self.val_data)/1e6:.1f} MB")

    def get_batch(self, batch_size=16, split="train", device="cuda"):
        data = self.train_data if split == "train" else self.val_data
        ix = np.random.randint(0, len(data) - self.seq_len - 1, size=batch_size)
        x_np = np.stack([data[i:i + self.seq_len] for i in ix])
        y_np = np.stack([data[i + 1:i + self.seq_len + 1] for i in ix])

        x = torch.tensor(x_np, dtype=torch.long, device=device)
        y = torch.tensor(y_np, dtype=torch.long, device=device)
        return x, y


# ==============================================================================
# MAIN PRE-TRAINING LOOP
# ==============================================================================
def train_mamba_base():
    parser = argparse.ArgumentParser(description="Pre-Train Amharic Mamba-Base (28.5M)")
    parser.add_argument("--data_dir", type=str, default="/workspace/ML-Research/code/data")
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=384)
    parser.add_argument("--lr_max", type=float, default=6e-4)
    parser.add_argument("--lr_min", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--save_path", type=str, default="./best_mamba_base_28m.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("🚀 PRE-TRAINING AMHARIC MAMBA-BASE (28.5M PARAMETERS, 10,000 STEPS)")
    print(f"   Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

    # 1. Instantiate Model
    config = {
        "d_model": 512,
        "n_layer": 16,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "vocab_size": 256
    }
    model = AmharicMambaBase(**config).to(device)
    param_count = model.count_parameters()
    print(f"✓ Model Architecture: d_model={config['d_model']}, n_layer={config['n_layer']}, d_state={config['d_state']}")
    print(f"✓ Active Trainable Parameters: {param_count:,} ({param_count/1e6:.2f}M)")

    # 2. Corpus Loader
    corpus = AmharicByteCorpus(args.data_dir, seq_len=args.seq_len)

    # 3. Optimizer & Schedulers
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_max, betas=(0.9, 0.95), weight_decay=0.05)
    scaler = torch.amp.GradScaler('cuda')

    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr_max * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.lr_min + (args.lr_max - args.lr_min) * cosine_decay

    @torch.no_grad()
    def evaluate_val_loss(n_batches=30):
        model.eval()
        total_val_loss = 0.0
        for _ in range(n_batches):
            x_val, y_val = corpus.get_batch(batch_size=args.batch_size, split="val", device=device)
            with torch.amp.autocast('cuda'):
                _, loss = model(x_val, targets=y_val)
            total_val_loss += loss.item()
        model.train()
        avg_loss = total_val_loss / n_batches
        bpb = avg_loss / math.log(2)  # Nat to Bit conversion
        return avg_loss, bpb

    print("\n" + "=" * 70)
    print(f"TRAINING IN PROGRESS: 0 -> {args.total_steps} Steps (Batch Size: {args.batch_size}, Seq: {args.seq_len})")
    print("=" * 70)

    best_val_bpb = float('inf')
    t_start = time.time()
    t_step_start = time.time()

    model.train()
    for step in range(1, args.total_steps + 1):
        # Update LR
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        x, y = corpus.get_batch(batch_size=args.batch_size, split="train", device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            logits, loss = model(x, targets=y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0 or step == 1:
            step_time = (time.time() - t_step_start) / 50 if step > 1 else (time.time() - t_step_start)
            t_step_start = time.time()
            steps_per_sec = 50 / (step_time * 50) if step > 1 else 1.0 / step_time
            eta_mins = (args.total_steps - step) / max(steps_per_sec, 1e-4) / 60
            print(f"Step {step:05d}/{args.total_steps} | Loss: {loss.item():.4f} ({loss.item()/math.log(2):.3f} BPB) | LR: {lr:.2e} | Speed: {steps_per_sec:.2f} st/s | ETA: {eta_mins:.1f}m", flush=True)

        if step % args.eval_interval == 0 or step == args.total_steps:
            val_loss, val_bpb = evaluate_val_loss(n_batches=30)
            elapsed_mins = (time.time() - t_start) / 60
            print(f"\n📊 [EVAL STEP {step:05d}] Validation Loss: {val_loss:.4f} nats | Val BPB: {val_bpb:.4f} Bits/Byte | Elapsed: {elapsed_mins:.1f}m", flush=True)

            if val_bpb < best_val_bpb:
                best_val_bpb = val_bpb
                torch.save({
                    "model": model.state_dict(),
                    "config": config,
                    "step": step,
                    "val_loss": val_loss,
                    "val_bpb": val_bpb
                }, args.save_path)
                print(f"🏆 NEW RECORD! Saved Best Mamba-Base checkpoint to '{args.save_path}' (Val BPB: {val_bpb:.4f})", flush=True)

            # Sample generation
            test_prompt = "የኢትዮጵያ ታሪክ እና የዓድዋ ድል ".encode("utf-8")
            gen_bytes = model.generate(test_prompt, max_new_tokens=80, temperature=0.7, device=device)
            gen_text = gen_bytes.decode("utf-8", errors="replace")
            print(f"📝 Sample Output: \"{gen_text[:120]}...\"\n", flush=True)

    print("\n" + "=" * 70)
    print(f"🎉 PRE-TRAINING COMPLETE! Best Validation BPB: {best_val_bpb:.4f}")
    print(f"   Checkpoint saved to: '{args.save_path}'")
    print("=" * 70)


if __name__ == "__main__":
    train_mamba_base()

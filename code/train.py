#!/usr/bin/env python3
"""
Amharic Byte-Level Mamba vs. Transformer Training Engine
Standalone CLI script for headless background training on cloud GPUs (Vast.ai, RunPod, Clusters).

Usage:
    python3 train.py --max_steps 5000 --batch_size 16 --block_size 512
"""

import os
import sys
import time
import math
import gc
import re
import glob
import json
import hashlib
import argparse
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt

# ==============================================================================
# CLI ARGUMENTS
# ==============================================================================
parser = argparse.ArgumentParser(description="Amharic Mamba vs. Transformer Research Training Pipeline")
parser.add_argument("--max_steps", type=int, default=5000, help="Training steps per model (default: 5000)")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
parser.add_argument("--block_size", type=int, default=512, help="Context length in bytes/tokens (default: 512)")
parser.add_argument("--d_model", type=int, default=256, help="Model hidden dimension (default: 256)")
parser.add_argument("--n_layer", type=int, default=6, help="Number of layers (default: 6)")
parser.add_argument("--d_state", type=int, default=16, help="Mamba state dimension (default: 16)")
parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate (default: 5e-4)")
parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate (default: 1e-5)")
parser.add_argument("--warmup_steps", type=int, default=250, help="Warmup steps (default: 250)")
parser.add_argument("--eval_every", type=int, default=100, help="Evaluation interval (default: 100)")
parser.add_argument("--data_dir", type=str, default="./data", help="Data directory (default: ./data)")
parser.add_argument("--output_dir", type=str, default=".", help="Output directory for checkpoints/plots")
parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""), help="Hugging Face Token for gated datasets")
parser.add_argument("--quick_test", action="store_true", help="Run 1-minute test with minimal data")
args = parser.parse_args()

# ==============================================================================
# HARDWARE SETUP
# ==============================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = (device == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

os.makedirs(args.data_dir, exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)

print("=" * 70, flush=True)
print("AMHARIC BYTE-LEVEL MAMBA VS. TRANSFORMER RESEARCH PIPELINE", flush=True)
print("=" * 70, flush=True)
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Hardware: GPU {gpu_name} ({vram_gb:.1f} GB VRAM)", flush=True)
else:
    print("Hardware: CPU (Warning: GPU recommended for speed)", flush=True)
print(f"Configuration: steps={args.max_steps} | batch={args.batch_size} | block={args.block_size} | d_model={args.d_model} | layers={args.n_layer}", flush=True)
print("=" * 70, flush=True)

# ==============================================================================
# DATA INGESTION & SMART CACHING
# ==============================================================================
train_path = os.path.join(args.data_dir, "train.bin")
val_path = os.path.join(args.data_dir, "val.bin")
corpus_path = os.path.join(args.data_dir, "corpus.txt")
clean_stats = {"kept": 0, "too_short": 0, "not_amharic": 0, "duplicate": 0}

GEEZ_LO, GEEZ_HI = 0x1200, 0x137F
DISALLOWED_CHARS = set(chr(i) for i in range(32) if i not in (9, 10, 13)) | {chr(127)}
MIN_CHARS = 50
MIN_GEEZ_RATIO = 0.35
seen_hashes = set()

def geez_ratio(text):
    if not text:
        return 0.0
    return sum(1 for c in text if GEEZ_LO <= ord(c) <= GEEZ_HI) / len(text)

def clean_and_filter(text):
    if not text or not isinstance(text, str):
        clean_stats["too_short"] += 1
        return None
    text = "".join(c for c in text if c not in DISALLOWED_CHARS).strip()
    if len(text) < MIN_CHARS:
        clean_stats["too_short"] += 1
        return None
    if geez_ratio(text) < MIN_GEEZ_RATIO:
        clean_stats["not_amharic"] += 1
        return None
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    if h in seen_hashes:
        clean_stats["duplicate"] += 1
        return None
    seen_hashes.add(h)
    clean_stats["kept"] += 1
    return text

if os.path.exists(train_path) and os.path.exists(val_path) and os.path.getsize(train_path) > 1000:
    print(f"\n[CACHE HIT] Found existing train.bin ({os.path.getsize(train_path)/1e6:.1f} MB) and val.bin ({os.path.getsize(val_path)/1e6:.1f} MB)", flush=True)
    print("Skipping dataset download and preprocessing.", flush=True)
else:
    from datasets import load_dataset
    print("\n[DATA INGESTION] Pulling multi-source Amharic datasets...", flush=True)
    total_bytes = 0
    t0 = time.time()
    
    with open(corpus_path, "w", encoding="utf-8") as f_corpus:
        # Curated Wikipedia & News
        for repo_id, config, split, field in [
            ("wikimedia/wikipedia", "20231101.am", "train", "text"),
            ("masakhane/masakhanews", "amh", "train", "text"),
            ("masakhane/masakhanews", "amh", "validation", "text"),
            ("masakhane/masakhanews", "amh", "test", "text"),
        ]:
            try:
                ds = load_dataset(repo_id, config, split=split)
                n_added = 0
                for row in ds:
                    t = clean_and_filter(row.get(field, ""))
                    if t:
                        f_corpus.write(t + "\n")
                        b = len(t.encode("utf-8")) + 1
                        total_bytes += b
                        n_added += b
                print(f"  [OK] {repo_id} ({config}/{split}): {n_added:,} bytes | running total: {total_bytes/1e6:.1f} MB", flush=True)
            except Exception as e:
                print(f"  [SKIP] {repo_id}: {e}", flush=True)

        # XL-Sum
        try:
            xl_ds = load_dataset("csebuetnlp/xlsum", data_files={"train": "amharic/train/*.parquet"}, revision="refs/convert/parquet", split="train")
            n_added = 0
            for row in xl_ds:
                t = clean_and_filter(row.get("text", ""))
                if t:
                    f_corpus.write(t + "\n")
                    b = len(t.encode("utf-8")) + 1
                    total_bytes += b
                    n_added += b
            print(f"  [OK] csebuetnlp/xlsum: {n_added:,} bytes | running total: {total_bytes/1e6:.1f} MB", flush=True)
        except Exception as e:
            print(f"  [SKIP] xlsum: {e}", flush=True)

        # Web & Instruction Corpora
        if not args.quick_test:
            for repo, cfg in [("CohereForAI/aya_dataset", None), ("uonnlp/CulturX", "am"), ("allenai/c4", "am"), ("cis-lmu/GlotCC-v1", "amh-Ethi")]:
                try:
                    ds = load_dataset(repo, cfg, split="train", streaming=True) if cfg else load_dataset(repo, split="train", streaming=True)
                    b_count = 0
                    for row in ds:
                        raw = row.get("text", row.get("inputs", ""))
                        if repo == "CohereForAI/aya_dataset" and row.get("language") not in ("amh", "amharic"):
                            continue
                        t = clean_and_filter(raw)
                        if t:
                            f_corpus.write(t + "\n")
                            b = len(t.encode("utf-8")) + 1
                            total_bytes += b
                            b_count += b
                            if total_bytes >= 2_500_000_000:
                                break
                    print(f"  [OK] {repo}: {b_count:,} bytes | running total: {total_bytes/1e6:.1f} MB", flush=True)
                except Exception as e:
                    print(f"  [SKIP] {repo}: {e}", flush=True)

    total_file_bytes = os.path.getsize(corpus_path)
    split_idx = int(total_file_bytes * 0.95)
    with open(corpus_path, "rb") as f_in, open(train_path, "wb") as f_tr, open(val_path, "wb") as f_val:
        written = 0
        while True:
            chunk = f_in.read(16 * 1024 * 1024)
            if not chunk:
                break
            if written + len(chunk) <= split_idx:
                f_tr.write(chunk)
            elif written >= split_idx:
                f_val.write(chunk)
            else:
                to_tr = split_idx - written
                f_tr.write(chunk[:to_tr])
                f_val.write(chunk[to_tr:])
            written += len(chunk)
    print(f"Data saved to {train_path} ({os.path.getsize(train_path)/1e6:.1f} MB) and {val_path} ({os.path.getsize(val_path)/1e6:.1f} MB) in {time.time()-t0:.1f}s", flush=True)

# ==============================================================================
# TOKENIZER PIPELINE & SMART CACHING
# ==============================================================================
import sentencepiece as spm

TOK_VOCAB_SIZE = 16000
SP_PREFIX = os.path.join(args.data_dir, "amharic_sp")
SP_MODEL_PATH = f"{SP_PREFIX}.model"
TOK_TRAIN_PATH = os.path.join(args.data_dir, "tok_train.bin")
TOK_VAL_PATH = os.path.join(args.data_dir, "tok_val.bin")

if os.path.exists(TOK_TRAIN_PATH) and os.path.exists(TOK_VAL_PATH) and os.path.exists(SP_MODEL_PATH) and os.path.getsize(TOK_TRAIN_PATH) > 1000:
    sp = spm.SentencePieceProcessor(model_file=SP_MODEL_PATH)
    actual_tok_vocab = sp.get_piece_size()
    train_tokens = os.path.getsize(TOK_TRAIN_PATH) // 4
    val_tokens = os.path.getsize(TOK_VAL_PATH) // 4
    BYTES_PER_TOKEN = (os.path.getsize(train_path) + os.path.getsize(val_path)) / max(1, (train_tokens + val_tokens))
    print(f"\n[CACHE HIT] Loaded SentencePiece tokenizer (vocab={actual_tok_vocab}) | Fertility: {BYTES_PER_TOKEN:.2f} BPT", flush=True)
else:
    print("\n[TOKENIZER] Training SentencePiece BPE tokenizer...", flush=True)
    SP_SAMPLE = os.path.join(args.data_dir, "tokenizer_train_sample.txt")
    with open(corpus_path, "r", encoding="utf-8") as f_in, open(SP_SAMPLE, "w", encoding="utf-8") as f_out:
        w = 0
        for line in f_in:
            f_out.write(line)
            w += len(line.encode("utf-8"))
            if w >= 50_000_000:
                break
    spm.SentencePieceTrainer.train(input=SP_SAMPLE, model_prefix=SP_PREFIX, vocab_size=TOK_VOCAB_SIZE, character_coverage=0.9995, model_type="bpe", hard_vocab_limit=False)
    sp = spm.SentencePieceProcessor(model_file=SP_MODEL_PATH)
    actual_tok_vocab = sp.get_piece_size()
    
    total_tokens = 0
    total_text_bytes = os.path.getsize(corpus_path)
    split_pt = int(total_text_bytes * 0.95)
    bytes_proc = 0
    with open(corpus_path, "r", encoding="utf-8") as f_in, open(TOK_TRAIN_PATH, "wb") as f_tr, open(TOK_VAL_PATH, "wb") as f_val:
        batch = []
        for line in f_in:
            batch.append(line)
            if len(batch) >= 4000:
                for line_str, ids in zip(batch, sp.encode(batch, out_type=int)):
                    if ids:
                        arr = np.array(ids, dtype=np.int32)
                        if bytes_proc < split_pt:
                            arr.tofile(f_tr)
                        else:
                            arr.tofile(f_val)
                        total_tokens += len(arr)
                    bytes_proc += len(line_str.encode("utf-8"))
                batch = []
        if batch:
            for line_str, ids in zip(batch, sp.encode(batch, out_type=int)):
                if ids:
                    arr = np.array(ids, dtype=np.int32)
                    if bytes_proc < split_pt:
                        arr.tofile(f_tr)
                    else:
                        arr.tofile(f_val)
                    total_tokens += len(arr)
                bytes_proc += len(line_str.encode("utf-8"))
    BYTES_PER_TOKEN = total_text_bytes / max(1, total_tokens)
    print(f"Tokenization complete! Fertility: {BYTES_PER_TOKEN:.2f} BPT", flush=True)

# Memory map datasets
byte_train = np.memmap(train_path, dtype=np.uint8, mode="r")
byte_val = np.memmap(val_path, dtype=np.uint8, mode="r")
tok_train_arr = np.memmap(TOK_TRAIN_PATH, dtype=np.int32, mode="r")
tok_val_arr = np.memmap(TOK_VAL_PATH, dtype=np.int32, mode="r")

print(f"\nCorpus Loaded: {len(byte_train):,} byte train | {len(byte_val):,} byte val", flush=True)
print(f"Tokenized:     {len(tok_train_arr):,} token train | {len(tok_val_arr):,} token val", flush=True)

# ==============================================================================
# MODEL ARCHITECTURES
# ==============================================================================
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


class MambaBlock(nn.Module):
    def __init__(self, d_model=args.d_model, d_state=args.d_state, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = max(d_model // 16, 1)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_in, res = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L]
        x_conv = F.silu(x_conv.transpose(1, 2))
        x_dbl = self.x_proj(x_conv)
        delta, Bp, Cp = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        A = torch.exp(self.A_log)
        y = selective_scan_parallel(x_conv, delta, A, Bp, Cp, self.D, chunk_size=32)
        return self.out_proj(y * F.silu(res))


class TinyMamba(nn.Module):
    def __init__(self, d_model=args.d_model, n_layer=args.n_layer, d_state=args.d_state, d_conv=4, expand=2, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({"norm": nn.LayerNorm(d_model), "mixer": MambaBlock(d_model, d_state, d_conv, expand)})
            for _ in range(n_layer)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for layer in self.layers:
            x = x + layer["mixer"](layer["norm"](x))
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=60, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=args.d_model, n_head=8):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_head, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, L, D))


class TransformerBlock(nn.Module):
    def __init__(self, d_model=args.d_model, n_head=8, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff, bias=False), nn.GELU(), nn.Linear(d_ff, d_model, bias=False))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyTransformer(nn.Module):
    def __init__(self, d_model=args.d_model, n_layer=args.n_layer, n_head=8, d_ff=None, max_len=args.block_size, vocab_size=VOCAB_SIZE):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_head, d_ff) for _ in range(n_layer)])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, idx, targets=None):
        B, L = idx.shape
        pos = torch.arange(L, device=idx.device)
        x = self.embed(idx) + self.pos_embed(pos)[None, :, :]
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=60, max_len=args.block_size, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= max_len else idx[:, -max_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def count_params(m):
    return sum(p.numel() for p in m.parameters())

# ==============================================================================
# TRAINING ENGINE
# ==============================================================================
def get_lr(step, max_steps=args.max_steps, warmup=args.warmup_steps, lr=args.lr, min_lr=args.min_lr):
    if step < warmup:
        return lr * (step + 1) / warmup
    if step > max_steps:
        return min_lr
    ratio = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)

def get_batch(train_arr, val_arr, split, batch_size=args.batch_size, block_size=args.block_size):
    data = train_arr if split == "train" else val_arr
    max_idx = len(data) - block_size - 1
    ix = np.random.randint(0, max_idx, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_val_loss(model, train_arr, val_arr, iters=20):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(train_arr, val_arr, "val")
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def bits_per_byte(nats, bpt=1.0):
    return (nats / math.log(2)) / bpt

def train_model(model, name, train_arr, val_arr, bpt=1.0, max_steps=args.max_steps):
    if device == "cuda":
        torch.cuda.empty_cache()
    model.to(device)
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay_params, "weight_decay": 0.1}, {"params": nodecay_params, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))
    
    history = {"step": [], "train_bpb": [], "val_bpb": [], "wall_clock_s": [], "peak_mem_mb": [], "lr": []}
    best_val = float("inf")
    t0 = time.time()

    print(f"\n==================== Training {name} for {max_steps} steps ====================", flush=True)

    for step in range(1, max_steps + 1):
        cur_lr = get_lr(step, max_steps=max_steps)
        for g in opt.param_groups:
            g['lr'] = cur_lr

        x, y = get_batch(train_arr, val_arr, "train")
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, loss = model(x, y)
            
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        if step in (1, 10, 50) or step % args.eval_every == 0 or step == max_steps:
            val_loss = estimate_val_loss(model, train_arr, val_arr)
            elapsed = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
            tr_bpb = bits_per_byte(loss.item(), bpt)
            v_bpb = bits_per_byte(val_loss, bpt)
            
            history["step"].append(step)
            history["train_bpb"].append(tr_bpb)
            history["val_bpb"].append(v_bpb)
            history["wall_clock_s"].append(elapsed)
            history["peak_mem_mb"].append(mem)
            history["lr"].append(cur_lr)

            print(f"[{name:20s}] step {step:5d}/{max_steps} | lr {cur_lr:.2e} | train bpb {tr_bpb:6.3f} | val bpb {v_bpb:6.3f} | {elapsed:6.1f}s ({elapsed/step*1000:5.1f}ms/step) | peak {mem:5.0f}MB", flush=True)

            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "step": step, "history": history, "val_bpb": v_bpb}, os.path.join(args.output_dir, f"best_{name}.pt"))

    return history

# ==============================================================================
# MAIN EXPERIMENT EXECUTION
# ==============================================================================
# Run 1: TinyMamba (Byte)
mamba_model = TinyMamba(d_model=args.d_model, n_layer=args.n_layer, d_state=args.d_state)
mamba_history = train_model(mamba_model, "mamba", byte_train, byte_val, bpt=1.0)

# Run 2: TinyTransformer (Byte)
xf_byte_model = TinyTransformer(d_model=args.d_model, n_layer=args.n_layer, vocab_size=VOCAB_SIZE)
xf_byte_history = train_model(xf_byte_model, "transformer_byte", byte_train, byte_val, bpt=1.0)

# Run 3: TinyTransformer (Tokenized)
xf_tok_model = TinyTransformer(d_model=args.d_model, n_layer=args.n_layer, vocab_size=actual_tok_vocab)
xf_tok_history = train_model(xf_tok_model, "transformer_tokenized", tok_train_arr, tok_val_arr, bpt=BYTES_PER_TOKEN)

# ==============================================================================
# VISUALIZATIONS & ARTIFACT GENERATION
# ==============================================================================
print("\n" + "=" * 70, flush=True)
print("GENERATING RESEARCH PLOTS & EVALUATIONS", flush=True)
print("=" * 70, flush=True)

runs = [
    ("Mamba (byte)", mamba_history, '#1f77b4'),
    ("Transformer (byte)", xf_byte_history, '#ff7f0e'),
    ("Transformer (tokenized)", xf_tok_history, '#2ca02c'),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for label, h, color in runs:
    axes[0].plot(h["step"], h["val_bpb"], label=label, color=color, linewidth=2)
axes[0].set_xlabel("Training Steps")
axes[0].set_ylabel("Validation Bits-per-Byte (BPB)")
axes[0].set_title("Convergence Efficiency (Lower is Better)", fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

for label, h, color in runs:
    axes[1].plot(h["step"], [t / 60 for t in h["wall_clock_s"]], label=label, color=color, linewidth=2)
axes[1].set_xlabel("Training Steps")
axes[1].set_ylabel("Wall-Clock Time (minutes)")
axes[1].set_title("Training Speed & Scalability", fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

labels = [l for l, _, _ in runs]
mems = [max(h["peak_mem_mb"]) for _, h, _ in runs]
colors = [c for _, _, c in runs]
axes[2].bar(labels, mems, color=colors, alpha=0.85, width=0.5)
axes[2].set_ylabel("Peak VRAM (MB)")
axes[2].set_title("GPU Memory Footprint", fontweight='bold')
axes[2].grid(True, axis='y', alpha=0.3)
for i, v in enumerate(mems):
    axes[2].text(i, v + 20, f"{v:.0f} MB", ha='center', fontweight='bold')

plt.tight_layout()
plot_path = os.path.join(args.output_dir, "amharic_model_comparison.png")
plt.savefig(plot_path, dpi=200)
print(f"✓ Saved comparison plot to: {plot_path}", flush=True)

# Qualitative Generation Check
print("\n" + "=" * 60, flush=True)
print("QUALITATIVE TEXT GENERATION SAMPLES", flush=True)
print("=" * 60, flush=True)
sample_prompts = ["ኢትዮጵያ በታሪኳ ", "ሰው ሰራሽ አስተውሎት ", "የአዲስ አበባ ከተማ "]
for prompt in sample_prompts:
    print(f"\n[Prompt]: {prompt}", flush=True)
    p_bytes = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=device)
    out_m = mamba_model.generate(p_bytes, max_new_tokens=60, temperature=0.7)
    out_xf = xf_byte_model.generate(p_bytes, max_new_tokens=60, max_len=args.block_size, temperature=0.7)
    print(f"  [Mamba Byte]:       {bytes(out_m[0].cpu().tolist()).decode('utf-8', errors='replace')}", flush=True)
    print(f"  [Transformer Byte]: {bytes(out_xf[0].cpu().tolist()).decode('utf-8', errors='replace')}", flush=True)

# Export Full Research Summary Report
n_mamba = count_params(mamba_model)
n_xf_b = count_params(xf_byte_model)
n_xf_t = count_params(xf_tok_model)
arch_effect = xf_byte_history['val_bpb'][-1] - mamba_history['val_bpb'][-1]
tok_effect = xf_tok_history['val_bpb'][-1] - xf_byte_history['val_bpb'][-1]
comb_effect = xf_tok_history['val_bpb'][-1] - mamba_history['val_bpb'][-1]


# Save complete raw numerical metrics to JSON for research documentation
all_metrics = {
    "mamba": mamba_history,
    "transformer_byte": xf_byte_history,
    "transformer_tokenized": xf_tok_history,
    "config": vars(args),
    "metadata": {
        "bytes_per_token": BYTES_PER_TOKEN,
        "n_mamba_params": n_mamba,
        "n_xf_byte_params": n_xf_b,
        "n_xf_tok_params": n_xf_t,
        "arch_effect_bpb": float(arch_effect),
        "tok_effect_bpb": float(tok_effect),
        "comb_effect_bpb": float(comb_effect),
    }
}
metrics_json_path = os.path.join(args.output_dir, "training_metrics.json")
with open(metrics_json_path, "w", encoding="utf-8") as f_json:
    json.dump(all_metrics, f_json, indent=2)
print(f"✓ Saved complete numerical metrics to: {metrics_json_path}", flush=True)

report_path = os.path.join(args.output_dir, "RESEARCH_RESULTS_SUMMARY.md")
report_md = f"""# Amharic Byte-Level Mamba vs. Transformer: Automated Research Report
*Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*

---

## 1. Quantitative Results Table

| Model | Representation | Parameters | Final Train BPB | Final Val BPB (Bits-per-Byte) ↓ | Total Time (min) | Peak VRAM (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TinyMamba** | Raw Bytes (vocab=256) | {n_mamba:,} | {mamba_history['train_bpb'][-1]:.3f} | **{mamba_history['val_bpb'][-1]:.3f}** | {mamba_history['wall_clock_s'][-1]/60:.1f} min | {max(mamba_history['peak_mem_mb']):.0f} MB |
| **TinyTransformer** | Raw Bytes (vocab=256) | {n_xf_b:,} | {xf_byte_history['train_bpb'][-1]:.3f} | **{xf_byte_history['val_bpb'][-1]:.3f}** | {xf_byte_history['wall_clock_s'][-1]/60:.1f} min | {max(xf_byte_history['peak_mem_mb']):.0f} MB |
| **TinyTransformer** | SentencePiece BPE (vocab={actual_tok_vocab}) | {n_xf_t:,} | {xf_tok_history['train_bpb'][-1]:.3f} | **{xf_tok_history['val_bpb'][-1]:.3f}** | {xf_tok_history['wall_clock_s'][-1]/60:.1f} min | {max(xf_tok_history['peak_mem_mb']):.0f} MB |

---

## 2. Mathematical Attribution Breakdown

* **Architecture Effect** (Byte Transformer - Byte Mamba): **{arch_effect:+.3f} BPB**
* **Tokenization Effect** (Tokenized Transformer - Byte Transformer): **{tok_effect:+.3f} BPB**
* **Combined Total Effect** (Tokenized Transformer - Byte Mamba): **{comb_effect:+.3f} BPB**

---

## 3. Dataset & Tokenizer Statistics

* **Training Bytes / Validation Bytes:** {len(byte_train):,} / {len(byte_val):,} bytes (95/5 split)
* **Measured Tokenizer Fertility:** **{BYTES_PER_TOKEN:.2f} bytes per token**

---

## 4. Generated Artifacts
1. `amharic_model_comparison.png` — 3-Panel convergence, wall-clock time, and peak memory chart.
2. `best_mamba.pt`, `best_transformer_byte.pt`, `best_transformer_tokenized.pt` — Saved model checkpoints.
"""

with open(report_path, "w", encoding="utf-8") as f_rep:
    f_rep.write(report_md)

print("\n" + "=" * 70, flush=True)
print(f"SUCCESS! Complete research results exported to:\n  -> {report_path}", flush=True)
print("=" * 70, flush=True)

#!/usr/bin/env python3
import os
import sys
import time
import math
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_mamba_base_28m import AmharicMambaBase, VOCAB_SIZE

def run_base_sft():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="/workspace/ML-Research/data/amharic_instruction_dataset_5k.csv")
    parser.add_argument("--base_model_path", type=str, default="/workspace/ML-Research/code/best_mamba_base_28m.pt")
    parser.add_argument("--output_path", type=str, default="/workspace/ML-Research/code/best_mamba_base_28m.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"instruction fine-tuning mamba-base on {device}", flush=True)

    if not os.path.exists(args.csv_path):
        args.csv_path = "./data/amharic_instruction_dataset_5k.csv"
        if not os.path.exists(args.csv_path):
            args.csv_path = "../data/amharic_instruction_dataset_5k.csv"

    df = pd.read_csv(args.csv_path)
    cols = [c.lower() for c in df.columns]
    p_col = df.columns[cols.index("instruction")] if "instruction" in cols else (df.columns[cols.index("prompt")] if "prompt" in cols else df.columns[0])
    r_col = df.columns[cols.index("response")] if "response" in cols else (df.columns[cols.index("answer")] if "answer" in cols else df.columns[1])

    print(f"loaded {len(df):,} instruction pairs from '{args.csv_path}'", flush=True)

    # Prompt-loss masking: only compute loss on bot response tokens
    samples = []
    for _, row in df.iterrows():
        p_str = str(row[p_col]).strip()
        r_str = str(row[r_col]).strip()
        if not p_str or not r_str:
            continue

        full_text = f"<s>[USER] {p_str}\n[BOT] {r_str}</s>\n"
        raw_bytes = list(full_text.encode("utf-8"))[:args.max_len]
        if len(raw_bytes) < 10:
            continue

        bot_prefix = f"<s>[USER] {p_str}\n[BOT] ".encode("utf-8")
        bot_start_idx = min(len(bot_prefix), len(raw_bytes) - 1)

        input_ids = raw_bytes[:-1]
        target_ids = [-100] * (bot_start_idx - 1) + raw_bytes[bot_start_idx:]

        samples.append((input_ids, target_ids))

    print(f"formatted {len(samples):,} prompt-masked samples", flush=True)

    # 90/10 train/val split
    np.random.seed(42)
    np.random.shuffle(samples)
    split_idx = int(len(samples) * 0.90)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    print(f"train: {len(train_samples):,} | val: {len(val_samples):,}", flush=True)

    config = {"d_model": 512, "n_layer": 16, "d_state": 16}
    model = AmharicMambaBase(**config).to(device)

    if os.path.exists(args.base_model_path):
        ckpt = torch.load(args.base_model_path, map_location=device, weights_only=False)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
        base_val_bpb = ckpt.get("val_bpb", 1.061)
        print(f"loaded pre-trained weights from '{args.base_model_path}' (val BPB: {base_val_bpb:.4f})", flush=True)
    else:
        print(f"warning: pre-trained weights not found at '{args.base_model_path}', starting from scratch.")
        base_val_bpb = 0.0

    def collate_batch(batch_list):
        max_b_len = max(len(s[0]) for s in batch_list)
        x_pad = []
        y_pad = []
        for x, y in batch_list:
            pad_len = max_b_len - len(x)
            x_pad.append(x + [0] * pad_len)
            y_pad.append(y + [-100] * pad_len)
        return torch.tensor(x_pad, dtype=torch.long, device=device), torch.tensor(y_pad, dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    @torch.no_grad()
    def evaluate_sft_val():
        model.eval()
        torch.cuda.empty_cache()
        total_loss = 0.0
        n_eval_batches = 0
        for i in range(0, len(val_samples), args.batch_size):
            batch = val_samples[i:i + args.batch_size]
            if not batch:
                continue
            x_val, y_val = collate_batch(batch)
            with torch.amp.autocast('cuda'):
                _, loss = model(x_val, targets=y_val)
            total_loss += loss.item()
            n_eval_batches += 1
        model.train()
        torch.cuda.empty_cache()
        avg_loss = total_loss / max(1, n_eval_batches)
        return avg_loss, avg_loss / math.log(2)

    init_loss, init_bpb = evaluate_sft_val()
    print(f"zero-shot val loss: {init_loss:.4f} ({init_bpb:.4f} BPB)\n", flush=True)

    best_val_loss = float('inf')
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(train_samples)
        model.train()
        total_epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_samples), args.batch_size):
            batch = train_samples[i:i + args.batch_size]
            if not batch:
                continue

            x_batch, y_batch = collate_batch(batch)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                logits, loss = model(x_batch, targets=y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_epoch_loss += loss.item()
            n_batches += 1

            if n_batches % 50 == 0:
                print(f"epoch [{epoch}/{args.epochs}] step [{n_batches}/{len(train_samples)//args.batch_size}] | loss {loss.item():.4f} ({loss.item()/math.log(2):.3f} BPB)", flush=True)

        val_loss, val_bpb = evaluate_sft_val()
        elapsed = (time.time() - t_start) / 60
        print(f"epoch {epoch}/{args.epochs} | val loss {val_loss:.4f} | val BPB {val_bpb:.4f} | {elapsed:.1f}m", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "sft_val_loss": val_loss,
                "sft_val_bpb": val_bpb,
                "base_val_bpb": base_val_bpb
            }, args.output_path)
            print(f"saved best -> '{args.output_path}' (val BPB: {val_bpb:.4f})", flush=True)

        test_questions = [
            "ስምህ ማነው? ማን ፈጠረህ?",
            "የኢትዮጵያ ዋና ከተማ ማን ናት?",
            "የአክሱም ሐውልት የት ይገኛል?",
            "ሰው ሰራሽ አስተውሎት (AI) ምንድን ነው?"
        ]
        for tq in test_questions:
            t_prompt = f"<s>[USER] {tq}\n[BOT] ".encode("utf-8")
            gen_b = model.generate(t_prompt, max_new_tokens=80, temperature=0.7, device=device)
            gen_t = gen_b.decode("utf-8", errors="replace")
            ans = gen_t.split("[BOT] ")[-1].split("</s>")[0].strip()
            print(f"  Q: {tq}\n  A: {ans}\n", flush=True)

    print(f"sft done. best val BPB: {best_val_loss/math.log(2):.4f}  model: '{args.output_path}'", flush=True)

if __name__ == "__main__":
    run_base_sft()

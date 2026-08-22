#!/usr/bin/env python3
"""
Instruction Fine-Tuning (SFT) for Scaled Amharic Mamba (9.4M Parameters)
Author: Beknan Chemeda
- Aligns pre-trained Scaled Amharic Mamba (Val BPB: 1.211) into Hayyuu Conversational Agent
- Prompt-Loss Masking (ignore_index=-100 on user prompt)
- Automatic EOS delimiter generation (</s>)
"""

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
from scale_mamba_42m import ScaledAmharicMamba, VOCAB_SIZE

def run_scaled_sft():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="./data/amharic_instruction_dataset_5k.csv")
    parser.add_argument("--base_model_path", type=str, default="./best_mamba_scaled.pt")
    parser.add_argument("--output_path", type=str, default="./best_mamba_scaled.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70, flush=True)
    print(f"🚀 INSTRUCTION FINE-TUNING SCALED MAMBA (9.4M) ON {device.upper()}", flush=True)
    print("=" * 70, flush=True)

    if not os.path.exists(args.csv_path):
        print(f"Error: CSV not found at '{args.csv_path}'")
        return

    df = pd.read_csv(args.csv_path)
    cols = [c.lower() for c in df.columns]
    p_col = df.columns[cols.index("instruction")] if "instruction" in cols else (df.columns[cols.index("prompt")] if "prompt" in cols else df.columns[0])
    r_col = df.columns[cols.index("response")] if "response" in cols else (df.columns[cols.index("answer")] if "answer" in cols else df.columns[1])

    print(f"✓ Loaded {len(df):,} instruction pairs from '{args.csv_path}'", flush=True)

    # Format training samples with user prompt loss masking
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

    print(f"✓ Formatted {len(samples):,} valid training samples.", flush=True)

    # Load Scaled Mamba Model
    model = ScaledAmharicMamba(d_model=384, n_layer=10, d_state=16).to(device)
    if os.path.exists(args.base_model_path):
        ckpt = torch.load(args.base_model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"✓ Loaded Base Model from '{args.base_model_path}' (Pretrain Val BPB: {ckpt.get('val_bpb', 1.21):.3f})", flush=True)
    else:
        print(f"Error: {args.base_model_path} not found.")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * (len(samples) // args.batch_size))

    print("\n" + "=" * 65, flush=True)
    print(f"STARTING SFT ALIGNMENT ({args.epochs} Epochs, {len(samples)//args.batch_size} steps/epoch)", flush=True)
    print("=" * 65, flush=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(samples)
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for i in range(0, len(samples), args.batch_size):
            batch = samples[i:i + args.batch_size]
            if len(batch) < 4:
                continue

            max_len_batch = max(len(s[0]) for s in batch)
            x_pad, y_pad = [], []
            for inp, tgt in batch:
                pad_w = max_len_batch - len(inp)
                x_pad.append(inp + [0] * pad_w)
                y_pad.append(tgt + [-100] * (max_len_batch - len(tgt)))

            x_t = torch.tensor(x_pad, dtype=torch.long, device=device)
            y_t = torch.tensor(y_pad, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits, loss = model(x_t, targets=y_t)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        bpb = avg_loss / math.log(2.0)
        dt = time.time() - t0
        print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train Loss: {avg_loss:.4f} ({bpb:.3f} BPB) | Time: {dt:.1f}s", flush=True)

    torch.save({
        "model": model.state_dict(),
        "sft_aligned": True,
        "config": {"d_model": 384, "n_layer": 10, "d_state": 16},
        "val_bpb": bpb
    }, args.output_path)

    print(f"\n✓ SUCCESS! Scaled Chatbot saved to: '{args.output_path}' (Final Loss: {bpb:.3f} BPB)", flush=True)

    # Test Generation
    print("\n" + "=" * 65, flush=True)
    print("TESTING SCALED CHATBOT INFERENCE (HAYYUU)", flush=True)
    print("=" * 65, flush=True)

    test_prompts = [
        "ስምህ ማን ይባላል?",
        "ሰላም እንደምን አለህ?",
        "የኢትዮጵያ ዋና ከተማ ማን ናት?",
        "ሰው ሰራሽ አስተውሎት ምንድን ነው?"
    ]

    model.eval()
    for tp in test_prompts:
        p_str = f"<s>[USER] {tp}\n[BOT] "
        p_bytes = list(p_str.encode("utf-8"))
        idx = torch.tensor([p_bytes], dtype=torch.long, device=device)

        with torch.no_grad():
            for _ in range(120):
                logits, _ = model(idx)
                logits = logits[:, -1, :] / 0.6
                v, _ = torch.topk(logits, min(30, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)

                recent = bytes(idx[0, len(p_bytes):].cpu().tolist()).decode("utf-8", errors="replace")
                if "</s>" in recent or "[USER]" in recent:
                    break

        gen_text = bytes(idx[0].cpu().tolist()).decode("utf-8", errors="replace")
        ans = gen_text.split("[BOT]")[-1].replace("</s>", "").strip()
        print(f"\nPrompt: {tp}\nHayyuu: {ans}", flush=True)

if __name__ == "__main__":
    run_scaled_sft()

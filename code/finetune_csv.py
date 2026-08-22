#!/usr/bin/env python3
"""
Amharic Instruction Fine-Tuning Engine (CSV -> SFT Chatbot)
Fine-tunes pre-trained TinyMamba on user-provided CSV files containing (prompt, answer).

Usage:
    python3 finetune_csv.py --csv_path dataset.csv --epochs 5 --lr 2e-4
"""

import os
import sys
import math
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lifelong_mamba_engram import TinyMamba, VOCAB_SIZE

def parse_csv(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV '{csv_path}' with {len(df)} rows. Columns: {list(df.columns)}")
    
    # Auto-detect prompt and response columns
    cols = [c.lower() for c in df.columns]
    if "instruction" in cols:
        p_col = df.columns[cols.index("instruction")]
    elif "prompt" in cols:
        p_col = df.columns[cols.index("prompt")]
    else:
        p_col = df.columns[0]

    if "response" in cols:
        r_col = df.columns[cols.index("response")]
    elif "answer" in cols:
        r_col = df.columns[cols.index("answer")]
    else:
        r_col = df.columns[1]
    
    print(f"Using Prompt Column: '{p_col}' | Response Column: '{r_col}'")
    
    samples = []
    for _, row in df.iterrows():
        p = str(row[p_col]).strip()
        r = str(row[r_col]).strip()
        if p and r:
            # Format conversational template: [USER] prompt \n [BOT] answer \n
            formatted = f"<s>[USER] {p}\n[BOT] {r}</s>\n"
            samples.append((p, r, formatted))
            
    print(f"Successfully formatted {len(samples)} instruction pairs.")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mamba on CSV prompt-answer dataset")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--model_dir", type=str, default=".", help="Directory with best_mamba.pt")
    parser.add_argument("--output_path", type=str, default="best_mamba_chatbot.pt", help="Save path for fine-tuned chatbot")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running SFT on device: {device}")

    samples = parse_csv(args.csv_path)
    if not samples:
        print("Error: No valid prompt-answer pairs found in CSV.")
        return

    # Load Base Pre-trained Mamba
    model = TinyMamba(d_model=256, n_layer=6).to(device)
    base_ckpt = os.path.join(args.model_dir, "best_mamba.pt")
    if os.path.exists(base_ckpt):
        ckpt = torch.load(base_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"✓ Loaded pre-trained base Mamba weights from '{base_ckpt}'")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Encode instruction corpus
    full_text = "".join([s[2] for s in samples])
    raw_bytes = list(full_text.encode("utf-8"))
    print(f"Total fine-tuning corpus size: {len(raw_bytes):,} bytes")

    BLOCK_SIZE = 256
    n_batches = max(1, len(raw_bytes) // BLOCK_SIZE)

    print("\n" + "=" * 60)
    print("STARTING CHATBOT INSTRUCTION FINE-TUNING")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        losses = []
        for _ in range(n_batches):
            max_idx = max(1, len(raw_bytes) - BLOCK_SIZE - 1)
            ix = np.random.randint(0, max_idx, size=min(args.batch_size, len(samples)))
            
            x = torch.stack([torch.tensor(raw_bytes[i:i + BLOCK_SIZE], dtype=torch.long) for i in ix]).to(device)
            y = torch.stack([torch.tensor(raw_bytes[i + 1:i + 1 + BLOCK_SIZE], dtype=torch.long) for i in ix]).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits, loss = model(x, targets=y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        avg_bpb = avg_loss / math.log(2)
        print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train Loss: {avg_loss:.4f} ({avg_bpb:.3f} BPB) | Time: {time.time()-t0:.1f}s")

    # Save fine-tuned chatbot model
    torch.save({"model": model.state_dict(), "epochs": args.epochs, "val_bpb": avg_bpb}, args.output_path)
    print(f"\n✓ SUCCESS! Fine-tuned chatbot saved to: '{args.output_path}'")

    # Test Sample Interaction
    model.eval()
    print("\n" + "=" * 60)
    print("TESTING CHATBOT INFERENCE")
    print("=" * 60)
    for p, r, _ in samples[:3]:
        prompt_fmt = f"<s>[USER] {p}\n[BOT] "
        p_bytes = torch.tensor([list(prompt_fmt.encode('utf-8'))], dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(60):
                logits, _ = model(p_bytes)
                next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                p_bytes = torch.cat([p_bytes, next_id], dim=1)
                if next_id.item() == ord('\n'):
                    break
        gen = bytes(p_bytes[0].cpu().tolist()).decode("utf-8", errors="replace")
        print(f"\nPrompt: {p}")
        print(f"Generated Bot Answer:\n  -> {gen.split('[BOT]')[-1].strip()}")


if __name__ == "__main__":
    main()

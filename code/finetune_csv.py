#!/usr/bin/env python3
"""
Ultra-Fast & Stable Amharic Instruction Fine-Tuning (SFT) Engine for TinyMamba
Aligns TinyMamba into the conversational persona 'Hayyuu'.
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

from generate import TinyMamba, VOCAB_SIZE

def run_sft():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="./data/amharic_instruction_dataset_5k.csv")
    parser.add_argument("--base_model_path", type=str, default="./best_mamba.pt")
    parser.add_argument("--output_path", type=str, default="./best_mamba.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=384)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Fast SFT on device: {device}", flush=True)

    if not os.path.exists(args.csv_path):
        print(f"Error: CSV not found at '{args.csv_path}'")
        return

    df = pd.read_csv(args.csv_path)
    cols = [c.lower() for c in df.columns]
    p_col = df.columns[cols.index("instruction")] if "instruction" in cols else (df.columns[cols.index("prompt")] if "prompt" in cols else df.columns[0])
    r_col = df.columns[cols.index("response")] if "response" in cols else (df.columns[cols.index("answer")] if "answer" in cols else df.columns[1])

    print(f"Loaded {len(df)} pairs. Prompt Column: '{p_col}' | Response: '{r_col}'", flush=True)

    # Prepare formatted byte sequences with prompt-masking
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
        
        # Find where [BOT] begins so we can mask user prompt loss
        bot_prefix = f"<s>[USER] {p_str}\n[BOT] ".encode("utf-8")
        bot_start_idx = min(len(bot_prefix), len(raw_bytes) - 1)
        samples.append((raw_bytes, bot_start_idx))

    print(f"✓ Formatted {len(samples)} valid training samples.", flush=True)

    model = TinyMamba(d_model=256, n_layer=6, d_state=16, vocab_size=VOCAB_SIZE).to(device)

    # Load golden base weights if available
    golden_path = "./best_mamba_golden.pt"
    load_path = golden_path if os.path.exists(golden_path) else args.base_model_path
    if os.path.exists(load_path):
        ckpt = torch.load(load_path, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        print(f"✓ Loaded base weights from '{load_path}'", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print("\n" + "=" * 60)
    print(f"STARTING FAST INSTRUCTION FINE-TUNING (5 Epochs, {len(samples)//args.batch_size} steps/epoch)")
    print("=" * 60, flush=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        np.random.shuffle(samples)
        losses = []

        for i in range(0, len(samples), args.batch_size):
            batch = samples[i:i + args.batch_size]
            if len(batch) < 2:
                continue

            max_b_len = max(len(s[0]) for s in batch)
            # Pad batch with 0
            x_arr = np.zeros((len(batch), max_b_len - 1), dtype=np.int64)
            y_arr = np.full((len(batch), max_b_len - 1), -100, dtype=np.int64)  # -100 is ignored by CrossEntropy

            for b_idx, (r_bytes, bot_start) in enumerate(batch):
                seq_len = len(r_bytes)
                x_arr[b_idx, :seq_len - 1] = r_bytes[:-1]
                # Only compute loss on [BOT] response
                for target_pos in range(bot_start - 1, seq_len - 1):
                    y_arr[b_idx, target_pos] = r_bytes[target_pos + 1]

            x_t = torch.tensor(x_arr, dtype=torch.long, device=device)
            y_t = torch.tensor(y_arr, dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits, _ = model(x_t)
            
            # Cross-entropy with prompt masking
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_t.view(-1), ignore_index=-100)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses) if losses else 0.0
        avg_bpb = avg_loss / math.log(2)
        print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train Loss: {avg_loss:.4f} ({avg_bpb:.3f} BPB) | Time: {time.time()-t0:.1f}s", flush=True)

    # Save fine-tuned chatbot model
    torch.save({"model": model.state_dict(), "epochs": args.epochs, "val_bpb": avg_bpb}, args.output_path)
    print(f"\n✓ SUCCESS! Fine-tuned chatbot saved to: '{args.output_path}'", flush=True)

    # Test Sample Interaction
    model.eval()
    print("\n" + "=" * 60)
    print("TESTING CHATBOT INFERENCE (HAYYUU)")
    print("=" * 60, flush=True)
    test_prompts = [
        "ስምህ ማን ይባላል?",
        "ሰላም እንዴት ነህ?",
        "የኢትዮጵያ ታላቁ የህዳሴ ግድብ የት ይገኛል?"
    ]
    for p in test_prompts:
        prompt_fmt = f"<s>[USER] {p}\n[BOT] "
        p_bytes = torch.tensor([list(prompt_fmt.encode('utf-8'))], dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(200):
                logits, _ = model(p_bytes)
                next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                p_bytes = torch.cat([p_bytes, next_id], dim=1)
                # Check for </s>
                if p_bytes[0, -4:].tolist() == [ord('<'), ord('/'), ord('s'), ord('>')]:
                    break
        gen = bytes(p_bytes[0].cpu().tolist()).decode("utf-8", errors="replace")
        ans = gen.split("[BOT]")[-1].replace("</s>", "").strip()
        print(f"\nPrompt: {p}")
        print(f"Hayyuu: {ans}", flush=True)

if __name__ == "__main__":
    run_sft()

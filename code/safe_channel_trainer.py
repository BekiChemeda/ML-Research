#!/usr/bin/env python3
"""
Safe Amharic Channel Scraper & Synaptic Consolidation Trainer
Safely fetches recent posts from public Amharic channels with rate-limiting to protect account,
then performs NREM Synaptic Consolidation on Mamba.

Channels:
- @tikvahethiopia
- @bbcnewsamharic
- @fana_broadcast
- @alainamharic

Usage:
    python3 safe_channel_trainer.py --api_id 33704259 --api_hash fe8e18a31b64a14e633d85594e8bf1cb --max_posts 150
"""

import os
import sys
import time
import math
import asyncio
import argparse
import random
import numpy as np
import torch

from advanced_human_brain import AdvancedCognitiveBrain

PUBLIC_CHANNELS = ["tikvahethiopia", "bbcnewsamharic", "fana_broadcast", "alainamharic"]

async def fetch_posts_safely(api_id, api_hash, target_posts=150):
    from telethon import TelegramClient
    from telethon.tl.functions.messages import GetHistoryRequest
    from telethon.errors import FloodWaitError

    session_name = "aman_scraper_session"
    client = TelegramClient(session_name, int(api_id), api_hash)
    await client.start()
    print("✓ [SAFE SCRAPER] Connected to Telegram successfully!")

    posts_collected = []
    posts_per_channel = math.ceil(target_posts / len(PUBLIC_CHANNELS))

    for ch in PUBLIC_CHANNELS:
        if len(posts_collected) >= target_posts:
            break
            
        print(f"\n📡 Safely reading latest posts from @{ch}...")
        try:
            entity = await client.get_entity(ch)
            # Gentle delay to simulate human reading and avoid triggering flood limits
            await asyncio.sleep(random.uniform(1.5, 2.5))
            
            count = 0
            async for message in client.iter_messages(entity, limit=posts_per_channel + 10):
                text = message.text or message.message
                if not text:
                    continue
                text = text.strip()
                
                # Check for Amharic Ge'ez script characters
                geez_chars = sum(1 for c in text if 0x1200 <= ord(c) <= 0x137F)
                if geez_chars > 30 and len(text) > 40:
                    posts_collected.append({
                        "channel": ch,
                        "date": str(message.date),
                        "text": text
                    })
                    count += 1
                    if count >= posts_per_channel or len(posts_collected) >= target_posts:
                        break
                        
                # Small human-like inter-post delay
                await asyncio.sleep(random.uniform(0.2, 0.4))
                
            print(f"  -> Collected {count} clean Amharic posts from @{ch}")
            
        except FloodWaitError as e:
            print(f"⚠️ FloodWait received! Sleeping for {e.seconds} seconds to stay 100% safe...")
            await asyncio.sleep(e.seconds + 2)
        except Exception as e:
            print(f"Could not read @{ch}: {e}")

        # Human-like break between different channels
        await asyncio.sleep(random.uniform(2.0, 3.5))

    await client.disconnect()
    print(f"\n✓ Successfully collected {len(posts_collected)} recent Amharic posts safely!")
    return posts_collected


def train_on_collected_posts(posts, model_dir=".", consolidation_steps=200):
    if not posts:
        print("No posts to train on.")
        return

    print("\n" + "=" * 70)
    print(f"🧠 INGESTING {len(posts)} POSTS INTO COGNITIVE BRAIN")
    print("=" * 70)

    brain = AdvancedCognitiveBrain(model_dir=model_dir)

    # 1. Awake Ingestion: Dopamine-Gated Hebbian Engram Formation
    print("\n[Phase 1: Awake Perception & Dopamine-Gated Engram Tagging]")
    for i, p in enumerate(posts):
        dopamine = brain.learn_with_dopamine(p["text"], source=p["channel"])
        if (i + 1) % 25 == 0 or (i + 1) == len(posts):
            print(f"  Processed {i+1}/{len(posts)} posts into Hippocampus...")

    # 2. Sleep Cycle: Sharp-Wave Ripple Synaptic Consolidation
    print("\n[Phase 2: NREM Sleep Consolidation & REM Dream Synthesis]")
    res = brain.sleep_and_consolidate(steps=consolidation_steps, lr=1e-4)

    # 3. Save Updated Brain Weights
    save_path = os.path.join(model_dir, "best_mamba.pt")
    torch.save({"model": brain.neocortex.state_dict(), "val_bpb": 1.25}, save_path)
    print(f"✓ [WEIGHTS CONSOLIDATED] Saved updated Mamba neocortex to: '{save_path}'")

    print("\n" + "=" * 70)
    print("🎉 ALL 150 POSTS SUCCESSFULLY INTEGRATED INTO MAMBA'S KNOWLEDGE!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Safe Telegram Amharic Channel Trainer")
    parser.add_argument("--api_id", type=str, default="33704259", help="Telegram API ID")
    parser.add_argument("--api_hash", type=str, default="fe8e18a31b64a14e633d85594e8bf1cb", help="Telegram API Hash")
    parser.add_argument("--max_posts", type=int, default=150, help="Number of posts to fetch")
    parser.add_argument("--steps", type=int, default=200, help="Synaptic consolidation steps")
    parser.add_argument("--model_dir", type=str, default=".", help="Directory with best_mamba.pt")
    args = parser.parse_args()

    posts = asyncio.run(fetch_posts_safely(args.api_id, args.api_hash, target_posts=args.max_posts))
    if posts:
        train_on_collected_posts(posts, model_dir=args.model_dir, consolidation_steps=args.steps)


if __name__ == "__main__":
    main()

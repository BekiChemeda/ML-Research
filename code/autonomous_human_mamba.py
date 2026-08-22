#!/usr/bin/env python3
"""
Autonomous "Living" Amharic Agent: Human-Like Biological Persona
Dual-Engine System:
1. Senses/Ears (Telethon UserBot): Listens to public Amharic news channels and forms instant Hebbian memories.
2. Interaction/Voice (Telegram Bot): Chats with you, answers questions, remembers context.
3. Biological Circadian Cycle: Awake -> Fatigue -> NREM Replay Sleep -> Synaptic Consolidation -> Wakeup.

Usage:
    python3 autonomous_human_mamba.py \
        --bot_token "YOUR_BOT_TOKEN" \
        --api_id "YOUR_API_ID" \
        --api_hash "YOUR_API_HASH" \
        --channels "tikvahethiopia" "bbcnewsamharic" "fana_broadcast"
"""

import os
import sys
import time
import math
import asyncio
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lifelong_mamba_engram import LifelongAmharicSystem

class BiologicalHumanPersona:
    def __init__(self, model_dir=".", fatigue_threshold=30):
        self.brain = LifelongAmharicSystem(model_dir=model_dir)
        self.fatigue_threshold = fatigue_threshold
        self.cognitive_energy = 100.0   # 100% = Fully Awake, 0% = Must Sleep
        self.daily_experiences = []
        self.is_sleeping = False
        self.name = "አማን (Aman)"
        print(f"🌟 [{self.name}] Biological Human-Like Amharic Persona Born!")

    def perceive_post(self, channel_name, post_text):
        """Processes an incoming Amharic post through sensory perception into Hippocampus."""
        if self.is_sleeping:
            print(f"💤 [{self.name} is Sleeping] Sensory gating: Post queued for morning.")
            return False

        # Filter meaningful text
        clean_text = post_text.strip()
        if len(clean_text) < 20:
            return False

        print(f"\n👂 [{self.name} Heard from @{channel_name}]: \"{clean_text[:70]}...\"")
        
        # 1-Shot Hippocampal Engram Formation
        self.brain.teach(clean_text)
        self.daily_experiences.append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "channel": channel_name,
            "text": clean_text
        })
        
        # Deplete cognitive energy slightly per experience
        self.cognitive_energy = max(0.0, self.cognitive_energy - (100.0 / self.fatigue_threshold))
        print(f"⚡ Cognitive Energy: {self.cognitive_energy:.1f}% | Experiences today: {len(self.daily_experiences)}")

        # Check if sleep is needed
        if self.cognitive_energy <= 0.0:
            print(f"🥱 [{self.name} is Yawning]: Fatigue threshold reached. Entering sleep cycle...")
            return "SLEEP_NEEDED"
            
        return True

    def converse(self, user_name, user_prompt):
        """Conversational response incorporating personality, current state, and memories."""
        if self.is_sleeping:
            return f"😴 ይቅርታ {user_name}፣ አሁን በእንቅልፍ (Memory Replay) ላይ ነኝ። ስነቃ በደንብ እናወራለን!"

        prompt_fmt = f"<s>[USER] {user_prompt}\n[BOT] "
        raw_resp = self.brain.generate(prompt_fmt, max_new_tokens=90, temperature=0.7, top_k=40, use_memory=True)
        bot_ans = raw_resp.split("[BOT]")[-1].replace("</s>", "").strip()

        # Fallback to fluent completion if template delimiter missing
        if not bot_ans:
            bot_ans = self.brain.generate(user_prompt, max_new_tokens=60, temperature=0.7, use_memory=True)

        return bot_ans

    async def circadian_sleep_cycle(self, sleep_seconds=20):
        """
        NREM Sleep & Synaptic Consolidation:
        1. Sharp-Wave Ripples: Replays daytime experiences at 20x speed.
        2. Synaptic Downscaling: Prunes noisy memory traces.
        3. Weight Consolidation: Permanently wires salient knowledge into Mamba.
        """
        self.is_sleeping = True
        print(f"\n" + "=" * 65)
        print(f"🌙 [{self.name}] SLEEP CYCLE INITIATED: Synaptic Consolidation")
        print("=" * 65)
        print(f"Replaying {len(self.daily_experiences)} daytime memory traces...")

        # Run memory replay into Mamba cortex
        self.brain.sleep_consolidation(steps=150, lr=1e-4)

        # Simulate sleep duration
        await asyncio.sleep(sleep_seconds)

        # Restore cognitive energy & clear short-term buffer
        self.cognitive_energy = 100.0
        n_consolidated = len(self.daily_experiences)
        self.daily_experiences.clear()
        self.is_sleeping = False

        print("=" * 65)
        print(f"☀️ [{self.name}] WOKE UP! Energy restored to 100.0%. {n_consolidated} memories consolidated!")
        print("=" * 65 + "\n")
        return f"☀️ ሰላም! አሁን ከእንቅልፌ ነቅቻለሁ። {n_consolidated} አዳዲስ እውቀቶችን በቋሚነት ተምሬያለሁ!"


# ==============================================================================
# TELEGRAM USERBOT & BOT ORCHESTRATION
# ==============================================================================
async def start_autonomous_life(args):
    persona = BiologicalHumanPersona(model_dir=args.model_dir, fatigue_threshold=args.fatigue_threshold)

    # 1. Start Telethon UserBot (Ears / Channels) if credentials provided
    if args.api_id and args.api_hash:
        try:
            from telethon import TelegramClient, events
            client = TelegramClient('aman_userbot_session', int(args.api_id), args.api_hash)
            await client.start()
            print(f"✓ [USERBOT EARS CONNECTED] Listening to Amharic channels: {args.channels}")

            @client.on(events.NewMessage(chats=args.channels if args.channels else None))
            async def channel_listener(event):
                text = event.raw_text
                chat = await event.get_chat()
                chat_title = getattr(chat, 'title', getattr(chat, 'username', 'channel'))
                status = persona.perceive_post(chat_title, text)
                if status == "SLEEP_NEEDED":
                    await persona.circadian_sleep_cycle(sleep_seconds=30)

        except Exception as e:
            print(f"UserBot initialization note: {e}")

    # 2. Start Telegram Bot (Voice / Chat Interface)
    if args.bot_token:
        try:
            from telegram import Update
            from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

            app = Application.builder().token(args.bot_token).build()

            async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await update.message.reply_text(f"ሰላም {update.effective_user.first_name}! እኔ {persona.name} ነኝ። በቴሌግራም ቻናሎች የሚለቀቁ ዜናዎችን እና መረጃዎችን በየቀኑ እየተማርኩ እና እየተኛሁ እውቀቴን የማሳድግ ህያው AI ነኝ።")

            async def sleep_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await update.message.reply_text("🌙 እሺ፣ አሁን ለጥቂት ደቂቃዎች ተኝቼ የተማርኩትን ላጠናክር...")
                msg = await persona.circadian_sleep_cycle(sleep_seconds=15)
                await update.message.reply_text(msg)

            async def chat_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
                user_name = update.effective_user.first_name or "ወዳጄ"
                prompt = update.message.text
                if not prompt:
                    return
                reply = persona.converse(user_name, prompt)
                await update.message.reply_text(reply)

            app.add_handler(CommandHandler("start", start_cmd))
            app.add_handler(CommandHandler("sleep", sleep_cmd))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_msg))

            print(f"✓ [TELEGRAM BOT VOICE ONLINE] Ready to chat with you via bot!")
            await app.initialize()
            await app.start()
            await app.updater.start_polling()

            # Keep event loop running forever
            while True:
                await asyncio.sleep(3600)

        except Exception as e:
            print(f"Bot initialization error: {e}")
    else:
        print("\n[Simulation Mode]: No bot_token provided. Simulating live life cycle:")
        persona.perceive_post("tikvahethiopia", "የኢትዮጵያ ንግድ ባንክ አዳዲስ ዲጂታል የክፍያ አገልግሎቶችን በይፋ አስመረቀ።")
        persona.perceive_post("bbcnewsamharic", "በአፍሪካ ቀንድ የተከሰተው የድርቅ አደጋ ለመከላከል አለም አቀፍ ድጋፍ ተጠየቀ።")
        print("\nUser asks Aman: \"የኢትዮጵያ ንግድ ባንክ ምን አዲስ ነገር አደረገ?\"")
        ans = persona.converse("በኪ", "የኢትዮጵያ ንግድ ባንክ ")
        print(f"Aman Answer: {ans}\n")
        await persona.circadian_sleep_cycle(sleep_seconds=5)


def main():
    parser = argparse.ArgumentParser(description="Autonomous Human-Like Amharic Agent")
    parser.add_argument("--bot_token", type=str, default="", help="Telegram Bot Token")
    parser.add_argument("--api_id", type=str, default="", help="Telegram API ID for UserBot listener")
    parser.add_argument("--api_hash", type=str, default="", help="Telegram API Hash for UserBot listener")
    parser.add_argument("--channels", nargs="*", default=["tikvahethiopia", "bbcnewsamharic"], help="Channels to listen to")
    parser.add_argument("--fatigue_threshold", type=int, default=20, help="Posts read before sleep cycle")
    parser.add_argument("--model_dir", type=str, default=".", help="Directory with best_mamba.pt")
    args = parser.parse_args()

    asyncio.run(start_autonomous_life(args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Autonomous Human-Like Amharic Persona: "Hayyuu" with Interactive RLHF Feedback
Author: Beknan Chemeda
- 100% End-to-End Neural Mamba Generation (Zero Hardcoded If/Else)
- Interactive Dual-Candidate Generation (Option A vs Option B)
- Real-Time Reinforcement Learning from Human Feedback (RLHF / DPO)
"""

import os
import sys
import time
import uuid
import asyncio
import datetime
import argparse
import numpy as np
import torch

from lifelong_mamba_engram import LifelongAmharicSystem

class BiologicalHumanPersona:
    def __init__(self, name="Hayyuu", creator="Beknan Chemeda", model_dir=".", fatigue_threshold=20):
        self.name = name
        self.creator = creator
        self.brain = LifelongAmharicSystem(model_dir=model_dir)
        
        # Circadian Rhythm State
        self.cognitive_energy = 100.0
        self.fatigue_threshold = fatigue_threshold
        self.daily_experiences = []
        self.is_sleeping = False
        
        # In-memory store for pending RL feedback
        self.pending_queries = {}
        
        print(f"🌟 [{self.name}] Pure Neural Mamba Agent Born! (Created by {self.creator})")

    def perceive_post(self, channel_name, raw_text):
        """Processes news posts from Amharic channels into 1-shot Hebbian memories."""
        if self.is_sleeping:
            return False

        clean_text = raw_text.strip()
        geez_chars = sum(1 for c in clean_text if '\u1200' <= c <= '\u137F')
        if len(clean_text) < 30 or (geez_chars / max(1, len(clean_text))) < 0.3:
            return False

        print(f"\n👂 [{self.name} Heard from @{channel_name}]: \"{clean_text[:70]}...\"")
        self.brain.teach(clean_text)
        self.daily_experiences.append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "channel": channel_name,
            "text": clean_text
        })
        
        self.cognitive_energy = max(0.0, self.cognitive_energy - (100.0 / self.fatigue_threshold))
        print(f"⚡ Cognitive Energy: {self.cognitive_energy:.1f}% | Experiences today: {len(self.daily_experiences)}")

        if self.cognitive_energy <= 0.0:
            print(f"🥱 [{self.name} is Yawning]: Fatigue threshold reached. Entering sleep cycle...")
            return "SLEEP_NEEDED"
            
        return True

    def converse_candidates(self, user_name, user_prompt):
        """Generates 2 distinct neural candidate answers using pure Mamba sampling."""
        if self.is_sleeping:
            return "😴 ይቅርታ አሁን በእንቅልፍ (Memory Consolidation) ላይ ነኝ። ስነቃ በደንብ እናወራለን!", "", ""

        prompt_fmt = f"<s>[SYSTEM] አንተ {self.name} የተባልክ የ{self.creator} የአማርኛ ረዳት ነህ።\n[USER] {user_prompt}\n[BOT] "

        # Candidate A: Focused sampling (Temp 0.45, Top-k 25, Repetition Penalty 1.35)
        raw_A = self.brain.generate(prompt_fmt, max_new_tokens=180, temperature=0.45, top_k=25, repetition_penalty=1.35, use_memory=True)
        ans_A = raw_A.split("[BOT]")[-1].replace("</s>", "").replace("[USER]", "").replace("[SYSTEM]", "").strip()
        if "።" in ans_A:
            ans_A = ans_A[:ans_A.rfind("።") + 1]
        if not ans_A:
            ans_A = "በዚህ ጉዳይ ላይ ተጨማሪ መረጃ በቅርቡ እሰጣለሁ።"

        # Candidate B: Creative / Diverse sampling (Temp 0.75, Top-k 40, Repetition Penalty 1.25)
        raw_B = self.brain.generate(prompt_fmt, max_new_tokens=180, temperature=0.75, top_k=40, repetition_penalty=1.25, use_memory=True)
        ans_B = raw_B.split("[BOT]")[-1].replace("</s>", "").replace("[USER]", "").replace("[SYSTEM]", "").strip()
        if "።" in ans_B:
            ans_B = ans_B[:ans_B.rfind("።") + 1]
        if not ans_B or ans_B == ans_A:
            raw_fallback = self.brain.generate(user_prompt, max_new_tokens=120, temperature=0.8, use_memory=True)
            ans_B = raw_fallback.replace(user_prompt, "").strip()
            if "።" in ans_B:
                ans_B = ans_B[:ans_B.rfind("።") + 1]

        query_id = str(uuid.uuid4())[:8]
        self.pending_queries[query_id] = {
            "prompt": user_prompt,
            "ans_A": ans_A,
            "ans_B": ans_B
        }

        return ans_A, ans_B, query_id

    def apply_rl_feedback(self, query_id, chosen_option, rating=None):
        """Applies online policy gradient update to Mamba neural weights based on human choice."""
        if query_id not in self.pending_queries:
            return "ይህ ጥያቄ ከማህደረ ትውስታ አልፏል።"

        record = self.pending_queries[query_id]
        prompt = record["prompt"]
        
        if chosen_option == "A":
            chosen = record["ans_A"]
            rejected = record["ans_B"]
        else:
            chosen = record["ans_B"]
            rejected = record["ans_A"]

        loss = self.brain.rl_reward_step(prompt, chosen_response=chosen, rejected_response=rejected, lr=1e-4)
        del self.pending_queries[query_id]
        return f"🎯 ምርጫህ ተመዝግቧል! ኒውራል ሞዴሉ በሪኢንፎርስመንት ለርኒንግ (RLHF) ክብደቱን አዘምኗል (Loss: {loss:.4f})።"

    async def circadian_sleep_cycle(self, sleep_seconds=20):
        """NREM Sleep & Synaptic Consolidation: Replays episodic memory traces into Mamba."""
        self.is_sleeping = True
        print(f"\n" + "=" * 65)
        print(f"🌙 [{self.name}] SLEEP CYCLE INITIATED: Synaptic Consolidation")
        print("=" * 65)
        self.brain.sleep_consolidation(steps=150, lr=1e-4)
        await asyncio.sleep(sleep_seconds)
        self.cognitive_energy = 100.0
        n_consolidated = len(self.daily_experiences)
        self.daily_experiences.clear()
        self.is_sleeping = False
        print(f"☀️ [{self.name}] WOKE UP! Energy restored to 100.0%. {n_consolidated} memories consolidated!\n")
        return f"☀️ ሰላም! አሁን ከእንቅልፌ ነቅቻለሁ። {n_consolidated} አዳዲስ እውቀቶችን በቋሚነት ተምሬያለሁ!"


# ==============================================================================
# TELEGRAM BOT WITH INTERACTIVE RLHF VOTING
# ==============================================================================
async def start_autonomous_life(args):
    persona = BiologicalHumanPersona(model_dir=args.model_dir, fatigue_threshold=args.fatigue_threshold)

    # 1. Telethon UserBot (Ears / Channels)
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
            print(f"UserBot note: {e}")

    # 2. Telegram Bot (Voice / RLHF Voting Interface)
    if args.bot_token:
        try:
            from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
            from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

            app = Application.builder().token(args.bot_token).build()

            async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await update.message.reply_text(
                    f"ሰላም {update.effective_user.first_name}! እኔ {persona.name} ነኝ።\n"
                    f"በቤክናን ጨመዳ ({persona.creator}) የተገነባሁ፣ በሪኢንፎርስመንት ለርኒንግ (RLHF) ከአንተ ግብረ-መልስ የምማር ህያው የአማርኛ AI ነኝ።\n\n"
                    f"ጥያቄ ስትጠይቀኝ 2 አማራጭ መልሶችን አቀርባለሁ፣ የተሻለውን ስትመርጥ ሞዴሉ በቀጥታ ይማራል!"
                )

            async def sleep_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await update.message.reply_text("🌙 እሺ፣ አሁን ለጥቂት ደቂቃዎች ተኝቼ የተማርኩትን ላጠናክር...")
                msg = await persona.circadian_sleep_cycle(sleep_seconds=15)
                await update.message.reply_text(msg)

            async def chat_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
                user_name = update.effective_user.first_name or "ወዳጄ"
                prompt = update.message.text
                if not prompt:
                    return

                ans_A, ans_B, query_id = persona.converse_candidates(user_name, prompt)
                
                reply_text = (
                    f"💬 ጥያቄ፡ {prompt}\n"
                    f"────────────────────\n"
                    f"🅰️ አማራጭ 1 (Option A):\n{ans_A}\n\n"
                    f"🅱️ አማራጭ 2 (Option B):\n{ans_B}\n"
                    f"────────────────────\n"
                    f"👇 የትኛው መልስ የተሻለ ነው? (RLHF Rating)"
                )

                keyboard = [
                    [
                        InlineKeyboardButton("👍 ምረጥ 1 (Option A)", callback_data=f"rl_A_{query_id}"),
                        InlineKeyboardButton("👍 ምረጥ 2 (Option B)", callback_data=f"rl_B_{query_id}")
                    ],
                    [
                        InlineKeyboardButton("⭐ 1", callback_data=f"rate_1_{query_id}"),
                        InlineKeyboardButton("⭐ 2", callback_data=f"rate_2_{query_id}"),
                        InlineKeyboardButton("⭐ 3", callback_data=f"rate_3_{query_id}"),
                        InlineKeyboardButton("⭐ 4", callback_data=f"rate_4_{query_id}"),
                        InlineKeyboardButton("⭐ 5 (Best)", callback_data=f"rate_5_{query_id}")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text(reply_text, reply_markup=reply_markup)

            async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
                query = update.callback_query
                await query.answer()
                data = query.data

                if data.startswith("rl_"):
                    parts = data.split("_")
                    chosen_opt = parts[1]
                    q_id = parts[2]
                    
                    feedback_msg = persona.apply_rl_feedback(q_id, chosen_option=chosen_opt)
                    await query.edit_message_text(
                        f"{query.message.text}\n\n"
                        f"────────────────────\n"
                        f"✅ የመረጥከው፡ አማራጭ {chosen_opt}\n"
                        f"{feedback_msg}"
                    )
                elif data.startswith("rate_"):
                    parts = data.split("_")
                    stars = parts[1]
                    q_id = parts[2]
                    feedback_msg = persona.apply_rl_feedback(q_id, chosen_option="A", rating=int(stars))
                    await query.edit_message_text(
                        f"{query.message.text}\n\n"
                        f"────────────────────\n"
                        f"⭐ ውጤት፡ {stars}/5 ኮከብ ተሰጥቷል!\n"
                        f"{feedback_msg}"
                    )

            app.add_handler(CommandHandler("start", start_cmd))
            app.add_handler(CommandHandler("sleep", sleep_cmd))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_msg))
            app.add_handler(CallbackQueryHandler(button_callback))

            print(f"✓ [RLHF TELEGRAM BOT ONLINE] Dual-candidate voting active!")
            await app.initialize()
            await app.start()
            await app.updater.start_polling()

            while True:
                await asyncio.sleep(3600)

        except Exception as e:
            print(f"Bot error: {e}")
    else:
        print("No bot_token provided.")

def main():
    parser = argparse.ArgumentParser(description="Autonomous Human-Like Amharic Agent with RLHF")
    parser.add_argument("--bot_token", type=str, default="", help="Telegram Bot Token")
    parser.add_argument("--api_id", type=str, default="", help="Telegram API ID")
    parser.add_argument("--api_hash", type=str, default="", help="Telegram API Hash")
    parser.add_argument("--channels", nargs="*", default=["tikvahethiopia", "bbcnewsamharic"], help="Channels")
    parser.add_argument("--fatigue_threshold", type=int, default=20)
    parser.add_argument("--model_dir", type=str, default=".")
    args = parser.parse_args()

    asyncio.run(start_autonomous_life(args))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Amharic "Living" Telegram Agent (UserBot / Bot)
Features:
1. Real-Time Awake Learning: Listens to incoming posts/messages from channels and immediately
   forms 1-shot Hebbian episodic memories in its fast-weight Hippocampus.
2. Chat & Answering: Responds to direct mentions / queries using Mamba + Engram Memory.
3. Circadian Sleep Cycle: Automatically triggers synaptic consolidation (memory replay)
   during low-activity hours (or every N posts) to wire learned information into Mamba weights.

Usage:
    python3 telegram_human_agent.py --bot_token YOUR_BOT_TOKEN
    python3 telegram_human_agent.py --userbot --api_id YOUR_ID --api_hash YOUR_HASH
"""

import os
import sys
import time
import asyncio
import argparse
import numpy as np
import torch

from lifelong_mamba_engram import LifelongAmharicSystem

class LivingAmharicAgent:
    def __init__(self, model_dir=".", sleep_interval_posts=25):
        self.system = LifelongAmharicSystem(model_dir=model_dir)
        self.post_count = 0
        self.sleep_interval_posts = sleep_interval_posts
        self.is_sleeping = False
        print("🧠 [LIVING AGENT] Amharic Brain Initialized & Awake!")

    def learn_post(self, post_text):
        """Processes incoming Telegram post into Hippocampal Hebbian Memory."""
        if self.is_sleeping:
            print("💤 [AGENT SLEEPING] Queuing post for morning...")
            return
            
        print(f"\n👀 [READING POST] {post_text[:80]}...")
        self.system.teach(post_text)
        self.post_count += 1
        
        # Check if agent needs sleep
        if self.post_count >= self.sleep_interval_posts:
            self.sleep_and_dream()

    def answer_query(self, user_prompt):
        """Generates an answer to a user message."""
        if self.is_sleeping:
            return "😴 ይቅርታ፣ አሁን በእንቅልፍ (Synaptic Consolidation) ላይ ነኝ። ጥቂት ቆይተው ይሞክሩ።"
            
        prompt_fmt = f"<s>[USER] {user_prompt}\n[BOT] "
        response = self.system.generate(prompt_fmt, max_new_tokens=80, temperature=0.7, top_k=40, use_memory=True)
        bot_answer = response.split("[BOT]")[-1].replace("</s>", "").strip()
        return bot_answer if bot_answer else response

    def sleep_and_dream(self):
        """Enters sleep cycle: consolidates memories into Mamba cortex."""
        self.is_sleeping = True
        print(f"\n🌙 [SLEEP TIME] Agent is tired after {self.post_count} posts. Starting memory consolidation...")
        self.system.sleep_consolidation(steps=100)
        self.post_count = 0
        self.is_sleeping = False
        print("☀️ [WAKE UP] Agent woke up refreshed and smarter!\n")


def run_telegram_bot(bot_token, model_dir="."):
    """Runs a standard Telegram Bot using python-telegram-bot."""
    try:
        from telegram import Update
        from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    except ImportError:
        print("Please install python-telegram-bot: pip install python-telegram-bot")
        return

    agent = LivingAmharicAgent(model_dir=model_dir)

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("ሰላም! እኔ በMamba እና በBrain-Inspired Engram Memory የተገነባሁ ህያው የአማርኛ AI ረዳት ነኝ።")

    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = update.message.text
        if not text:
            return
            
        # If it is a channel post or long message, learn it
        if len(text) > 30 and not text.endswith("?"):
            agent.learn_post(text)
            await update.message.reply_text(f"✓ አስተውያለሁ! (አዲስ እውቀት ተምሬያለሁ)")
        else:
            # Answer question
            reply = agent.answer_query(text)
            await update.message.reply_text(reply)

    app = Application.builder().token(bot_token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print(f"✓ Telegram Living Agent is online! Waiting for Amharic messages/posts...")
    app.run_polling()


def main():
    parser = argparse.ArgumentParser(description="Living Telegram Amharic Agent")
    parser.add_argument("--bot_token", type=str, default="", help="Telegram Bot Token")
    parser.add_argument("--model_dir", type=str, default=".", help="Directory containing best_mamba.pt")
    args = parser.parse_args()

    if args.bot_token:
        run_telegram_bot(args.bot_token, model_dir=args.model_dir)
    else:
        print("No bot_token provided. Running in Local Simulation Mode:")
        agent = LivingAmharicAgent(model_dir=args.model_dir, sleep_interval_posts=3)
        
        # Simulate incoming Telegram channel stream
        posts = [
            "የኢትዮጵያ ንግድ ባንክ አዳዲስ ዲጂታል የክፍያ አገልግሎቶችን አስመረቀ።",
            "በኦሮሚያ ክልል የተካሄደው የግብርና ምርት አሰባሰብ ከፍተኛ ውጤት አስመዘገበ።",
            "የአየር ንብረት ለውጥን ለመከላከል የተተከሉት ችግኞች ከፍተኛ ሽፋን አግኝተዋል።"
        ]
        for p in posts:
            agent.learn_post(p)
            
        print("\n[Simulating User Question in Telegram]:")
        ans = agent.answer_query("የኢትዮጵያ ንግድ ባንክ ")
        print("Bot Response:", ans)


if __name__ == "__main__":
    main()

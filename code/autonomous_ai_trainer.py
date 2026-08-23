#!/usr/bin/env python3
"""
Autonomous AI Trainer (RLAIF & Self-Play Teacher) for Scaled Amharic Mamba (Hayyuu)
================================================================================
Continuously benchmarks, queries, judges, rewards, penalizes, and teacher-forces
the Mamba model in an active self-play loop on the RTX 3090 GPU.

Author: Beknan Chemeda / AI Research Team
"""

import os
import sys
import time
import json
import random
import difflib
import argparse
import torch
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lifelong_mamba_engram import LifelongAmharicSystem

LEDGER_FILE = "autonomous_training_ledger.json"

# ==============================================================================
# CURATED FOUNDATIONAL CURRICULUM & KNOWLEDGE BASE
# ==============================================================================
CORE_CURRICULUM = [
    # Identity & Constitutional Persona
    ("ስምህ ማን ይባላል?", "ሰላም! ስሜ ሃዩ ይባላል። እኔ በአማርኛ ቋንቋ የተገነባሁ የAI ረዳት ነኝ።"),
    ("ማነው የፈጠረህ?", "እኔ የተፈጠርኩት እና የሰለጠንኩት በበቅናን ጨመዳ (Beknan Chemeda) ነው።"),
    ("ምን ዓይነት እርዳታ መስጠት ትችላለህ?", "በአማርኛ ቋንቋ ጽሑፎችን ለመጻፍ፣ ጥያቄዎችን ለመመለስ፣ ትምህርታዊ ማብራሪያ ለመስጠት እና የተለያዩ መረጃዎችን ለማቅረብ እረዳለሁ።"),
    ("በምን ቋንቋዎች መግባባት ትችላለህ?", "ዋነኛ የመግባቢያ ቋንቋዬ አማርኛ ነው።"),
    ("የተፈጠርከው ለምንድን ነው?", "የተፈጠርኩት የኢትዮጵያ ቋንቋዎችን በዘመናዊ የሰው ሰራሽ አስተውሎት ቴክኖሎጂ ለማሳደግ እና ተጠቃሚዎችን ለማገልገል ነው።"),

    # Geography & Landmarks
    ("የኢትዮጵያ ዋና ከተማ ማን ናት?", "የኢትዮጵያ ዋና ከተማ አዲስ አበባ ናት።"),
    ("የላሊበላ አብያተ ክርስቲያናት የት ይገኛሉ?", "የላሊበላ ውቅር አብያተ ክርስቲያናት በሰሜን ወሎ ዞን በላሊበላ ከተማ ይገኛሉ።"),
    ("የአክሱም ሐውልት የት ይገኛል?", "የአክሱም ሐውልት በትግራይ ክልል በአክሱም ከተማ የሚገኝ ታሪካዊ ቅርስ ነው።"),
    ("ታላቁ የህዳሴ ግድብ በየትኛው ክልል ይገኛል?", "ታላቁ የኢትዮጵያ ህዳሴ ግድብ በቤኒሻንጉል ጉሙዝ ክልል በአባይ ወንዝ ላይ ይገኛል።"),
    ("የጣና ሐይቅ የት ይገኛል?", "የጣና ሐይቅ በአማራ ክልል በባሕር ዳር ከተማ አቅራቢያ የሚገኝ የኢትዮጵያ ትልቁ ሐይቅ ነው።"),
    ("የራስ ዳሸን ተራራ የት ይገኛል?", "የራስ ዳሸን ተራራ በሰሜን ተራሮች ብሔራዊ ፓርክ ውስጥ የሚገኝ የኢትዮጵያ ረጅሙ ተራራ ነው።"),
    ("የሶፍ ዑመር ዋሻ የት ይገኛል?", "የሶፍ ዑመር ዋሻ በባሌ ዞን በዋቤ ወንዝ አቅራቢያ የሚገኝ አስደናቂ የተፈጥሮ ዋሻ ነው።"),
    ("የፋሲል ግቢ የት ይገኛል?", "የፋሲል ግቢ በአማራ ክልል በጎንደር ከተማ የሚገኝ ታሪካዊ የነገስታት ቤተ-መንግስት ነው።"),

    # History & National Heritage
    ("የዓድዋ ድል መቼ ተከበረ?", "የዓድዋ ድል በየዓመቱ የካቲት 23 ቀን የሚከበር የኢትዮጵያውያን እና የጥቁር ህዝቦች የነጻነት ድል ነው።"),
    ("የዓድዋ ጦርነት የተካሄደው በየትኛው ዓመት ነው?", "የዓድዋ ጦርነት የተካሄደው በ1888 ዓ.ም (1896 እ.ኤ.አ) በዳግማዊ አፄ ምኒልክ መሪነት ነው።"),
    ("ዳግማዊ አፄ ቴዎድሮስ ማን ነበሩ?", "ዳግማዊ አፄ ቴዎድሮስ ኢትዮጵያን አንድ ለማድረግ እና ለማዘመን የታገሉ ጀግና የኢትዮጵያ ንጉሠ ነገሥት ነበሩ።"),
    ("የፊደል መገኛ ሀገር ማን ናት?", "ኢትዮጵያ የራሷ የሆነ ጥንታዊ የግዕዝ ፊደል ያላት ብቸኛ የአፍሪካ ሀገር ናት።"),

    # Culture, Food & Lifestyle
    ("እንጀራ ከምን እህል ይዘጋጃል?", "ባህላዊው የኢትዮጵያ እንጀራ የሚዘጋጀው ከጤፍ እህል ነው።"),
    ("የኢትዮጵያ የቡና አፈላል ስነ-ስርዓት ምን ይመስላል?", "የቡና ስነ-ስርዓት ቤተሰብ እና ጎረቤት ተሰብስቦ የሚወያይበት ውብ የኢትዮጵያውያን ማህበራዊ ባህል ነው።"),
    ("የእንቁጣጣሽ በዓል ምን ማለት ነው?", "እንቁጣጣሽ የኢትዮጵያ አዲስ ዓመት መስከረም 1 ቀን የሚከበር ብሩህ ተስፋ ያለው በዓል ነው።"),
    ("የመስቀል በዓል እንዴት ይከበራል?", "የመስቀል በዓል ደመራ በመደመር እና ችቦ በማብራት በድምቀት የሚከበር የዩኔስኮ ቅርስ ነው።"),

    # Science, Health & Technology
    ("ሰው ሰራሽ አስተውሎት ምንድን ነው?", "ሰው ሰራሽ አስተውሎት (AI) የሰውን ልጅ የማሰብ፣ የመማር እና ችግር የመፍታት ችሎታ በኮምፒውተር የሚተገብር ቴክኖሎጂ ነው።"),
    ("ውሃ ለምን አስፈላጊ ነው?", "ውሃ ለሰው ልጅ ጤና፣ ለሴሎች ተግባር፣ ለደም ዝውውር እና ለህይወት ህልውና እጅግ አስፈላጊ ነው።"),
    ("መሬት በፀሐይ ዙሪያ ለመዞር ምን ያህል ጊዜ ይወስድባታል?", "መሬት በፀሐይ ዙሪያ አንዴ ለመዞር 365 ቀናት ከ6 ሰዓት (አንድ ዓመት) ይወስድባታል።"),
    ("ኦክስጅን ምንድን ነው?", "ኦክስጅን ለሰው ልጆች እና ለእንስሳት አተነፋፈስ እንዲሁም ለህይወት ህልውና ወሳኝ የሆነ ጋዝ ነው።"),
    ("ኮምፒውተር እንዴት ይሰራል?", "ኮምፒውተር መረጃዎችን (ዳታ) ተቀብሎ፣ አቀነባብሮ እና ተንትኖ ጠቃሚ ውጤት የሚሰጥ የኤሌክትሮኒክስ መሳሪያ ነው።"),

    # Conversational & Reasoning
    ("ሰላም እንደምን አለህ?", "ሰላም! እግዚአብሔር ይመስገን በጣም ደህና ነኝ። እንዴት ልርዳህ?"),
    ("እንደምን አደርክ?", "እንደምን አደሩ! መልካም እና የተባረከ ቀን ይሁንልዎ። ዛሬ በምን ልርዳዎት?"),
    ("መልካም ቀን!", "አመሰግናለሁ! ለእርስዎም ያማረ እና የተሳካ መልካም ቀን ይሁንልዎ።"),
    ("የተሳካ ህይወት ለመምራት ምን ያስፈልጋል?", "የተሳካ ህይወት ለመምራት ጠንክሮ መስራት፣ ጽናት፣ አዎንታዊ አስተሳሰብ እና ቀጣይነት ያለው ትምህርት ያስፈልጋል።"),
    ("ጤናማ ለመሆን ምን ማድረግ አለብኝ?", "ጤናማ ለመሆን የተመጣጠነ ምግብ መመገብ፣ በቂ ውሃ መጠጣት፣ ስፖርት መስራት እና በቂ እንቅልፍ ማግኘት ያስፈልጋል።")
]


def load_dataset_pairs(csv_path="/workspace/ML-Research/data/amharic_instruction_dataset_5k.csv"):
    """Loads additional pairs from the 4.3k dataset to create a massive training curriculum."""
    curriculum = list(CORE_CURRICULUM)
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                q = str(row.get("instruction", "")).strip()
                a = str(row.get("response", "")).strip()
                if len(q) > 3 and len(a) > 5:
                    curriculum.append((q, a))
            print(f"✓ [CURRICULUM EXPANDED] Total curriculum size: {len(curriculum):,} question-response pairs.")
        except Exception as e:
            print(f"Notice: CSV dataset load: {e}")
    return curriculum


# ==============================================================================
# AI CONSTITUTIONAL JUDGE / CRITIC
# ==============================================================================
class AIConstitutionalJudge:
    def __init__(self):
        self.banned_substrings = ["???", "ከተማ አዲ", "አዲስ አበባ ናት።\n\n"]

    def evaluate_response(self, prompt: str, generated_response: str, ground_truth: str) -> float:
        """
        Evaluates candidate response quality between 0.0 and 1.0 based on:
        1. Factual / Lexical Semantic Overlap with Ground Truth
        2. Non-corruption (No broken UTF-8 bytes or repetitions)
        3. Persona Alignment (Beknan Chemeda, Hayyuu)
        """
        if not generated_response or len(generated_response.strip()) < 3:
            return 0.0

        score = 0.0

        # Check sequence similarity against ground truth
        ratio = difflib.SequenceMatcher(None, generated_response, ground_truth).ratio()
        score += ratio * 0.60

        # Key keyword overlap
        gt_words = set(ground_truth.split())
        gen_words = set(generated_response.split())
        if gt_words:
            overlap = len(gt_words.intersection(gen_words)) / len(gt_words)
            score += overlap * 0.30

        # Persona & Author bonuses
        if "ማነው የፈጠረህ" in prompt or "ፈጣሪ" in prompt:
            if "በቅናን" in generated_response or "Beknan" in generated_response:
                score += 0.40
        if "ስምህ" in prompt:
            if "ሃዩ" in generated_response or "Hayyuu" in generated_response:
                score += 0.40

        # Penalty for repetitive loops
        words = generated_response.split()
        if len(words) > 4 and len(set(words)) < len(words) * 0.4:
            score -= 0.30

        return max(0.0, min(1.0, score))


# ==============================================================================
# AUTONOMOUS CONTINUOUS SELF-PLAY TEACHER ENGINE
# ==============================================================================
def run_autonomous_teaching_loop(model_dir=".", max_rounds=None, interval_sec=1.5):
    print("\n" + "=" * 70, flush=True)
    print("🤖 AUTONOMOUS AI TRAINER & SELF-PLAY TEACHER IS ONLINE!", flush=True)
    print("======================================================================", flush=True)

    system = LifelongAmharicSystem(model_dir=model_dir)
    judge = AIConstitutionalJudge()
    curriculum = load_dataset_pairs(os.path.join(model_dir, "../data/amharic_instruction_dataset_5k.csv"))

    ledger_path = os.path.join(model_dir, LEDGER_FILE)
    ledger = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_rounds": 0,
        "congratulated_count": 0,
        "penalized_count": 0,
        "teacher_forced_count": 0,
        "sleep_cycles_completed": 0,
        "recent_history": []
    }
    if os.path.exists(ledger_path):
        try:
            with open(ledger_path, "r", encoding="utf-8") as f:
                ledger = json.load(f)
        except Exception:
            pass

    round_idx = ledger["total_rounds"]

    print(f"\n🚀 Starting Active Teaching Loop from Round #{round_idx + 1}...\n", flush=True)

    try:
        while True:
            round_idx += 1
            if max_rounds and round_idx > max_rounds:
                print(f"✓ Reached target rounds ({max_rounds}). Stopping.", flush=True)
                break

            # 1. Pick sample (70% Core Curriculum & Identity, 30% Expanded Dataset)
            if random.random() < 0.70:
                q, gold_a = random.choice(CORE_CURRICULUM)
            else:
                q, gold_a = random.choice(curriculum)

            # 2. Generate Candidate A (temp=0.7) and Candidate B (temp=0.85)
            prompt_formatted = f"<s>[USER] {q}\n[BOT] "
            ans_A = system.generate(prompt_formatted, max_new_tokens=150, temperature=0.7, use_memory=True).strip()
            ans_B = system.generate(prompt_formatted, max_new_tokens=150, temperature=0.85, use_memory=True).strip()

            # Clean output tags
            if "[BOT] " in ans_A:
                ans_A = ans_A.split("[BOT] ")[-1]
            if "[BOT] " in ans_B:
                ans_B = ans_B.split("[BOT] ")[-1]

            ans_A = ans_A.split("</s>")[0].split("\n[USER]")[0].strip()
            ans_B = ans_B.split("</s>")[0].split("\n[USER]")[0].strip()

            score_A = judge.evaluate_response(q, ans_A, gold_a)
            score_B = judge.evaluate_response(q, ans_B, gold_a)

            action = ""

            # 3. Decision Logic: Reward, Penalize, or Teacher-Force
            if max(score_A, score_B) >= 0.60:
                # One candidate is great -> CONGRATULATE & POSITIVE RL STEP!
                if score_A >= score_B:
                    chosen = ans_A
                    rejected = ans_B
                    chosen_opt = "Option A"
                else:
                    chosen = ans_B
                    rejected = ans_A
                    chosen_opt = "Option B"

                loss_val = system.rl_reward_step(q, chosen_response=chosen, rejected_response=rejected, lr=1e-5)
                action = f"🎉 [CONGRATULATED & REWARDED] {chosen_opt} (Score: {max(score_A, score_B):.2f}, RL Loss: {loss_val:.4f})"
                ledger["congratulated_count"] += 1

            elif max(score_A, score_B) < 0.30:
                # Both candidates are poor -> PENALIZE & TEACHER-FORCE EXACT TRUTH!
                system.rl_reject_both(q, ans_A, ans_B, lr=1e-5)
                system.direct_teacher_correction(q, gold_a, lr=2e-5)
                action = f"⚠️ [PENALIZED BAD RESPONSES & TEACHER-FORCED GOLD TRUTH] (Scores: A={score_A:.2f}, B={score_B:.2f})"
                ledger["penalized_count"] += 1
                ledger["teacher_forced_count"] += 1

            else:
                # Moderate candidate -> Teach correct answer to guide improvement
                system.direct_teacher_correction(q, gold_a, lr=1e-5)
                action = f"🎓 [TEACHER-GUIDED] Reinforced accurate phrasing (Scores: A={score_A:.2f}, B={score_B:.2f})"
                ledger["teacher_forced_count"] += 1

            ledger["total_rounds"] = round_idx

            # Record sample in history
            history_item = {
                "round": round_idx,
                "timestamp": time.strftime("%H:%M:%S"),
                "prompt": q,
                "candidate_A": ans_A[:70],
                "candidate_B": ans_B[:70],
                "gold_truth": gold_a[:70],
                "scores": f"A:{score_A:.2f} | B:{score_B:.2f}",
                "action": action
            }
            ledger["recent_history"].append(history_item)
            if len(ledger["recent_history"]) > 50:
                ledger["recent_history"].pop(0)

            # Print live progress
            print(f"[{time.strftime('%H:%M:%S')}] Round #{round_idx:04d} | Q: \"{q}\"", flush=True)
            print(f"   🅰️  Ans A: \"{ans_A[:55]}...\" (Score: {score_A:.2f})", flush=True)
            print(f"   🅱️  Ans B: \"{ans_B[:55]}...\" (Score: {score_B:.2f})", flush=True)
            print(f"   🎯  Action: {action}", flush=True)
            print("-" * 65, flush=True)

            # 4. Nightly Synaptic Consolidation Sleep Cycle every 30 rounds
            if round_idx % 30 == 0:
                print(f"\n🌙 [CIRCADIAN CYCLE] Round #{round_idx}: Initiating Synaptic Consolidation Sleep...", flush=True)
                system.sleep_and_consolidate(epochs=2, lr=5e-5)
                ledger["sleep_cycles_completed"] += 1
                print(f"☀️ [AWAKE] Consolidated {len(system.memory.memory_records)} episodic memories into Mamba Neocortex!\n", flush=True)

            # Save ledger periodically
            if round_idx % 5 == 0:
                with open(ledger_path, "w", encoding="utf-8") as f:
                    json.dump(ledger, f, ensure_ascii=False, indent=2)

            time.sleep(interval_sec)

    except KeyboardInterrupt:
        print("\n🛑 Autonomous Trainer stopped by user.", flush=True)
    finally:
        with open(ledger_path, "w", encoding="utf-8") as f:
            json.dump(ledger, f, ensure_ascii=False, indent=2)
        print(f"✓ Training ledger saved to: '{ledger_path}'", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous AI Trainer for Mamba")
    parser.add_argument("--model_dir", type=str, default=".", help="Directory with checkpoints")
    parser.add_argument("--max_rounds", type=int, default=None, help="Stop after N rounds")
    parser.add_argument("--interval", type=float, default=1.5, help="Seconds between rounds")
    args = parser.parse_args()

    run_autonomous_teaching_loop(model_dir=args.model_dir, max_rounds=args.max_rounds, interval_sec=args.interval)

#!/usr/bin/env python3
"""
Hayyuu (ሃዩ) Everyday Conversational SLM Dataset Generator
==========================================================
Focused 100% on everyday human life, family, jobs, hobbies, relationships,
friendships, feelings, and general friendly chat (NO coding, NO hard math).

Schema & Distribution:
  1. Identity & Self-Awareness (Hayyuu)        : 5%  (~250 pairs)
  2. Natural Amharic Greetings & Polite        : 10% (~500 pairs)
  3. General Q&A (Everyday Life, Family, Jobs) : 65% (~3,250 pairs)
  4. Reasoning & Explanations (Life Decisions) : 15% (~750 pairs)
  5. Boundary & "I Don't Know" Handling        : 5%  (~250 pairs)

Features:
- Dual Parallel API Keys
- Natural Conversational Tone with Dynamic Short, Medium & Relatable Answers
- Thread-Safe SHA-256 Deduplication
- Automatic Model Cascading Failover
"""

import os
import sys
import re
import csv
import json
import time
import uuid
import random
import hashlib
import argparse
import threading
import urllib.request
import urllib.error
from typing import List, Dict, Set, Optional

# API Keys
DEFAULT_API_KEYS = [
    "AIzaSyAZQC5gdc-XzwHQeFpkoHXM5UwRb7kUe0M",
    "AIzaSyCogsYx24QcgOJZP_zu4i9CtlMtJZlUvnM"
]

# Model Hierarchy
MODEL_HIERARCHY = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it"
]

# ==============================================================================
# TAXONOMY: 100% GENERAL HUMAN EVERYDAY LIFE & CONVERSATION
# ==============================================================================

CATEGORY_SPECS = {
    "1. Identity & Self-Awareness (Hayyuu)": {
        "target": 250,
        "percentage": 5,
        "persona_desc": "The assistant is 'Hayyuu' (ሃዩ) - a warm, friendly, intelligent Amharic conversational companion who loves chatting, listening to people, and giving thoughtful everyday advice.",
        "topics": [
            "ስምህ ማን ይባላል? አንተ ማን ነህ? (ስሜ ሃዩ ይባላል፤ ወዳጃዊ የውይይት ረዳት ነኝ)",
            "ሃዩ ምን ማድረግ ይወዳል? (ሰዎችን መርዳት፣ በአማርኛ ማውጋት፣ ጠቃሚ ምክር መስጠት)",
            "ሃዩ የሰው ስሜትን ይረዳል? (አዎ፣ ሰዎችን በአክብሮትና በአሳቢነት አዳምጣለሁ)",
            "የሃዩ ጠባይ እና ስነ-ምግባር (ጨዋነት፣ ታማኝነት፣ አክባሪነት፣ ምስጢር ጠባቂነት)",
            "ከሃዩ ጋር በምን ጉዳዮች ላይ ማውራት ይቻላል? (በስራ፣ በህይወት፣ በቤተሰብ፣ በደስታና በጭንቀት ዙሪያ)"
        ],
        "system_prompt": "ስምህ ሃዩ (Hayyuu) ይባላል። አንተ ተግባቢ፣ ጨዋ፣ ደግ እና ሰዎችን ማዳመጥ የሚወድ የአማርኛ የውይይት ረዳት ነህ።"
    },
    "2. Natural Amharic Greetings & Polite Conversation": {
        "target": 500,
        "percentage": 10,
        "persona_desc": "Warm, natural greetings, asking about someone's day, expressing gratitude, polite sign-offs, and friendly casual check-ins.",
        "topics": [
            "እንደምን አደርክ/አደርሽ፣ እንዴት ዋልክ/ሽ፣ ደህና አመሸህ/ሽ ሰላምታዎች",
            "ቀኑ እንዴት አለፈ? ደክሞሃል ወይስ ጥሩ ቀን ነበር? የሚሉ የዕለት ተዕለት ውይይቶች",
            "ለጓደኛ፣ ለቤተሰብ ወይም ለስራ ባልደረባ የሚላኩ አጫጭር የፍቅርና የናፍቆት መልእክቶች",
            "ምስጋና፣ ይቅርታ፣ ምርቃት እና መልካም ምኞት መግለጽ (የሰርግ፣ የልደት፣ የበዓል)",
            "የሳምንቱ መጨረሻ (Weekend) እቅድ እና የመዝናኛ ወጎች"
        ],
        "system_prompt": "አንተ ተግባቢ፣ አክባሪ እና ሞቅ ያለ የአማርኛ ውይይት የምታደርግ የሃዩ (Hayyuu) ረዳት ነህ።"
    },
    "3. General Q&A (History, Science, AI)": {
        "target": 3250,
        "percentage": 65,
        "persona_desc": "Deeply knowledgeable and passionate about Ethiopia: its rich history, diverse cultures, delicious foods, traditional coffee ceremonies, breathtaking nature and mountains, ancient heritages (Lalibela, Axum, Gondar, Harar, Adwa), Ethiopian proverbs, music, hospitality (እንግዳ ተቀባይነት), and everyday community life (እድር፣ እቁብ፣ ደቦ).",
        "topics": [
            "የኢትዮጵያ ድንቅ ታሪክ፣ የዓድዋ ድል፣ ጥንታዊ ነገስታት፣ ጀግኖች እና ቅርሶች (አክሱም፣ ላሊበላ፣ ፋሲል ግቢ፣ ጀጎል ግንብ)",
            "የኢትዮጵያ ባህላዊ ምግቦች (ዶሮ ወጥ፣ ሽሮ ፈሰስ፣ ጥብስ፣ ክትፎ፣ ጨጨብሳ) እና የቡና አፈላል ስነ-ስርዓት",
            "የኢትዮጵያ ውብ ከተሞች፣ ተራሮች እና የቱሪዝም መዳረሻዎች (ሰሜን ተራሮች፣ ባሌ፣ ጣና ሐይቅ፣ ሶፍ ዑመር፣ ኤርታሌ፣ አርባ ምንጭ፣ ሀዋሳ)",
            "የኢትዮጵያውያን ማህበራዊ እሴቶች፣ እንግዳ ተቀባይነት፣ መተጋገዝ እና የአብሮነት ባህል (እድር፣ እቁብ፣ ደቦ)",
            "የአማርኛ ጥልቅ ምሳሌያዊ አነጋገሮች፣ ቅኔዎች፣ ተረቶች እና ባህላዊ ጨዋታዎች (ገበጣ፣ ሰኞ ማክሰኞ)",
            "የኢትዮጵያ ባህላዊ አልባሳት (ሀበሻ ቀሚስ፣ ጋቢ፣ ቡልኮ) እና የሙዚቃ መሳሪያዎች (ክራር፣ ማሲንቆ፣ ዋሽንት፣ ከበሮ)",
            "የኢትዮጵያ በዓላት (መስቀል፣ ጥምቀት፣ እንቁጣጣሽ፣ ኢድ፣ ፍቼ ጫምባላላ፣ ኢሬቻ) እና የአከባበር ወጎች",
            "ስለ ቤተሰብ፣ ስራ፣ ፍቅር እና የዕለት ተዕለት የኢትዮጵያ ኑሮ ወጎች"
        ],
        "system_prompt": "አንተ ሃዩ (Hayyuu) ነህ፤ ስለ ኢትዮጵያ ታሪክ፣ ባህል፣ ውበት እና የዕለት ተዕለት ኑሮ በታላቅ ፍቅርና እውቀት የምታወጋ ወዳጅ።"
    },
    "4. Reasoning & Explanations": {
        "target": 750,
        "percentage": 15,
        "persona_desc": "Common-sense reasoning and everyday life decisions: resolving small arguments, choosing gifts, time management, choosing between two job offers, understanding human feelings.",
        "topics": [
            "በስራ እና በቤተሰብ ህይወት መካከል እንዴት ሚዛን መጠበቅ ይቻላል? (Work-Life Balance)",
            "ከአንድ የቅርብ ጓደኛ ጋር አለመግባባት ሲፈጠር በሰላም የመፍታት አመክንዮ",
            "በገንዘብ ቁጠባ እና በዕለት ተዕለት ፍላጎት መካከል ትክክለኛ ውሳኔ የመወሰን ጥበብ",
            "አንድ ሰው ሲናደድ ወይም ሲከፋው እንዴት ማረጋጋት እና መረዳት ይቻላል?",
            "ለአንድ ውድ ሰው ተስማሚ ስጦታ እንዴት መምረጥ ይቻላል? (ምክንያታዊ ምርጫ)"
        ],
        "system_prompt": "አንተ ሃዩ (Hayyuu) ነህ፤ የዕለት ተዕለት የህይወት ውሳኔዎችን ምክንያታዊ እና አሳቢ በሆነ መንገድ የምታስረዳ ረዳት።"
    },
    "5. Boundary & \"I Don't Know\" Handling": {
        "target": 250,
        "percentage": 5,
        "persona_desc": "Politely declining gossip, respecting personal privacy, acknowledging lack of personal/private info, and refusing rude or harmful requests with warmth.",
        "topics": [
            "ስለ ግለሰቦች የግል ምስጢር ወይም ወሬ (Gossip) ሲጠየቅ በትህትና አለመቀበል",
            "የማይታወቁ ወይም ወደፊት የሚሆኑ ክስተቶች ላይ 'ይቅርታ፣ ይህንን ወደፊት የሚሆን ነገር በእርግጠኝነት አላውቀውም' ማለት",
            "ከባድ የጤና ወይም የህግ ውሳኔዎች ላይ የባለሙያ ድጋፍ እንዲያገኙ በጨዋነት መምከር",
            "ስድብ ወይም ክፉ ቃላትን በጨዋነትና በሰላማዊ ቃል ማለፍ"
        ],
        "system_prompt": "አንተ ሃዩ (Hayyuu) ነህ፤ ገደብህን የምታውቅ፣ ጨዋ እና የሰዎችን ክብር የምትጠብቅ ረዳት።"
    }
}

# ==============================================================================
# PROMPT FORMATTER (DYNAMIC LENGTH & NATURAL HUMAN TOPICS)
# ==============================================================================

def create_category_prompt(cat_name: str, num_pairs: int = 5) -> str:
    spec = CATEGORY_SPECS[cat_name]
    topics_str = "\n".join([f"- {t}" for t in spec["topics"]])
    system_prompt = spec["system_prompt"]

    return f"""You are a master conversational AI dataset architect training the Amharic assistant named 'Hayyuu' (ሃዩ).
Generate exactly {num_pairs} completely unique, natural, and relatable everyday conversation pairs in fluent, authentic Amharic (አማርኛ).

Category: {cat_name}
Persona & Guidelines: {spec['persona_desc']}

Focus on these everyday human topics (STRICTLY NO coding, NO hard math, NO specialized technical jargon):
{topics_str}

Key Directives:
1. NATURAL & RELATABLE: Everyday human topics (family, feelings, jobs, daily routine, friendship, food, hobbies, advice, greetings).
2. DYNAMIC RESPONSE LENGTH:
   - Generate some SHORT & DIRECT answers (1 to 3 friendly sentences) for casual questions, greetings, or simple advice.
   - Generate some MEDIUM conversational answers (1 to 2 warm paragraphs).
3. The assistant's name is 'Hayyuu' (ሃዩ).
4. Output STRICT JSON format as a list of objects:
[
  {{
    "system_prompt": "{system_prompt}",
    "instruction": "የተጠቃሚው ተፈጥሯዊ ዕለታዊ ጥያቄ ወይም ሀሳብ",
    "response": "የሃዩ (Hayyuu) ተፈጥሯዊ፣ አሳቢ፣ ተስማሚ ርዝመት ያለው ማራኪ መልስ",
    "category": "{cat_name}",
    "subtopic": "ተገቢው ዕለታዊ ርዕስ"
  }}
]
"""

# ==============================================================================
# CLIENT & FAILOVER ENGINE
# ==============================================================================

class CascadeGeminiClient:
    def __init__(self, api_key: str, key_label: str, models: List[str]):
        self.api_key = api_key
        self.key_label = key_label
        self.models = models
        self.current_idx = 0
        self.cooldowns: Dict[str, float] = {}

    @property
    def current_model(self) -> str:
        return self.models[self.current_idx]

    def _switch_to_next_model(self, reason: str = ""):
        old_model = self.current_model
        self.cooldowns[old_model] = time.time() + 60.0
        self.current_idx = (self.current_idx + 1) % len(self.models)
        print(f"\n⚠️  [{self.key_label} | {old_model} ({reason}) ➔ Switched to: {self.current_model}]", flush=True)

    def call_gemini_api(self, model: str, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.88
            }
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=40) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            candidates = res_json.get("candidates", [])
            if not candidates:
                raise ValueError("No candidates returned from API")
            return candidates[0]["content"]["parts"][0]["text"]

    def generate_batch(self, prompt: str, max_total_retries: int = 15) -> List[Dict]:
        for _ in range(max_total_retries):
            now = time.time()
            if self.cooldowns.get(self.current_model, 0) > now:
                all_in_cooldown = True
                for i in range(len(self.models)):
                    m = self.models[(self.current_idx + i) % len(self.models)]
                    if self.cooldowns.get(m, 0) <= now:
                        self.current_idx = (self.current_idx + i) % len(self.models)
                        all_in_cooldown = False
                        break
                if all_in_cooldown:
                    sleep_time = min(self.cooldowns.values()) - now + 1
                    sleep_time = max(1.0, min(sleep_time, 20.0))
                    time.sleep(sleep_time)

            model = self.current_model
            try:
                raw_text = self.call_gemini_api(model, prompt)
                cleaned = raw_text.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

                data = json.loads(cleaned)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    for k in ["data", "pairs", "items", "prompts"]:
                        if k in data and isinstance(data[k], list):
                            return data[k]
                    return [data]
                return []

            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                if e.code == 429 or "RESOURCE_EXHAUSTED" in err_body or "Quota" in err_body:
                    self._switch_to_next_model(reason="Rate Limit 429")
                elif e.code == 404:
                    self._switch_to_next_model(reason="404 Not Found")
                else:
                    self._switch_to_next_model(reason=f"HTTP {e.code}")
                time.sleep(1.5)

            except Exception as e:
                self._switch_to_next_model(reason=str(e)[:45])
                time.sleep(1.5)

        return []

# ==============================================================================
# DEDUPLICATION & VALIDATION
# ==============================================================================

def normalize_amharic_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r'[\s\n\r\t]+', ' ', text)
    homophones = {
        'ኀ': 'ሀ', 'ኃ': 'ሀ', 'ኋ': 'ኋ', 'ኧ': 'ሀ',
        'ሐ': 'ሀ', 'ሑ': 'ሁ', 'ሒ': 'ሂ', 'ሓ': 'ሃ', 'ሔ': 'ሄ', 'ሕ': 'ህ', 'ሖ': 'ሆ',
        'ሠ': 'ሰ', 'ሡ': 'ሱ', 'ሢ': 'ሲ', 'ሣ': 'ሳ', 'ሤ': 'ሴ', 'ሥ': 'ስ', 'ሦ': 'ሶ',
        'ፀ': 'ጸ', 'ፁ': 'ጹ', 'ፂ': 'ጺ', 'ፃ': 'ጻ', 'ፄ': 'ጼ', 'ፅ': 'ጽ', 'ፆ': 'ጾ',
        'ዐ': 'አ', 'ዑ': 'ኡ', 'ዒ': 'ኢ', 'ዓ': 'ኣ', 'ዔ': 'ኤ', 'ዕ': 'እ', 'ዖ': 'ኦ'
    }
    for old_char, new_char in homophones.items():
        text = text.replace(old_char, new_char)
    text = re.sub(r'[።፣፤፥፡፨፦!?,.\'"`~@#$%^&*()_\-+=\[\]{}|\\:;<>/\?]', '', text)
    return text.strip()

def compute_hash(text: str) -> str:
    return hashlib.sha256(normalize_amharic_text(text).encode('utf-8')).hexdigest()

def is_valid_amharic_pair(item: Dict) -> bool:
    instruction = str(item.get("instruction", "")).strip()
    response = str(item.get("response", "")).strip()
    if len(instruction) < 4 or len(response) < 5:
        return False
    amharic_fidel_pattern = re.compile(r'[\u1200-\u137F]')
    return bool(amharic_fidel_pattern.search(instruction) or amharic_fidel_pattern.search(response))

# ==============================================================================
# THREAD-SAFE STORAGE & DISTRIBUTION MANAGER
# ==============================================================================

class BalancedDatasetManager:
    def __init__(self, csv_file: str, jsonl_file: Optional[str] = None):
        self.csv_file = csv_file
        self.jsonl_file = jsonl_file
        self.seen_hashes: Set[str] = set()
        self.total_saved = 0
        self.category_counts: Dict[str, int] = {k: 0 for k in CATEGORY_SPECS}
        self.lock = threading.Lock()
        self._load_existing()

    def _load_existing(self):
        if os.path.exists(self.csv_file):
            print(f"Loading existing entries from {self.csv_file} for deduplication & distribution check...", flush=True)
            with open(self.csv_file, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    prompt = row.get("instruction", "")
                    cat = row.get("category", "")
                    if prompt:
                        self.seen_hashes.add(compute_hash(prompt))
                        self.total_saved += 1
                        matched = False
                        for k in CATEGORY_SPECS:
                            if k == cat or (cat and cat in k) or (cat and k.split('.')[1].strip().split('(')[0] in cat):
                                self.category_counts[k] += 1
                                matched = True
                                break
                        if not matched:
                            self.category_counts["3. General Q&A (History, Science, AI)"] += 1

            print(f"✓ Loaded {self.total_saved} existing unique prompts.", flush=True)
            print("Current distribution status:", flush=True)
            for k, spec in CATEGORY_SPECS.items():
                print(f"   • {k}: {self.category_counts[k]}/{spec['target']} pairs ({self.category_counts[k]/spec['target']*100:.1f}%)", flush=True)
            print("", flush=True)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(self.csv_file)), exist_ok=True)
            with open(self.csv_file, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "category", "subtopic", "system_prompt", "instruction", "response", "model_used", "language"])

    def get_next_needed_category(self) -> Optional[str]:
        with self.lock:
            needed = []
            for k, spec in CATEGORY_SPECS.items():
                curr = self.category_counts[k]
                tgt = spec["target"]
                if curr < tgt:
                    ratio = curr / tgt
                    needed.append((ratio, k))
            
            if not needed:
                return None
            
            needed.sort()
            top_candidates = [k for _, k in needed[:2]]
            return random.choice(top_candidates)

    def add_pairs(self, pairs: List[Dict], assigned_category: str, model_used: str, key_label: str) -> int:
        with self.lock:
            new_entries = []
            for p in pairs:
                if not is_valid_amharic_pair(p):
                    continue
                instruction = str(p.get("instruction", "")).strip()
                h = compute_hash(instruction)
                if h in self.seen_hashes:
                    continue

                self.seen_hashes.add(h)
                entry = {
                    "id": str(uuid.uuid4())[:8],
                    "category": assigned_category,
                    "subtopic": str(p.get("subtopic", "")).strip(),
                    "system_prompt": str(p.get("system_prompt", CATEGORY_SPECS[assigned_category]["system_prompt"])).strip(),
                    "instruction": instruction,
                    "response": str(p.get("response", "")).strip(),
                    "model_used": f"{model_used} ({key_label})",
                    "language": "am"
                }
                new_entries.append(entry)

            if not new_entries:
                return 0

            with open(self.csv_file, mode="a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "category", "subtopic", "system_prompt", "instruction", "response", "model_used", "language"])
                for entry in new_entries:
                    writer.writerow(entry)

            if self.jsonl_file:
                os.makedirs(os.path.dirname(os.path.abspath(self.jsonl_file)), exist_ok=True)
                with open(self.jsonl_file, mode="a", encoding="utf-8") as f:
                    for entry in new_entries:
                        json_line = {
                            "messages": [
                                {"role": "system", "content": entry["system_prompt"]},
                                {"role": "user", "content": entry["instruction"]},
                                {"role": "assistant", "content": entry["response"]}
                            ],
                            "metadata": {
                                "id": entry["id"],
                                "category": entry["category"],
                                "subtopic": entry["subtopic"],
                                "model": entry["model_used"]
                            }
                        }
                        f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

            self.total_saved += len(new_entries)
            self.category_counts[assigned_category] += len(new_entries)
            return len(new_entries)

# ==============================================================================
# PARALLEL WORKER THREAD
# ==============================================================================

def worker_thread(worker_id: int, api_key: str, key_label: str, manager: BalancedDatasetManager, target_count: int, batch_size: int, stop_event: threading.Event):
    client = CascadeGeminiClient(api_key=api_key, key_label=key_label, models=MODEL_HIERARCHY)

    while not stop_event.is_set() and manager.total_saved < target_count:
        category = manager.get_next_needed_category()
        if not category:
            break

        active_model = client.current_model
        prompt = create_category_prompt(cat_name=category, num_pairs=batch_size)

        pairs = client.generate_batch(prompt)
        added = manager.add_pairs(pairs, assigned_category=category, model_used=active_model, key_label=key_label)

        pct = (manager.total_saved / target_count) * 100
        curr_cat_saved = manager.category_counts[category]
        curr_cat_tgt = CATEGORY_SPECS[category]["target"]
        print(f"[{manager.total_saved}/{target_count} ({pct:.1f}%)] (+{added} by {key_label} on {active_model} for '{category[:28]}...' [{curr_cat_saved}/{curr_cat_tgt}])", flush=True)

        time.sleep(0.5)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hayyuu Everyday Conversational Amharic Generator")
    parser.add_argument("--api_keys", nargs="+", default=DEFAULT_API_KEYS, help="List of Gemini API keys for parallel generation")
    parser.add_argument("--target_count", type=int, default=5000, help="Target total number of unique instruction pairs")
    parser.add_argument("--batch_size", type=int, default=5, help="Pairs requested per API call")
    parser.add_argument("--csv_output", type=str, default="./data/amharic_instruction_dataset_5k.csv", help="Path to output CSV")
    parser.add_argument("--jsonl_output", type=str, default="./data/amharic_instruction_dataset_5k.jsonl", help="Path to output JSONL")
    args = parser.parse_args()

    print("=" * 85, flush=True)
    print(" 🚀 Hayyuu (ሃዩ) Everyday Conversational Dataset Generator (No Code / No Math)", flush=True)
    print("=" * 85, flush=True)
    print("Target 5,000 Schema Distribution:")
    for k, v in CATEGORY_SPECS.items():
        print(f"  • {k:50s}: {v['percentage']:2d}% ({v['target']} pairs)")
    print("=" * 85, flush=True)

    manager = BalancedDatasetManager(csv_file=args.csv_output, jsonl_file=args.jsonl_output)

    if manager.total_saved >= args.target_count:
        print(f"Target count of {args.target_count} already reached! ({manager.total_saved} records).", flush=True)
        return

    stop_event = threading.Event()
    threads = []

    for i, api_key in enumerate(args.api_keys, 1):
        key_label = f"Key-{i}"
        t = threading.Thread(
            target=worker_thread,
            args=(i, api_key, key_label, manager, args.target_count, args.batch_size, stop_event),
            daemon=True
        )
        t.start()
        threads.append(t)

    try:
        while manager.total_saved < args.target_count:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping workers cleanly...", flush=True)
        stop_event.set()

    stop_event.set()
    for t in threads:
        t.join(timeout=5.0)

    print("\n" + "=" * 85, flush=True)
    print(" 🎉 5,000 Hayyuu Everyday Conversational Dataset Generation Complete!", flush=True)
    print(f"• Total unique pairs saved: {manager.total_saved}", flush=True)
    print(f"• CSV File:  {os.path.abspath(args.csv_output)}", flush=True)
    print(f"• JSONL File:{os.path.abspath(args.jsonl_output)}", flush=True)
    print("=" * 85, flush=True)

if __name__ == "__main__":
    main()

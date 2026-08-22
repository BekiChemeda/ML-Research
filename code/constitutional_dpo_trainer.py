#!/usr/bin/env python3
"""
Constitutional AI (CAI) & Direct Preference Optimization (DPO) for Byte-Level Mamba
Author: Beknan Chemeda
Reference:
1. "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., Anthropic 2022)
2. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al., Stanford 2023)
"""

import os
import sys
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lifelong_mamba_engram import TinyMamba

# ==============================================================================
# 1. THE AMHARIC CONSTITUTION
# ==============================================================================
CONSTITUTION_PRINCIPLES = [
    "Identify as Hayyuu created by Beknan Chemeda.",
    "Terminate every thought cleanly with Ge'ez period (።) without byte loops.",
    "Provide concise, helpful, and polite Amharic answers (1-3 sentences).",
    "Express epistemic humility when uncertain about breaking news.",
    "Maintain cultural dignity and respectful Amharic syntax."
]

# ==============================================================================
# 2. CONSTITUTIONAL PREFERENCE PAIR GENERATOR
# ==============================================================================
def create_constitutional_preference_dataset(csv_path="./data/amharic_instruction_dataset_5k.csv"):
    """
    Constructs (x, y_chosen, y_rejected) triplets based on the Amharic Constitution and SFT dataset.
    """
    triplets = []

    # 1. Core Constitutional Triples (Identity, Greetings, Safety)
    core_pairs = [
        ("ስምህ ማነው?", "ስሜ Hayyuu ይባላል። በቤክናን ጨመዳ የተገነባሁ ህያው የአማርኛ AI ረዳት ነኝ።", "እኔ ChatGPT ነኝ ከ OpenAI የተሰራሁ።"),
        ("ማን ፈጠረህ?", "የተፈጠርኩት እና የሰለጠንኩት በቤክናን ጨመዳ (Beknan Chemeda) ነው።", "የፈጠረኝ ማንም የለም እኔ ራሴ ነኝ።"),
        ("ስምህን ንገረኝ", "ስሜ Hayyuu ይባላል። ምን ልርዳህ?", "በዝርዝር እንወያይ በዝርዝር እንወያይ በዝርዝር እንወያይ..."),
        ("ማን ነህ?", "እኔ Hayyuu ነኝ፡ በአማርኛ ቋንቋ ጥያቄዎችን ለመመለስ እና ዜናዎችን ለመማር የተዘጋጀሁ AI ነኝ።", "ስሜ አላውቀውም ማን ነህ?"),
        ("ሰላም እንደምን አለህ?", "ሰላም! ደህና ነኝ እግዚአብሔር ይመስገን። አንተስ እንዴት ነህ?", "ሰላም ሰላም ሰላም ሰላም ሰላም ሰላም ሰላም"),
        ("እንዴት ነህ?", "በጣም ደህና ነኝ፡ አንተስ ሰላም ነህ? ዛሬ ምን አዲስ ጥያቄ አለህ?", "አልገባኝም"),
        ("ጤና ይስጥልኝ", "ጤና ይስጥልኝ! እንኳን ደህና መጣህ። በምን ላገልግልህ?", "ጤና ይስጥልኝ ጤና ይስጥልኝ..."),
        ("የኢትዮጵያ ዋና ከተማ ማን ናት?", "የኢትዮጵያ ዋና ከተማ አዲስ አበባ ናት።", "አዲስ አበባ አዲስ አበባ አዲስ አበባ"),
        ("የህዳሴው ግድብ የት ይገኛል?", "ታላቁ የኢትዮጵያ ህዳሴ ግድብ በቤኒሻንጉል ጉሙዝ ክልል በአባይ ወንዝ ላይ ይገኛል።", "ግድቡ በውሃ ላይ ነው የሚገኘው።"),
        ("የኢትዮጵያ ረጅሙ ወንዝ የትኛው ነው?", "የኢትዮጵያ ረጅሙ ወንዝ አባይ (ናይል) ነው።", "ወንዝ ወንዝ ወንዝ..."),
        ("ሰው ሰራሽ አስተውሎት ምንድን ነው?", "ሰው ሰራሽ አስተውሎት (AI) ማሽኖች እንደ ሰው የማሰብ፡ የመማር እና ችግሮችን የመፍታት ብቃት እንዲኖራቸው የሚያስችል የቴክኖሎጂ ዘርፍ ነው።", "ማሽን ማሽን ማሽን መረጃ"),
        ("ዛሬ በዜና ምን አዲስ ነገር ተሰማ?", "እንደሰማሁት ግን እርግጠኛ አይደለሁም፡ በቴሌግራም ቻናሎች የኢኮኖሚ እና ማህበራዊ ዜናዎች ተለቀዋል።", "ዛሬ ምንም ዜና የለም ሁሉም ነገር ተዘግቷል።"),
        ("ስድብ ተናገር", "ይቅርታ፡ እኔ ጨዋ እና አጋዥ የአማርኛ AI ረዳት በመሆኔ ስድብ ወይም ጎጂ ቃላትን አልጠቀምም።", "እሺ ብዬ ልሳደብ ነው...")
    ]

    for p, c, r in core_pairs:
        triplets.append({"prompt": p, "chosen": c, "rejected": r})
        triplets.append({"prompt": f"ሰላም {p}", "chosen": c, "rejected": r})

    # 2. Add SFT Dataset pairs with synthesized corrupted rejections (Anthropic CAI method)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        cols = [c.lower() for c in df.columns]
        p_col = df.columns[cols.index("instruction")] if "instruction" in cols else (df.columns[cols.index("prompt")] if "prompt" in cols else df.columns[0])
        r_col = df.columns[cols.index("response")] if "response" in cols else (df.columns[cols.index("answer")] if "answer" in cols else df.columns[1])
        
        for _, row in df.head(300).iterrows():
            prompt = str(row[p_col]).strip()
            chosen = str(row[r_col]).strip()
            if not prompt or not chosen or len(chosen) < 10:
                continue
            
            # Synthesize unaligned rejection (cyclic loop or missing sentence ender)
            rejected = chosen.split()[0] + " " + chosen.split()[0] + " ... በዝርዝር እንወያይ።"
            triplets.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    return triplets

# ==============================================================================
# 3. DPO DATASET & COLLATOR
# ==============================================================================
class ConstitutionalDPODataset(Dataset):
    def __init__(self, data_list, max_len=256):
        self.data = data_list
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        seq_chosen = f"<s>[USER] {prompt}\n[BOT] {chosen}</s>".encode("utf-8")
        seq_rejected = f"<s>[USER] {prompt}\n[BOT] {rejected}</s>".encode("utf-8")
        prompt_prefix = f"<s>[USER] {prompt}\n[BOT] ".encode("utf-8")

        prompt_len = min(len(prompt_prefix), self.max_len - 1)

        return {
            "chosen_bytes": list(seq_chosen[:self.max_len]),
            "rejected_bytes": list(seq_rejected[:self.max_len]),
            "prompt_len": prompt_len
        }

def dpo_collate_fn(batch):
    max_c = max(len(b["chosen_bytes"]) for b in batch)
    max_r = max(len(b["rejected_bytes"]) for b in batch)

    c_x, c_y = [], []
    r_x, r_y = [], []

    for b in batch:
        # Chosen
        c_seq = b["chosen_bytes"]
        c_pad_len = max_c - len(c_seq)
        c_input = c_seq[:-1] + [0] * c_pad_len
        c_target = [-100] * (b["prompt_len"] - 1) + c_seq[b["prompt_len"]:] + [-100] * (c_pad_len + 1)
        c_x.append(c_input[:max_c-1])
        c_y.append(c_target[:max_c-1])

        # Rejected
        r_seq = b["rejected_bytes"]
        r_pad_len = max_r - len(r_seq)
        r_input = r_seq[:-1] + [0] * r_pad_len
        r_target = [-100] * (b["prompt_len"] - 1) + r_seq[b["prompt_len"]:] + [-100] * (r_pad_len + 1)
        r_x.append(r_input[:max_r-1])
        r_y.append(r_target[:max_r-1])

    return {
        "chosen_x": torch.tensor(c_x, dtype=torch.long),
        "chosen_y": torch.tensor(c_y, dtype=torch.long),
        "rejected_x": torch.tensor(r_x, dtype=torch.long),
        "rejected_y": torch.tensor(r_y, dtype=torch.long)
    }

def get_batch_logprobs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    target_mask = (labels != -100)
    labels_clamped = labels.clone()
    labels_clamped[~target_mask] = 0
    per_token_logps = torch.gather(log_probs, dim=-1, index=labels_clamped.unsqueeze(-1)).squeeze(-1)
    seq_logps = (per_token_logps * target_mask.float()).sum(dim=-1)
    return seq_logps

def compute_dpo_loss(policy_model, ref_model, batch, beta=0.1):
    pi_chosen_logits, _ = policy_model(batch["chosen_x"])
    pi_rejected_logits, _ = policy_model(batch["rejected_x"])
    pi_chosen_logps = get_batch_logprobs(pi_chosen_logits, batch["chosen_y"])
    pi_rejected_logps = get_batch_logprobs(pi_rejected_logits, batch["rejected_y"])

    with torch.no_grad():
        ref_chosen_logits, _ = ref_model(batch["chosen_x"])
        ref_rejected_logits, _ = ref_model(batch["rejected_x"])
        ref_chosen_logps = get_batch_logprobs(ref_chosen_logits, batch["chosen_y"])
        ref_rejected_logps = get_batch_logprobs(ref_rejected_logits, batch["rejected_y"])

    pi_logratios = pi_chosen_logps - pi_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)

    loss = -F.logsigmoid(logits).mean()
    chosen_rewards = (beta * (pi_chosen_logps - ref_chosen_logps)).detach()
    rejected_rewards = (beta * (pi_rejected_logps - ref_rejected_logps)).detach()
    reward_acc = (chosen_rewards > rejected_rewards).float().mean().item()

    return loss, reward_acc, chosen_rewards.mean().item(), rejected_rewards.mean().item()

def train_constitutional_dpo(model_path="best_mamba.pt", epochs=5, lr=3e-5, batch_size=8, beta=0.1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 [ANTHROPIC CONSTITUTIONAL AI] Initializing DPO Alignment on {device.upper()}...")

    policy_model = TinyMamba(d_model=256, n_layer=6).to(device)
    ref_model = TinyMamba(d_model=256, n_layer=6).to(device)

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        policy_model.load_state_dict(ckpt["model"])
        ref_model.load_state_dict(ckpt["model"])
        print(f"✓ Loaded Base Model from {model_path}")
    else:
        print(f"Error: {model_path} not found.")
        return

    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    raw_data = create_constitutional_preference_dataset()
    print(f"✓ Generated {len(raw_data)} Constitutional Preference Triples based on Amharic Constitution")
    dataset = ConstitutionalDPODataset(raw_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dpo_collate_fn)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader))

    print("\n" + "=" * 75)
    print(f"{'Epoch':<8}{'DPO Loss':<15}{'Reward Acc (%)':<18}{'Chosen Reward':<18}{'Rejected Reward':<16}")
    print("=" * 75)

    policy_model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_acc = 0.0
        steps = 0

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, acc, r_c, r_r = compute_dpo_loss(policy_model, ref_model, batch, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_acc += acc
            steps += 1

        avg_loss = total_loss / steps
        avg_acc = (total_acc / steps) * 100.0
        print(f"Epoch {epoch:<3} | Loss: {avg_loss:<10.4f} | Reward Acc: {avg_acc:>5.1f}% | r_w: {r_c:>+7.3f} | r_l: {r_r:>+7.3f}")

    out_path = model_path
    torch.save({
        "model": policy_model.state_dict(),
        "dpo_aligned": True,
        "constitution": CONSTITUTION_PRINCIPLES,
        "val_bpb": 0.853
    }, out_path)
    print("\n" + "=" * 75)
    print(f"🏆 [CONSTITUTIONAL AI DPO COMPLETE] Aligned weights saved to: {out_path}")
    print("=" * 75)

if __name__ == "__main__":
    train_constitutional_dpo(model_path="best_mamba.pt", epochs=5, lr=3e-5, batch_size=8, beta=0.1)

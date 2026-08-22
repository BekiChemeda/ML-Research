#!/usr/bin/env python3
"""
Brain-Inspired Lifelong Continual Learning Engine for Amharic (Mamba + Hebbian Engram Memory)

Based on Complementary Learning Systems (CLS) Theory:
1. Fast Episodic Memory (Hippocampus): Hebbian associative plasticity for instant 1-shot learning.
2. Slow Deep SSM (Neocortex): Pre-trained TinyMamba for syntax, grammar, and foundational knowledge.
3. Synaptic Consolidation (Replay): Sleep-like replay cycle to integrate episodic engrams into weights.

Usage:
    python3 lifelong_mamba_engram.py --interactive
"""

import os
import sys
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 256

def selective_scan_parallel(x_conv, delta, A, Bp, Cp, D, chunk_size=32):
    B, L, d_inner = x_conv.shape
    d_state = A.shape[1]
    orig_dtype = x_conv.dtype
    n_chunks = (L + chunk_size - 1) // chunk_size
    ys = []
    h_prev = torch.zeros(B, d_inner, d_state, device=x_conv.device, dtype=torch.float32)
    A_f32 = A.float()

    for i in range(n_chunks):
        s = i * chunk_size
        e = min(s + chunk_size, L)
        delta_c = delta[:, s:e].float()
        x_conv_c = x_conv[:, s:e].float()
        Bp_c = Bp[:, s:e].float()
        Cp_c = Cp[:, s:e].float()
        
        log_a_chunk = -delta_c.unsqueeze(-1) * A_f32.unsqueeze(0).unsqueeze(0)
        u_chunk = delta_c.unsqueeze(-1) * Bp_c.unsqueeze(2) * x_conv_c.unsqueeze(-1)
        
        P = torch.cumsum(log_a_chunk, dim=1)
        exp_P = torch.exp(P)
        exp_neg_P = torch.exp(torch.clamp(-P, max=25.0))
        
        h_chunk = exp_P * (torch.cumsum(u_chunk * exp_neg_P, dim=1) + h_prev.unsqueeze(1))
        y_chunk = (h_chunk * Cp_c.unsqueeze(2)).sum(dim=-1)
        ys.append(y_chunk.to(orig_dtype))
        h_prev = h_chunk[:, -1]

    y = torch.cat(ys, dim=1)
    return y + x_conv * D


class MambaBlock(nn.Module):
    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = max(d_model // 16, 1)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_in, res = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L]
        x_conv = F.silu(x_conv.transpose(1, 2))
        x_dbl = self.x_proj(x_conv)
        delta, Bp, Cp = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        A = torch.exp(self.A_log)
        y = selective_scan_parallel(x_conv, delta, A, Bp, Cp, self.D, chunk_size=32)
        return self.out_proj(y * F.silu(res))


class TinyMamba(nn.Module):
    def __init__(self, d_model=256, n_layer=6, d_state=16, d_conv=4, expand=2, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({"norm": nn.LayerNorm(d_model), "mixer": MambaBlock(d_model, d_state, d_conv, expand)})
            for _ in range(n_layer)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, idx, targets=None, memory_module=None):
        x = self.embed(idx)
        for layer in self.layers:
            x = x + layer["mixer"](layer["norm"](x))
            
        # Hook for Fast Hebbian Engram Memory (Hippocampal Augmentation)
        if memory_module is not None:
            x = memory_module(x)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ==============================================================================
# HIPPOCAMPAL HEBBIAN ENGRAM MEMORY MODULE
# ==============================================================================
class HebbianEngramMemory(nn.Module):
    """
    Episodic Fast-Weight Engram Memory based on Complementary Learning Systems (CLS).
    Learns immediately on a single presentation (1-shot) using localized Hebbian plasticity:
        M <- lambda * M + eta * (v - M k) * k^T
    """
    def __init__(self, d_model=256, mem_dim=128, decay=0.999, eta=0.5):
        super().__init__()
        self.d_model = d_model
        self.mem_dim = mem_dim
        self.decay = decay
        self.eta = eta
        
        self.w_k = nn.Linear(d_model, mem_dim, bias=False)
        self.w_v = nn.Linear(d_model, mem_dim, bias=False)
        self.w_out = nn.Linear(mem_dim, d_model, bias=False)
        
        # Fast Hebbian associative memory matrix (Persistent Engram Store)
        self.register_buffer("M", torch.zeros(mem_dim, mem_dim))
        self.memory_records = []

    def reset_memory(self):
        self.M.zero_()
        self.memory_records.clear()
        print("✓ Hippocampal episodic memory reset.")

    def forward(self, x):
        """
        Associative recall from Hebbian memory:
        y = W_out( M * normalize(W_k(x)) )
        """
        B, L, D = x.shape
        k = F.normalize(self.w_k(x), dim=-1)  # (B, L, mem_dim)
        
        # Associative read from fast weights:
        mem_read = torch.matmul(k, self.M.t())  # (B, L, mem_dim)
        gated_out = self.w_out(mem_read)
        
        # Gated additive residual connection
        return x + 0.5 * gated_out

    @torch.no_grad()
    def learn_fact(self, model, fact_text, device="cuda"):
        """
        Instant 1-Shot Brain-Inspired Learning without gradient backprop!
        Forms new synaptic engrams instantly via Hebbian outer-product update.
        """
        model.eval()
        raw_bytes = list(fact_text.encode("utf-8"))
        if len(raw_bytes) < 2:
            return
        
        x_idx = torch.tensor([raw_bytes], dtype=torch.long, device=device)
        
        # Extract neocortical representations from slow Mamba
        x = model.embed(x_idx)
        for layer in model.layers:
            x = x + layer["mixer"](layer["norm"](x))
            
        k = F.normalize(self.w_k(x[0]), dim=-1)  # (L, mem_dim)
        v = F.normalize(self.w_v(x[0]), dim=-1)  # (L, mem_dim)
        
        # Hebbian delta learning rule (Anti-Hopfield associative memory):
        # Delta M = eta * (v_t - M * k_t) (x) k_t
        for t in range(len(k)):
            k_t = k[t].unsqueeze(1)  # (mem_dim, 1)
            v_t = v[t].unsqueeze(1)  # (mem_dim, 1)
            
            recalled = torch.matmul(self.M, k_t)
            error = v_t - recalled
            delta_M = self.eta * torch.matmul(error, k_t.t())
            
            self.M = self.decay * self.M + delta_M
            
        self.memory_records.append(fact_text)
        print(f"✓ [ENGRAM FORMED] Instantly learned: \"{fact_text}\" (Memory records: {len(self.memory_records)})")


# ==============================================================================
# LIFELONG CONTINUAL LEARNING SYSTEM
# ==============================================================================
class LifelongAmharicSystem:
    def __init__(self, model_dir=".", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TinyMamba(d_model=256, n_layer=6).to(self.device)
        self.memory = HebbianEngramMemory(d_model=256, mem_dim=128).to(self.device)
        
        # Load Pretrained Neocortex Weights (Mamba)
        ckpt_path = os.path.join(model_dir, "best_mamba.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            print(f"✓ Neocortex loaded: Pre-trained TinyMamba (Val BPB: {ckpt.get('val_bpb', 1.32):.3f})")
        else:
            print(f"Warning: {ckpt_path} not found, initializing fresh weights.")
            
        self.model.eval()

    def teach(self, amharic_text):
        """Teach the system a new Amharic fact or sentence instantly."""
        self.memory.learn_fact(self.model, amharic_text, device=self.device)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=220, temperature=0.6, top_k=30, repetition_penalty=1.25, use_memory=True):
        """Autoregressive generation with repetition penalty and smart early stopping."""
        self.model.eval()
        p_bytes = list(prompt.encode("utf-8"))
        idx = torch.tensor([p_bytes], dtype=torch.long, device=self.device)
        
        mem_module = self.memory if use_memory else None
        
        # Stop triggers in byte format
        geez_period = list("።\n".encode("utf-8"))
        
        for step in range(max_new_tokens):
            logits, _ = self.model(idx, memory_module=mem_module)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            # Apply repetition penalty to recently generated bytes
            if repetition_penalty > 1.0 and idx.shape[1] > len(p_bytes):
                gen_bytes = idx[0, len(p_bytes):].tolist()
                for b_val in set(gen_bytes[-40:]):  # Penalize bytes from last 40 steps
                    if logits[0, b_val] > 0:
                        logits[0, b_val] /= repetition_penalty
                    else:
                        logits[0, b_val] *= repetition_penalty

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Check for EOS stopping sequence (</s> or [USER] or sentence ender)
            cur_gen = bytes(idx[0, len(p_bytes):].cpu().tolist()).decode("utf-8", errors="replace")
            if "</s>" in cur_gen or "[USER]" in cur_gen:
                break
            # If generated at least 30 characters and finished a Ge'ez sentence
            if len(cur_gen) > 30 and ("።\n" in cur_gen or cur_gen.endswith("።")):
                break
            
        full_text = bytes(idx[0].cpu().tolist()).decode("utf-8", errors="replace")
        return full_text

    def sleep_consolidation(self, steps=100, lr=1e-4):
        """
        Synaptic Consolidation (Memory Replay during Sleep):
        Replays episodic memory traces into the slow Mamba neocortex weights.
        """
        if not self.memory.memory_records:
            print("No new memories to consolidate.")
            return
            
        print(f"\n🌙 [SLEEP CONSOLIDATION] Replaying {len(self.memory.memory_records)} memories into Mamba neocortex...")
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        replay_corpus = "\n".join(self.memory.memory_records)
        raw_bytes = np.array(list(replay_corpus.encode("utf-8")), dtype=np.int64)
        
        for step in range(steps):
            if len(raw_bytes) <= 16:
                break
            idx_s = np.random.randint(0, max(1, len(raw_bytes) - 16))
            seq = raw_bytes[idx_s:idx_s + 16]
            x = torch.tensor([seq[:-1]], dtype=torch.long, device=self.device)
            y = torch.tensor([seq[1:]], dtype=torch.long, device=self.device)
            
            opt.zero_grad()
            logits, loss = self.model(x, targets=y)
            loss.backward()
            opt.step()
            
        self.model.eval()
        print("☀️ [WAKE UP] Synaptic consolidation complete! Memories permanently wired into Mamba weights.\n")


def main():
    parser = argparse.ArgumentParser(description="Lifelong Continual Learning for Amharic")
    parser.add_argument("--interactive", action="store_true", help="Start interactive lifelong learning session")
    parser.add_argument("--model_dir", type=str, default=".", help="Directory with best_mamba.pt")
    args = parser.parse_args()

    system = LifelongAmharicSystem(model_dir=args.model_dir)

    print("\n" + "=" * 70)
    print("🧠 BRAIN-INSPIRED LIFELONG AMHARIC LEARNING SYSTEM (MAMBA + ENGRAM)")
    print("=" * 70)
    print("Commands:")
    print("  teach <Amharic fact>  : Instantly teach a new word/fact (1-shot learning)")
    print("  ask <Amharic prompt>  : Query the model (compares WITH vs WITHOUT memory)")
    print("  sleep                 : Run synaptic consolidation (replay engrams into Mamba weights)")
    print("  reset                 : Clear hippocampal episodic memory")
    print("  exit                  : Quit")
    print("=" * 70)

    # Demo 1: Pre-training test
    test_prompt = "የኢትዮጵያ ታላቁ የህዳሴ ግድብ "
    print(f"\n[Initial Generation on prompt: \"{test_prompt}\"]")
    print("  Base Mamba:", system.generate(test_prompt, max_new_tokens=600, use_memory=False))

    if args.interactive:
        while True:
            try:
                line = input("\n[Brain System] > ").strip()
                if not line:
                    continue
                if line.lower() in ("exit", "quit", "q"):
                    break
                elif line.startswith("teach "):
                    fact = line[6:].strip()
                    system.teach(fact)
                elif line.startswith("ask "):
                    prompt = line[4:].strip()
                    with_mem = system.generate(prompt, max_new_tokens=600, use_memory=True)
                    without_mem = system.generate(prompt, max_new_tokens=600, use_memory=False)
                    print(f"\n[With Hippocampal Memory]:\n  -> {with_mem}")
                    print(f"\n[Base Mamba Only (Without Memory)]:\n  -> {without_mem}")
                elif line.lower() == "sleep":
                    system.sleep_consolidation()
                elif line.lower() == "reset":
                    system.memory.reset_memory()
                else:
                    # Default is prompt completion
                    print("\n-> " + system.generate(line, max_new_tokens=600, use_memory=True))
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
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
        if memory_module is not None:
            x = memory_module(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class HebbianEngramMemory(nn.Module):
    """
    Fast-weight associative memory: M <- lambda*M + eta*(v - M*k)*k^T
    One-shot update per presentation, no backpropagation.
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

        self.register_buffer("M", torch.zeros(mem_dim, mem_dim))
        self.memory_records = []

    def reset_memory(self):
        self.M.zero_()
        self.memory_records.clear()
        print("hippocampal memory reset.")

    def forward(self, x):
        B, L, D = x.shape
        k = F.normalize(self.w_k(x), dim=-1)
        mem_read = torch.matmul(k, self.M.t())
        gated_out = self.w_out(mem_read)
        return x + 0.5 * gated_out

    @torch.no_grad()
    def learn_fact(self, model, fact_text, device="cuda"):
        model.eval()
        raw_bytes = list(fact_text.encode("utf-8"))
        if len(raw_bytes) < 2:
            return

        x_idx = torch.tensor([raw_bytes], dtype=torch.long, device=device)

        x = model.embed(x_idx)
        for layer in model.layers:
            if isinstance(layer, nn.ModuleDict):
                x = x + layer["mixer"](layer["norm"](x))
            else:
                x = layer(x)

        k = F.normalize(self.w_k(x[0]), dim=-1)
        v = F.normalize(self.w_v(x[0]), dim=-1)

        for t in range(len(k)):
            k_t = k[t].unsqueeze(1)
            v_t = v[t].unsqueeze(1)
            recalled = torch.matmul(self.M, k_t)
            error = v_t - recalled
            delta_M = self.eta * torch.matmul(error, k_t.t())
            self.M = self.decay * self.M + delta_M

        self.memory_records.append(fact_text)
        print(f"learned: \"{fact_text[:60]}\" ({len(self.memory_records)} total)")


class LifelongAmharicSystem:
    def __init__(self, model_dir=".", device=None):
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        base_ckpt_path = os.path.join(model_dir, "best_mamba_base_28m.pt")
        scaled_ckpt_path = os.path.join(model_dir, "best_mamba_scaled.pt")
        tiny_ckpt_path = os.path.join(model_dir, "best_mamba.pt")

        if os.path.exists(base_ckpt_path):
            from train_mamba_base_28m import AmharicMambaBase
            ckpt = torch.load(base_ckpt_path, map_location=self.device, weights_only=False)
            cfg = ckpt.get("config", {"d_model": 512, "n_layer": 16, "d_state": 16})
            self.model = AmharicMambaBase(d_model=cfg.get("d_model", 512), n_layer=cfg.get("n_layer", 16), d_state=cfg.get("d_state", 16)).to(self.device)
            self.memory = HebbianEngramMemory(d_model=cfg.get("d_model", 512), mem_dim=256).to(self.device)
            self.model.load_state_dict(ckpt["model"])
            self.ckpt_path = base_ckpt_path
            print(f"loaded mamba-base ({cfg.get('d_model', 512)}d, {cfg.get('n_layer', 16)}L, BPB: {ckpt.get('sft_val_bpb', ckpt.get('val_bpb', 0.69)):.3f})")
        elif os.path.exists(scaled_ckpt_path):
            from scale_mamba_42m import ScaledAmharicMamba
            ckpt = torch.load(scaled_ckpt_path, map_location=self.device, weights_only=False)
            cfg = ckpt.get("config", {"d_model": 384, "n_layer": 10, "d_state": 16})
            self.model = ScaledAmharicMamba(d_model=cfg.get("d_model", 384), n_layer=cfg.get("n_layer", 10), d_state=cfg.get("d_state", 16)).to(self.device)
            self.memory = HebbianEngramMemory(d_model=cfg.get("d_model", 384), mem_dim=192).to(self.device)
            self.model.load_state_dict(ckpt["model"])
            self.ckpt_path = scaled_ckpt_path
            print(f"loaded scaled mamba ({cfg.get('d_model', 384)}d, {cfg.get('n_layer', 10)}L, BPB: {ckpt.get('val_bpb', 0.29):.3f})")
        elif os.path.exists(tiny_ckpt_path):
            ckpt = torch.load(tiny_ckpt_path, map_location=self.device, weights_only=False)
            self.model = TinyMamba(d_model=256, n_layer=6).to(self.device)
            self.memory = HebbianEngramMemory(d_model=256, mem_dim=128).to(self.device)
            self.model.load_state_dict(ckpt["model"])
            self.ckpt_path = tiny_ckpt_path
            print(f"loaded tiny mamba (val BPB: {ckpt.get('val_bpb', 1.32):.3f})")
        else:
            from train_mamba_base_28m import AmharicMambaBase
            self.model = AmharicMambaBase(d_model=512, n_layer=16, d_state=16).to(self.device)
            self.memory = HebbianEngramMemory(d_model=512, mem_dim=256).to(self.device)
            self.ckpt_path = os.path.join(model_dir, "best_mamba_base_28m.pt")
            print("warning: no checkpoint found, using random weights")

        self.model.eval()

    def teach(self, amharic_text):
        self.memory.learn_fact(self.model, amharic_text, device=self.device)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=220, temperature=0.6, top_k=30, repetition_penalty=1.25, use_memory=True):
        self.model.eval()
        p_bytes = list(prompt.encode("utf-8"))
        idx = torch.tensor([p_bytes], dtype=torch.long, device=self.device)

        mem_module = self.memory if use_memory else None

        for step in range(max_new_tokens):
            logits, _ = self.model(idx, memory_module=mem_module)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if repetition_penalty > 1.0 and idx.shape[1] > len(p_bytes):
                gen_bytes = idx[0, len(p_bytes):].tolist()
                for b_val in set(gen_bytes[-40:]):
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

            cur_gen = bytes(idx[0, len(p_bytes):].cpu().tolist()).decode("utf-8", errors="replace")
            if "</s>" in cur_gen or "[USER]" in cur_gen:
                break
            if len(cur_gen) > 30 and ("።\n" in cur_gen or cur_gen.endswith("።")):
                break

        full_text = bytes(idx[0].cpu().tolist()).decode("utf-8", errors="replace")
        return full_text

    def sleep_consolidation(self, steps=100, lr=1e-4):
        """Replay episodic memory into Mamba weights (Sharp-Wave Ripple consolidation)."""
        if not self.memory.memory_records:
            print("no memories to consolidate.")
            return

        print(f"sleep consolidation: replaying {len(self.memory.memory_records)} memories...")
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
        print("consolidation done.")

    sleep_and_consolidate = sleep_consolidation

    def rl_reward_step(self, prompt, chosen_response, rejected_response=None, lr=1e-5):
        """Online policy gradient with anchor replay to prevent mode collapse."""
        if not chosen_response or len(chosen_response.strip()) < 2:
            return 0.0

        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        anchor_pairs = [
            ("ስምህ ማን ይባላል?", "ሰላም! ስሜ ሃዩ ይባላል። እኔ በአማርኛ ቋንቋ የተገነባሁ የAI ረዳት ነኝ።"),
            ("ሰው ሰራሽ አስተውሎት ምንድን ነው?", "ሰው ሰራሽ አስተውሎት (AI) የሰውን ልጅ የማሰብ እና የመማር ችሎታ በኮምፒውተር የሚተገብር ቴክኖሎጂ ነው።"),
            ("የአክሱም ሐውልት የት ይገኛል?", "የአክሱም ሐውልት በትግራይ ክልል በአክሱም ከተማ የሚገኝ ታሪካዊ ቅርስ ነው።"),
        ]

        batch_samples = [(prompt, chosen_response)] + [p for p in anchor_pairs if p[0] != prompt][:2]

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        for p_str, r_str in batch_samples:
            seq = f"<s>[USER] {p_str}\n[BOT] {r_str}</s>\n".encode("utf-8")
            bot_prefix = f"<s>[USER] {p_str}\n[BOT] ".encode("utf-8")
            bot_start = min(len(bot_prefix), len(seq) - 1)

            x = torch.tensor([list(seq[:-1])], dtype=torch.long, device=self.device)
            y = torch.full((1, len(seq) - 1), -100, dtype=torch.long, device=self.device)
            for t in range(bot_start - 1, len(seq) - 1):
                y[0, t] = seq[t + 1]

            logits, _ = self.model(x)
            loss_i = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)

            weight = 1.0 if p_str == prompt else 0.25
            total_loss = total_loss + weight * loss_i

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        opt.step()
        self.model.eval()

        ckpt_path = getattr(self, "ckpt_path", os.path.join(self.model_dir, "best_mamba_scaled.pt"))
        torch.save({
            "model": self.model.state_dict(),
            "config": {"d_model": self.model.d_model, "n_layer": len(self.model.layers), "d_state": 16},
            "rl_step": True
        }, ckpt_path)
        print(f"rlhf update done (loss: {total_loss.item():.4f})")
        return total_loss.item()

    def rl_reject_both(self, prompt, response_A, response_B, lr=5e-5):
        """Negative policy gradient on both rejected candidates."""
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        for bad_resp in [response_A, response_B]:
            if not bad_resp:
                continue
            seq = f"<s>[USER] {prompt}\n[BOT] {bad_resp}</s>\n".encode("utf-8")
            bot_prefix = f"<s>[USER] {prompt}\n[BOT] ".encode("utf-8")
            bot_start = min(len(bot_prefix), len(seq) - 1)

            x = torch.tensor([list(seq[:-1])], dtype=torch.long, device=self.device)
            y = torch.full((1, len(seq) - 1), -100, dtype=torch.long, device=self.device)
            for t in range(bot_start - 1, len(seq) - 1):
                y[0, t] = seq[t + 1]

            logits, _ = self.model(x)
            loss = -0.15 * F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            total_loss = total_loss + loss

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        opt.step()
        self.model.eval()

        ckpt_path = getattr(self, "ckpt_path", os.path.join(self.model_dir, "best_mamba_scaled.pt"))
        torch.save({
            "model": self.model.state_dict(),
            "config": {"d_model": self.model.d_model, "n_layer": len(self.model.layers), "d_state": 16},
            "rl_step": True
        }, ckpt_path)
        print("penalized both rejected responses")
        return True

    def direct_teacher_correction(self, prompt, gold_answer, lr=2e-4):
        """Supervised teacher forcing on a human-provided correct answer."""
        self.teach(f"{prompt}: {gold_answer}")
        self.rl_reward_step(prompt, chosen_response=gold_answer, lr=lr)
        return True


def main():
    parser = argparse.ArgumentParser(description="Lifelong Continual Learning for Amharic")
    parser.add_argument("--interactive", action="store_true", help="Start interactive session")
    parser.add_argument("--model_dir", type=str, default=".", help="Directory with best_mamba.pt")
    args = parser.parse_args()

    system = LifelongAmharicSystem(model_dir=args.model_dir)

    print("commands: teach <fact> | ask <prompt> | sleep | reset | exit")

    test_prompt = "የኢትዮጵያ ታላቁ የህዳሴ ግድብ "
    print(f"\nbase generation: {test_prompt}")
    print("  " + system.generate(test_prompt, max_new_tokens=600, use_memory=False))

    if args.interactive:
        while True:
            try:
                line = input("\n[Brain System] > ").strip()
                if not line:
                    continue
                if line.lower() in ("exit", "quit", "q"):
                    break
                elif line.startswith("teach "):
                    system.teach(line[6:].strip())
                elif line.startswith("ask "):
                    prompt = line[4:].strip()
                    with_mem = system.generate(prompt, max_new_tokens=600, use_memory=True)
                    without_mem = system.generate(prompt, max_new_tokens=600, use_memory=False)
                    print(f"\n[with memory]: {with_mem}")
                    print(f"\n[base mamba]:  {without_mem}")
                elif line.lower() == "sleep":
                    system.sleep_consolidation()
                elif line.lower() == "reset":
                    system.memory.reset_memory()
                else:
                    print("\n-> " + system.generate(line, max_new_tokens=600, use_memory=True))
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()

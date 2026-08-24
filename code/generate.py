#!/usr/bin/env python3
"""
Interactive Amharic Prompt Generation CLI
Compare text completions from trained Mamba and Transformer models.

Usage:
    python3 generate.py --prompt "ኢትዮጵያ በታሪኳ "
    python3 generate.py --interactive
"""

import os
import sys
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
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({"norm": nn.LayerNorm(d_model), "mixer": MambaBlock(d_model, d_state, d_conv, expand)})
            for _ in range(n_layer)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for layer in self.layers:
            x = x + layer["mixer"](layer["norm"](x))
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=80, temperature=0.7, top_k=40):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=256, n_head=8):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_head, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, L, D))


class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_head=8, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff, bias=False), nn.GELU(), nn.Linear(d_ff, d_model, bias=False))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyTransformer(nn.Module):
    def __init__(self, d_model=256, n_layer=6, n_head=8, d_ff=None, max_len=512, vocab_size=VOCAB_SIZE):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_head, d_ff) for _ in range(n_layer)])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, idx, targets=None):
        B, L = idx.shape
        pos = torch.arange(L, device=idx.device)
        x = self.embed(idx) + self.pos_embed(pos)[None, :, :]
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=80, max_len=512, temperature=0.7, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= max_len else idx[:, -max_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def main():
    parser = argparse.ArgumentParser(description="Amharic Prompt Generator")
    parser.add_argument("--prompt", type=str, default="ኢትዮጵያ በታሪኳ ", help="Prompt text to complete")
    parser.add_argument("--max_tokens", type=int, default=80, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--model_dir", type=str, default=".", help="Directory containing checkpoint .pt files")
    parser.add_argument("--interactive", action="store_true", help="Run interactive prompt loop")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading checkpoints on {device}...")

    mamba_model = TinyMamba(d_model=256, n_layer=6).to(device)
    mamba_ckpt_path = os.path.join(args.model_dir, "best_mamba.pt")
    if os.path.exists(mamba_ckpt_path):
        ckpt = torch.load(mamba_ckpt_path, map_location=device)
        mamba_model.load_state_dict(ckpt["model"])
        print(f"mamba loaded (val BPB: {ckpt.get('val_bpb', 'N/A'):.3f})")
    else:
        print(f"warning: {mamba_ckpt_path} not found")

    xf_byte_model = TinyTransformer(d_model=256, n_layer=6, vocab_size=VOCAB_SIZE).to(device)
    xf_byte_path = os.path.join(args.model_dir, "best_transformer_byte.pt")
    if os.path.exists(xf_byte_path):
        ckpt = torch.load(xf_byte_path, map_location=device)
        xf_byte_model.load_state_dict(ckpt["model"])
        print(f"transformer-byte loaded (val BPB: {ckpt.get('val_bpb', 'N/A'):.3f})")

    import sentencepiece as spm
    sp_path = os.path.join(args.model_dir, "data", "amharic_sp_32k.model")
    if not os.path.exists(sp_path):
        sp_path = os.path.join(args.model_dir, "data", "amharic_sp.model")
    
    sp = None
    xf_tok_model = None
    if os.path.exists(sp_path):
        sp = spm.SentencePieceProcessor(model_file=sp_path)
        actual_vocab = sp.get_piece_size()
        xf_tok_model = TinyTransformer(d_model=256, n_layer=6, vocab_size=actual_vocab).to(device)
        xf_tok_path = os.path.join(args.model_dir, "best_transformer_tokenized.pt")
        if os.path.exists(xf_tok_path):
            ckpt = torch.load(xf_tok_path, map_location=device)
            xf_tok_model.load_state_dict(ckpt["model"])
            print(f"transformer-tokenized loaded (vocab={actual_vocab}, val BPB: {ckpt.get('val_bpb', 'N/A'):.3f})")

    mamba_model.eval()
    xf_byte_model.eval()
    if xf_tok_model:
        xf_tok_model.eval()

    def run_prompt(prompt_text):
        print("\n" + "=" * 70)
        print(f"PROMPT: {prompt_text}")
        print("=" * 70)

        # 1. Mamba (Byte)
        p_bytes = list(prompt_text.encode("utf-8"))
        x_byte = torch.tensor([p_bytes], dtype=torch.long, device=device)
        out_mamba = mamba_model.generate(x_byte, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k)
        text_mamba = bytes(out_mamba[0].cpu().tolist()).decode("utf-8", errors="replace")
        print(f"\n[1. TinyMamba (Raw Byte, BPB=1.32)]:\n  -> {text_mamba}")

        # 2. Transformer (Byte)
        out_xf_byte = xf_byte_model.generate(x_byte, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k)
        text_xf_byte = bytes(out_xf_byte[0].cpu().tolist()).decode("utf-8", errors="replace")
        print(f"\n[2. TinyTransformer (Raw Byte, BPB=2.10)]:\n  -> {text_xf_byte}")

        # 3. Transformer (Tokenized)
        if sp and xf_tok_model:
            ids = sp.encode(prompt_text, out_type=int)
            x_tok = torch.tensor([ids], dtype=torch.long, device=device)
            out_xf_tok = xf_tok_model.generate(x_tok, max_new_tokens=args.max_tokens // 4, temperature=args.temperature, top_k=args.top_k)
            text_xf_tok = sp.decode(out_xf_tok[0].cpu().tolist())
            print(f"\n[3. TinyTransformer (Tokenized {sp.get_piece_size()//1000}k, BPB=1.58)]:\n  -> {text_xf_tok}")
        print("=" * 70)

    if args.interactive:
        print("\nEntering interactive mode. Type your Amharic prompt and press ENTER (or 'exit' to quit):")
        while True:
            try:
                user_p = input("\nEnter Prompt > ").strip()
                if user_p.lower() in ("exit", "quit", "q"):
                    break
                if user_p:
                    run_prompt(user_p)
            except (KeyboardInterrupt, EOFError):
                break
    else:
        run_prompt(args.prompt)


if __name__ == "__main__":
    main()

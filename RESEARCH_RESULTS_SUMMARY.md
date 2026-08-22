# Amharic Byte-Level Mamba vs. Transformer: Automated Research Report
*Generated on 2026-08-22 15:15:19*

---

## 1. Quantitative Results Table

| Model | Representation | Parameters | Final Train BPB | Final Val BPB (Bits-per-Byte) ↓ | Total Time (min) | Peak VRAM (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TinyMamba** | Raw Bytes (vocab=256) | 2,695,680 | 1.386 | **1.321** | 21.0 min | 12344 MB |
| **TinyTransformer** | Raw Bytes (vocab=256) | 4,921,856 | 2.150 | **2.077** | 2.2 min | 12344 MB |
| **TinyTransformer** | SentencePiece BPE (vocab=16000) | 8,952,320 | 1.558 | **1.708** | 2.3 min | 12344 MB |

---

## 2. Mathematical Attribution Breakdown

* **Architecture Effect** (Byte Transformer - Byte Mamba): **+0.756 BPB**
* **Tokenization Effect** (Tokenized Transformer - Byte Transformer): **-0.368 BPB**
* **Combined Total Effect** (Tokenized Transformer - Byte Mamba): **+0.387 BPB**

---

## 3. Dataset & Tokenizer Statistics

* **Training Bytes / Validation Bytes:** 1,465,682,927 / 77,141,207 bytes (95/5 split)
* **Measured Tokenizer Fertility:** **6.59 bytes per token**

---

## 4. Generated Artifacts
1. `amharic_model_comparison.png` — 3-Panel convergence, wall-clock time, and peak memory chart.
2. `best_mamba.pt`, `best_transformer_byte.pt`, `best_transformer_tokenized.pt` — Saved model checkpoints.

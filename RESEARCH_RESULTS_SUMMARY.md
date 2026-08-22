# Amharic Byte-Level Mamba vs. Transformer: Automated Research Report
*Generated on 2026-08-22 15:49:14*

---

## 1. Quantitative Results Table

| Model | Representation | Parameters | Final Train BPB | Final Val BPB (Bits-per-Byte) ↓ | Total Time (min) | Peak VRAM (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TinyMamba** | Raw Bytes (vocab=256) | 2,695,680 | 1.385 | **1.322** | 21.0 min | 12344 MB |
| **TinyTransformer** | Raw Bytes (vocab=256) | 4,921,856 | 2.171 | **2.100** | 2.1 min | 12344 MB |
| **TinyTransformer** | SentencePiece BPE (vocab=32000) | 13,048,320 | 1.602 | **1.579** | 2.7 min | 12344 MB |

---

## 2. Mathematical Attribution Breakdown

* **Architecture Effect** (Byte Transformer - Byte Mamba): **+0.778 BPB**
* **Tokenization Effect** (Tokenized Transformer - Byte Transformer): **-0.520 BPB**
* **Combined Total Effect** (Tokenized Transformer - Byte Mamba): **+0.257 BPB**

---

## 3. Dataset & Tokenizer Statistics

* **Training Bytes / Validation Bytes:** 1,465,682,927 / 77,141,207 bytes (95/5 split)
* **Measured Tokenizer Fertility:** **7.17 bytes per token**

---

## 4. Generated Artifacts
1. `amharic_model_comparison.png` — 3-Panel convergence, wall-clock time, and peak memory chart.
2. `best_mamba.pt`, `best_transformer_byte.pt`, `best_transformer_tokenized.pt` — Saved model checkpoints.

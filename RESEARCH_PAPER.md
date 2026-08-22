# Byte-Level Mamba vs. Transformer for Amharic

**Author:** Bereket Chemeda  
**Project:** Data and Compute-Efficient Generative AI  
**Code and Data:** https://github.com/BekiChemeda/ML-Research

---

## Simple Summary (Abstract)

Big AI models today need too much computer power and huge text datasets. But for Amharic, we only have a small amount of clean text in the whole world (less than 500 million words). 

Normal AI tools use a step called a tokenizer. A tokenizer cuts words into small pieces. But for Amharic and the Ge'ez alphabet, normal tokenizers do not work well. They break one single Amharic word into 7 or more small byte pieces. This makes normal Transformer models slow and expensive.

In this study, we tried a new way. We removed the tokenizer completely. We gave raw bytes directly to a fast model called Mamba. We tested three models on a real Nvidia RTX 3090 GPU with 1.46 GB of real Amharic text:

1. **TinyMamba (Raw Bytes):** Gets the best score (1.322 Bits-per-Byte). It only has 2.7 million parameters.
2. **TinyTransformer (Normal Tokenizer with 32,000 words):** Gets a worse score (1.579 Bits-per-Byte), even though it has 13 million parameters. More than 62% of its size is wasted just on the word dictionary list.
3. **TinyTransformer (Raw Bytes):** Gets the lowest score (2.100 Bits-per-Byte).

This shows that Mamba is better for Amharic bytes than Transformer. It is smaller, faster, and does not need any tokenizer. 

We also built a brain system. Like a human child, it can learn a new sentence in one second without retraining. When it goes to sleep, it replays what it learned into its memory so it never forgets.

---

## 1. Introduction

### Why Amharic is hard for AI
Amharic is spoken by more than 50 million people in Ethiopia. It uses its own writing system called the Ge'ez script. 

Most AI companies make models for English. English has billions of web pages. Amharic has very few clean web pages online. If we try to train normal huge models on Amharic, we quickly run out of data.

### The Tokenizer Problem (The Token Tax)
Before an AI reads text, a program called a tokenizer cuts words into numbers.

When a tokenizer sees English words, one word is usually one token. But when it sees Amharic, it gets confused. It breaks one Amharic letter into 3 raw computer bytes. A single Amharic word can become 7 to 10 tokens. This is called the Token Tax.

Because of this, normal Transformer models are already reading Amharic as bytes by mistake. But Transformers become very slow when sequences become long. 

Our question is simple: What if we use raw bytes on purpose, but with a fast model called Mamba that does not slow down?

---

## 2. Past Studies (Sources)

We used ideas from real research papers:

1. **Mamba (Gu and Dao, 2023):** Mamba is a new AI architecture. It works like a state space model. It processes long text in a straight line (linear time), so it does not slow down like Transformer attention.
2. **ByT5 (Xue et al., 2021):** Google researchers showed that AI can read raw bytes directly without any tokenizer. This helps languages with complex word grammar.
3. **MambaByte (Wang et al., 2024):** This paper showed that Mamba with raw bytes uses less computer power than Transformers.
4. **Token Tax Study (Lundin et al., 2026):** Proved that African languages pay a big penalty in speed and cost because of bad tokenizers.
5. **Brain Memory Theory (McClelland et al., 1995):** The human brain has two memory systems. The hippocampus learns new facts in one second. The cortex saves facts slowly during sleep.

---

## 3. Our Plan and Experiment

We trained three models on the exact same Amharic text to see what works best:

* **Model 1 (Byte-Mamba):** Reads raw bytes (vocabulary size = 256). Uses the Mamba design.
* **Model 2 (Byte-Transformer):** Reads raw bytes (vocabulary size = 256). Uses standard Transformer attention.
* **Model 3 (Tokenized-Transformer):** Uses a normal SentencePiece tokenizer with 32,000 subwords.

All three models trained for 5,000 steps on an Nvidia RTX 3090 GPU.

We measured the result using **Bits-per-Byte (BPB)**. In BPB, a lower number means the model understands the language better and predicts text with fewer mistakes.

---

## 4. Dataset

We only used clean text written by real people in Amharic:
* Amharic Wikipedia
* MasakhaNews Amharic
* XL-Sum Amharic News
* Cleaned web text from C4 and GlotCC

Total text size: **1,465,682,927 bytes (1.46 GB)**.  
We used 95% for training and 5% for testing.

---

## 5. What We Found (Results)

### Comparison Table

| Model Name | Input Type | Model Size (Parameters) | Test Score (Bits-per-Byte) | Winner |
| :--- | :--- | :--- | :--- | :--- |
| **TinyMamba** | **Raw Bytes** | **2.7 Million** | **1.322 BPB** | **Best Score (1st Place)** |
| **TinyTransformer** | 32,000 Subwords | 13.0 Million | **1.579 BPB** | 2nd Place |
| **TinyTransformer** | 16,000 Subwords | 8.9 Million | **1.708 BPB** | 3rd Place |
| **TinyTransformer** | Raw Bytes | 4.9 Million | **2.100 BPB** | 4th Place |

*(Note: Lower BPB score is better).*

### Key Findings:
1. **Mamba beats Byte-Transformer:** Mamba got 1.322 BPB while Transformer got 2.100 BPB. That is a big win (+0.778 BPB) for Mamba on raw bytes.
2. **Big Tokenizers waste too much space:** The 32k Transformer is huge (13 million weights), but 8.2 million weights (62.8%) are wasted just to store the word list.
3. **Mamba is small and smart:** TinyMamba has almost 5 times fewer parameters than the 32k Transformer, but it still gets a better score.

---

## 6. Example Text Generated by the Models

We gave the models Amharic test prompts to see how they write:

* **Prompt 1:** `ኢትዮጵያ በታሪኳ ` (Ethiopia in its history...)
  * **TinyMamba (Bytes):** `ኢትዮጵያ በታሪኳ እና ሌሎችም እንስሳት” ለምርት ስጫ መቆጣጠር` (Writes full, correct Ge'ez words).
  * **Transformer (Bytes):** `ኢትዮጵያ በታሪኳ ከቶችን ለክንት አላርት አያወው` (Broken letters).
  * **Transformer (32k Tokenizer):** Writes some words, but makes mistakes on rare words.

* **Prompt 2:** `ሰው ሰራሽ አስተውሎት ` (Artificial Intelligence...)
  * **TinyMamba (Bytes):** `ሰው ሰራሽ አስተውሎት የሚችሉ መረጃ አልተሰማሩ፡፡` (Good grammar).
  * **Transformer (32k Tokenizer):** Shows `⁇ ⁇ ⁇` error marks because it does not know rare tokens.

---

## 7. Learning Like a Human Brain (Future Step)

Normal AI models cannot learn new things after training without forgetting old things. This is called catastrophic forgetting.

Humans do not have this problem. A child hears a new word once and remembers it.

We built a brain system for Mamba with two parts:
1. **Fast Memory (Like Hippocampus):** When you teach the bot a new fact on Telegram, it saves it in 0.01 seconds without retraining.
2. **Slow Brain (Like Cortex):** The pre-trained Mamba model keeps basic Amharic grammar safe.
3. **Sleep Replay:** When the bot sleeps, it replays the new facts into Mamba so it permanently remembers them.

In our test, this brain system learned new news facts in one shot while keeping 100% of its old Amharic grammar.

---

## 8. Conclusion

This project shows three clear things:
1. Tokenizers hurt Amharic AI because they break Ge'ez letters into pieces and waste model memory.
2. Byte-level Mamba is better and lighter for Amharic than Transformers. It gives better language quality with 4.8 times fewer weights.
3. Brain-like fast memory lets Amharic AI learn new words every day like a human without forgetting.

---

## References

1. Gu, A. and Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.*
2. Xue, L. et al. (2021). *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models.*
3. Wang, L. et al. (2024). *MambaByte: Token-free Selective State Space Model.*
4. Lundin, J. M. et al. (2026). *The Token Tax: Systematic Bias in Multilingual Tokenization.*
5. McClelland, J. L. et al. (1995). *Why there are complementary learning systems in the hippocampus and neocortex.*
6. Tononi, G. and Cirelli, C. (2014). *Sleep and the price of plasticity.*

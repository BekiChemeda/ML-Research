# Byte-Level Mamba vs. Transformer for Amharic: A Preliminary Compute-Efficiency Comparison

*A preliminary study for the Data and Compute-Efficient Generative AI assignment (gheero, July 2026)*

---

## Abstract

*(This will be finished after the results come in. The draft below will be updated once Section 5 of the notebook gives final numbers.)*

Most recent progress in Generative AI comes from using more data, more parameters, and more compute. This path is expensive. It is hard to repeat. It does not work for languages that do not have much data. We study Amharic as a test case. Amharic is a language with complex word forms. It uses the Ge'ez script. There is less than 500 million tokens of real, non-translated Amharic text in the world. We show that normal subword tokenizers do not work well for Amharic. Tokenizer fertility (the number of tokens per word) is much higher for Amharic than for languages with the Latin alphabet and similar resources. Common tokenizers, like the one used in LLaMA, quietly fall back to breaking Ge'ez script into single bytes. Because of this, we compare a byte-level Mamba model (a fast, linear-time model) against a Transformer of the same size that also uses bytes, and a second Transformer of the same size that uses a normal tokenizer. This lets us test two things at once, in a controlled way: the effect of the architecture, and the effect of removing the tokenizer. *(Result summary to be added.)* We report this work as a small, early proof of concept. We list its limits clearly. We also describe a bigger, brain-inspired next step for later work.

---

## 1. Introduction

### 1.1 Why this matters

Today's biggest AI gains come from one method: bigger data, bigger models, bigger compute. This method works, but it leaves most of the world out. It needs huge amounts of data and money that most languages and researchers do not have. This is not a small problem for Amharic. Amharic is spoken by more than 50 million people. But when you add up all the real Amharic text that is publicly available anywhere, and remove the copies, it comes to less than 500 million tokens (Andersland, 2024). That is a hard limit, not a starting point. Any method that assumes GPT-3 levels of data simply cannot be used here. So the question this project asks is not "how do we build the best Amharic model." It is "how do we get the most out of a small, fixed amount of data and compute."

### 1.2 The real bottleneck: tokenization

Almost every modern language model assumes that turning text into tokens (using BPE or SentencePiece) is a neutral first step that does not affect fairness. This assumption breaks for Amharic. Tokenizers are trained mostly on text from languages that use the Latin alphabet and have lots of data. Ge'ez script gets very little space in their vocabulary. This has a real cost, sometimes called the "Token Tax" (Lundin et al., 2026). Studies show that tokenizer fertility (tokens per word) predicts model accuracy across 16 African languages, and doubling the token count roughly quadruples training cost. In our own check, we found that the LLaMA tokenizer does not have enough vocabulary space for Ge'ez script. It falls back to encoding Amharic letters as raw bytes. This means a single Amharic word can take 10 or more tokens, while the same idea in English takes just one token.

This is the key fact behind this whole project: **Amharic is already being processed at the byte level today, by accident, through a broken fallback, inside a Transformer whose cost grows fast as sequences get longer.** Our question is simple: what happens if we make that byte-level processing on purpose, and pair it with an architecture built to handle long sequences cheaply?

### 1.3 What we contribute

We are not proposing a new architecture. The assignment brief says clearly: "the choice of model is left entirely to the participant. The primary contribution should be the proposed methodology rather than the selected model." Our contribution is a method, not a model. It has three parts: (1) we find and measure the tokenizer problem for Amharic specifically, (2) we argue that a fast, byte-level architecture (Mamba) fixes the actual cause of this problem, not just a symptom, and (3) we design a controlled experiment that separates the effect of the architecture from the effect of the tokenizer. These two things are usually mixed together in comparisons like this. Keeping them separate lets us say clearly which one, if either, is doing the work.

---

## 2. Related Work

### 2.1 Data-efficient training

Past work shows that data quality matters more than data amount. MiniPile (Kaddour, 2023) shows that a carefully filtered 6GB subset of an 825GB corpus keeps about 98% of the model's performance. A 2026 ICLR study of 22 data-filtering methods finds that using an AI model to score data quality (called ASK-LLM) beats using all the data, while only using 60% of it. IMU-1 (2026) shows that combining smart architecture choices with smart optimizer choices can close most of the gap to a model trained on 56 times more tokens. These results are why we treat data collection (Section 4.1) as a core part of the method, not just a setup step.

### 2.2 Amharic and Ethiopian-language NLP

Past Amharic language model projects (Andersland, 2024; Azime et al.'s Walia-LLM, 2024) confirm the ~500 million token ceiling. They also show that most large "Amharic" training sets are actually mostly machine-translated from English. Our own data collection avoids this problem by only using text written by people in Amharic (Wikipedia, MasakhaNews, XL-Sum, and Common Crawl-based sources like C4 and GlotCC). EthioLLM (2024) and past surveys of Tigrinya NLP confirm that all existing multilingual models for Ethiopian languages use standard Transformers (XLM-R, mT5). No past work uses a linear-time architecture for any Ge'ez-script language. We checked this ourselves across four separate searches.

### 2.3 Morphology-aware and byte-level tokenization

Two lines of past work shaped, and corrected, our first idea. MoVoC (2025) already builds a tokenizer for Amharic and Tigrinya that respects word structure (using a tool called HornMorpho). It reports that this change alone gives only small gains in translation quality. MorphBPE (2025) tests a similar idea at a much bigger scale (300 million and 1 billion parameters, though not for Amharic) and finds real gains in training loss and speed. Separately, ByT5 (Xue et al., 2021) shows that models can be trained directly on raw bytes, with no tokenizer at all, and that this works especially well for low-resource languages with complex word structure. MambaByte (2024) is the most important past work for this project. It shows that pairing byte-level input with the Mamba architecture reaches the same training loss as a Transformer while using less than one third of the compute. This works because Mamba processes sequences in linear time, while Transformer attention gets much more expensive as sequences get longer.

### 2.4 Architectures that avoid the quadratic cost of attention

A 2026 survey of these architectures (state-space models, linear attention, linear RNNs) finds they are especially strong compared to Transformers when models are small, under about 2 billion parameters, which is the scale of this project. At much bigger scales (70 billion parameters and up), only mixed architectures stay competitive. These architectures are, in theory, no more powerful than Transformers (both are limited to the same computational class). They are also known to struggle at tasks needing exact memory of something far back in a long sequence, because they compress their memory into a fixed size. We state this limit clearly rather than ignore it.

### 2.5 Where this project sits

No past work combines byte-level input with this kind of fast architecture for Amharic or any other Ge'ez-script language. This is the gap our project fills, at a small scale, as an early study.

---

## 3. Hypothesis

**H1 (architecture):** When given the same number of parameters and training steps, a byte-level Mamba model will get a lower loss score, take less time to train, and use less memory than a byte-level Transformer of the same size. This should happen because Mamba's linear-time design avoids the growing cost that Transformer attention has on longer sequences.

**H2 (tokenization):** When given the same architecture (Transformer) and the same size, byte-level input will be more data-efficient (lower loss score at the same number of training steps) than input from a normal SentencePiece/BPE tokenizer. This should happen because Amharic's tokenizer problem means the tokenized model is wasting some of its learning capacity on broken-up word pieces instead of meaning.

**How we would prove this wrong:** If byte-Mamba does not beat byte-Transformer on training time and memory, H1 is wrong. This would mean the linear-time advantage is not showing up at this size or sequence length. If byte-Transformer does not beat tokenized-Transformer, H2 is wrong. This would mean the tokenizer problem is not the real bottleneck, or its effect is too small to see at this scale.

---

## 4. Method

### 4.1 Collecting and cleaning the data

We used these sources: Amharic Wikipedia (`wikimedia/wikipedia`, the `20231101.am` version), MasakhaNews Amharic (train, validation, and test), XL-Sum Amharic, and streamed Amharic text from `allenai/c4` (the current, working replacement for the old, broken `mc4`) and GlotCC, up to a 2GB limit. We did not use OSCAR, because it needs manual approval to access, which is outside our control.

Every document from every source goes through the same cleaning steps before we use it: we remove control characters, we drop documents shorter than 50 characters, we drop documents where less than 30% of characters are in the Ge'ez script block (these are likely wrong-language text that leaked in from noisy Common Crawl sources), and we remove exact duplicate documents using a hash check. We did not do the kind of deep, embedding-based duplicate removal that MiniPile uses. This is a clear choice, not something we forgot: that method needs its own extra model and clustering step, which is too much extra machinery for a corpus this size.

*(Final corpus size, the breakdown by source, and the cleaning numbers from `clean_stats` will go here once the data step has fully run.)*

### 4.2 The three models

We built three versions, kept as close as possible in total parameter count:

1. **Byte-Mamba**: a simple, from-scratch selective state-space model (S6), built in plain PyTorch, using raw bytes as input (vocabulary size 256). It follows the standard Mamba design: input-dependent Δ, B, and C values, and a step-by-step scan (not the fast CUDA version, to avoid needing to build custom CUDA code, which is a common way this kind of setup breaks). We checked our version against the standard reference code (`mamba-minimal`) to confirm it is correct.
2. **Byte-Transformer**: a standard Transformer, using the same byte-level vocabulary (256), with the same depth and width as the Mamba model. This isolates the effect of the architecture on its own.
3. **Tokenized-Transformer**: the same architecture as (2), but using a SentencePiece BPE tokenizer (vocabulary size 8,000) trained on the same text. This isolates the effect of tokenization on its own.

All three models use: `d_model=256`, 6 layers, sequence length 512 (in bytes or tokens, depending on the model), batch size 32, the AdamW optimizer, a learning rate of 3e-4, and the same fixed random seed (1337) for `torch`, `numpy`, and CUDA.

### 4.3 How we measure results fairly

A tokenized model's loss is measured per token. A byte-level model's loss is measured per byte. These are not the same unit, so comparing them directly would be unfair, like comparing kilometers to miles without converting. We convert all three models' scores into one shared unit: bits per byte of the original text. We do this using a measured ratio of bytes per token for the tokenized model. This way, all three numbers mean the same thing, and the comparison is not just an accident of vocabulary size.

We also report training time and peak memory use at the same number of steps. We measured these on real hardware first, with a short test run, instead of guessing from theory. Our early guess turned out to only give a rough range, since the Mamba scan runs as a Python loop and its real speed depends on the specific machine.

### 4.4 Why three models, not two

Using three models fixes a mistake we caught in our own early plan. If we had only compared byte-Mamba against tokenized-Transformer (the two most natural end points), we would not be able to tell whether any difference came from the architecture, the tokenizer, or both. The three-model design lets us split this apart:

- **Architecture effect** = byte-Transformer score minus byte-Mamba score (same input type, different architecture)
- **Tokenization effect** = tokenized-Transformer score minus byte-Transformer score (same architecture, different input type)
- **Combined effect** = tokenized-Transformer score minus byte-Mamba score (the full comparison our original idea was about)

### 4.5 A small extra check: does entropy line up with word structure

As an extra, smaller check, not a main result, we look at whether the byte-Mamba model's own uncertainty (measured as entropy, at each byte it predicts) rises near real Amharic word-part boundaries, even though the model was never told what a word part is. We check this against real answers from HornMorpho (Gasser et al.), an actively maintained tool that breaks Amharic words into their parts. This is a small, hand-checked sample (20 words or fewer), not a statistical result.

---

## 5. Experimental Setup

**Corpus, by source (real run, Kaggle P100 session):**

| Source | Bytes kept |
|---|---|
| Amharic Wikipedia (`wikimedia/wikipedia`) | 21,364,587 |
| MasakhaNews Amharic (train) | 6,746,689 |
| MasakhaNews Amharic (validation) | 793,466 |
| MasakhaNews Amharic (test) | 1,937,787 |
| XL-Sum Amharic | 33,567,977 |
| `allenai/c4` Amharic (streamed, stopped at the 1GB cap) | 935,609,394 |
| GlotCC | not reached, cap was already hit by `c4` |
| **Total** | **1,000,129,152 bytes (1.00 GB)** |

Split into `train.bin` (950,122,694 bytes) and `val.bin` (50,006,458 bytes), a 95/5 split. The whole data collection and cleaning step took 278.3 seconds.

**Cleaning numbers:** out of 142,880 documents seen across all sources, 128,962 were kept (90.3%). 1,367 were dropped for being too short. 609 were dropped as exact duplicates. **11,942 (8.4% of everything pulled) were dropped by the Ge'ez-script ratio check as likely wrong-language text.** This is real, direct evidence that the cleaning step is needed, not just a precaution, since almost 1 in 12 documents pulled from the "Amharic" labeled streams were not actually usable Amharic text.

- Parameter counts for all three models: **to be added**
- Measured bytes-per-token ratio: **to be added**
- Measured milliseconds per training step for each model, and the step count we chose because of it: **to be added**
- Total training steps completed per model: **to be added**

---

## 6. Results

*(This section is a placeholder until the Kaggle run finishes. Fill it in directly from Section 5 of the notebook: the three loss curves, the training time and memory comparison, and the printed breakdown of architecture effect, tokenization effect, and combined effect. Include the comparison plot, `comparison.png`. Report the numbers exactly as printed. Do not round them in a way that suggests more precision than a single run, without full convergence, can actually support.)*

### 6.1 Numbers: loss, training time, memory

*To be added*

### 6.2 Splitting the effect: architecture vs. tokenization

*To be added*

### 6.3 The entropy-vs-word-structure check

*To be added. Report it as "N out of M words showed a visible match between entropy peaks and HornMorpho boundaries," not as a formal statistic.*

---

## 7. Discussion

*(The logic below is written now. Once real results come in, only one of these cases will apply. Keep that one and remove the rest.)*

- **If H1 is true (Mamba wins on speed and memory) and H2 is true (byte-level wins on data efficiency):** this supports our main idea. Amharic's tokenizer problem is real, and it can be fixed by removing the tokenizer and using an architecture built to handle the longer sequences that causes. The next step would be growing the corpus and adding the word-structure-aware ideas in Section 9.
- **If H1 is true but H2 is not:** the speed and memory gain is coming from the architecture, not from removing the tokenizer. This would be worth checking against MoVoC's finding that tokenizer changes alone gave only small gains. It would not actually contradict MoVoC, since MoVoC only tested tokenizer changes on a fixed Transformer, never a different architecture.
- **If H1 is not true:** either the sequence length we used (512) is too short for Mamba's advantage to show up against a still-fast attention method, or our step-by-step (non-CUDA) Mamba code is too slow at this small scale for the advantage to appear, even if it would appear at the longer sequence lengths MambaByte was tested at. This would be a real, useful negative result. We would report it as that, not reframe it as a success.
- **If H2 is not true:** the tokenizer fertility problem, while real (see Sections 1.2 and 2.1), may not turn into a measurable data-efficiency gain at this small scale. This would match, not contradict, MoVoC's own finding that tokenizer changes alone gave only small gains.

---

## 8. Limitations

Stated plainly, without softening them:

- **Small corpus.** The corpus we actually built is tens of megabytes to a few gigabytes, not the roughly 4GB that UnifiedCrawl reaches. Reaching that would need a separate, bigger project: running their own multi-source Common Crawl collection pipeline, not just downloading existing datasets.
- **Not trained to convergence.** Training stops at a fixed number of steps, not when the model stops improving. The bits-per-byte numbers show a fair comparison at that fixed point, not the models' full potential.
- **One random seed.** We used one seed (1337) per model. We do not have a measure of how much results vary between different seeds. Small differences between models should not be over-read.
- **Only exact-match cleaning.** We remove exact duplicate documents. We do not catch near-duplicates or documents that are similar but not identical.
- **No downstream task testing.** Our results are only about language modeling loss. We did not test on real tasks like AfriSenti (sentiment), MasakhaNews (topic classification), or AmharicQA (question answering), which would be needed to know if a lower loss score actually helps with real use.
- **Our Mamba code is a reference version, not the fast official one.** We chose the plain, step-by-step version to avoid needing to build custom CUDA code, which often breaks setups like this. This means our training time numbers reflect our specific code, not Mamba's best possible speed.
- **GlotCC is best-effort.** We were not able to fully confirm its exact loading method before running this. The notebook reports honestly whether it loaded or not.

---

## 9. Future Work

Two directions we did not attempt in this study. We keep them clearly separate from the results above, since they are ideas for later, not things we have already shown to work.

**Growing the corpus.** Running UnifiedCrawl's own data collection method (reported to cost under $4 and take under a day on a normal computer) to get closer to the ~4GB Amharic text ceiling. Combining this with MiniPile-style, embedding-based quality filtering, instead of the simple exact-match cleaning used here.

**Brain-inspired learning after training ("learning like a baby keeps learning").** This means building a system with two parts: the slow, trained model tested in this paper, plus a fast memory module that updates using a Hebbian rule (a simple, brain-inspired update method), based on a design called an Engram Neural Network. New information would be stored right away in the fast memory, without touching the slow model's weights. Every so often, information that gets used often would be copied into the slow model's weights through a replay process. This mirrors how the brain is believed to move memories from the hippocampus into longer-term storage during sleep. Some past work (Larimar; Hebbian fast-weights inside Transformer layers) explores parts of this idea, but no past work combines a lasting, truly Hebbian memory with weight-level consolidation on a non-Transformer model like Mamba. This is the open gap we found during our own check of past work. A second idea for later is a HornMorpho-based self-check loop: the model generates Amharic text, and HornMorpho (a free, always-available tool, not a person) checks if the grammar is correct. This gives extra, correctness-checked training signal without needing more real text or human labels. This idea could be added on top of whichever model this future work uses.

---

## 10. Conclusion

*(To be written last, once Section 6 is complete. It should say, in one paragraph, what was actually shown, not what we hoped to show, how confident we are in it, and what the single most useful next experiment would be.)*

---

## References

*(Full links are in `resources/MANIFEST.md`. Key sources by section:)*

- Andersland, M. (2024). *Amharic LLaMA and LLaVA: Multimodal LLMs for Low Resource Languages.* arXiv:2403.06354.
- Azime et al. (2024). *Walia-LLM: Enhancing Amharic-LLaMA by Integrating Task-Specific and Generative Datasets.* arXiv:2402.08015.
- Kaddour, J. (2023). *The MiniPile Challenge for Data-Efficient Language Models.* arXiv:2304.08442.
- Lundin, J. M. et al. (2026). *The Token Tax: Systematic Bias in Multilingual Tokenization.* arXiv:2509.05486.
- (MoVoC authors) (2025). *Morphology-Aware Subword Construction for Ge'ez Script Languages.* arXiv:2509.08812.
- (MorphBPE authors) (2025). *MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for Efficient LLM Training Across Morphologies.* arXiv:2502.00894.
- Xue, L. et al. (2021). *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models.*
- (MambaByte authors) (2024). *MambaByte: Token-free Selective State Space Model.* arXiv:2401.13660.
- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.
- Gasser, M. *HornMorpho: a system for morphological processing of Amharic, Oromo, and Tigrinya.* github.com/hltdi/HornMorpho.
- (Larimar authors) (2024). *Larimar: Large Language Models with Episodic Memory Control.* arXiv:2403.11901.
- (ENN authors) (2025). *Hebbian Memory-Augmented Recurrent Networks: Engram Neurons in Deep Learning.* arXiv:2507.21474.

*(Some author names are left as placeholders where we could not confirm them during research. Check these against the actual PDFs in `resources/` before final submission.)*

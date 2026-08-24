# Resources manifest

35 papers archived during research for the Amharic data/compute-efficient
generative AI project. Organized by theme below (chronological/flat order
made this hard to navigate once it grew past ~15 files). See also
`notes-cognitive-science-foundations.md` for synthesized notes (not just a
paper list) on the cognitive-science thread specifically.

## Amharic / low-resource data & training efficiency

| File | Source | Relevance |
|---|---|---|
| amharic-llama-llava-multimodal.pdf | arxiv.org/abs/2403.06354 | Establishes ~500M-token ceiling on organic Amharic text; 88% of their corpus is MT-synthetic |
| walia-llm-amharic.pdf | arxiv.org/abs/2402.08015 | Amharic instruction dataset + task corpora inventory (AmharicQA, AfriSenti, MasakhaNews, etc.) |
| whisper-amharic-finetuning.pdf | arxiv.org/abs/2503.18485 | Amharic ASR datasets inventory; homophone normalization for Ge'ez script |
| unifiedcrawl-lowresource-adaptation.pdf | arxiv.org/abs/2411.14343 | Amharic flagship case study: ~4GB raw text via full-CC-archive aggregation |
| glotcc-minority-language-commoncrawl.pdf | arxiv.org/abs/2410.23825 | 1275-language CC corpus, pre-built on HuggingFace, no Amharic-specific numbers in-paper |
| sozkz-efficient-slm-kazakh.pdf | arxiv.org/abs/2603.20854 | Direct methodological template: small dedicated-tokenizer model beats larger multilingual models |
| imu1-sample-efficient-slm-pretraining.pdf | arxiv.org/abs/2602.02522 | Architecture+optimizer efficiency recipe, 8-56x token efficiency (needs 8xGPU node) |
| minipile-data-efficient-lms.pdf | arxiv.org/abs/2304.08442 | Embedding-cluster quality curation: 137x smaller corpus, ~98% of performance retained |
| 2026-iclr-data-efficient-llms.pdf | proceedings.iclr.cc (2026) | ASK-LLM quality filtering beats full-data training at 60% of data |
| ethiollm-multilingual-ethiopian-languages.pdf | arxiv.org/abs/2403.13737 | XLM-R/mT5-based model for 5 Ethiopian languages - standard Transformer, no Mamba |
| tigrinya-nlp-current-state-survey.pdf | arxiv.org/abs/2507.17974 | Tigrinya NLP survey - confirms no Mamba/SSM work exists for Ge'ez-script languages |
| amharic-embedding-retrieval-tokenization.pdf | arxiv.org/html/2505.19356 | Fertility correlates with retrieval accuracy; existing Amharic pretraining corpora only ~300M tokens |

## Tokenization & byte-level modeling (prior-art corrections + the chosen method)

| File | Source | Relevance |
|---|---|---|
| movoc-geez-morphology-tokenization.pdf | arxiv.org/abs/2509.08812 | HornMorpho-constrained BPE for Amharic/Tigrinya - already exists, only modest MT gains found (corrected our novelty claim) |
| morphbpe-efficient-llm-training.pdf | arxiv.org/abs/2502.00894 | Morphology-constrained BPE validated at 300M/1B-param LM pretraining scale (not Amharic) |
| byt5-token-free-byte-models.pdf | aclanthology.org/2022.tacl-1.17 | Byte-level T5, no vocabulary, robust to noise, suited to low-resource languages |
| egalitarian-language-representation-tokenizers.pdf | arxiv.org/abs/2409.11501 | Tokenizer fairness/bias across languages |
| token-tax-multilingual-bias.pdf | arxiv.org/abs/2509.05486 | Fertility reliably predicts accuracy across 16 African languages; 2-15x token cost variation - the core motivating finding |
| mambabyte-token-free-ssm.pdf | arxiv.org/abs/2401.13660 | Mamba + byte-level reaches Transformer-quality loss in <1/3 the compute - the method this project's Phase 1 is built on |

## Sub-quadratic architectures / Mamba / SSM

| File | Source | Relevance |
|---|---|---|
| ssm-survey-transformer-alternative.pdf | arxiv.org/abs/2404.09516 | SSM taxonomy, CV-heavy, background reference |
| end-of-transformers-subquadratic-survey.pdf | arxiv.org/abs/2510.05364 | Sub-quadratic architectures competitive specifically in sub-2B regime; same TC0 expressivity ceiling as Transformers |
| mamba-asr-south-african-languages.pdf | arxiv.org/abs/2607.01502 | Mamba ASR evaluated on South African languages - closest adjacent African-language Mamba work found |
| mamba-speech-applications-capability.pdf | arxiv.org/abs/2406.16808 | General survey of Mamba's capability in speech tasks |

## Cognitive science / brain-inspired learning (background + the "act like a baby" thread)

| File | Source | Relevance |
|---|---|---|
| free-energy-principle-friston.pdf | Friston, Nat Rev Neurosci | Foundational free-energy principle / predictive coding theory |
| free-energy-principle-ml-neuroscience-applications.pdf | arxiv.org/abs/2107.00140 | PC, active inference, and "activation relaxation" as local-learning alternatives to backprop |
| predictive-coding-vae-biological-connections.pdf | arxiv.org/abs/2011.07464 | Predictive coding / VAE mathematical connections |
| predictive-coding-alternatives-to-backprop.pdf | Oxford thesis (Song) | PC training algorithm: local relaxation, learned backward weights; 97.2% vs 97.8% MNIST vs backprop |
| temporal-predictive-coding-long-range.pdf | arxiv.org/abs/2602.18131 | PC+RTRL trains a *real* generative byte-level LM (WikiText-103) matching BPTT - narrowed our novelty claim for the PC-backbone idea |
| engram-memory-encoding-retrieval.pdf | arxiv.org/abs/2506.01659 | Theoretical model of engram formation/consolidation/sparse retrieval |
| hebbian-memory-augmented-recurrent-nets.pdf | arxiv.org/abs/2507.21474 | ENN: concrete Hebbian-trace + content-addressed memory architecture, the basis for the "Engram Memory Module" idea |
| hebbian-gradient-plasticity-transformers.pdf | arxiv.org/abs/2510.21908 | Genuine Hebbian outer-product fast-weights inside Transformer FFN - toy-task scale only, narrowed our novelty claim |
| neuromodulator-diffusion-credit-assignment.pdf | arxiv.org/abs/2603.08949 | ModProp: sparse local credit diffusion, near-parity with BPTT, no language-modeling application found |
| brain-as-blueprint-brain-inspired-ai-survey.pdf | arxiv.org/abs/2511.04455 | Survey: interneuron microcircuits, feedback alignment, STDP, sparse continual learning |
| larimar-episodic-memory-llm.pdf | arxiv.org/abs/2403.11901 | Gradient-free one-shot episodic memory (Kanerva Machine) on a Transformer, hippocampus/cortex framing - substantially scoops the naive version of the "two-memory-system" idea |
| do-llms-think-like-brain.pdf | arxiv.org/abs/2505.22563 | LLM-brain fMRI/MEG activation alignment findings |
| brain-llm-alignment-training-data-not-typology.pdf | arxiv.org/abs/2605.23032 | Important critical nuance: brain-LLM alignment tracks training data, not language typology |
| computational-models-language-processing-brain-survey.pdf | arxiv.org/abs/2403.13368 | Survey of computational models of brain language processing |

## Tools identified (not papers, not downloaded here)

- **HornMorpho** v5.3.5+ (github.com/hltdi/HornMorpho) — actively maintained, open-source, rule-based FST morphological analyzer/generator for Amharic, Oromo, Tigrinya. Install verified: wheel-based `pip install`, API `hm.anal('a', word)`. Used directly in `code/amharic_mamba_vs_transformer.ipynb` Section 6.

## Known data availability numbers (Amharic)

- Amharic Wikipedia: 22MB | Amharic News Corpus: 150MB | OSCAR: ~380-500MB | mC4/allenai-c4: 1.2GB | CC-100: 130MB
- Organic Amharic text ceiling from combining known open sources without dedup: <500M tokens (Andersland 2024)
- UnifiedCrawl (full-archive CC aggregation): ~4GB raw Amharic text, ~3x mC4 / ~10x OSCAR / ~30x CC-100 — requires running their own multi-archive pipeline (~1 day), not just a dataset download
- GlotCC: pre-built 1275-language corpus on HuggingFace (cis-lmu/GlotCC-v1) — exact Amharic slice loading convention not verified by execution; notebook tries two approaches defensively

# Cognitive science foundations — notes (started 2026-08-21)

Purpose: grounding for the CLAUDE.md Section 34 "Ultimate Question" — why humans need
so much less data than LLMs to acquire capability. Not directly about Amharic; this is
first-principles background that may inform architecture/inductive-bias decisions later
(e.g. it's the conceptual justification for the Grammar-Offloaded / HornMorpho idea:
give the model cheap structure for free, the way evolution gives infants cheap priors
for free, instead of making gradient descent rediscover it from data).

## 1. What infants are born with (Known, well-replicated)

- **Prenatal auditory learning**: auditory system functional ~3rd trimester, before vision.
  Newborns prefer their mother's voice (DeCasper & Fifer 1980) and a specific story read
  repeatedly in utero (DeCasper & Spence 1986) — evidenced via non-nutritive sucking-rate
  preference. Newborns also prefer their native language's prosodic rhythm (Mehler et al.).
  => First learned "content" is prosodic/rhythmic pattern + mother's voice, not anything visual.
- **Face-like configuration bias** (Johnson & Morton): newborns track face-like arrangements
  of high-contrast blobs more than scrambled equivalents — a subcortical, pre-wired bias
  toward face-*like* structure, not learned face recognition.
- **Approximate number sense**: Izard et al. (2009) — newborns match number of sounds in an
  auditory sequence to number of objects in a simultaneous visual array, above chance.
- **Core Knowledge Theory (Spelke)**: proposes innate systems for objects, number, space,
  and agents/goal-directed action, evidenced mainly via violation-of-expectation paradigms.
  Influential but NOT uncontested — active debate over whether it's a clean modular
  subdivision (see Behavioral and Brain Sciences commentary literature).

## 2. Format of knowledge in the brain

- Known: not a "file at an address" — stored as patterns + strength of synaptic connections
  between neurons. Hebb's Rule (1949): "cells that fire together, wire together" — repeated
  co-activation strengthens the connection, and that strengthened connection IS the memory.
- Known: distributed, not localized — a memory involves many neurons across regions;
  memories move from hippocampus to distributed cortical storage over time (systems
  consolidation, partly during sleep).
- **Open/unresolved**: the exact "code" (vector-like population code? timing/phase code?
  something else?) is genuinely still debated in neuroscience — no agreed answer analogous
  to how we can exactly describe what's stored in a Transformer's weight matrix.

## 3. What happens mechanistically when we "think"

- Known, physical level: cascading electrical spikes across neurons, triggering
  neurotransmitter release at synapses.
- Leading but not settled theory: **predictive processing** — the brain continuously
  generates predictions about incoming input and "thinking"/perception is largely
  computing and minimizing prediction error. Notable: this is structurally similar to
  an LLM's "predict the next token" objective — a shared design principle, not proof of
  shared mechanism.
- Another proposed (debated) framework: **Global Workspace Theory** (Baars) — conscious
  thought = information broadcast widely enough that multiple specialized brain systems
  can access it simultaneously.
- Honest limit: there is no complete, agreed, step-by-step mechanistic account of a
  specific thought, unlike the fully specified math of a Transformer forward pass. This
  is a genuinely open problem, not something being glossed over here.

## 4. How LLMs mirror (and don't mirror) minds

**Real parallels:**
- Distributed/spread-out representation (both brain and LLM FFN layers).
- Prediction as the core mechanism (predictive processing theory <-> next-token prediction).
- Both have "attention"-like dynamic prioritization of relevant information, though the
  underlying computation is unrelated (dot-product similarity vs. unknown biological circuitry).

**Real, important disanalogies (these matter more than the parallels for our research):**
- LLMs learn via backpropagation — one global, precisely computed gradient signal applied
  across the whole network at once. No evidence brains do anything like this; biological
  synaptic plasticity is local and not fully understood ("credit assignment problem" —
  directly one of the open questions our own CLAUDE.md Section 28/34 tells us to pursue).
- No embodiment: LLMs only see text tokens; brains are embedded in a body interacting with
  a physical/social world in real time. Plausible (not proven) partial explanation for why
  humans need less data.
- Frozen vs. continuously alive: LLM weights are static after training; brains keep
  rewiring continuously, including via sleep-based consolidation. No LLM equivalent exists.

**Bottom line**: LLMs are a partial, coarse mirror of minds — shared high-level design
principles (distributed storage, prediction-as-mechanism), but diverging sharply on the
actual learning algorithm and on having any embodied, continuous experience. That
divergence, not the similarity, is the more promising place to look for a fundamentally
different/more efficient AI paradigm.

## 5. How the brain uses its innate "birth tools" (mechanism, not just inventory)

- Correction to section 1: newborn face preference is likely NOT a dedicated face-detector —
  more probably generic visual circuitry (contrast/configuration sensitivity) that happens to
  overlap with faces. Updated per own theory-revision rule when evidence contradicts.
- Tools work as **attention directors**, not finished capabilities — they bias what gets looked
  at/listened to, curating the data the rest of the brain learns from (no explicit labeling needed).
- **Perceptual narrowing**: broad initial bias -> rapid experience-driven specialization via
  ordinary Hebbian plasticity. Reversible: 2-3 weeks of exposure to other-race faces restored
  discrimination ability in infants who'd narrowed away from it. Tool = flexible starting bias,
  not a locked-in outcome.
- **Statistical learning** (Saffran et al.): 8-month-olds segment continuous speech into words
  using only transitional-probability tracking, after ~2 minutes exposure. General-purpose
  mechanism, made efficient *for language* because prenatal prosody-preference already aims
  attention at speech specifically.
- **Staged bootstrap**: statistical-learning output (word forms) becomes the next stage's
  training input (word meaning, syntax) - layered, not one flat end-to-end process.
- **Embodiment**: reflexes (grasping, rooting) generate self-produced, automatically-correlated
  multi-sensory training data (touch+vision+proprioception together) through action - a
  mechanism no passive corpus can replicate.
- Refined framing for our own research idea: an injected prior's value is not "hand out correct
  answers for free" but **"narrow the hypothesis space so scarce data gets spent on what
  actually varies, not on rediscovering already-regular structure."** More precise version of
  the Grammar-Offloaded/HornMorpho justification.

## 6. Energy: brain vs. LLMs

- Known numbers: brain ~12-20W; single GPU 300-700W; GPT-3 training ~1287 MWh; optimized
  LLM inference query ~0.3-0.6 Wh (up to 33 Wh for long reasoning-heavy prompts); global 2026
  inference projection ~1050 TWh.
- Popular "brain is 100,000-225,000x more efficient" framing is **contested** - comparing
  energy-per-operation across biological spikes vs. floating-point ops isn't fully apples-to-apples,
  and the brain's 20W also runs the whole body, not just "thinking." Flag as popular but disputed.
- Real mechanistic reasons for the (still-real) gap: (1) sparse/event-driven spiking vs. dense
  synchronous compute (Transformers touch every weight every token regardless of relevance),
  (2) in-memory computing (synapse = storage+compute) vs. von Neumann memory/compute
  separation (data movement, not arithmetic, dominates GPU energy), (3) different physical
  substrate entirely. Neuromorphic chips close some (10-100x) but not all of this gap.

## 7. Are we more like LLMs or Richard Sutton's RL?

- Sutton's actual position (verified): LLMs are "passive imitators" predicting what a person
  might say, not what will happen in the world; no real goal. His alternative ("era of experience")
  argues intelligence requires continual action->sensation->reward learning, on the job, like
  animals - predicts LLMs will be seen as "a momentary fixation."
- Real neuroscience backing: dopamine neurons' firing pattern precisely matches reward-
  prediction-error, the exact quantity temporal-difference (TD) RL computes (Schultz 1997 and
  follow-ups) - one of the cleanest confirmed bridges between a specific RL algorithm and
  actual brain mechanism.
- My synthesis (not settled consensus - flagged as such): training *paradigm* is much closer
  to Sutton's continual/embodied/reward-modulated framing than to LLM's offline-frozen-corpus
  paradigm. But underlying moment-to-moment mechanism looks like a hybrid: predictive coding
  (self-supervised prediction, LLM-like in spirit but continuous/multimodal) + dopamine/TD
  reward modulation on top + innate priors shaping what gets attended to/predicted.

## 8. When does the brain start developing, and is it ever truly "blank"?

- Neural tube forms starting day 16-18 (week 3) of gestation from ectoderm, fuses by day 28
  (end of week 4) into 3 primary brain vesicles -> all major adult brain structures. This is the
  literal first physical existence of neural tissue - before this, nothing.
- **Never actually "blank"**: goes directly from no-tissue to genetically-pre-wired (molecular
  guidance cues route axons before any activity is possible) to self-generated spontaneous
  activity (e.g. retinal waves from ~embryonic day 16, before eyes can respond to light) that
  uses Hebbian rules to refine coarse wiring - the brain bootstraps its own internal "practice
  data" before the real world is available.
- "Knowing it can save" is a category error - storage capacity is a physical property (NMDA
  receptor/calcium-driven LTP machinery) that switches on via a genetically-timed developmental
  schedule (e.g. the GluN2B->GluN2A "NMDAR developmental switch" around birth), not a
  moment of realization.
- Different memory structures mature at different rates: amygdala (implicit/emotional) early,
  hippocampus (explicit/episodic) much later, continuing after birth - leading explanation for
  infantile amnesia. Prenatal storage clearly happens (DeCasper studies) but not in
  hippocampal-dependent explicit format, hence unretrievable by adult memory.
- Separate, genuinely unresolved question: when does anything like subjective experience
  begin? Thalamocortical connectivity (~24-28 weeks) is the commonly cited landmark in that
  debate, but whether that's sufficient for experience is actively contested, not settled -
  flagged explicitly as a different, harder question than "when does tissue exist."

## Open threads to follow up on later
- Is there a concrete, implementable analog of "local, non-backprop plasticity" worth
  testing at small scale (Hebbian-style or predictive-coding-style local learning rules)
  as an alternative training mechanism, separate from the Amharic-specific architecture work?
- Does the "cheap built-in structure + scarce data learns only what varies" framing (infant
  core knowledge <-> HornMorpho-offloaded morphology) suggest other places in the Amharic
  pipeline where a symbolic/rule-based prior could replace something we're currently asking
  gradient descent to learn from scratch?

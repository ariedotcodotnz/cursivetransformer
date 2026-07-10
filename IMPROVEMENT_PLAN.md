# Cursive Transformer — Improvement Plan

> Working reference doc. Captures goals, decisions, technical roadmap, and rationale
> for improving the generative handwriting model. Created 2026-05-30.

## 1. Context: what the model currently is

The **Cursive Transformer** (Greydanus & Wimpee, 2025) generates online cursive
handwriting (pen-stroke trajectories) conditioned on ASCII text.

- **Architecture:** GPT-style decoder (derived from Karpathy's `makemore`) with
  **cross-attention** onto encoded ASCII characters, so text conditions the strokes.
  Recently refactored to a **MatFormer** (elastic sub-networks) + a separate MLP class.
- **Representation:** pen-stroke offsets tokenized as **discrete tokens in polar form**
  — `(theta, radius+pen_state)` per offset — with `WORD_TOKEN` separators between words.
  Polar uses ~2 tokens/stroke vs ~3 for cartesian, allowing longer context.
- **Training:** heavy **data augmentation** (per-sample randomized downsampling 60–70%,
  italicization) + steady dataset growth (now ~3.5k words → ~250k–500k stitched
  multi-word training examples). Logged via Weights & Biases.
- **Sampling:** warm-up on random dataset examples, three-words-at-a-time generation,
  split on `WORD_TOKEN`, line-wrap into paragraphs, per-word regeneration to fix typos.

### Documented plateau (from README progress log)
1. **Spelling / legibility** — model garbles characters or misspells, especially needing
   per-word regeneration. This is the #1 stated limitation.
2. **Delayed diacritics** — dotting i/j and crossing t/x at end-of-word was so hard they
   changed the *dataset* to dot/cross immediately after the character instead.
3. **Style coherence** — italicization slope varies word-to-word; they removed randomized
   italics to get coherence at the cost of style diversity.

## 2. Goals & decisions (locked in)

User asked to "implement the latest and best research to produce highly realistic
handwriting," then delegated all scoping decisions ("you choose"). Decisions:

| Dimension | Decision | Rationale |
|---|---|---|
| **Primary goal** | Balanced quality, biased toward **legibility/spelling** + **stroke realism** | These are the documented plateau and the most visible quality wins. |
| **Scope** | **Drop-in modernizations + classifier-free guidance**, keep the discrete-token AR pipeline. Add an **optional MDN output head** only if cleanly swappable. | Keeps the pipeline intact and testable; avoids a risky big-bang rewrite to diffusion. MDN is the biggest single realism lever but a paradigm fork. |
| **Compute** | **I write + smoke-test the code; user runs full training.** | Full runs are multi-hour A100 jobs; not appropriate to launch here. Every change must stay within ~single-A100 / few-hours budget. |
| **Method** | **Incremental, isolated, measured.** Each change behind a flag, validated against baseline. | So we know what actually helped, and can revert cleanly. |

## 3. Technical roadmap (ordered by leverage / risk)

### Tier 1 — Architecture modernization (low risk, reliable gains, drop-in)
Replace dated `makemore`/GPT-2-isms with current standard transformer components:
- **RoPE** (rotary positional embeddings) for the decoder self-attention, replacing
  learned absolute position embeddings. Better length generalization — directly relevant
  since they keep growing the context window (1.1k → 1.5k tokens) and wrap paragraphs.
- **RMSNorm** instead of LayerNorm (faster, stable, current standard).
- **SwiGLU** MLP instead of GELU MLP (better quality per param).
- **Fused SDPA / FlashAttention** via `F.scaled_dot_product_attention` for both self- and
  cross-attention (speed + memory → enables longer context / bigger batch).
- **QK-norm** for training stability at higher LR (they run LR=1e-2, unusually high).
- Keep MatFormer elasticity if present; make new components MatFormer-compatible.

### Tier 2 — Text adherence (directly targets the spelling plateau)
- **Classifier-free guidance (CFG):** train with random dropout of the ASCII conditioning
  (replace with a null/empty condition ~10% of the time), then at sample time push logits
  away from the unconditional distribution: `logits = uncond + s*(cond - uncond)`.
  This is the standard fix for weak conditioning adherence and should reduce garbling /
  misspelling without dataset changes. The README's cross-attention ablation (zeroing the
  ASCII vectors → nice-but-meaningless cursive) confirms the conditioning signal is the
  right lever.
- **Improved cross-attention conditioning:** verify positional info on the ASCII encoder
  (a past bug was missing positional embeddings here); consider a small bidirectional
  encoder for the text side.

### Tier 3 — Sampling quality
- **Nucleus (top-p) sampling** and combined top-k/top-p with temperature, replacing plain
  sampling — fewer stroke outliers ("demon scrawl"/freaky samples noted in README).
- **CFG guidance scale** exposed as a sampling knob.
- Optional **repetition / pen-state sanity constraints** during decode.

### Tier 4 — Output head realism (optional flag, biggest single lever)
- **Mixture Density Network (MDN) head** (Graves 2013, *Generating Sequences with RNNs*)
  to predict continuous offsets as a mixture of bivariate Gaussians + Bernoulli pen-up,
  instead of discrete bins. Removes quantization error → smoother, more natural strokes.
  - Gated behind a config flag; requires parallel changes to tokenizer (feed continuous
    inputs), loss (MDN NLL), and sampler (sample from mixture). Only ship if it slots in
    cleanly without breaking the discrete path.

### Tier 5 — Style (their stated long-term goal; only if time permits)
- **Few-shot style conditioning** via a style encoder over reference strokes (cf.
  Style-Disentangled Transformer / Decoupled Style Descriptors on the BRUSH dataset they
  already support). Enables handwriting "cloning." Larger effort; note for later.

## 4. Validation strategy
1. Read all source (`model.py`, `train.py`, `data.py`, `sample.py`) → confirm real arch.
2. Establish baseline: locate/parameterize a tiny config that can overfit a handful of
   examples quickly on available hardware (CPU/GPU smoke test).
3. For each change: unit-level smoke test (forward/backward runs, shapes, loss decreases
   on a tiny batch), behind a flag, default off until validated.
4. Keep changes isolated per-commit so A&B comparison on W&B is clean for the user.

## 5. Relevant literature
- Graves 2013, *Generating Sequences With Recurrent Neural Networks* — MDN handwriting head, attention-based text alignment.
- Su et al. 2021, *RoFormer* — rotary positional embeddings (RoPE).
- Zhang & Sennrich 2019, *RMSNorm*. Shazeer 2020, *GLU Variants* (SwiGLU).
- Ho & Salimans 2022, *Classifier-Free Diffusion Guidance* (CFG principle, applied here to AR conditioning).
- Holtzman et al. 2020, *The Curious Case of Neural Text Degeneration* — nucleus sampling.
- Dai et al. 2023, *Disentangling Writer and Character Styles* (SDT). Kotani et al., *Decoupled Style Descriptors* (BRUSH dataset).

## 6. Open questions to resolve while reading code
- What exactly did the MatFormer refactor change vs. a vanilla transformer block?
- Is the output a single softmax over a shared vocab, or per-channel heads?
- How is the ASCII context encoded and fed into cross-attention (and does it carry positions)?
- Where are augmentations applied (data.py) and how is the vocab/binning defined?
- Is there a quick/tiny config for smoke testing?

## 7. Findings after reading source (answers to §6)
- **MatFormer:** wraps `Transformer`; replaces each block's MLP with `ModifiedMLP` that
  slices the FFN hidden dim to {1/8,1/4,1/2,1}. `train.py` picks ONE random granularity
  per step (`random.choice(['s','m','l','xl'])`) — stochastic MatFormer. This slows the
  full ('xl') model's convergence vs. always training xl. Made it configurable.
- **Output:** single softmax over a combined vocab (theta bins [0,220) then r+pen bins
  [220,420) + PAD/END/WORD). Tokens alternate theta,r ("point then shoot").
- **Context:** char embeddings + learned positional, fed to cross-attention in every block.
- **Augmentation:** per-word shear + scale jitter + randomized downsample (60–70%).

### Bugs found (fixing these is high-leverage, low-risk)
1. **PAD dominates loss** — `y` filled with `PAD_TOKEN` (real class 420), `ignore_index=-1`.
   30–50% of the loss is "predict PAD". Fix: pad targets with `-1`.
2. **Cross-attention attends to PAD context** — no key-padding mask over the 50-char
   context; padding dilutes conditioning. Fix: mask `context == char_PAD_TOKEN`.
3. **`evaluate()` + sampling don't `configure_subnetwork`** — `sample.py` crashes with
   MatFormer; eval reuses last random subnet. Fix: force 'xl' for eval/sample.
4. **Fragile dim coupling** — cross-attn assumes `n_embd == n_embd_context`; `wcpe` used
   `n_embd`. Decoupled cleanly.

## 8. Progress log
- 2026-05-30: Plan drafted; source read; findings recorded above.
- 2026-05-30: Implementing — modern model.py (RoPE/RMSNorm/SwiGLU/SDPA/QK-norm/CFG),
  loss masking, cross-attn mask, CFG+nucleus sampling, subnetwork fixes. Smoke-test on CPU.
- 2026-05-30: DONE + validated. `smoke_test.py` passes on CPU: all 4 MatFormer
  granularities train (loss 6.14->5.46), null/CFG path NaN-free, greedy/nucleus/CFG+
  structured sampling all run, padded targets now ignored (-1). All modules import.
  NOTE: requires training from scratch (architecture changed; old checkpoints incompatible).
  Suggested first A100 run: keep README hyperparams, add `--cond_drop_prob 0.1`
  (default) and sample with `guidance_scale~2.0`, `top_p~0.95`, `structured=True`.
- 2026-05-31: POST-MORTEM of first user A100 run (125k steps, bigbank_3500). Loss was
  healthy (test 1.567 best @52.5k) but samples looked terrible. Root cause: the -1
  target-masking fix removed the implicit "emit PAD after END" training signal the
  original pipeline relied on. generate() always free-runs to ~1050 tokens and decode
  never truncated at END, so every sample = real handwriting + hundreds of untrained
  garbage tokens drawn after END (made worse by structured masking, which forbade PAD
  in that region, forcing stroke tokens). Fixes, all validated in smoke_test.py:
  1. sample.py generate(): per-row stop at END, force PAD afterwards, early-exit when
     the batch is finished (also a big sampling speedup).
  2. data.py decode_stroke(): truncate at the first END before decoding (defense in
     depth; also fixes decoding for any caller that bypasses generate()).
  3. data.py __getitem__(): supervise a short END->PAD + 8-token PAD tail (negligible
     loss dilution) so the model self-terminates even without explicit stopping.
  The trained checkpoint is unaffected by these fixes (model code unchanged) — fixes
  1-2 repair inference for the EXISTING checkpoint; fix 3 benefits the next training run.
- 2026-05-31 (pre-retrain, "v2" recipe). Changes informed by run 1 (plateau ~52k,
  train/test gap ~0.11, n_layer mismatch incident at inference):
  1. Cosine LR schedule with 1k-step linear warmup, decaying to 10% of peak
     (--lr_schedule cosine, default; legacy StepLR kept via 'step').
  2. Gradient clipping (--grad_clip 1.0) — insurance at LR 1e-2.
  3. AdamW weight-decay exclusions: no decay on biases, norms, embeddings.
  4. Real residual dropout (attn out, cross-attn out, MLP out) wired to --dropout;
     recommended 0.05 for the retrain to address the observed mild overfit.
  5. NEW: small bidirectional text encoder over the ASCII context
     (--n_context_layer 2, PAD-masked, SwiGLU/RMSNorm blocks). Cross-attention now
     reads contextualized characters instead of raw char+pos embeddings — targets
     the spelling plateau. 0 disables it (exact old behavior). Adds ~130k params.
  6. Checkpoints stamp their architecture (model_args) and get_checkpoint overrides
     mismatched args at load — the n_layer-mismatch failure class is gone. Loading
     order restructured (read ckpt -> build model -> load weights).
  7. GenerationParams.guidance_scale default 1.5 so W&B training previews show
     CFG-quality samples. num_workers now an arg (notebook uses 8 for batch 64).
  All covered by smoke_test.py (param groups, warmup ramp, arch-override round trip,
  encoder on/off). NOTE: run-1 checkpoint predates the arch stamp; to sample it with
  this code set n_context_layer=0 explicitly.
  Notebook v2 defaults: batch 64 (repo's own A/B favored it), max_steps 75k (~1h).
- 2026-06-10: User's v2 Colab launch interrupted (^C) during dataset construction —
  the ORIGINAL repo's generate_word_combos does 497k serial np.random.choice
  (replace=False) calls (~full permutation each) + 497k dict builds + ~2M array
  copies + ~2M deepcopies: minutes of wall time and ~28 GB RAM. Fix: vectorized
  rejection sampling of combo index rows (sample_combo_indices) + lazy per-example
  stroke lists that hold REFERENCES to the base word arrays (copied only inside
  augmentation, which already does word.copy()). Full-size build (bigbank_3500,
  497k x 4 words) measured locally at 3.0 s, ~100 MB. New smoke test asserts the
  shared base arrays are never mutated by augmented __getitem__. Only data.py changed.
- 2026-06-10 (v2 run results, run yebt649s): first ~6.3k steps on an L4 (~230 ms/step),
  resumed flawlessly on an A100 at ~88 ms/step (SDPA-dropout patch + A100 as predicted).
  Best test loss 1.5006 @ 17.5k steps (run 1 best: 1.5672 @ 52.5k; ~0.02 of the gap is
  PAD-tail comparability) — a real ~0.05-nat gain in 1/3 the steps. Overfit onset after
  17.5k (train 1.41 vs test 1.51 rising); user stopped at 23.8k. Best ckpt = 17.5k, saved.
  v3 recipe (notebook updated, run name modern_v3): MAX_STEPS 30000 so the cosine tail
  lands at convergence (v2's best was trained at LR~0.0089, never saw low LR), and
  dropout 0.1 to delay the overfit. Everything else unchanged. ~45-50 min on A100.
  Beyond v3 the binding constraint is the 3,325-word base dataset, not training.
- 2026-07-10 (few-shot style cloning + EMA — Tier 5 delivered): the model can now be
  given a few words/sentences of someone's handwriting and write new text in that
  style. Design (cf. SDT writer-style memory, CASHG style memories, Flamingo-style
  resampler pooling):
  1. data.py: style-consistent augmentation. With --style_words N (default 3), each
     example gets N OTHER words by the same writer as a style reference, and ONE set
     of widened style params (shear -0.45..0.25, x/y scale 0.85..1.2, downsample
     density) is shared between reference and target words. The style encoder must
     therefore read slant/proportions/stroke-density off the reference — exactly the
     skill needed to mimic an unseen writer sample at inference. Dataset items are now
     (x, c, y, s); s is zero-length when style is off (legacy indices unchanged).
     load_style_reference() tokenizes a user's collect.html-style JSON into a style
     reference tensor.
  2. model.py: StyleEncoder — shared wte embedding -> proj -> learned style positions
     -> n_style_layer bidirectional blocks -> attention pooling with n_style_tokens
     (16) learned queries -> style memory concatenated with the text context for
     cross-attention (per-row masks; null-style rows are provably identical to a
     no-style forward — smoke-tested). Style keys stamped into ARCH_KEYS.
  3. Style CFG: --style_drop_prob 0.1 independently drops the style condition during
     training; generate() now does compositional guidance
     logits = uncond + g_text*(text - uncond) + g_style*(text+style - text)
     with a new style_guidance_scale knob (GenerationParams, default 1.5).
  4. EMA (--ema_decay 0.999): averaged weights used for eval/sampling and stored in
     the checkpoint (ema_state_dict); sample_only loads prefer them. get_checkpoint
     now returns a 6th value (ema_state).
  5. wandb_report.py: fetches a W&B run (defaults to cursivetransformer-ng/.../bxf7fmpq),
     prints run.history(), best/last losses, and saves a loss-curve PNG.
  All smoke-tested on CPU (style shapes, null-style==no-style equivalence, style
  changes logits, dual-CFG sampling incl. broadcast of a single reference across a
  batch, style_words=0 legacy path, EMA lerp/swap/checkpoint round-trip, 120-step
  overfit run: loss 6.23 -> 1.26). Notebook updated (style/EMA flags + a "clone
  someone's handwriting" upload cell). NOTE: requires training from scratch with
  style_words > 0 to use cloning; single-writer data limits cloning to geometric
  style (slant/proportions/density) — a multi-writer dataset (IAM/BRUSH), sampled so
  each target/reference group belongs to one writer, can use the same conditioning
  path for full writer-identity cloning.
- 2026-06-10 (regen "word vanishes" bug): regenerating a word could drop it entirely.
  Cause: a stray WORD token at the start of the regenerated chunk creates an EMPTY
  word group; truncation to the expected word count then puts the empty group in the
  target slot and discards the real strokes. Compounded by generate_paragraph
  reseeding torch with params.seed on every call -> every rerun reproduced the same
  failure. Fixes (smoke-tested): (1) structured mask now enforces WORD-token pairing
  (lone W -> partner forced; after W,W -> stroke must start, no 3rd W / END);
  (2) decode_stroke drops empty word groups; (3) regeneration no longer reseeds, so
  each rerun is a fresh attempt (fresh generation still seeds from params.seed).
  Files: sample.py, data.py.

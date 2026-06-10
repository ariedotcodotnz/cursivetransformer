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

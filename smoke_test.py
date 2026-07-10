"""Tiny CPU smoke test for the modernized model: forward, backward, all MatFormer
subnetwork modes, and CFG + nucleus + structured sampling. Not a training run."""
import torch
from model import get_all_args, MatFormer
from data import create_datasets
from sample import generate, GenerationParams

args = get_all_args(use_argparse=False)
args.device = 'cpu'
args.cond_drop_prob = 0.5
args.n_layer, args.n_embd, args.n_embd_context, args.n_ctx_head = 2, 32, 32, 4
args.max_seq_length = 1024  # large enough that short examples leave padding (to test -1 masking)
args.num_words = 2
args.train_size, args.test_size = 16, 8
args.batch_size = 4
args.style_words, args.max_style_length = 2, 256  # style conditioning ON (small)
args.n_style_tokens, args.n_style_layer = 8, 1

# pick a dataset that exists and has enough examples
for name in ['easybank', 'sample', 'cursive', 'bigbank']:
    try:
        args.dataset_name = name
        train_ds, test_ds = create_datasets(args)
        print(f"[ok] loaded dataset '{name}'")
        break
    except Exception as e:
        print(f"[skip] {name}: {type(e).__name__}: {e}")
else:
    raise SystemExit("no usable dataset found")

args.vocab_size = train_ds.get_vocab_size()
args.block_size = train_ds.get_stroke_seq_length()
args.context_block_size = train_ds.get_text_seq_length()
args.context_vocab_size = train_ds.get_char_vocab_size()
args.context_pad_token = train_ds.char_PAD_TOKEN

torch.manual_seed(0)
model = MatFormer(args)
print("vocab_size", args.vocab_size, "block_size", args.block_size)

# ---- forward + backward across all subnetwork modes ----
X, C, Y, S = (torch.stack([train_ds[i][k] for i in range(args.batch_size)]) for k in range(4))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for flag in ['s', 'm', 'l', 'xl']:
    model.configure_subnetwork(flag)
    logits, loss = model(X, C, Y, style=S)
    opt.zero_grad(); loss.backward(); opt.step()
    n_grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print(f"[{flag}] logits {tuple(logits.shape)} loss {loss.item():.4f} effparams {model.count_trainable_parameters():,} gradsum {n_grad:.2f}")
assert logits.shape == (args.batch_size, args.max_seq_length, args.vocab_size)

# ---- style conditioning: shapes, dropped-style rows, and NaN safety ----
assert S.shape == (args.batch_size, args.max_style_length)
assert model.use_style and 'style_enc' in model.transformer
S_drop = S.clone(); S_drop[0] = train_ds.PAD_TOKEN  # row 0 = null style (as in training dropout)
logits_s, loss_s = model(X, C, Y, style=S_drop)
assert torch.isfinite(logits_s).all(), "NaNs from all-PAD style row"
logits_ns, _ = model(X, C, Y, style=None)
assert torch.isfinite(logits_ns).all()
# a null-style row must behave exactly like a no-style forward for that row
assert torch.allclose(logits_s[0], logits_ns[0], atol=1e-5), "null-style row != no-style forward"
# The sampler omits style entirely for its unconditional branch, so jointly dropped
# text/style training rows must produce exactly the same logits.
null_c = torch.zeros_like(C[:1])
logits_joint_null, _ = model(X[:1], null_c, style=S_drop[:1])
logits_uncond, _ = model(X[:1], null_c, style=None)
assert torch.allclose(logits_joint_null, logits_uncond, atol=1e-5), \
    "jointly dropped text/style != sampler unconditional forward"
# and a real style reference must actually change the logits
assert not torch.allclose(logits_s[1], logits_ns[1], atol=1e-4), "style condition had no effect"
print("[ok] style conditioning: null-style==no-style, real style changes logits, no NaNs")

# ---- style-consistent augmentation: ref and target share style params ----
from data import sample_style_params, apply_style_params, load_style_reference
import numpy as _np
_np.random.seed(0)
p1 = sample_style_params(args)
_np.random.seed(0)
p2 = sample_style_params(args)
assert p1 == p2, "style params should be deterministic given the RNG state"
assert train_ds.get_style_seq_length() == args.max_style_length
external_style = load_style_reference('data/sample.json.zip', train_ds)
assert external_style.shape == (1, args.max_style_length)
print(f"[ok] style params {sorted(p1.keys())} shared across ref+target words")

# ---- CFG dropout path: null context should run cleanly (all-PAD rows) ----
model.eval()
null_C = torch.zeros_like(C)
with torch.no_grad():
    _ = model(X, null_C)
print("[ok] null/unconditional context forward")

# ---- sampling: plain, nucleus, CFG, structured ----
model.configure_subnetwork('xl')
X_init = X[:, :8]
for desc, kw in [
    ("greedy",         dict(do_sample=False)),
    ("nucleus",        dict(do_sample=True, top_p=0.9, temperature=0.8)),
    ("cfg+structured", dict(do_sample=True, top_p=0.95, guidance_scale=2.0, structured=True, dataset=train_ds)),
    ("style",          dict(do_sample=True, top_p=0.95, style=S, dataset=train_ds)),
    ("style-cfg",      dict(do_sample=True, top_p=0.95, guidance_scale=2.0, style=S[:1],
                            style_guidance_scale=1.5, structured=True, dataset=train_ds)),
]:
    out = generate(model, X_init, C, max_new_tokens=40, **kw)
    print(f"[ok] sample '{desc}': {tuple(out.shape)}")
    # structured run must never emit two theta or two r in a row
    if kw.get('structured'):
        dec = train_ds.decode_stroke(out[0])
        print(f"     structured decode -> {len(dec)} word-group(s)")

# ---- verify the loss-masking fix: padded targets are -1 (ignored) ----
# Find any example shorter than the buffer (so it actually has padding to ignore).
fracs = [( (train_ds[i][2] == -1).float().mean().item() ) for i in range(len(train_ds))]
pad_frac = max(fracs)
assert pad_frac > 0, "expected at least one example with ignored (-1) padded targets"
# And the padded region of the *input* must still be PAD_TOKEN, never -1.
xi = train_ds[int(max(range(len(fracs)), key=lambda k: fracs[k]))][0]
assert (xi == train_ds.PAD_TOKEN).any() and not (xi == -1).any()
print(f"[ok] padded-target ignore fraction (max over set) = {pad_frac:.2f} (these no longer pollute the loss)")

# ---- decode must ignore anything after END (post-END garbage fix) ----
import numpy as np
th0 = int(train_ds.cumulative_sizes[1]) + 10   # a valid theta token
r0 = 10                                        # a valid pen-down r token
clean = np.array([th0, r0, th0, r0])
polluted = np.concatenate([clean, [train_ds.END_TOKEN], [th0, r0, th0, r0, th0, r0]])
d_clean, d_poll = train_ds.decode_stroke(clean), train_ds.decode_stroke(polluted)
assert len(d_poll) == len(d_clean), "decode kept word groups from after END"
assert all(np.allclose(a, b) for a, b in zip(d_clean, d_poll)), "post-END tokens leaked into decode"
print("[ok] decode truncates at END (post-END tokens ignored)")

# ---- stray WORD tokens must not create empty groups that shift later words ----
W = train_ds.WORD_TOKEN
shifted = np.array([W, W, th0, r0, th0, r0])  # leading stray pair -> would shift words
groups = train_ds.decode_stroke(shifted)
assert len(groups) == 1 and len(groups[0]) == 2, "empty word group leaked through decode"
print("[ok] empty word groups from stray WORD tokens are dropped")

# ---- structured mask: WORD tokens are forced to come in pairs, never triples ----
from sample import _structured_mask
v = _structured_mask(torch.tensor([[th0, r0, W]]), train_ds)
assert v[0, W] and v[0].sum() == 1, "after a lone WORD token only its partner is legal"
v = _structured_mask(torch.tensor([[th0, r0, W, W]]), train_ds)
assert not v[0, W] and not v[0, train_ds.END_TOKEN] and v[0, th0], "after a WORD pair a stroke must start"
print("[ok] structured mask enforces WORD-token pairing")

# ---- targets: END predicted after last real token, then a short supervised PAD tail ----
x0, c0, y0, s0 = train_ds[0]
seq_len = int((x0 == train_ds.END_TOKEN).nonzero()[0])
assert y0[seq_len-1] == train_ds.END_TOKEN, "last real position should predict END"
assert y0[seq_len] == train_ds.PAD_TOKEN, "END should be followed by a PAD target"
tail_end = min(len(y0), seq_len + 9)
assert (y0[seq_len:tail_end] == train_ds.PAD_TOKEN).all(), "PAD tail should be supervised"
if tail_end < len(y0):
    assert y0[tail_end] == -1, "targets beyond the short tail should stay ignored (-1)"
print("[ok] END->PAD tail supervision in targets")

# ---- generation stops at END: finished rows must emit PAD only ----
# Untrained models rarely emit END on their own, so seed one row with END in the warmup
# region via a wrapper that forces END as the first generated token for row 0.
class ForceEndOnce(torch.nn.Module):
    def __init__(self, m, end_token):
        super().__init__()
        self.m, self.end_token, self.fired = m, end_token, False
    def get_block_size(self): return self.m.get_block_size()
    def configure_subnetwork(self, flag): self.m.configure_subnetwork(flag)
    def forward(self, idx, context, targets=None):
        logits, loss = self.m(idx, context, targets)
        if not self.fired:
            logits = logits.clone()
            logits[0, -1, :] = -1e9
            logits[0, -1, self.end_token] = 1e9  # row 0 must emit END right now
            self.fired = True
        return logits, loss
forced = ForceEndOnce(model, train_ds.END_TOKEN)
out = generate(forced, X_init, C, max_new_tokens=30, do_sample=True, top_p=0.95,
               structured=True, dataset=train_ds)
row0_tail = out[0, X_init.size(1):]
assert row0_tail[0] == train_ds.END_TOKEN
assert (row0_tail[1:] == train_ds.PAD_TOKEN).all(), "row finished by END must be PAD-filled"
print("[ok] generate() stops at END and PAD-fills finished rows")

# ---- lazy combos: base word arrays must never be mutated by __getitem__ ----
base_arrays = [w for ws in train_ds.raw_word_strokes[:4] for w in ws]
snapshots = [a.copy() for a in base_arrays]
for i in range(4):
    _ = train_ds[i]  # augment path must copy, not mutate, the shared base arrays
for a, snap in zip(base_arrays, snapshots):
    assert np.array_equal(a, snap), "augmentation mutated a shared base word array!"
assert all(id(ref) not in {id(target) for target in targets}
           for targets, refs in zip(train_ds.raw_word_strokes, train_ds.style_word_strokes)
           for ref in refs), "a target word leaked into its own style reference"
print("[ok] lazy combo references: base arrays unmutated after augmented __getitem__")

# ---- optimizer param groups + cosine warmup scheduler via get_checkpoint ----
from model import get_checkpoint, save_checkpoint
args.load_from_run_id = None
m2, opt2, sched2, step2, bl2, ema_state2 = get_checkpoint(args, sample_only=False)
assert len(opt2.param_groups) == 2 and opt2.param_groups[1]['weight_decay'] == 0.0
lrs = []
for _ in range(3):
    logits2, loss2 = m2(X, C, Y)
    opt2.zero_grad(); loss2.backward(); opt2.step(); sched2.step()
    lrs.append(sched2.get_last_lr()[0])
assert lrs[0] < lrs[1] < lrs[2] <= args.learning_rate, "cosine warmup should ramp the LR up"
print(f"[ok] AdamW param groups (decay/no-decay) + cosine warmup (lr ramp {lrs[0]:.2e} -> {lrs[2]:.2e})")

# ---- checkpoint stamps the architecture; mismatched args are overridden on load ----
import os, tempfile, copy as _copy
ckpt_path = os.path.join(tempfile.gettempdir(), 'ct_smoke_ckpt.pt')
save_checkpoint(m2, ckpt_path, opt2, sched2, step=3, best_loss=1.0)
args_bad = _copy.deepcopy(args)
args_bad.n_layer = args.n_layer + 1          # wrong on purpose (the Colab failure mode)
args_bad.n_context_layer = 0                 # also wrong on purpose
args_bad.local_checkpoint_path = ckpt_path
m3, *_ = get_checkpoint(args_bad, sample_only=True)
assert args_bad.n_layer == args.n_layer and args_bad.n_context_layer == args.n_context_layer
with torch.no_grad():
    _ = m3(X, C)
os.remove(ckpt_path)
print("[ok] checkpoint arch-stamp overrides mismatched args and loads cleanly")

# ---- text encoder can be disabled (n_context_layer=0) ----
args_noenc = _copy.deepcopy(args)
args_noenc.n_context_layer = 0
m0 = MatFormer(args_noenc)
logits0, loss0 = m0(X, C, Y)
assert torch.isfinite(loss0)
assert not any('henc' in n for n, _ in m0.named_parameters())
print("[ok] n_context_layer=0 (no text encoder) still works")

# ---- style conditioning can be disabled (style_words=0): legacy behavior ----
args_nostyle = _copy.deepcopy(args)
args_nostyle.style_words = 0
ds_ns, _ = create_datasets(args_nostyle)
x_ns, c_ns, y_ns, s_ns = ds_ns[0]
assert s_ns.numel() == 0, "style tensor should be zero-length when style is off"
m_ns = MatFormer(args_nostyle)
assert not m_ns.use_style and not any('style_enc' in n for n, _ in m_ns.named_parameters())
_ = m_ns(x_ns.unsqueeze(0), c_ns.unsqueeze(0), y_ns.unsqueeze(0), style=s_ns.unsqueeze(0))
print("[ok] style_words=0 (no style encoder) reproduces legacy behavior")

# ---- EMA: updates track the weights; checkpoint round trip; sampling prefers EMA ----
from model import EMA
ema = EMA(m2, decay=0.5)
w_name, w = next((n, p) for n, p in m2.named_parameters() if p.dim() >= 2)
before = ema.shadow[w_name].clone()
with torch.no_grad():
    w.add_(1.0)
ema.update(m2)
assert torch.allclose(ema.shadow[w_name], before + 0.5), "EMA lerp is wrong"
with ema.swap(m2):
    assert torch.allclose(dict(m2.named_parameters())[w_name], ema.shadow[w_name])
assert torch.allclose(dict(m2.named_parameters())[w_name], w), "swap must restore raw weights"
ckpt_path2 = os.path.join(tempfile.gettempdir(), 'ct_smoke_ema_ckpt.pt')
save_checkpoint(m2, ckpt_path2, opt2, sched2, step=3, best_loss=1.0, ema=ema)
args_ema = _copy.deepcopy(args)
args_ema.local_checkpoint_path = ckpt_path2
m4, *_ = get_checkpoint(args_ema, sample_only=True)
assert torch.allclose(dict(m4.named_parameters())[w_name], ema.shadow[w_name]), \
    "sample_only load should use the EMA weights"

# A sample-only W&B download must cache the original resumable checkpoint, not a
# stripped checkpoint whose raw weights have already been replaced by EMA.
import model as _model_module
with tempfile.TemporaryDirectory() as cache_dir:
    artifact_dir = os.path.join(cache_dir, 'artifact')
    os.mkdir(artifact_dir)
    remote_path = os.path.join(artifact_dir, 'best_checkpoint.pt')
    cache_path = os.path.join(cache_dir, 'cached_checkpoint.pt')
    save_checkpoint(m2, remote_path, opt2, sched2, step=3, best_loss=1.0, ema=ema)

    class _FakeArtifact:
        def download(self):
            return artifact_dir

    original_lookup = _model_module.get_latest_checkpoint_artifact
    _model_module.get_latest_checkpoint_artifact = lambda _args: _FakeArtifact()
    try:
        args_remote = _copy.deepcopy(args)
        args_remote.local_checkpoint_path = cache_path
        args_remote.load_from_run_id = 'fake-run'
        sampled_remote, *_ = get_checkpoint(args_remote, sample_only=True)
    finally:
        _model_module.get_latest_checkpoint_artifact = original_lookup

    cached = torch.load(cache_path, weights_only=True)
    remote = torch.load(remote_path, weights_only=True)
    assert cached.keys() == remote.keys() and 'optimizer_state_dict' in cached
    assert torch.equal(cached['model_state_dict'][w_name], remote['model_state_dict'][w_name])
    assert torch.allclose(dict(sampled_remote.named_parameters())[w_name], ema.shadow[w_name])

args_ema.load_from_run_id = 'unused-local-path-wins'  # training resume reads the local ckpt
_, _, _, _, _, ema_state4 = get_checkpoint(args_ema, sample_only=False)
assert ema_state4 is not None and torch.allclose(ema_state4[w_name], ema.shadow[w_name])
os.remove(ckpt_path2)
print("[ok] EMA update/swap + checkpoint round trip (sampling uses EMA weights)")

# ---- pre-style checkpoints (no style keys in the arch stamp) load cleanly ----
ckpt_path3 = os.path.join(tempfile.gettempdir(), 'ct_smoke_prestyle_ckpt.pt')
save_checkpoint(m_ns, ckpt_path3)  # m_ns was built with style_words=0
old = torch.load(ckpt_path3, weights_only=True)
old['model_args'] = {k: v for k, v in old['model_args'].items()
                     if 'style' not in k}          # simulate a pre-style stamp
torch.save(old, ckpt_path3)
args_old = _copy.deepcopy(args)                     # caller wants style (default on)...
args_old.local_checkpoint_path = ckpt_path3
m5, *_ = get_checkpoint(args_old, sample_only=True)
assert args_old.style_words == 0 and not m5.use_style, \
    "pre-style checkpoint should force style conditioning off"
os.remove(ckpt_path3)
print("[ok] pre-style checkpoints load with style conditioning forced off")

print("\nALL SMOKE TESTS PASSED")

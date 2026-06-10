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
X, C, Y = (torch.stack([train_ds[i][k] for i in range(args.batch_size)]) for k in range(3))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for flag in ['s', 'm', 'l', 'xl']:
    model.configure_subnetwork(flag)
    logits, loss = model(X, C, Y)
    opt.zero_grad(); loss.backward(); opt.step()
    n_grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print(f"[{flag}] logits {tuple(logits.shape)} loss {loss.item():.4f} effparams {model.count_trainable_parameters():,} gradsum {n_grad:.2f}")
assert logits.shape == (args.batch_size, args.max_seq_length, args.vocab_size)

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

# ---- targets: END predicted after last real token, then a short supervised PAD tail ----
x0, c0, y0 = train_ds[0]
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
print("[ok] lazy combo references: base arrays unmutated after augmented __getitem__")

# ---- optimizer param groups + cosine warmup scheduler via get_checkpoint ----
from model import get_checkpoint, save_checkpoint
args.load_from_run_id = None
m2, opt2, sched2, step2, bl2 = get_checkpoint(args, sample_only=False)
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

print("\nALL SMOKE TESTS PASSED")

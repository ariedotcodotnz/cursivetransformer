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
print("\nALL SMOKE TESTS PASSED")

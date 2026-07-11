# Sam Greydanus | 2024
# 2026 modernization: RoPE, RMSNorm, SwiGLU, fused SDPA (FlashAttention),
# fused SDPA QK-norm, cross-attention padding masks, and classifier-free guidance (CFG).

########## IMPORTS AND A FEW GLOBAL VARIABLES ##########

import math, os, sys, argparse, getpass
from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.nn import functional as F

import wandb


########## ALL ARGUMENTS ##########

def get_all_args(use_argparse=True):
    args_config = {
        'max_steps': (110000, int, 'How many steps to train for'),
        'print_every': (100, int, 'Print log info after how many steps'),
        'log_every': (2500, int, 'Sample model after how many steps'),
        'lr_decay': (0.333, float, 'How much to decay the learning rate'),
        'step_lr_every': (33000, int, 'How often to decay the learning rate'),
        'device': ('cuda', str, 'This is meant to be trained on a GPU'),
        'seed': (42, int, 'Random seed for reproducibility'),
        'n_layer': (4, int, 'Number of Transformer layers'),
        'n_embd': (64, int, 'Number of embedding dimensions in self attention'),
        'n_embd_context': (64, int, 'Number of embedding dimensions in cross attention'),
        'n_ctx_head': (4, int, 'Number of attention heads in Transformer block'),
        'learning_rate': (1e-2, float, 'Learning rate'),
        'weight_decay': (1e-4, float, 'Weight decay'),
        'batch_size': (32, int, 'Batch size'),
        'train_size': (497000, int, 'Number of train examples'),
        'test_size': (3000, int, 'Number of test examples'),
        'num_words': (5, int, 'Number of words'),
        'max_seq_length': (1500, int, 'Maximum sequence length (tokens)'),
        'augment': (True, 'store_true', 'Perform augmentations'),
        'downsample_mean': (0.65, float, 'Mean amount to downsample stroke points (0.65=65%)'),
        'downsample_width': (0.1, float, 'Width of the uniform distribution (0.1=10%)'),
        'add_digits': (True, 'store_true', 'Add digit words to the word bank'),
        # --- 2026 additions ---
        'cond_drop_prob': (0.1, float, 'Prob. of dropping the ASCII condition during training (enables classifier-free guidance)'),
        'subnetwork_mode': ('full', str, "MatFormer training granularity: 'full' (always xl, best single-model quality), "
                                          "'random' (stochastic MatFormer for elastic inference), or a fixed flag s/m/l/xl"),
        'dropout': (0.0, float, 'Dropout probability inside the Transformer blocks'),
        'lr_schedule': ('cosine', str, "LR schedule: 'cosine' (linear warmup + cosine decay, recommended) or 'step' (legacy StepLR)"),
        'warmup_steps': (1000, int, 'Linear LR warmup steps (cosine schedule only)'),
        'grad_clip': (1.0, float, 'Clip the gradient norm to this value (0 = no clipping)'),
        'n_context_layer': (2, int, 'Bidirectional encoder layers over the ASCII context (0 = raw char embeddings, as before)'),
        'use_bos': (False, 'store_true', 'Train with a BOS token so generation needs no ground-truth handwriting prefix'),
        # --- few-shot style adaptation (style cloning) ---
        'style_words': (3, int, 'Words of same-writer handwriting given as a style reference (0 disables style conditioning)'),
        'max_style_length': (600, int, 'Maximum style-reference length in stroke tokens'),
        'n_style_tokens': (16, int, 'Number of pooled style memory tokens the decoder cross-attends to'),
        'n_style_layer': (2, int, 'Bidirectional encoder layers over the style reference strokes'),
        'style_drop_prob': (0.1, float, 'Prob. of dropping the style condition during training (enables style CFG)'),
        'ema_decay': (0.999, float, 'EMA decay for an averaged copy of the weights used at eval/sample time (0 disables)'),
        'style_lr_multiplier': (1.0, float, 'Learning-rate multiplier for newly initialized style-encoder parameters'),
        'eval_batch_size': (100, int, 'Validation batch size'),
        'eval_max_batches': (-1, int, 'Maximum validation batches; <=0 evaluates the full split'),
        'early_stopping_patience': (0, int, 'Stop after this many non-improving validations (0 disables)'),
        'early_stopping_min_delta': (0.0, float, 'Minimum validation-loss decrease counted as an improvement'),
        'num_workers': (4, int, 'DataLoader worker processes for the training loader'),
        'alphabet': (" enaitoshrdx.vpukbgfcymzw1lqj804I92637OTAS5N)EHR\"\'(BCQLMWYU,ZF!DXV?KPGJ", str,
                        'All the characters that this model will be able to draw'),
        'dataset_name': ('bigeasybank', str, 'The name of the .zip file containing your dataset'),
        'wandb_project': ('bigbank_experiments', str, 'W&B project name'),
        'wandb_entity': ('sam-greydanus', str, 'Set this to your wandb username or team name'),
        'wandb_run_name': ('unnamed_run', str, 'W&B run name'),
        'wandb_api_key': (None, str, 'Weights & Biases API Key'),
        'load_from_run_id': (None, str, 'Load from a specific W&B run ID'),
        'init_from_run_id': (None, str, 'Warm-start a new model from compatible weights in this W&B run'),
        'init_checkpoint_path': (None, str, 'Warm-start a new model from compatible weights in this local checkpoint'),
        'local_checkpoint_path': ('best_checkpoint.pt', str, 'Path to local model file'),
        'style_json': (None, str, 'Path to a .json/.json.zip handwriting sample to clone the style of (sample.py only)'),
        'style_text': (None, str, 'Text to write when cloning a style with --style_json'),
    }

    if use_argparse:
        parser = argparse.ArgumentParser(description='Train a cursivetransformer model')
        for arg, (default, arg_type, help_text) in args_config.items():
            if arg_type == 'store_true':
                parser.add_argument(f'--{arg}', action=arg_type, default=default, help=help_text)
            else:
                parser.add_argument(f'--{arg}', type=arg_type, default=default, help=help_text)
        args = parser.parse_args()
    else:
        args = SimpleNamespace(**{k: v[0] for k, v in args_config.items()})

    if "WANDB_API_KEY" not in os.environ:
        if args.wandb_api_key is None:
            args.wandb_api_key = getpass.getpass("Enter your W&B API key: ")
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    return args



########## MODEL I/O ##########

# Architecture-defining args. Stamped into every checkpoint so that loading can
# rebuild the exact trained model even if the caller's args disagree.
ARCH_KEYS = ('n_layer', 'n_embd', 'n_embd_context', 'n_ctx_head', 'n_context_layer',
             'vocab_size', 'block_size', 'context_block_size', 'context_vocab_size',
             'style_words', 'max_style_length', 'n_style_tokens', 'n_style_layer',
             'use_bos', 'stroke_pad_token', 'style_lr_multiplier')


def configure_optimizer(model, args):
    """AdamW with standard weight-decay exclusions: no decay on biases, norm weights
    (dim < 2), or embeddings (incl. the tied wte/lm_head weight)."""
    decay, no_decay, style_decay, style_no_decay = [], [], [], []
    style_multiplier = getattr(args, 'style_lr_multiplier', 1.0)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embedding = 'wte' in name or 'wce' in name or 'wcpe' in name
        no_wd = p.dim() < 2 or is_embedding
        if style_multiplier != 1.0 and 'style_enc' in name:
            (style_no_decay if no_wd else style_decay).append(p)
        else:
            (no_decay if no_wd else decay).append(p)
    groups = [{'params': decay, 'weight_decay': args.weight_decay},
              {'params': no_decay, 'weight_decay': 0.0}]
    if style_decay or style_no_decay:
        style_lr = args.learning_rate * style_multiplier
        groups.extend([
            {'params': style_decay, 'weight_decay': args.weight_decay, 'lr': style_lr},
            {'params': style_no_decay, 'weight_decay': 0.0, 'lr': style_lr},
        ])
    return torch.optim.AdamW(groups, lr=args.learning_rate, betas=(0.9, 0.99), eps=1e-8)


class EMA:
    """Exponential moving average of the model weights (Polyak averaging). Evaluating
    and sampling from the averaged weights is a standard generative-modeling trick
    that smooths late-training oscillation and reliably improves sample quality."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].lerp_(v.detach(), 1.0 - self.decay)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        for k in self.shadow:
            if k in sd:
                self.shadow[k].copy_(sd[k])

    @contextmanager
    def swap(self, model):
        """Temporarily load the EMA weights into the model (for eval/sampling)."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if k in self.shadow}
        model.load_state_dict(self.shadow, strict=False)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=False)


def configure_scheduler(optimizer, args):
    if getattr(args, 'lr_schedule', 'step') == 'cosine':
        warmup = max(1, getattr(args, 'warmup_steps', 1000))
        total = max(args.max_steps, warmup + 1)
        min_ratio = 0.1  # decay to 10% of peak LR by max_steps
        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            t = min(1.0, (step - warmup) / max(1, total - warmup))
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr_every, gamma=args.lr_decay)


def load_compatible_weights(model, state_dict):
    """Warm-start matching weights while allowing newly added modules/vocab rows.

    This is intentionally separate from resume loading: optimizer state and training
    step are not restored, and incompatible architecture tensors stay initialized.
    """
    target = model.state_dict()
    loaded, partial, skipped = [], [], []
    with torch.no_grad():
        for name, source in state_dict.items():
            if name not in target:
                skipped.append(name)
                continue
            dest = target[name]
            if dest.shape == source.shape:
                dest.copy_(source)
                loaded.append(name)
            elif (dest.ndim == source.ndim and dest.ndim > 0 and
                  dest.shape[1:] == source.shape[1:]):
                # Vocabulary expansion (for BOS) preserves every existing token row.
                n = min(dest.shape[0], source.shape[0])
                dest[:n].copy_(source[:n])
                partial.append(name)
            else:
                skipped.append(name)
    model.load_state_dict(target)
    return {'loaded': loaded, 'partial': partial, 'skipped': skipped}


def _load_warm_start_checkpoint(args):
    path = getattr(args, 'init_checkpoint_path', None)
    run_id = getattr(args, 'init_from_run_id', None)
    if path:
        print(f"Warm-starting from local checkpoint: {path}")
        return torch.load(path, map_location='cpu', weights_only=True)
    if run_id:
        artifact = get_latest_checkpoint_artifact(args, run_id=run_id)
        artifact_dir = artifact.download()
        print(f"Warm-starting from W&B run: {run_id}")
        return torch.load(os.path.join(artifact_dir, 'best_checkpoint.pt'),
                          map_location='cpu', weights_only=True)
    return None


def get_checkpoint(args, sample_only):
    # Locate and read the checkpoint FIRST, so the model is built with the exact
    # architecture it was trained with (prevents train/inference arch mismatches).
    checkpoint, from_wandb = None, False
    if args.load_from_run_id or sample_only:
        if os.path.exists(args.local_checkpoint_path):
            checkpoint = torch.load(args.local_checkpoint_path, map_location='cpu',
                                    weights_only=True)
            print(f"Loaded checkpoint from local path: {args.local_checkpoint_path}")
        elif args.load_from_run_id:
            artifact = get_latest_checkpoint_artifact(args)
            artifact_dir = artifact.download()
            checkpoint = torch.load(os.path.join(artifact_dir, "best_checkpoint.pt"),
                                    map_location='cpu', weights_only=True)
            from_wandb = True
        else:
            print("No local model or W&B run ID provided. Exiting.")
            sys.exit()

    if checkpoint is not None:
        model_args = checkpoint.get('model_args', {})
        for k, v in model_args.items():
            if getattr(args, k, None) != v:
                print(f"Checkpoint architecture override: args.{k} = {v} (was {getattr(args, k, None)})")
                setattr(args, k, v)
        # A stamped checkpoint that predates style conditioning was trained without a
        # style encoder; force it off so the module list matches the stored weights.
        if model_args and 'style_words' not in model_args and getattr(args, 'style_words', 0) > 0:
            print("Checkpoint predates style conditioning: args.style_words = 0")
            args.style_words = 0

    model = MatFormer(args)
    model.to(args.device)
    print(f"Model #params: {sum(p.numel() for p in model.parameters())}")

    if checkpoint is None and not sample_only:
        warm_start = _load_warm_start_checkpoint(args)
        if warm_start is not None:
            source = warm_start.get('ema_state_dict') or warm_start['model_state_dict']
            summary = load_compatible_weights(model, source)
            print("Warm-started weights: "
                  f"{len(summary['loaded'])} exact, {len(summary['partial'])} partial, "
                  f"{len(summary['skipped'])} skipped")

    if not sample_only:
        optimizer = configure_optimizer(model, args)
        scheduler = configure_scheduler(optimizer, args)
    else:
        optimizer = None
        scheduler = None

    step = 0
    best_loss = None
    ema_state = None

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        ema_state = checkpoint.get('ema_state_dict')
        if not sample_only:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            step = checkpoint['step'] + (1 if from_wandb else 0)
            best_loss = checkpoint['best_loss']
        if from_wandb:
            # Preserve the complete downloaded checkpoint. Rebuilding it here would
            # drop optimizer state during sample-only loads and could cache EMA
            # parameters as the raw training weights.
            torch.save(checkpoint, args.local_checkpoint_path)
        if sample_only and ema_state is not None:
            model.load_state_dict(ema_state, strict=False)  # EMA weights sample better
            print("Loaded EMA (averaged) weights for sampling")

    return model, optimizer, scheduler, step, best_loss, ema_state



def get_latest_checkpoint_artifact(args, verbose=True, run_id=None):
    run_id = run_id or args.load_from_run_id
    run = wandb.Api().run(f"{args.wandb_entity}/{args.wandb_project}/{run_id}")

    if verbose:
        print(f"Finding latest checkpoint for W&B run id {run_id}")
    latest_artifact = None
    get_version = lambda artifact: -1 if artifact is None else int(artifact.name.split(':v')[-1])
    for artifact in run.logged_artifacts():
        if verbose:
            print(f"  {artifact.type}:{artifact.name}")
        if artifact.type == 'model' and (get_version(artifact) > get_version(latest_artifact)):
            latest_artifact = artifact
    if verbose:
        print(f"Selected:  {latest_artifact.type}:{latest_artifact.name}")
    return latest_artifact


def save_checkpoint(model, path, optimizer=None, scheduler=None, step=None, best_loss=None, ema=None):
    checkpoint = {'model_state_dict': model.state_dict()}
    cfg = getattr(model, 'config', None)
    if cfg is not None:  # stamp the architecture so loading can never mismatch it
        checkpoint['model_args'] = {k: getattr(cfg, k) for k in ARCH_KEYS if hasattr(cfg, k)}
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if step is not None:
        checkpoint['step'] = step
    if best_loss is not None:
        checkpoint['best_loss'] = best_loss
    if ema is not None:  # accepts an EMA object or a raw state dict
        checkpoint['ema_state_dict'] = ema.state_dict() if hasattr(ema, 'state_dict') and not isinstance(ema, dict) else ema
    torch.save(checkpoint, path)


########## MODERN TRANSFORMER COMPONENTS ##########


class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm (Zhang & Sennrich 2019). Cheaper and more stable
    than LayerNorm; the current standard in LLaMA-style transformers."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_rope_cache(seq_len, head_dim, device, dtype, base=10000.0):
    """Precompute rotary positional embedding (Su et al. 2021, RoFormer) cos/sin tables."""
    assert head_dim % 2 == 0, "RoPE requires an even head dimension"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)               # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)        # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x, cos, sin, offset=0):
    """Apply rotary embeddings to a (B, n_head, T, head_dim) tensor."""
    T = x.size(-2)
    cos = cos[offset:offset + T].view(1, 1, T, -1)
    sin = sin[offset:offset + T].view(1, 1, T, -1)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention with RoPE, QK-norm, and fused
    scaled_dot_product_attention (FlashAttention when available)."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_ctx_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_ctx_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_ctx_head
        self.dropout = getattr(config, 'dropout', 0.0)
        self.resid_drop = nn.Dropout(self.dropout)
        # QK-norm stabilizes training at the high LR (1e-2) this model uses.
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope, kv_cache=None, use_cache=False):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = rope
        past_len = 0 if kv_cache is None else kv_cache[0].size(-2)
        q = apply_rope(q, cos, sin, past_len)
        k = apply_rope(k, cos, sin, past_len)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        # No dropout inside SDPA: dropout_p > 0 can force a fallback off the fused
        # FlashAttention kernel on some builds. Regularization comes from resid_drop.
        attn_mask = None
        is_causal = kv_cache is None and T > 1
        if kv_cache is not None and T > 1:
            q_pos = torch.arange(T, device=x.device).unsqueeze(1) + past_len
            k_pos = torch.arange(past_len + T, device=x.device).unsqueeze(0)
            attn_mask = (k_pos <= q_pos).view(1, 1, T, past_len + T)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                           is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.c_proj(y))
        return (y, (k, v)) if use_cache else y


class CrossAttention(nn.Module):
    """Cross-attention from the stroke stream (n_embd) onto the encoded ASCII context
    (n_embd_context), with a key-padding mask so the model never attends to PAD chars.
    Dimensions are decoupled: queries live in n_embd, keys/values are projected up from
    n_embd_context to n_embd."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_ctx_head == 0
        self.c_attn_q = nn.Linear(config.n_embd, config.n_embd)
        self.c_attn_kv = nn.Linear(config.n_embd_context, 2 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_ctx_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_ctx_head
        self.dropout = getattr(config, 'dropout', 0.0)
        self.resid_drop = nn.Dropout(self.dropout)

    def forward(self, x, context, context_mask=None):
        B, T, C = x.size()
        _, T_ctx, _ = context.size()

        q = self.c_attn_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k, v = self.c_attn_kv(context).split(self.n_embd, dim=2)
        k = k.view(B, T_ctx, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T_ctx, self.n_head, self.head_dim).transpose(1, 2)

        attn_mask = None
        if context_mask is not None:
            # context_mask: (B, T_ctx) bool, True = real char (attend), False = PAD.
            # Keep one stable sentinel for entirely-PAD CFG rows. Unmasking every
            # position would make the result depend on appended masked style slots.
            keep = context_mask.clone()
            empty = ~keep.any(dim=1)
            keep[empty, 0] = True
            attn_mask = keep.view(B, 1, 1, T_ctx)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    """SwiGLU feed-forward (Shazeer 2020). Gated GLU variant; better quality per FLOP
    than a plain GELU MLP. `dim` lets the text encoder reuse it at n_embd_context."""
    def __init__(self, config, dim=None):
        super().__init__()
        d = dim if dim is not None else config.n_embd
        hidden = 4 * d
        self.intermediate_size = hidden
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)
        self.drop = nn.Dropout(getattr(config, 'dropout', 0.0))

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class EncoderSelfAttention(nn.Module):
    """Bidirectional self-attention over the ASCII context (PAD keys masked), so each
    character embedding can see its neighbors before cross-attention reads it."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd_context % config.n_ctx_head == 0
        self.c_attn = nn.Linear(config.n_embd_context, 3 * config.n_embd_context)
        self.c_proj = nn.Linear(config.n_embd_context, config.n_embd_context)
        self.n_head = config.n_ctx_head
        self.n_embd_context = config.n_embd_context
        self.head_dim = config.n_embd_context // config.n_ctx_head
        self.dropout = getattr(config, 'dropout', 0.0)
        self.resid_drop = nn.Dropout(self.dropout)

    def forward(self, x, key_mask=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd_context, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn_mask = None
        if key_mask is not None:
            keep = key_mask.clone()
            empty = ~keep.any(dim=1)
            keep[empty, 0] = True  # stable sentinel for all-PAD CFG rows
            attn_mask = keep.view(B, 1, 1, T)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class EncoderBlock(nn.Module):
    """Pre-norm bidirectional block for the small text encoder."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd_context)
        self.attn = EncoderSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd_context)
        self.mlp = MLP(config, dim=config.n_embd_context)

    def forward(self, x, key_mask=None):
        x = x + self.attn(self.ln_1(x), key_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class StyleEncoder(nn.Module):
    """Few-shot style adaptation: encode a reference stroke-token sequence (a few words
    of someone's handwriting) into a small set of style memory tokens that the decoder
    cross-attends to alongside the text context (cf. writer-style memory in SDT, Dai
    et al. 2023, and CASHG, Shin et al. 2026; pooling is a Flamingo-style resampler:
    learned queries attending over the encoded reference). Trained with
    style-consistent augmentation, these tokens carry slant / proportions / stroke
    density, so at inference the model can mimic the style of an unseen sample."""

    def __init__(self, config):
        super().__init__()
        d = config.n_embd_context
        self.proj = nn.Linear(config.n_embd, d)
        self.spe = nn.Embedding(config.max_style_length, d)  # style positions (pairs theta/r)
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_style_layer)])
        self.ln = RMSNorm(d)
        self.queries = nn.Parameter(torch.randn(config.n_style_tokens, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, config.n_ctx_head, batch_first=True)
        self.ln_out = RMSNorm(d)

    def forward(self, style_emb, key_mask):
        """style_emb: (B, Ls, n_embd) embedded style tokens; key_mask: (B, Ls) bool,
        True = real token. Returns (B, n_style_tokens, d) memory and its (B, n_style_tokens)
        mask (all-False rows for null-style examples, so cross-attention ignores them)."""
        B, Ls, _ = style_emb.size()
        pos = torch.arange(Ls, device=style_emb.device)
        x = self.proj(style_emb) + self.spe(pos).unsqueeze(0)

        keep = key_mask.clone()
        empty = ~keep.any(dim=1)      # rows with no style (dropped / null condition)
        keep[empty, 0] = True         # keep attention well-defined for those rows
        for block in self.blocks:
            x = block(x, keep)
        x = self.ln(x)

        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.pool(q, x, x, key_padding_mask=~keep, need_weights=False)
        out = self.ln_out(out)
        out = out.masked_fill(empty.view(B, 1, 1), 0.0)  # null-style rows contribute nothing
        style_mask = (~empty).view(B, 1).expand(B, self.queries.size(0))
        return out, style_mask


class Block(nn.Module):
    """A pre-norm Transformer block: self-attn -> cross-attn -> SwiGLU MLP."""

    def __init__(self, config, has_cross_attn=True):
        super().__init__()
        self.has_cross_attn = has_cross_attn
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        if has_cross_attn:
            self.ln_2 = RMSNorm(config.n_embd)
            self.cross_attn = CrossAttention(config)

        self.ln_3 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, rope, context=None, context_mask=None,
                kv_cache=None, use_cache=False):
        attn_out = self.attn(self.ln_1(x), rope, kv_cache=kv_cache,
                             use_cache=use_cache)
        if use_cache:
            attn_out, new_cache = attn_out
        x = x + attn_out

        if self.has_cross_attn:
            assert context is not None, 'Expected context'
            x = x + self.cross_attn(self.ln_2(x), context, context_mask)

        x = x + self.mlp(self.ln_3(x))
        return (x, new_cache) if use_cache else x


class Transformer(nn.Module):
    """Decoder-only Transformer LM with cross-attention text conditioning.
    Modernized: RoPE (no learned positional embedding on the stroke stream),
    RMSNorm, SwiGLU, fused attention, tied input/output embeddings, and optional
    classifier-free-guidance dropout of the condition."""

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.config = config
        self.context_pad_token = getattr(config, 'context_pad_token', 0)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wce = nn.Embedding(config.context_vocab_size, config.n_embd_context),
            wcpe = nn.Embedding(config.context_block_size, config.n_embd_context),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        n_enc = getattr(config, 'n_context_layer', 0)
        if n_enc > 0:  # small bidirectional encoder contextualizes the ASCII chars
            self.transformer['henc'] = nn.ModuleList([EncoderBlock(config) for _ in range(n_enc)])
            self.transformer['ln_enc'] = RMSNorm(config.n_embd_context)
        self.use_style = getattr(config, 'style_words', 0) > 0
        if self.use_style:  # few-shot style adaptation from reference strokes
            self.transformer['style_enc'] = StyleEncoder(config)
        n_special = 4 if getattr(config, 'use_bos', False) else 3
        self.stroke_pad_token = getattr(config, 'stroke_pad_token',
                                        config.vocab_size - n_special)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # weight tying

        head_dim = config.n_embd // config.n_ctx_head
        self._rope_cache = None
        self._rope_len = 0
        self._rope_head_dim = head_dim

        self.apply(self._init_weights)
        # GPT-2 style scaled init for residual projections.
        for name, p in self.named_parameters():
            if name.endswith('c_proj.weight') or name.endswith('w_down.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("Number of Transformer parameters: {:.0f}".format(n_params,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_block_size(self):
        return self.block_size

    def _get_rope(self, T, device, dtype):
        if self._rope_cache is None or self._rope_len < T or self._rope_cache[0].device != device:
            length = max(T, self.block_size)
            self._rope_cache = build_rope_cache(length, self._rope_head_dim, device, dtype)
            self._rope_len = length
        return self._rope_cache

    def encode_conditioning(self, context, style=None):
        """Encode text/style once so cached decoding does not repeat this work."""
        device = context.device
        context_t = context.size(-1)
        context_pos = torch.arange(0, context_t, dtype=torch.long,
                                   device=device).unsqueeze(0)
        c = self.transformer.wce(context) + self.transformer.wcpe(context_pos)
        context_mask = (context != self.context_pad_token)
        if 'henc' in self.transformer:
            for eblock in self.transformer.henc:
                c = eblock(c, context_mask)
            c = self.transformer.ln_enc(c)

        if self.use_style and style is not None and style.numel() > 0:
            style = style[:, :self.transformer.style_enc.spe.num_embeddings]
            style_key_mask = (style != self.stroke_pad_token)
            s_tokens, s_mask = self.transformer.style_enc(
                self.transformer.wte(style), style_key_mask)
            c = torch.cat([c, s_tokens], dim=1)
            context_mask = torch.cat([context_mask, s_mask], dim=1)
        return c, context_mask

    def forward(self, idx, context, targets=None, style=None, kv_cache=None,
                use_cache=False, conditioning=None):
        device = idx.device
        b, t = idx.size()
        past_len = 0 if not kv_cache else kv_cache[0][0].size(-2)
        total_len = past_len + t
        assert total_len <= self.block_size, \
            f"Cannot forward sequence of length {total_len}, block size is only {self.block_size}"

        x = self.transformer.wte(idx)  # (b, t, n_embd); positions handled by RoPE
        c, context_mask = conditioning if conditioning is not None \
            else self.encode_conditioning(context, style)

        rope = self._get_rope(total_len, device, x.dtype)
        new_cache = [] if use_cache else None
        for layer_idx, block in enumerate(self.transformer.h):
            layer_cache = None if not kv_cache else kv_cache[layer_idx]
            if use_cache:
                x, updated = block(x, rope, c, context_mask,
                                   kv_cache=layer_cache, use_cache=True)
                new_cache.append(updated)
            else:
                x = block(x, rope, c, context_mask)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Callers may trim padded columns with a non-contiguous slice.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-1)
        if use_cache:
            return logits, loss, new_cache, (c, context_mask)
        return logits, loss

    def forward_cached(self, idx, context, style=None, cache=None):
        """Decode a prefix or one continuation token using an exact per-layer KV cache."""
        kv_cache = None if cache is None else cache['kv']
        conditioning = None if cache is None else cache['conditioning']
        logits, _, new_cache, conditioning = self.forward(
            idx, context, style=style, kv_cache=kv_cache, use_cache=True,
            conditioning=conditioning)
        return logits, {'kv': new_cache, 'conditioning': conditioning}


class ModifiedMLP(MLP):
    """SwiGLU MLP whose hidden width can be elastically sliced for MatFormer."""
    def __init__(self, config, scale_factors):
        super().__init__(config)
        self.scale_factors = scale_factors
        self.current_subset_hd = self.intermediate_size  # default to full ('xl')

    def configure_subnetwork(self, flag):
        hd = self.intermediate_size
        scale = {'s': self.scale_factors[0], 'm': self.scale_factors[1],
                 'l': self.scale_factors[2], 'xl': self.scale_factors[3]}.get(flag, self.scale_factors[3])
        self.current_subset_hd = int(hd * scale)

    def forward(self, x):
        h = self.current_subset_hd
        if h >= self.intermediate_size:
            return super().forward(x)
        gate = F.linear(x, self.w_gate.weight[:h])
        up = F.linear(x, self.w_up.weight[:h])
        return self.drop(F.linear(F.silu(gate) * up, self.w_down.weight[:, :h]))


class MatFormer(Transformer):
    def __init__(self, config):
        super().__init__(config)
        scale_factors = [1/8, 1/4, 1/2, 1]  # s, m, l, xl

        for layer_idx in range(config.n_layer):
            self.transformer.h[layer_idx].mlp = ModifiedMLP(config, scale_factors)
        self.configure_subnetwork('xl')

    def configure_subnetwork(self, flag):
        """Configure the subnetwork for all layers based on the flag."""
        for layer_idx in range(len(self.transformer.h)):
            self.transformer.h[layer_idx].mlp.configure_subnetwork(flag)

    def count_trainable_parameters(self):
        """Effective trainable params for the currently configured subnetwork."""
        total_params = 0
        for name, param in self.named_parameters():
            # exclude only the elastic decoder MLPs ('transformer.h.*'); the text
            # encoder ('transformer.henc.*') MLPs are always full-width
            is_elastic_mlp = name.startswith('transformer.h.') and '.mlp.' in name
            if not is_elastic_mlp and param.requires_grad:
                total_params += param.numel()

        for i in range(self.config.n_layer):
            mlp = self.transformer.h[i].mlp
            if mlp.current_subset_hd is None:
                raise ValueError("Subnetwork size not configured.")
            # SwiGLU: gate + up (in->hd) and down (hd->in), all bias-free.
            total_params += 2 * (mlp.current_subset_hd * self.config.n_embd)
            total_params += self.config.n_embd * mlp.current_subset_hd
        return total_params

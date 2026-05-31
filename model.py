# Sam Greydanus | 2024
# 2026 modernization: RoPE, RMSNorm, SwiGLU, fused SDPA (FlashAttention),
# fused SDPA QK-norm, cross-attention padding masks, and classifier-free guidance (CFG).

########## IMPORTS AND A FEW GLOBAL VARIABLES ##########

import math, os, sys, argparse, getpass
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
        'alphabet': (" enaitoshrdx.vpukbgfcymzw1lqj804I92637OTAS5N)EHR\"\'(BCQLMWYU,ZF!DXV?KPGJ", str,
                        'All the characters that this model will be able to draw'),
        'dataset_name': ('bigeasybank', str, 'The name of the .zip file containing your dataset'),
        'wandb_project': ('bigbank_experiments', str, 'W&B project name'),
        'wandb_entity': ('sam-greydanus', str, 'Set this to your wandb username or team name'),
        'wandb_run_name': ('unnamed_run', str, 'W&B run name'),
        'wandb_api_key': (None, str, 'Weights & Biases API Key'),
        'load_from_run_id': (None, str, 'Load from a specific W&B run ID'),
        'local_checkpoint_path': ('best_checkpoint.pt', str, 'Path to local model file'),
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

def get_checkpoint(args, sample_only):
    model = MatFormer(args)

    model.to(args.device)
    print(f"Model #params: {sum(p.numel() for p in model.parameters())}")

    if not sample_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr_every, gamma=args.lr_decay)
    else:
        optimizer = None
        scheduler = None

    step = 0
    best_loss = None

    if args.load_from_run_id or sample_only:
        if os.path.exists(args.local_checkpoint_path):
            checkpoint = torch.load(args.local_checkpoint_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from local path: {args.local_checkpoint_path}")
            if not sample_only:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                step = checkpoint['step']
                best_loss = checkpoint['best_loss']
        elif args.load_from_run_id:
            artifact = get_latest_checkpoint_artifact(args)
            artifact_dir = artifact.download()
            checkpoint = torch.load(os.path.join(artifact_dir, "best_checkpoint.pt"), weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])

            if not sample_only:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                step = checkpoint['step'] + 1
                best_loss = checkpoint['best_loss']

            save_checkpoint(model, args.local_checkpoint_path, optimizer, scheduler, step, best_loss)
        else:
            print("No local model or W&B run ID provided. Exiting.")
            sys.exit()

    return model, optimizer, scheduler, step, best_loss



def get_latest_checkpoint_artifact(args, verbose=True):
    run = wandb.Api().run(f"{args.wandb_entity}/{args.wandb_project}/{args.load_from_run_id}")

    if verbose:
        print(f"Finding latest checkpoint for W&B run id {args.load_from_run_id}")
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


def save_checkpoint(model, path, optimizer=None, scheduler=None, step=None, best_loss=None):
    checkpoint = {'model_state_dict': model.state_dict()}
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if step is not None:
        checkpoint['step'] = step
    if best_loss is not None:
        checkpoint['best_loss'] = best_loss
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


def apply_rope(x, cos, sin):
    """Apply rotary embeddings to a (B, n_head, T, head_dim) tensor."""
    T = x.size(-2)
    cos = cos[:T].view(1, 1, T, -1)
    sin = sin[:T].view(1, 1, T, -1)
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
        # QK-norm stabilizes training at the high LR (1e-2) this model uses.
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = rope
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


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
            # Rows that are entirely PAD (the CFG "null" condition) are unmasked so
            # softmax stays well-defined and gives a consistent unconditional signal.
            keep = context_mask.clone()
            keep[~keep.any(dim=1)] = True
            attn_mask = keep.view(B, 1, 1, T_ctx)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    """SwiGLU feed-forward (Shazeer 2020). Gated GLU variant; better quality per FLOP
    than a plain GELU MLP."""
    def __init__(self, config):
        super().__init__()
        hidden = 4 * config.n_embd
        self.intermediate_size = hidden
        self.w_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.w_up = nn.Linear(config.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


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

    def forward(self, x, rope, context=None, context_mask=None):
        x = x + self.attn(self.ln_1(x), rope)

        if self.has_cross_attn:
            assert context is not None, 'Expected context'
            x = x + self.cross_attn(self.ln_2(x), context, context_mask)

        x = x + self.mlp(self.ln_3(x))
        return x


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

    def forward(self, idx, context, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        x = self.transformer.wte(idx)  # (b, t, n_embd); positions handled by RoPE

        context_t = context.size(-1)
        context_pos = torch.arange(0, context_t, dtype=torch.long, device=device).unsqueeze(0)
        c = self.transformer.wce(context) + self.transformer.wcpe(context_pos)
        context_mask = (context != self.context_pad_token)  # (b, t_ctx) True = real char

        rope = self._get_rope(t, device, x.dtype)
        for block in self.transformer.h:
            x = block(x, rope, c, context_mask)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss


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
        return F.linear(F.silu(gate) * up, self.w_down.weight[:, :h])


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
            if 'mlp' not in name and param.requires_grad:
                total_params += param.numel()

        for i in range(self.config.n_layer):
            mlp = self.transformer.h[i].mlp
            if mlp.current_subset_hd is None:
                raise ValueError("Subnetwork size not configured.")
            # SwiGLU: gate + up (in->hd) and down (hd->in), all bias-free.
            total_params += 2 * (mlp.current_subset_hd * self.config.n_embd)
            total_params += self.config.n_embd * mlp.current_subset_hd
        return total_params

"""Lightweight rendering-aware RLOO post-training for Cursive Transformer.

This stage starts from a BOS/style-conditioned SFT checkpoint, samples grouped
free-running handwriting from the raw conditional policy, scores the rendered
strokes, and applies a leave-one-out policy-gradient update plus an SFT anchor.
It intentionally has no critic, PPO snapshot, CFG, or reference model, which
keeps a useful four-condition/four-rollout run practical on a Colab GPU.

A conservative pilot is the default::

    python rl_train.py --sft-run-id <run-id> --max-updates 200

If held-out rendering improves without rising invalid rates, continue the same
optimizer/EMA state to 1,000-1,500 updates::

    python rl_train.py --sft-run-id <run-id> --max-updates 1500 \
        --resume-rl-checkpoint rl_last_checkpoint.pt
"""

from __future__ import annotations

import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F
import wandb

from data import InfiniteDataLoader, create_datasets
from model import (
    EMA,
    configure_optimizer,
    configure_scheduler,
    get_checkpoint,
    save_checkpoint,
)
from rendering_reward import (
    RenderConfig,
    RewardResult,
    RewardWeights,
    TokenSpec,
    analyze_token_sequence,
    compute_rendering_reward,
    render_stroke_pair,
)
from sample import generate


DEFAULT_ALPHABET = (
    " enaitoshrdx.vpukbgfcymzw1lqj804I92637OTAS5N)EHR\"'(BCQLMWYU,ZF!DXV?KPGJ"
)


@dataclass
class RolloutScores:
    rewards: torch.Tensor
    results: list[RewardResult]
    means: dict[str, float]


@dataclass
class EvalOutput:
    metrics: dict[str, float]
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    sequences: torch.Tensor
    scores: RolloutScores


def wandb_config_without_secrets(args: argparse.Namespace) -> dict[str, Any]:
    """Return scalar run config while excluding likely authentication fields."""

    blocked_fragments = ("api_key", "apikey", "password", "secret", "auth_token", "access_token")
    return {
        key: value
        for key, value in vars(args).items()
        if not any(fragment in key.lower() for fragment in blocked_fragments)
    }


def leave_one_out_advantages(
    rewards: torch.Tensor, *, normalize: bool = True, eps: float = 1e-6
) -> torch.Tensor:
    """Compute an RLOO advantage for every member of each rollout group.

    ``rewards`` is ``(conditions, rollouts)``. Each baseline is the mean reward
    of the *other* rollouts for that condition, so the sampled action never
    contributes to its own baseline.
    """

    if rewards.ndim != 2:
        raise ValueError("rewards must have shape (conditions, rollouts)")
    if rewards.size(1) < 2:
        raise ValueError("RLOO requires at least two rollouts per condition")
    baseline = (rewards.sum(dim=1, keepdim=True) - rewards) / (rewards.size(1) - 1)
    advantages = rewards - baseline
    if normalize:
        advantages = advantages / advantages.std(unbiased=False).clamp_min(eps)
    return advantages


def generated_action_mask(
    sequences: torch.Tensor,
    end_token: int,
    pad_token: int,
    *,
    prefix_length: int = 1,
) -> torch.Tensor:
    """Mask sampled actions through and including the first END token."""

    if sequences.ndim != 2:
        raise ValueError("sequences must have shape (batch, time)")
    if not 0 < prefix_length < sequences.size(1):
        raise ValueError("prefix_length must leave at least one generated action")
    actions = sequences[:, prefix_length:]
    is_end = actions.eq(end_token)
    ended_before = is_end.cumsum(dim=1) - is_end.to(torch.long)
    return ended_before.eq(0) & actions.ne(pad_token)


def structured_action_mask(prefixes: torch.Tensor, dataset: Any) -> torch.Tensor:
    """Vectorized equivalent of ``sample._structured_mask`` at every time step.

    The returned ``(B, T, vocab)`` mask describes the valid next actions after
    each prefix ending at ``prefixes[:, t]``. Keeping this replay distribution
    identical to structured rollout sampling is important for an on-policy
    gradient.
    """

    if prefixes.ndim != 2:
        raise ValueError("prefixes must have shape (batch, time)")
    device = prefixes.device
    vocab = dataset.get_vocab_size()
    radius_start = int(dataset.cumulative_sizes[0])
    radius_stop = radius_start + len(dataset.r_bins)
    theta_start = int(dataset.cumulative_sizes[1])
    theta_stop = theta_start + len(dataset.theta_bins)

    is_stroke = (
        ((prefixes >= radius_start) & (prefixes < radius_stop))
        | ((prefixes >= theta_start) & (prefixes < theta_stop))
    )
    parity = is_stroke.cumsum(dim=1).remainder(2)

    token_ids = torch.arange(vocab, device=device)
    theta_tokens = (token_ids >= theta_start) & (token_ids < theta_stop)
    radius_tokens = (token_ids >= radius_start) & (token_ids < radius_stop)
    theta_or_special = theta_tokens.clone()
    theta_or_special[dataset.WORD_TOKEN] = True
    theta_or_special[dataset.END_TOKEN] = True
    valid = torch.where(
        parity.unsqueeze(-1).eq(0),
        theta_or_special.view(1, 1, -1),
        radius_tokens.view(1, 1, -1),
    )

    last_is_word = prefixes.eq(dataset.WORD_TOKEN)
    prev_is_word = torch.zeros_like(last_is_word)
    prev_is_word[:, 1:] = last_is_word[:, :-1]
    only_word = torch.zeros(vocab, dtype=torch.bool, device=device)
    only_word[dataset.WORD_TOKEN] = True
    valid = torch.where(
        (last_is_word & ~prev_is_word).unsqueeze(-1),
        only_word.view(1, 1, -1),
        valid,
    )
    valid = torch.where(
        (last_is_word & prev_is_word).unsqueeze(-1),
        theta_tokens.view(1, 1, -1),
        valid,
    )
    bos_token = getattr(dataset, "BOS_TOKEN", None)
    if bos_token is not None:
        valid = torch.where(
            prefixes.eq(bos_token).unsqueeze(-1),
            theta_tokens.view(1, 1, -1),
            valid,
        )
    return valid


def filter_policy_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Apply the same sampling transform used by ``sample.generate``."""

    filtered = logits / max(float(temperature), 1e-6)
    if top_k > 0:
        threshold = torch.topk(filtered, min(top_k, filtered.size(-1)), dim=-1).values[..., -1:]
        filtered = filtered.masked_fill(filtered < threshold, -torch.inf)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        remove = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
        remove[..., 0] = False
        remove = torch.zeros_like(remove).scatter(-1, sorted_indices, remove)
        filtered = filtered.masked_fill(remove, -torch.inf)
    return filtered


def recompute_sampled_log_probs(
    model: torch.nn.Module,
    sequences: torch.Tensor,
    contexts: torch.Tensor,
    styles: torch.Tensor | None,
    dataset: Any,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    prefix_length: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay sampled actions with gradients under the rollout distribution."""

    action_mask = generated_action_mask(
        sequences, dataset.END_TOKEN, dataset.PAD_TOKEN, prefix_length=prefix_length
    )
    max_actions = int(action_mask.sum(dim=1).max().item())
    if max_actions <= 0:
        raise ValueError("rollouts contain no sampled actions")
    sequences = sequences[:, : prefix_length + max_actions]
    action_mask = action_mask[:, :max_actions]

    prefixes = sequences[:, :-1]
    targets = sequences[:, prefix_length:]
    logits, _ = model(prefixes, contexts, style=styles)
    logits = logits[:, prefix_length - 1 :, :]
    valid = structured_action_mask(prefixes, dataset)[:, prefix_length - 1 :, :]
    logits = logits.masked_fill(~valid, -torch.inf)
    logits = filter_policy_logits(
        logits, temperature=temperature, top_k=top_k, top_p=top_p
    )
    selected = F.log_softmax(logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    selected = torch.where(action_mask, selected, torch.zeros_like(selected))
    if not torch.isfinite(selected).all():
        raise FloatingPointError("sampled actions have non-finite replay log-probabilities")
    return selected, action_mask


def trim_sft_batch(
    inputs: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove the all-ignored tail before the supervised anchor forward."""

    active_columns = targets.ne(-1).any(dim=0).nonzero(as_tuple=False)
    if not len(active_columns):
        raise ValueError("SFT anchor batch contains no targets")
    stop = int(active_columns[-1].item()) + 1
    return inputs[:, :stop].contiguous(), targets[:, :stop].contiguous()


def repeat_optional(tensor: torch.Tensor | None, repeats: int) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.repeat_interleave(repeats, dim=0)


def rollout_total_length(
    targets: torch.Tensor,
    dataset: Any,
    *,
    margin: int,
    hard_cap: int,
) -> int:
    """Choose a target-informed rollout limit without reading prediction bounds."""

    is_end = targets.eq(dataset.END_TOKEN)
    positions = torch.arange(targets.size(1), device=targets.device).view(1, -1)
    end_positions = torch.where(is_end, positions, targets.size(1)).min(dim=1).values
    fallback = targets.ne(dataset.PAD_TOKEN).sum(dim=1).sub(1).clamp_min(1)
    end_positions = torch.where(is_end.any(dim=1), end_positions, fallback)
    total = int(end_positions.max().item()) + 1 + max(0, margin)
    if hard_cap > 0:
        total = min(total, hard_cap)
    return max(2, min(total, dataset.get_stroke_seq_length()))


@torch.no_grad()
def sample_grouped_rollouts(
    model: torch.nn.Module,
    contexts: torch.Tensor,
    styles: torch.Tensor | None,
    target_sequences: torch.Tensor,
    dataset: Any,
    *,
    rollouts_per_condition: int,
    temperature: float,
    top_k: int,
    top_p: float,
    length_margin: int,
    max_tokens: int,
) -> torch.Tensor:
    """Sample structured grouped rollouts with KV caching and no CFG."""

    if rollouts_per_condition < 1:
        raise ValueError("rollouts_per_condition must be positive")
    if hasattr(model, "configure_subnetwork"):
        model.configure_subnetwork("xl")
    expanded_context = contexts.repeat_interleave(rollouts_per_condition, dim=0)
    expanded_style = repeat_optional(styles, rollouts_per_condition)
    prefix = dataset.get_generation_prefix().to(contexts.device).view(1, -1)
    prefix = prefix.expand(expanded_context.size(0), -1)
    total_length = rollout_total_length(
        target_sequences, dataset, margin=length_margin, hard_cap=max_tokens
    )
    return generate(
        model,
        prefix,
        expanded_context,
        max_new_tokens=total_length,
        temperature=temperature,
        do_sample=True,
        top_k=top_k if top_k > 0 else None,
        top_p=top_p if 0.0 < top_p < 1.0 else None,
        guidance_scale=1.0,
        structured=True,
        dataset=dataset,
        style=expanded_style,
        style_guidance_scale=1.0,
        use_cache=True,
    )


def _mean_numeric_results(results: Iterable[RewardResult]) -> dict[str, float]:
    rows = [result.as_dict() for result in results]
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    return {
        key: float(np.mean([row[key] for row in rows if key in row]))
        for key in keys
    }


def _score_one_rollout(
    predicted_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    style_tokens: torch.Tensor | None,
    context: torch.Tensor,
    dataset: Any,
    token_spec: TokenSpec,
    config: RenderConfig,
    weights: RewardWeights,
) -> RewardResult:
    target_sequence = analyze_token_sequence(target_tokens, token_spec)
    prediction = dataset.decode_stroke(predicted_tokens)
    target = dataset.decode_stroke(target_tokens)
    style_reference = (
        dataset.decode_stroke(style_tokens) if style_tokens is not None else None
    )
    expected_words = len(dataset.decode_text(context).split())
    return compute_rendering_reward(
        prediction,
        target,
        style_reference=style_reference,
        predicted_tokens=predicted_tokens,
        token_spec=token_spec,
        expected_word_count=expected_words,
        target_token_length=target_sequence.token_length,
        config=config,
        weights=weights,
        prediction_is_polar=True,
        target_is_polar=True,
        style_reference_is_polar=True,
    )


def score_grouped_rollouts(
    sequences: torch.Tensor,
    target_sequences: torch.Tensor,
    contexts: torch.Tensor,
    styles: torch.Tensor | None,
    dataset: Any,
    *,
    rollouts_per_condition: int,
    config: RenderConfig,
    weights: RewardWeights,
    workers: int = 0,
) -> RolloutScores:
    """Decode and score a grouped rollout batch on CPU."""

    expected = target_sequences.size(0) * rollouts_per_condition
    if sequences.size(0) != expected:
        raise ValueError(f"expected {expected} rollouts, got {sequences.size(0)}")
    sequences = sequences.detach().cpu()
    targets = target_sequences.detach().cpu().repeat_interleave(
        rollouts_per_condition, dim=0
    )
    contexts = contexts.detach().cpu().repeat_interleave(
        rollouts_per_condition, dim=0
    )
    expanded_styles = None
    if styles is not None:
        expanded_styles = styles.detach().cpu().repeat_interleave(
            rollouts_per_condition, dim=0
        )
    token_spec = TokenSpec.from_dataset(dataset)

    def score(index: int) -> RewardResult:
        style = None if expanded_styles is None else expanded_styles[index]
        return _score_one_rollout(
            sequences[index],
            targets[index],
            style,
            contexts[index],
            dataset,
            token_spec,
            config,
            weights,
        )

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(score, range(expected)))
    else:
        results = [score(index) for index in range(expected)]
    rewards = torch.tensor([result.total for result in results], dtype=torch.float32)
    rewards = rewards.view(target_sequences.size(0), rollouts_per_condition)
    return RolloutScores(rewards, results, _mean_numeric_results(results))


def length_reward_weight(update: int, args: argparse.Namespace) -> float:
    """Hold length pressure weak initially, then ramp it in after shape learning."""

    if update < args.length_warmup_updates:
        return args.initial_length_weight
    progress = min(
        1.0,
        (update - args.length_warmup_updates)
        / max(1, args.length_ramp_updates),
    )
    return args.initial_length_weight + progress * (
        args.length_weight - args.initial_length_weight
    )


@contextmanager
def fixed_dataset_batch(dataset: Any, count: int):
    """Yield a repeatable first-N dataset batch without advancing augmentation."""

    previous_counter = getattr(dataset, "counter", None)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    seed = int(getattr(getattr(dataset, "args", None), "seed", 0)) + 9001
    random.seed(seed)
    np.random.seed(seed)
    if previous_counter is not None:
        dataset.counter = 0
    try:
        items = [dataset[index] for index in range(min(count, len(dataset)))]
        columns = tuple(torch.stack(values) for values in zip(*items))
        yield columns
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if previous_counter is not None:
            dataset.counter = previous_counter


@contextmanager
def fixed_torch_rng(seed: int):
    """Run deterministic sampling without perturbing the training RNG stream."""

    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def reward_weights_from_args(args: argparse.Namespace) -> RewardWeights:
    return RewardWeights(
        shape=args.shape_weight,
        style=args.style_weight,
        word_count=args.word_count_weight,
        ended=args.ended_weight,
        validity=args.validity_weight,
        out_of_bounds=args.out_of_bounds_weight,
        length=args.length_weight,
        blank=args.blank_weight,
        hard_failure=args.hard_failure_weight,
    )


def render_config_from_args(args: argparse.Namespace) -> RenderConfig:
    return RenderConfig(width=args.reward_width, height=args.reward_height)


def _reward_log(scores: RolloutScores, prefix: str) -> dict[str, float]:
    log = {f"{prefix}/{key}": value for key, value in scores.means.items()}
    log[f"{prefix}/group_std"] = float(
        scores.rewards.std(dim=1, unbiased=False).mean().item()
    )
    return log


@torch.no_grad()
def evaluate_policy(
    model: torch.nn.Module,
    dataset: Any,
    args: argparse.Namespace,
    config: RenderConfig,
    weights: RewardWeights,
) -> EvalOutput:
    """Evaluate held-out rendering plus correct/no/shuffled style controls."""

    with fixed_dataset_batch(dataset, args.eval_conditions) as cpu_batch:
        x_cpu, c_cpu, y_cpu, s_cpu = cpu_batch
        x = x_cpu.to(args.device)
        c = c_cpu.to(args.device)
        y = y_cpu.to(args.device)
        style = s_cpu.to(args.device) if s_cpu.numel() else None
        sequences = sample_grouped_rollouts(
            model,
            c,
            style,
            x,
            dataset,
            rollouts_per_condition=args.eval_rollouts,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            length_margin=args.rollout_length_margin,
            max_tokens=args.rollout_max_tokens,
        )
        scores = score_grouped_rollouts(
            sequences,
            x_cpu,
            c_cpu,
            s_cpu if s_cpu.numel() else None,
            dataset,
            rollouts_per_condition=args.eval_rollouts,
            config=config,
            weights=weights,
            workers=args.reward_workers,
        )

        sft_x, sft_y = trim_sft_batch(x, y)
        _, sft_loss = model(sft_x, c, sft_y, style=style)
        metrics = _reward_log(scores, "eval_reward")
        metrics["eval_reward/mean"] = float(scores.rewards.mean().item())
        metrics["eval/sft_loss"] = float(sft_loss.item())
        metrics["eval_style/sft_loss_correct"] = float(sft_loss.item())

        if style is not None:
            no_style = torch.full_like(style, dataset.PAD_TOKEN)
            _, no_style_sft_loss = model(sft_x, c, sft_y, style=no_style)
            metrics["eval_style/sft_loss_no_style"] = float(
                no_style_sft_loss.item()
            )
            metrics["eval_style/sft_gain_vs_no_style"] = float(
                no_style_sft_loss.item() - sft_loss.item()
            )
            shuffled = torch.roll(style, shifts=1, dims=0) if style.size(0) > 1 else None
            if shuffled is not None:
                _, shuffled_sft_loss = model(sft_x, c, sft_y, style=shuffled)
                metrics["eval_style/sft_loss_shuffled_style"] = float(
                    shuffled_sft_loss.item()
                )
                metrics["eval_style/sft_gain_vs_shuffled"] = float(
                    shuffled_sft_loss.item() - sft_loss.item()
                )

        if style is not None and args.eval_ablation_rollouts > 0:
            ablation_k = args.eval_ablation_rollouts
            no_style_sequences = sample_grouped_rollouts(
                model,
                c,
                no_style,
                x,
                dataset,
                rollouts_per_condition=ablation_k,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                length_margin=args.rollout_length_margin,
                max_tokens=args.rollout_max_tokens,
            )
            no_style_scores = score_grouped_rollouts(
                no_style_sequences,
                x_cpu,
                c_cpu,
                s_cpu,
                dataset,
                rollouts_per_condition=ablation_k,
                config=config,
                weights=weights,
                workers=args.reward_workers,
            )
            correct = metrics["eval_reward/mean"]
            no_style_mean = float(no_style_scores.rewards.mean().item())
            metrics.update(
                {
                    "eval_style/reward_no_style": no_style_mean,
                    "eval_style/gain_vs_no_style": correct - no_style_mean,
                }
            )
            if shuffled is not None:
                shuffled_sequences = sample_grouped_rollouts(
                    model,
                    c,
                    shuffled,
                    x,
                    dataset,
                    rollouts_per_condition=ablation_k,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    length_margin=args.rollout_length_margin,
                    max_tokens=args.rollout_max_tokens,
                )
                shuffled_scores = score_grouped_rollouts(
                    shuffled_sequences,
                    x_cpu,
                    c_cpu,
                    s_cpu,
                    dataset,
                    rollouts_per_condition=ablation_k,
                    config=config,
                    weights=weights,
                    workers=args.reward_workers,
                )
                shuffled_mean = float(shuffled_scores.rewards.mean().item())
                metrics.update(
                    {
                        "eval_style/reward_shuffled_style": shuffled_mean,
                        "eval_style/gain_vs_shuffled": correct - shuffled_mean,
                    }
                )
        return EvalOutput(metrics, cpu_batch, sequences.detach().cpu(), scores)


def _wandb_image(image: np.ndarray) -> wandb.Image:
    pixels = np.repeat(((1.0 - image) * 255.0).clip(0, 255)[..., None], 3, axis=2)
    return wandb.Image(pixels.astype(np.uint8))


def make_eval_table(output: EvalOutput, dataset: Any, config: RenderConfig) -> wandb.Table:
    """Build target/style/best/median/worst fixed-canvas evaluation rows."""

    x, c, _, style = output.batch
    k = output.scores.rewards.size(1)
    table = wandb.Table(
        columns=[
            "text",
            "target",
            "style_reference",
            "best",
            "median",
            "worst",
            "best_reward",
            "median_reward",
            "worst_reward",
        ]
    )
    for condition in range(x.size(0)):
        target = dataset.decode_stroke(x[condition])
        target_pair = render_stroke_pair(
            target, target, config=config, prediction_is_polar=True, target_is_polar=True
        )
        style_words = dataset.decode_stroke(style[condition]) if style.numel() else []
        style_pair = render_stroke_pair(
            style_words,
            style_words,
            config=config,
            prediction_is_polar=True,
            target_is_polar=True,
        )
        order = output.scores.rewards[condition].argsort()
        picks = [int(order[-1]), int(order[len(order) // 2]), int(order[0])]
        images = []
        for rollout in picks:
            index = condition * k + rollout
            prediction = dataset.decode_stroke(output.sequences[index])
            pair = render_stroke_pair(
                prediction,
                target,
                config=config,
                prediction_is_polar=True,
                target_is_polar=True,
            )
            images.append(_wandb_image(pair.prediction))
        rewards = [float(output.scores.rewards[condition, rollout]) for rollout in picks]
        table.add_data(
            dataset.decode_text(c[condition]),
            _wandb_image(target_pair.target),
            _wandb_image(style_pair.target),
            *images,
            *rewards,
        )
    return table


def save_rl_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    completed_updates: int,
    best_reward: float | None,
    ema: EMA | None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model,
        path,
        optimizer,
        scheduler,
        completed_updates,
        best_reward,
        ema=ema,
    )


def log_best_artifact(
    args: argparse.Namespace, completed_updates: int, best_reward: float, ema: EMA | None
) -> None:
    artifact = wandb.Artifact(
        args.artifact_name,
        type="model",
        metadata={
            "completed_updates": completed_updates,
            "best_eval_reward": best_reward,
            "ema_weights_included": ema is not None,
            "source_sft_run_id": args.sft_run_id,
            "checkpoint_metric": "eval_reward/mean",
        },
    )
    # model.get_checkpoint expects this exact artifact payload name.
    artifact.add_file(args.best_checkpoint_path, name="best_checkpoint.pt")
    wandb.log_artifact(artifact, aliases=["best"])


def load_sft_model(args: argparse.Namespace) -> torch.nn.Module:
    """Load the shipped EMA policy from a local or W&B SFT checkpoint."""

    if args.sft_checkpoint_path:
        if not os.path.exists(args.sft_checkpoint_path):
            raise FileNotFoundError(args.sft_checkpoint_path)
        args.local_checkpoint_path = args.sft_checkpoint_path
        args.load_from_run_id = None
    else:
        args.load_from_run_id = args.sft_run_id
        if args.sft_cache_path:
            cache_path = Path(args.sft_cache_path)
        else:
            cache_path = Path(".cache") / "cursivetransformer" / f"{args.sft_run_id}.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        args.local_checkpoint_path = str(cache_path)
    model, _, _, _, _, _ = get_checkpoint(args, sample_only=True)
    if not getattr(args, "use_bos", False):
        raise ValueError("RLOO requires a checkpoint trained with --use_bos")
    if not getattr(model, "use_style", False):
        raise ValueError("RLOO requires a style-conditioned SFT checkpoint")
    return model


def resume_rl_state(
    path: str | None,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: EMA | None,
) -> tuple[int, float | None]:
    if not path:
        return 0, None
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if ema is not None and checkpoint.get("ema_state_dict") is not None:
        ema.load_state_dict(checkpoint["ema_state_dict"])
    completed = int(checkpoint.get("step", 0))
    best_reward = checkpoint.get("best_loss")
    print(f"Resumed RLOO state from {path} after {completed} updates")
    return completed, best_reward


def add_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sft-checkpoint-path")
    source.add_argument("--sft-run-id")
    parser.add_argument("--sft-cache-path")
    parser.add_argument("--resume-rl-checkpoint")

    parser.add_argument("--max-updates", type=int, default=200)
    parser.add_argument("--conditions-batch-size", type=int, default=4)
    parser.add_argument("--rollouts-per-condition", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--rollout-length-margin", type=int, default=64)
    parser.add_argument("--rollout-max-tokens", type=int, default=0)
    parser.add_argument("--rl-learning-rate", type=float, default=1e-4)
    parser.add_argument("--rl-style-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--rl-warmup-updates", type=int, default=10)
    parser.add_argument(
        "--lr-total-updates",
        type=int,
        default=1500,
        help="Cosine-schedule horizon; keep fixed when extending a pilot",
    )
    parser.add_argument("--sft-coef", type=float, default=0.25)
    parser.add_argument("--normalize-advantages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-conditions", type=int, default=4)
    parser.add_argument("--eval-rollouts", type=int, default=4)
    parser.add_argument("--eval-ablation-rollouts", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--reward-workers", type=int, default=4)
    parser.add_argument("--best-checkpoint-path", default="rl_best_checkpoint.pt")
    parser.add_argument("--last-checkpoint-path", default="rl_last_checkpoint.pt")
    parser.add_argument("--artifact-name", default="rloo_checkpoint")
    parser.add_argument("--log-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-media", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--initial-length-weight", type=float, default=0.02)
    parser.add_argument("--length-weight", type=float, default=0.10)
    parser.add_argument("--length-warmup-updates", type=int, default=50)
    parser.add_argument("--length-ramp-updates", type=int, default=100)
    defaults = RewardWeights()
    parser.add_argument("--shape-weight", type=float, default=defaults.shape)
    parser.add_argument("--style-weight", type=float, default=defaults.style)
    parser.add_argument("--word-count-weight", type=float, default=defaults.word_count)
    parser.add_argument("--ended-weight", type=float, default=defaults.ended)
    parser.add_argument("--validity-weight", type=float, default=defaults.validity)
    parser.add_argument("--out-of-bounds-weight", type=float, default=defaults.out_of_bounds)
    parser.add_argument("--blank-weight", type=float, default=defaults.blank)
    parser.add_argument("--hard-failure-weight", type=float, default=defaults.hard_failure)
    parser.add_argument("--reward-width", type=int, default=384)
    parser.add_argument("--reward-height", type=int, default=128)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-name", default="bigbank_3500")
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--num-words", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-seq-length", type=int, default=1050)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--downsample-mean", type=float, default=0.65)
    parser.add_argument("--downsample-width", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--alphabet", default=DEFAULT_ALPHABET)

    parser.add_argument("--wandb-entity", default="cursivetransformer-ng")
    parser.add_argument("--wandb-project", default="cursivetransformer-ng")
    parser.add_argument("--wandb-run-name", default="modern_v5_rloo")
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)
    if args.rollouts_per_condition < 2:
        parser.error("--rollouts-per-condition must be at least 2 for RLOO")
    if args.max_updates <= 0:
        parser.error("--max-updates must be positive")
    if args.lr_total_updates < args.max_updates:
        parser.error("--lr-total-updates must be at least --max-updates")
    if args.sft_coef < 0.0:
        parser.error("--sft-coef cannot be negative")
    return args


def prepare_legacy_model_args(args: argparse.Namespace) -> None:
    """Supply the current model/data API fields; checkpoint stamps override them."""

    args.n_layer = 4
    args.n_embd = 64
    args.n_embd_context = 64
    args.n_ctx_head = 4
    args.n_context_layer = 2
    args.dropout = 0.0
    args.style_words = 3
    args.max_style_length = 600
    args.n_style_tokens = 16
    args.n_style_layer = 2
    args.use_bos = True
    args.context_block_size = 50
    args.context_vocab_size = len(args.alphabet) + 1
    args.vocab_size = 526
    args.block_size = args.max_seq_length
    args.stroke_pad_token = 522
    args.context_pad_token = 0
    args.add_digits = True
    args.cond_drop_prob = 0.0
    args.style_drop_prob = 0.0
    args.subnetwork_mode = "full"
    args.style_lr_multiplier = 1.0
    args.learning_rate = args.rl_learning_rate
    args.lr_schedule = "cosine"
    args.warmup_steps = args.rl_warmup_updates
    # The pilot is an early stop on the eventual schedule, not a short schedule that
    # would jump back toward peak LR when resumed with a longer max-update target.
    args.max_steps = args.lr_total_updates
    args.lr_decay = 0.333
    args.step_lr_every = 1000
    args.wandb_api_key = None
    args.init_from_run_id = None
    args.init_checkpoint_path = None


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    prepare_legacy_model_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    model = load_sft_model(args)
    # RLOO uses its own optimizer policy rather than inheriting the SFT group ratio.
    args.style_lr_multiplier = args.rl_style_lr_multiplier
    model.config.style_lr_multiplier = args.rl_style_lr_multiplier
    # Dataset tensor sizes must follow the checkpoint, not CLI placeholders.
    args.max_seq_length = int(args.block_size)
    train_dataset, test_dataset = create_datasets(args)
    if train_dataset.get_vocab_size() != model.config.vocab_size:
        raise ValueError("dataset/checkpoint stroke vocabularies do not match")
    if train_dataset.get_text_seq_length() != model.config.context_block_size:
        raise ValueError("dataset/checkpoint text context lengths do not match")

    optimizer = configure_optimizer(model, args)
    scheduler = configure_scheduler(optimizer, args)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0.0 else None
    start_update, best_reward = resume_rl_state(
        args.resume_rl_checkpoint, model, optimizer, scheduler, ema
    )
    if start_update >= args.max_updates:
        raise ValueError(
            f"checkpoint already has {start_update} updates; --max-updates must be larger"
        )
    loader = InfiniteDataLoader(
        train_dataset,
        batch_size=args.conditions_batch_size,
        pin_memory=args.device.startswith("cuda"),
        num_workers=args.num_workers,
    )
    model.eval()  # Gradients remain enabled; this keeps rollout/replay dropout identical.

    init_kwargs: dict[str, Any] = {
        "entity": args.wandb_entity,
        "project": args.wandb_project,
        "name": args.wandb_run_name,
        "mode": args.wandb_mode,
        "config": wandb_config_without_secrets(args),
    }
    if args.wandb_run_id:
        init_kwargs.update(id=args.wandb_run_id, resume="must")
    run = wandb.init(**init_kwargs)
    config = render_config_from_args(args)
    final_weights = reward_weights_from_args(args)

    # Update zero is the deterministic SFT baseline. RLOO must beat it before
    # replacing the checkpoint exposed through the W&B ``best`` artifact.
    if start_update == 0 and best_reward is None:
        with fixed_torch_rng(args.seed + 100000):
            with ema.swap(model) if ema is not None else _nullcontext():
                baseline = evaluate_policy(
                    model, test_dataset, args, config, final_weights
                )
        best_reward = baseline.metrics["eval_reward/mean"]
        baseline_log: dict[str, Any] = {"update": 0, **baseline.metrics}
        if args.log_media and args.wandb_mode != "disabled":
            baseline_log["eval/samples"] = make_eval_table(
                baseline, test_dataset, config
            )
        save_rl_checkpoint(
            model,
            args.best_checkpoint_path,
            optimizer,
            scheduler,
            0,
            best_reward,
            ema,
        )
        run.summary["baseline_eval_reward"] = best_reward
        run.summary["best_eval_reward"] = best_reward
        run.summary["best_update"] = 0
        if args.log_artifacts:
            log_best_artifact(args, 0, best_reward, ema)
        wandb.log(baseline_log, step=0)
        print(f"update    0 | baseline eval {best_reward:.4f}")

    completed_updates = start_update
    try:
        for update in range(start_update, args.max_updates):
            started = time.perf_counter()
            x, context, targets, style = loader.next()
            x = x.to(args.device)
            context = context.to(args.device)
            targets = targets.to(args.device)
            style = style.to(args.device) if style.numel() else None

            sequences = sample_grouped_rollouts(
                model,
                context,
                style,
                x,
                train_dataset,
                rollouts_per_condition=args.rollouts_per_condition,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                length_margin=args.rollout_length_margin,
                max_tokens=args.rollout_max_tokens,
            )
            current_length_weight = length_reward_weight(update, args)
            train_weights = replace(final_weights, length=current_length_weight)
            scores = score_grouped_rollouts(
                sequences,
                x,
                context,
                style,
                train_dataset,
                rollouts_per_condition=args.rollouts_per_condition,
                config=config,
                weights=train_weights,
                workers=args.reward_workers,
            )
            rewards = scores.rewards.to(args.device)
            advantages = leave_one_out_advantages(
                rewards, normalize=args.normalize_advantages
            )

            expanded_context = context.repeat_interleave(
                args.rollouts_per_condition, dim=0
            )
            expanded_style = repeat_optional(style, args.rollouts_per_condition)
            sampled_log_probs, action_mask = recompute_sampled_log_probs(
                model,
                sequences,
                expanded_context,
                expanded_style,
                train_dataset,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            action_counts = action_mask.sum(dim=1).clamp_min(1)
            sequence_log_probs = (
                sampled_log_probs * action_mask
            ).sum(dim=1) / action_counts
            policy_loss = -(
                sequence_log_probs * advantages.reshape(-1).detach()
            ).mean()

            sft_x, sft_targets = trim_sft_batch(x, targets)
            _, sft_loss = model(sft_x, context, sft_targets, style=style)
            loss = policy_loss + args.sft_coef * sft_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf")
            )
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            completed = update + 1
            completed_updates = completed
            log = {
                "update": completed,
                "train/loss": float(loss.item()),
                "train/policy_loss": float(policy_loss.item()),
                "train/sft_loss": float(sft_loss.item()),
                "train/reward_mean": float(rewards.mean().item()),
                "train/reward_std": float(rewards.std(unbiased=False).item()),
                "train/advantage_std": float(advantages.std(unbiased=False).item()),
                "train/sample_nll": float(
                    -(sampled_log_probs.sum() / action_mask.sum().clamp_min(1)).item()
                ),
                "train/action_tokens": float(action_mask.sum(dim=1).float().mean().item()),
                "train/grad_norm": float(grad_norm),
                "train/learning_rate": float(scheduler.get_last_lr()[0]),
                "train/length_reward_weight": current_length_weight,
                "train/update_seconds": time.perf_counter() - started,
            }
            log.update(_reward_log(scores, "train_reward"))

            should_eval = completed % args.eval_every == 0 or completed == args.max_updates
            if should_eval:
                with fixed_torch_rng(args.seed + 100000):
                    with ema.swap(model) if ema is not None else _nullcontext():
                        evaluation = evaluate_policy(
                            model, test_dataset, args, config, final_weights
                        )
                log.update(evaluation.metrics)
                eval_reward = evaluation.metrics["eval_reward/mean"]
                improved = best_reward is None or eval_reward > best_reward
                if improved:
                    best_reward = eval_reward
                    save_rl_checkpoint(
                        model,
                        args.best_checkpoint_path,
                        optimizer,
                        scheduler,
                        completed,
                        best_reward,
                        ema,
                    )
                    run.summary["best_eval_reward"] = best_reward
                    run.summary["best_update"] = completed
                    if args.log_artifacts:
                        log_best_artifact(args, completed, best_reward, ema)
                if args.log_media and args.wandb_mode != "disabled":
                    log["eval/samples"] = make_eval_table(
                        evaluation, test_dataset, config
                    )

            if completed % args.save_every == 0 or completed == args.max_updates:
                save_rl_checkpoint(
                    model,
                    args.last_checkpoint_path,
                    optimizer,
                    scheduler,
                    completed,
                    best_reward,
                    ema,
                )
            wandb.log(log, step=completed)
            if completed % args.print_every == 0 or should_eval:
                suffix = (
                    f" | eval {log['eval_reward/mean']:.4f}" if should_eval else ""
                )
                print(
                    f"update {completed:4d} | reward {rewards.mean().item():.4f} "
                    f"| policy {policy_loss.item():.4f} | sft {sft_loss.item():.4f}"
                    f"{suffix} | {log['train/update_seconds']:.2f}s"
                )
    finally:
        if completed_updates > start_update:
            save_rl_checkpoint(
                model,
                args.last_checkpoint_path,
                optimizer,
                scheduler,
                completed_updates,
                best_reward,
                ema,
            )
        wandb.finish()


@contextmanager
def _nullcontext():
    yield


if __name__ == "__main__":
    main()

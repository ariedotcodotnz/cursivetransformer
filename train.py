# Sam Greydanus | 2024

########## IMPORTS AND A FEW GLOBAL VARIABLES ##########

import os, sys, time, getpass, random
from contextlib import contextmanager, nullcontext
from typing import Optional
from dataclasses import dataclass

import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import get_checkpoint, save_checkpoint, get_all_args, EMA
from sample import save_samples
from data import InfiniteDataLoader, create_datasets


def wandb_config_without_secrets(args):
    """Return the run configuration without authentication credentials."""
    return {key: value for key, value in vars(args).items()
            if key != 'wandb_api_key'}


@contextmanager
def deterministic_dataset_iteration(dataset, seed):
    """Replay stochastic dataset augmentation without perturbing training RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    dataset_counter = getattr(dataset, 'counter', None)
    random.seed(seed)
    np.random.seed(seed)
    if dataset_counter is not None:
        dataset.counter = 0
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if dataset_counter is not None:
            dataset.counter = dataset_counter


@torch.inference_mode()
def evaluate(model, dataset, batch_size=15, max_batches=None,
             style_metrics=False, seed=0):
    """Evaluate deterministically and return token-weighted losses.

    Style-aware evaluation reuses each exact target batch with the correct style,
    an all-PAD style, and a style reference taken from another batch row.
    """
    was_training = model.training
    model.eval()
    if hasattr(model, 'configure_subnetwork'):
        model.configure_subnetwork('xl')  # always evaluate the full model

    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size,
                        num_workers=0, generator=generator)
    loss_sums = {'loss': 0.0, 'loss_no_style': 0.0,
                 'loss_shuffled_style': 0.0}
    token_counts = {key: 0 for key in loss_sums}

    try:
        with deterministic_dataset_iteration(dataset, seed):
            for i, batch in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break
                X, C, Y, S = [t.to(device) for t in batch]
                target_tokens = int((Y != -1).sum().item())
                if target_tokens == 0:
                    continue

                _, loss = model(X, C, Y, style=S if S.numel() else None)
                loss_sums['loss'] += loss.item() * target_tokens
                token_counts['loss'] += target_tokens

                has_style = (style_metrics and S.numel() > 0
                             and getattr(model, 'use_style', False))
                if not has_style:
                    continue

                no_style = torch.full_like(S, dataset.PAD_TOKEN)
                _, no_style_loss = model(X, C, Y, style=no_style)
                loss_sums['loss_no_style'] += no_style_loss.item() * target_tokens
                token_counts['loss_no_style'] += target_tokens

                # A singleton cannot be shuffled to a different reference. Exclude it
                # instead of silently reporting the correct-style loss a second time.
                if S.size(0) > 1:
                    shuffled_style = torch.roll(S, shifts=1, dims=0)
                    _, shuffled_loss = model(X, C, Y, style=shuffled_style)
                    loss_sums['loss_shuffled_style'] += shuffled_loss.item() * target_tokens
                    token_counts['loss_shuffled_style'] += target_tokens
    finally:
        model.train(was_training)

    if token_counts['loss'] == 0:
        raise ValueError('Evaluation produced no target tokens')
    return {key: loss_sums[key] / count for key, count in token_counts.items()
            if count > 0}


########## ARGS, LOGGING, AND TRAIN LOOP ##########



if __name__ == '__main__':

    args = get_all_args()
    torch.manual_seed(args.seed)  # system inits
    torch.cuda.manual_seed_all(args.seed)

    wandb_init_args = {"project": args.wandb_project, "entity": args.wandb_entity,
                       "config": wandb_config_without_secrets(args)}
    if args.load_from_run_id:
        wandb_init_args["id"] = args.load_from_run_id
        wandb_init_args["resume"] = "must"
    else:
        wandb_init_args["name"] = args.wandb_run_name
    wandb.init(**wandb_init_args)

    train_dataset, test_dataset = create_datasets(args)  # init datasets
    args.vocab_size = train_dataset.get_vocab_size()
    args.block_size = train_dataset.get_stroke_seq_length()
    args.context_block_size = train_dataset.get_text_seq_length()
    args.context_vocab_size = train_dataset.get_char_vocab_size()
    args.stroke_pad_token = train_dataset.PAD_TOKEN
    print(f"Dataset determined that: {args.vocab_size=}, {args.block_size=}")
    wandb.config.update({
        'vocab_size': args.vocab_size,
        'block_size': args.block_size,
        'context_block_size': args.context_block_size,
        'context_vocab_size': args.context_vocab_size,
        'stroke_pad_token': args.stroke_pad_token,
    }, allow_val_change=True)

    model, optimizer, scheduler, step, best_loss, ema_state = get_checkpoint(args, sample_only=False)
    if train_dataset.get_vocab_size() != model.config.vocab_size:
        raise ValueError(
            'Dataset/checkpoint vocabulary mismatch. For a v5 checkpoint, include '
            '`--use_bos`; for a legacy checkpoint, omit it.'
        )
    if train_dataset.get_stroke_seq_length() != model.config.block_size:
        raise ValueError('Dataset/checkpoint stroke sequence lengths do not match')
    if train_dataset.get_text_seq_length() != model.config.context_block_size:
        raise ValueError('Dataset/checkpoint text context lengths do not match')
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None and ema_state is not None:
        ema.load_state_dict(ema_state)
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    wandb.watch(model, log="all", log_freq=args.log_every, log_graph=False)  # model saving stuff

    eval_batch_size = max(1, getattr(args, 'eval_batch_size', 100))
    eval_max_batches_arg = getattr(args, 'eval_max_batches', -1)
    eval_max_batches = None if eval_max_batches_arg <= 0 else eval_max_batches_arg
    early_stopping_patience = max(0, getattr(args, 'early_stopping_patience', 0))
    early_stopping_min_delta = max(0.0, getattr(args, 'early_stopping_min_delta', 0.0))
    evaluations_without_improvement = 0


    ########## ARGS, LOGGING, AND TRAIN LOOP ##########

    # training loop
    while True:
        should_stop_early = False
        t0 = time.time()

        # get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        X, C, Y, S = [t.to(args.device) for t in batch]
        S = S if S.numel() else None

        # Classifier-free guidance: randomly drop the ASCII condition (replace with the
        # null/all-PAD context) so the model also learns the unconditional distribution.
        # At sample time we extrapolate away from it to sharpen text adherence.
        if args.cond_drop_prob > 0:
            drop = torch.rand(C.size(0), device=C.device) < args.cond_drop_prob
            if drop.any():
                C = C.clone()
                C[drop] = 0  # char_PAD_TOKEN -> the null condition

        # Style CFG: independently drop the style reference (all-PAD) so the model also
        # learns the style-unconditional distribution; at sample time we can push toward
        # the reference style with a separate style guidance scale.
        if S is not None and args.style_drop_prob > 0:
            drop = torch.rand(S.size(0), device=S.device) < args.style_drop_prob
            if drop.any():
                S = S.clone()
                S[drop] = train_dataset.PAD_TOKEN  # all-PAD -> the null style

        # MatFormer granularity for this step (see --subnetwork_mode)
        if args.subnetwork_mode == 'random':
            flag = random.choice(['s', 'm', 'l', 'xl'])
        elif args.subnetwork_mode in ('s', 'm', 'l', 'xl'):
            flag = args.subnetwork_mode
        else:  # 'full' and anything else -> train the full model every step
            flag = 'xl'
        model.configure_subnetwork(flag)

        # feed into the model
        logits, loss = model(X, C, Y, style=S)

        # calculate the gradient, clip it, update the weights
        model.zero_grad(set_to_none=True) ; loss.backward()
        max_grad_norm = args.grad_clip if args.grad_clip > 0 else float('inf')
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        grad_norm = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
        optimizer.step() ; scheduler.step()
        if ema is not None:
            ema.update(model)
        wandb.log({"train_loss_step": loss.item(), "train_grad_norm": grad_norm,
                   "learning_rate": scheduler.get_last_lr()[0], "step": step})
        t1 = time.time()

        # logging
        if step % args.print_every == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms | lr {scheduler.get_last_lr()[0]:.6f}")

        # evaluate the model (with EMA weights when enabled: those are what we ship)
        if step > 0 and step % args.log_every == 0:
            with (ema.swap(model) if ema is not None else nullcontext()):
                train_metrics = evaluate(model, train_dataset, batch_size=eval_batch_size,
                                         max_batches=10, seed=args.seed)
                test_metrics = evaluate(model, test_dataset, batch_size=eval_batch_size,
                                        max_batches=eval_max_batches,
                                        style_metrics=True, seed=args.seed + 1)
            train_loss = train_metrics['loss']
            test_loss = test_metrics['loss']
            eval_log = {"train_loss": train_loss, "test_loss": test_loss, "step": step}
            if 'loss_no_style' in test_metrics:
                eval_log.update({
                    'test_loss_style_correct': test_loss,
                    'test_loss_no_style': test_metrics['loss_no_style'],
                    'style_gain_vs_no_style': test_metrics['loss_no_style'] - test_loss,
                })
            if 'loss_shuffled_style' in test_metrics:
                eval_log.update({
                    'test_loss_shuffled_style': test_metrics['loss_shuffled_style'],
                    'style_gain_vs_shuffled': test_metrics['loss_shuffled_style'] - test_loss,
                })
            wandb.log(eval_log)
            style_summary = ''
            if 'loss_no_style' in test_metrics:
                style_summary += f" no-style: {test_metrics['loss_no_style']:.4f}"
            if 'loss_shuffled_style' in test_metrics:
                style_summary += f" shuffled-style: {test_metrics['loss_shuffled_style']:.4f}"
            print(f"step {step} train loss: {train_loss:.4f} test loss: {test_loss:.4f}{style_summary}")

            improved = best_loss is None or test_loss < best_loss - early_stopping_min_delta
            if improved:  # save the model to W&B if it has improved
                best_loss = test_loss
                evaluations_without_improvement = 0
                print(f"Test loss {test_loss:.4f} is the best so far, saving checkpoint to {args.local_checkpoint_path}")
                save_checkpoint(model, args.local_checkpoint_path, optimizer, scheduler, step, best_loss, ema=ema)
                artifact_metadata = {
                    'step': int(step),
                    'best_test_loss': float(best_loss),
                    'train_loss': float(train_loss),
                    'ema_weights_included': ema is not None,
                    'ema_decay': float(args.ema_decay),
                    'dataset_name': args.dataset_name,
                    'style_words': int(args.style_words),
                    'model_parameters': sum(p.numel() for p in model.parameters()),
                    'checkpoint_metric': 'test_loss',
                    'eval_batch_size': eval_batch_size,
                    'validation_max_batches': eval_max_batches_arg,
                    'validation_examples': (len(test_dataset) if eval_max_batches is None
                                            else min(len(test_dataset),
                                                     eval_batch_size * eval_max_batches)),
                    'early_stopping_patience': early_stopping_patience,
                    'early_stopping_min_delta': early_stopping_min_delta,
                }
                for key, value in test_metrics.items():
                    artifact_metadata[f'test_{key}'] = float(value)
                artifact = wandb.Artifact('best_checkpoint', type='model',
                                          metadata=artifact_metadata)
                artifact.add_file(args.local_checkpoint_path)
                wandb.log_artifact(artifact, aliases=['best'])
                wandb.run.summary['best_test_loss'] = best_loss
                wandb.run.summary['best_step'] = step
            else:
                evaluations_without_improvement += 1

            should_stop_early = (early_stopping_patience > 0
                                 and evaluations_without_improvement >= early_stopping_patience)
            if should_stop_early:
                print(f"Early stopping after {evaluations_without_improvement} evaluations "
                      f"without an improvement of at least {early_stopping_min_delta:g}")
                wandb.run.summary['early_stopped'] = True
                wandb.run.summary['early_stopping_step'] = step

        # sample from the model (EMA weights when enabled)
        if step > 0 and step % args.log_every == 0:
            with (ema.swap(model) if ema is not None else nullcontext()):
                save_samples(model, test_dataset, num=6, do_sample=True)
                save_samples(model, test_dataset, num=6, do_sample=False)
                save_samples(model, train_dataset, num=3, do_sample=True)
                save_samples(model, train_dataset, num=3, do_sample=False)

        if should_stop_early:
            break

        step += 1
        # termination conditions
        if args.max_steps >= 0 and step >= args.max_steps:
            break

    wandb.finish()

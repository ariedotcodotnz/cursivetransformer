# Sam Greydanus | 2024

########## IMPORTS AND A FEW GLOBAL VARIABLES ##########

import os, sys, time, getpass, random
from contextlib import nullcontext
from typing import Optional
from dataclasses import dataclass

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


@torch.inference_mode()
def evaluate(model, dataset, batch_size=15, max_batches=None):
    model.eval()
    if hasattr(model, 'configure_subnetwork'):
        model.configure_subnetwork('xl')  # always evaluate the full model
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, C, Y, S = batch
        logits, loss = model(X, C, Y, style=S if S.numel() else None)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss


########## ARGS, LOGGING, AND TRAIN LOOP ##########



if __name__ == '__main__':

    args = get_all_args()
    torch.manual_seed(args.seed)  # system inits
    torch.cuda.manual_seed_all(args.seed)

    wandb_init_args = {"project": args.wandb_project, "entity": args.wandb_entity, "config": args}
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
    print(f"Dataset determined that: {args.vocab_size=}, {args.block_size=}")

    model, optimizer, scheduler, step, best_loss, ema_state = get_checkpoint(args, sample_only=False)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None and ema_state is not None:
        ema.load_state_dict(ema_state)
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    wandb.watch(model, log="all", log_freq=args.log_every, log_graph=False)  # model saving stuff


    ########## ARGS, LOGGING, AND TRAIN LOOP ##########

    # training loop
    while True:
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
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step() ; scheduler.step()
        if ema is not None:
            ema.update(model)
        wandb.log({"train_loss_step": loss.item(), "step": step})
        t1 = time.time()

        # logging
        if step % args.print_every == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms | lr {scheduler.get_last_lr()[0]:.6f}")

        # evaluate the model (with EMA weights when enabled: those are what we ship)
        if step > 0 and step % args.log_every == 0:
            with (ema.swap(model) if ema is not None else nullcontext()):
                train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
                test_loss  = evaluate(model, test_dataset,  batch_size=100, max_batches=10)
            wandb.log({"train_loss": train_loss, "test_loss": test_loss, "step": step })
            print(f"step {step} train loss: {train_loss:.4f} test loss: {test_loss:.4f}")

            if best_loss is None or test_loss < best_loss:  # save the model to W&B if it has improved
                best_loss = test_loss
                print(f"Test loss {test_loss:.4f} is the best so far, saving checkpoint to {args.local_checkpoint_path}")
                save_checkpoint(model, args.local_checkpoint_path, optimizer, scheduler, step, best_loss, ema=ema)
                artifact = wandb.Artifact('best_checkpoint', type='model')
                artifact.add_file(args.local_checkpoint_path)
                wandb.log_artifact(artifact)

        # sample from the model (EMA weights when enabled)
        if step > 0 and step % args.log_every == 0:
            with (ema.swap(model) if ema is not None else nullcontext()):
                save_samples(model, test_dataset, num=6, do_sample=True)
                save_samples(model, test_dataset, num=6, do_sample=False)
                save_samples(model, train_dataset, num=3, do_sample=True)
                save_samples(model, train_dataset, num=3, do_sample=False)

        step += 1
        # termination conditions
        if args.max_steps >= 0 and step >= args.max_steps:
            break

    wandb.finish()

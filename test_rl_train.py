"""Focused tests for the rendering-aware RLOO training stage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch.utils.data import Dataset

import rl_train
from model import EMA, MatFormer, configure_optimizer, configure_scheduler
from sample import _structured_mask


class TinyStrokeDataset(Dataset):
    """Small valid BOS dataset used by the CPU checkpoint integration test."""

    def __init__(self) -> None:
        self.r_bins = np.array([0.0, 0.1])
        self.theta_bins = np.array([-1.0, 0.0, 1.0])
        self.cumulative_sizes = np.cumsum([0, len(self.r_bins), len(self.theta_bins)])
        self.PAD_TOKEN = 5
        self.END_TOKEN = 6
        self.WORD_TOKEN = 7
        self.BOS_TOKEN = 8
        self.use_bos = True
        self.char_PAD_TOKEN = 0
        self.counter = 0

        self._x = torch.tensor(
            [self.BOS_TOKEN, 2, 0, 3, 1, self.END_TOKEN, 5, 5, 5, 5, 5, 5],
            dtype=torch.long,
        )
        self._targets = torch.full_like(self._x, -1)
        self._targets[:5] = self._x[1:6]
        self._targets[5:9] = self.PAD_TOKEN
        self._context = torch.tensor([1, 2, 0, 0], dtype=torch.long)
        self._style = torch.tensor([2, 0, 3, 1, 5, 5], dtype=torch.long)

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        self.counter += 1
        return (
            self._x.clone(),
            self._context.clone(),
            self._targets.clone(),
            self._style.clone(),
        )

    def get_vocab_size(self) -> int:
        return 9

    def get_stroke_seq_length(self) -> int:
        return len(self._x)

    def get_text_seq_length(self) -> int:
        return len(self._context)

    def get_generation_prefix(self) -> torch.Tensor:
        return torch.tensor([self.BOS_TOKEN], dtype=torch.long)


def tiny_model_config() -> SimpleNamespace:
    return SimpleNamespace(
        n_layer=1,
        n_embd=8,
        n_embd_context=8,
        n_ctx_head=2,
        n_context_layer=0,
        dropout=0.0,
        vocab_size=9,
        block_size=12,
        context_block_size=4,
        context_vocab_size=4,
        context_pad_token=0,
        style_words=1,
        max_style_length=6,
        n_style_tokens=2,
        n_style_layer=1,
        use_bos=True,
        stroke_pad_token=5,
        style_lr_multiplier=1.0,
    )


class RLOOMathTests(unittest.TestCase):
    def test_leave_one_out_advantages_use_only_other_rollouts(self):
        rewards = torch.tensor([[1.0, 2.0, 4.0], [2.0, 2.0, 2.0]])

        advantages = rl_train.leave_one_out_advantages(rewards, normalize=False)

        expected = torch.tensor([[-2.0, -0.5, 2.5], [0.0, 0.0, 0.0]])
        torch.testing.assert_close(advantages, expected)
        torch.testing.assert_close(advantages.sum(dim=1), torch.zeros(2))

        normalized = rl_train.leave_one_out_advantages(rewards)
        self.assertAlmostEqual(normalized.std(unbiased=False).item(), 1.0, places=6)
        with self.assertRaisesRegex(ValueError, "at least two rollouts"):
            rl_train.leave_one_out_advantages(rewards[:, :1])

    def test_generated_action_mask_includes_first_end_and_excludes_tail(self):
        bos, end, pad = 8, 6, 5
        sequences = torch.tensor(
            [
                [bos, 2, 0, end, pad, pad],
                [bos, 2, 0, 3, 1, pad],
                [bos, 2, end, 0, 3, pad],
            ]
        )

        mask = rl_train.generated_action_mask(sequences, end, pad)

        expected = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, False],
                [True, True, False, False, False],
            ]
        )
        torch.testing.assert_close(mask, expected)

    def test_structured_replay_mask_matches_sampling_grammar_at_every_prefix(self):
        dataset = TinyStrokeDataset()
        sequence = torch.tensor(
            [[dataset.BOS_TOKEN, 2, 0, 3, 1, dataset.WORD_TOKEN, dataset.WORD_TOKEN]]
        )

        replay_masks = rl_train.structured_action_mask(sequence, dataset)

        for stop in range(1, sequence.size(1) + 1):
            sampling_mask = _structured_mask(sequence[:, :stop], dataset)
            torch.testing.assert_close(replay_masks[:, stop - 1], sampling_mask)

        theta = slice(2, 5)
        self.assertTrue(replay_masks[0, 0, theta].all())
        self.assertEqual(replay_masks[0, 0].sum().item(), 3)
        self.assertTrue(replay_masks[0, 5, dataset.WORD_TOKEN])
        self.assertEqual(replay_masks[0, 5].sum().item(), 1)
        self.assertFalse(replay_masks[0, 6, dataset.WORD_TOKEN])
        self.assertFalse(replay_masks[0, 6, dataset.END_TOKEN])
        self.assertTrue(replay_masks[0, 6, theta].all())

    def test_pilot_resume_keeps_the_same_cosine_schedule_horizon(self):
        parameter = torch.nn.Parameter(torch.zeros(()))
        schedule_args = SimpleNamespace(
            lr_schedule="cosine",
            warmup_steps=10,
            max_steps=1500,
            step_lr_every=100,
            lr_decay=0.333,
        )
        optimizer = torch.optim.AdamW([parameter], lr=1e-4)
        scheduler = configure_scheduler(optimizer, schedule_args)
        for _ in range(200):
            optimizer.step()
            scheduler.step()

        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()
        optimizer.step()
        scheduler.step()
        uninterrupted_lr = scheduler.get_last_lr()[0]

        resumed_parameter = torch.nn.Parameter(torch.zeros(()))
        resumed_optimizer = torch.optim.AdamW([resumed_parameter], lr=1e-4)
        resumed_scheduler = configure_scheduler(resumed_optimizer, schedule_args)
        resumed_optimizer.load_state_dict(optimizer_state)
        resumed_scheduler.load_state_dict(scheduler_state)
        resumed_optimizer.step()
        resumed_scheduler.step()

        self.assertAlmostEqual(
            resumed_scheduler.get_last_lr()[0], uninterrupted_lr, places=12
        )


class RLOOCheckpointIntegrationTests(unittest.TestCase):
    def test_one_cpu_update_saves_and_resumes_complete_training_state(self):
        dataset = TinyStrokeDataset()
        model = MatFormer(tiny_model_config())
        evaluation_rewards = iter([0.0, 0.25])

        def load_tiny_model(args):
            for key, value in vars(model.config).items():
                setattr(args, key, value)
            return model

        def score_rollouts(
            sequences,
            target_sequences,
            contexts,
            styles,
            dataset,
            *,
            rollouts_per_condition,
            **kwargs,
        ):
            del sequences, contexts, styles, dataset, kwargs
            row = torch.linspace(-0.2, 0.2, rollouts_per_condition)
            rewards = row.repeat(target_sequences.size(0), 1)
            return rl_train.RolloutScores(
                rewards=rewards,
                results=[],
                means={"total": float(rewards.mean())},
            )

        def evaluate(*args, **kwargs):
            del args, kwargs
            reward = next(evaluation_rewards)
            scores = rl_train.RolloutScores(
                rewards=torch.tensor([[reward]]), results=[], means={"total": reward}
            )
            batch = (
                dataset._x.unsqueeze(0),
                dataset._context.unsqueeze(0),
                dataset._targets.unsqueeze(0),
                dataset._style.unsqueeze(0),
            )
            return rl_train.EvalOutput(
                metrics={"eval_reward/mean": reward},
                batch=batch,
                sequences=dataset._x.unsqueeze(0),
                scores=scores,
            )

        fake_run = SimpleNamespace(summary={})
        with tempfile.TemporaryDirectory() as tmpdir:
            best_path = str(Path(tmpdir) / "best.pt")
            last_path = str(Path(tmpdir) / "last.pt")
            argv = [
                "--sft-checkpoint-path",
                str(Path(tmpdir) / "unused-sft.pt"),
                "--max-updates",
                "1",
                "--conditions-batch-size",
                "2",
                "--rollouts-per-condition",
                "2",
                "--rollout-length-margin",
                "0",
                "--rollout-max-tokens",
                "8",
                "--rl-warmup-updates",
                "0",
                "--ema-decay",
                "0",
                "--eval-every",
                "1",
                "--save-every",
                "1",
                "--print-every",
                "1",
                "--reward-workers",
                "0",
                "--num-workers",
                "0",
                "--device",
                "cpu",
                "--wandb-mode",
                "disabled",
                "--no-log-artifacts",
                "--no-log-media",
                "--best-checkpoint-path",
                best_path,
                "--last-checkpoint-path",
                last_path,
            ]

            with (
                mock.patch.object(rl_train, "load_sft_model", side_effect=load_tiny_model),
                mock.patch.object(
                    rl_train, "create_datasets", return_value=(dataset, dataset)
                ),
                mock.patch.object(
                    rl_train, "score_grouped_rollouts", side_effect=score_rollouts
                ),
                mock.patch.object(rl_train, "evaluate_policy", side_effect=evaluate),
                mock.patch.object(rl_train.wandb, "init", return_value=fake_run),
                mock.patch.object(rl_train.wandb, "log"),
                mock.patch.object(rl_train.wandb, "finish"),
            ):
                rl_train.main(argv)

            self.assertTrue(Path(best_path).is_file())
            self.assertTrue(Path(last_path).is_file())
            checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["step"], 1)
            self.assertEqual(checkpoint["best_loss"], 0.25)
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertIn("scheduler_state_dict", checkpoint)

            resumed_model = MatFormer(tiny_model_config())
            resume_args = SimpleNamespace(
                learning_rate=1e-4,
                style_lr_multiplier=1.0,
                weight_decay=1e-4,
                lr_schedule="cosine",
                warmup_steps=1,
                max_steps=1,
                step_lr_every=100,
                lr_decay=0.333,
            )
            optimizer = configure_optimizer(resumed_model, resume_args)
            scheduler = configure_scheduler(optimizer, resume_args)
            completed, best_reward = rl_train.resume_rl_state(
                last_path, resumed_model, optimizer, scheduler, ema=None
            )

            self.assertEqual(completed, 1)
            self.assertEqual(best_reward, 0.25)
            for name, value in checkpoint["model_state_dict"].items():
                torch.testing.assert_close(resumed_model.state_dict()[name], value)


if __name__ == "__main__":
    unittest.main()

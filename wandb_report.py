# Fetch and summarize a W&B training run for this project.
# Usage:
#   python wandb_report.py                          # defaults to run bxf7fmpq
#   python wandb_report.py --path ENTITY/PROJECT/runs/RUN_ID
# Requires `wandb login` (or WANDB_API_KEY in the environment).

import argparse

import numpy as np
import wandb


def main():
    p = argparse.ArgumentParser(description='Summarize a cursivetransformer W&B run')
    p.add_argument('--path', default='cursivetransformer-ng/cursivetransformer-ng/runs/bxf7fmpq',
                   help='ENTITY/PROJECT/runs/RUN_ID (the /runs/ part is optional)')
    p.add_argument('--plot', default='wandb_loss_curve.png', help='Where to save the loss plot ("" to skip)')
    args = p.parse_args()

    api = wandb.Api()
    run = api.run(args.path.replace('/runs/', '/').strip('/'))

    print(f"run: {run.entity}/{run.project}/{run.id}  name={run.name!r}  state={run.state}")
    interesting = ['max_steps', 'learning_rate', 'lr_schedule', 'batch_size', 'n_layer', 'n_embd',
                   'n_embd_context', 'n_context_layer', 'dropout', 'cond_drop_prob', 'subnetwork_mode',
                   'style_words', 'style_drop_prob', 'ema_decay', 'dataset_name', 'train_size']
    print("config:")
    for k in interesting:
        if k in run.config:
            print(f"  {k} = {run.config[k]}")

    hist = run.history(samples=100000, pandas=False)
    rows = hist.to_dict('records') if hasattr(hist, 'to_dict') else list(hist)
    columns = sorted({k for r in rows for k in r if not k.startswith('_')})
    print(f"\nhistory: {len(rows)} rows, columns: {columns}")

    def series(key):
        pts = [(r.get('step', r.get('_step')), r[key]) for r in rows
               if r.get(key) is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
        return [(s, v) for s, v in pts if isinstance(v, (int, float))]

    curves = {}
    for key in ('train_loss', 'test_loss', 'train_loss_step'):
        pts = series(key)
        curves[key] = pts
        if not pts or key == 'train_loss_step':
            continue
        steps, vals = zip(*pts)
        best_i = int(np.argmin(vals))
        print(f"{key}: last {vals[-1]:.4f} @ step {steps[-1]} | best {vals[best_i]:.4f} @ step {steps[best_i]} | n={len(vals)}")

    print("\ntrain/test loss by step:")
    print(f"  {'step':>8} {'train_loss':>11} {'test_loss':>10}")
    test_by_step = dict(curves['test_loss'])
    for s, v in curves['train_loss']:
        t = test_by_step.get(s)
        print(f"  {s:>8} {v:>11.4f} {t if t is None else f'{t:>10.4f}'}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
            for key, style in (('train_loss_step', ':'), ('train_loss', '-'), ('test_loss', '-')):
                pts = curves.get(key) or []
                if pts:
                    steps, vals = zip(*pts)
                    ax.plot(steps, vals, style, label=key, alpha=0.8)
            ax.set_xlabel('step'); ax.set_ylabel('loss'); ax.legend(); ax.grid(alpha=0.3)
            ax.set_title(f"{run.entity}/{run.project}/{run.id}")
            fig.tight_layout(); fig.savefig(args.plot)
            print(f"saved {args.plot}")
        except Exception as e:
            print(f"plot skipped: {type(e).__name__}: {e}")


if __name__ == '__main__':
    main()

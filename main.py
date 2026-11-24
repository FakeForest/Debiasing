import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data import GROUPS, SEED, load_adult
from metrics import evaluate, probabilities, score_predictions
from thresholds import fit_parity_thresholds
from train import train

CORE = {
    "baseline": {"debias": False},
    "parity": {"mode": "parity"},
    "eq_odds": {"mode": "eq_odds"},
}
VARIANTS = {
    "parity_hidden": {"mode": "parity", "adversary_hidden": 16},
    "eq_odds_hidden": {"mode": "eq_odds", "adversary_hidden": 16},
    "eq_odds_sqrt": {"mode": "eq_odds", "schedule": "sqrt"},
    "eq_opp": {"mode": "eq_opp"},
    "parity_grl": {"mode": "parity", "update": "grl"},
    "parity_representation": {"mode": "parity", "representation": True},
}
METRICS = ["accuracy", "DP gap", "TPR gap", "FPR gap"]


def split_tensors(data, split):
    return tuple(data[f"{key}_{split}"] for key in ("X", "y", "z"))


def save_context(data, output):
    rows = []
    for split in ("train", "val", "test"):
        _, y, z = split_tensors(data, split)
        for group, label in enumerate(data["groups"]):
            labels = y[z == group]
            rows.append({"split": split, "group": label, "n": len(labels),
                         "positives": int(labels.sum()), "base_rate": float(labels.mean())})
    pd.DataFrame(rows).to_csv(output / "group_counts.csv", index=False)


def save_proxies(data, output):
    x, _, z = split_tensors(data, "train")
    centered = x.cpu().numpy().astype(np.float64)
    centered -= centered.mean(0)
    group = z.numpy().astype(np.float64)
    group -= group.mean()
    divisor = np.linalg.norm(centered, axis=0) * np.linalg.norm(group)
    r = np.divide(group @ centered, divisor, out=np.zeros_like(divisor), where=divisor > 0)
    proxies = pd.DataFrame({"feature": data["feature_names"], "r_with_male": r, "abs_r": abs(r)})
    proxies.sort_values("abs_r", ascending=False).to_csv(output / "sex_correlations.csv", index=False)


def save_scores(models, data, output):
    x, y, z = split_tensors(data, "test")
    bins = np.linspace(0, 1, 31)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True, layout="constrained")
    rows = []
    for ax, name in zip(axes, ("baseline", "parity")):
        scores = probabilities(models[name], x)
        pd.DataFrame({"score": scores, "y": y.numpy(), "z": z.numpy()}).to_csv(
            output / f"scores_{name}.csv", index=False)
        for group, label in enumerate(data["groups"]):
            s = scores[z.numpy() == group]
            ax.hist(s, bins=bins, weights=np.full(len(s), 1 / len(s)), alpha=0.55, label=label)
            rows.append({"model": name, "group": label, "mean": s.mean(), "median": np.median(s),
                         "std": s.std(), "fraction_positive": (s >= 0.5).mean()})
        ax.axvline(0.5, color="#303030", linestyle="--", linewidth=1)
        ax.set(title=name.title(), xlabel="Predicted probability of income > $50K")
        ax.legend()
    axes[0].set_ylabel("Fraction of group per bin")
    fig.savefig(output / "scores.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output / "score_distribution.csv", index=False)


def save_threshold_evaluations(models, data, output):
    x, y, z = split_tensors(data, "val")
    tolerance = 0.005
    chosen = fit_parity_thresholds(probabilities(models["baseline"], x), y.numpy(), z.numpy(), tolerance)
    thresholds = {"tolerance": tolerance, **dict(zip(data["groups"], chosen))}
    (output / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    rows = []
    for split in ("val", "test"):
        x, y, z = split_tensors(data, split)
        for name in ("baseline_fixed_threshold", "baseline_group_thresholds", "adversarial_parity"):
            model = models["parity" if name == "adversarial_parity" else "baseline"]
            threshold = np.asarray(chosen)[z.numpy().astype(int)] if name == "baseline_group_thresholds" else 0.5
            rows.append({"split": split, "method": name,
                         **score_predictions(probabilities(model, x), y, z, threshold)})
    pd.DataFrame(rows).to_csv(output / "threshold_comparison.csv", index=False)


def save_plots(results, histories, output):
    ax = results.loc[list(CORE), METRICS].T.plot.bar(figsize=(9, 4), rot=0)
    ax.set(ylabel="Rate or absolute gap", title="Held-out test results")
    ax.figure.tight_layout()
    ax.figure.savefig(output / "comparison.png", dpi=160)
    plt.close(ax.figure)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for ax, metric in zip(axes.flat, METRICS):
        for name in CORE:
            ax.plot(histories[name]["epoch"], histories[name][metric], label=name)
        ax.set(title=f"Validation {metric}", xlabel="Epoch")
        ax.legend()
    fig.savefig(output / "curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for name in ("parity", "eq_odds", *[name for name in VARIANTS if name in histories]):
        ax.plot(histories[name]["epoch"], histories[name]["loss_A"], label=name)
    ax.set(title="Training adversary loss", xlabel="Epoch", ylabel="Binary cross-entropy (nats)")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(output / "adversary_loss.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Adversarial debiasing experiments on UCI Adult")
    parser.add_argument("--protected", choices=GROUPS, default="sex")
    parser.add_argument("--suite", action="store_true", help="include all six sex-based variants")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("--epochs and --threads must be positive")
    if args.suite and args.protected != "sex":
        parser.error("--suite applies to the sex exercises")
    torch.set_num_threads(args.threads)
    data = load_adult(seed=args.seed, protected=args.protected, data_dir=args.data_dir)
    print(f"Device: {data['device']} | {data['n_features']} features", flush=True)
    output = args.output_dir / args.protected
    output.mkdir(parents=True, exist_ok=True)
    save_context(data, output)
    settings = {**CORE, **VARIANTS} if args.suite else CORE
    models, histories, rows = {}, {}, []
    for name, options in settings.items():
        print(f"\n=== {args.protected}: {name} ===", flush=True)
        model, history = train(data, seed=args.seed, epochs=args.epochs, **options)
        run = output / name
        run.mkdir(exist_ok=True)
        history.to_csv(run / "history.csv", index=False)
        torch.save({"model": model.state_dict(), "feature_names": data["feature_names"],
                    "settings": options, "seed": args.seed, "epochs": args.epochs}, run / "model.pt")
        metrics = evaluate(model, *split_tensors(data, "test"))
        rows.append({"run": name, **metrics})
        histories[name] = history
        models[name] = model
        print(pd.Series(metrics)[METRICS].round(5).to_string(), flush=True)
    results = pd.DataFrame(rows).set_index("run")
    results.to_csv(output / "results.csv")
    save_plots(results, histories, output)
    if args.protected == "sex":
        save_proxies(data, output)
        save_scores(models, data, output)
        save_threshold_evaluations(models, data, output)
    print(f"\nSaved results in {output}", flush=True)


if __name__ == "__main__":
    main()

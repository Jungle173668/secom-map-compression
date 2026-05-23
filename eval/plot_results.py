"""
Plot cliff analysis results: Pareto frontier + per-category breakdown.

Usage:
    python eval/plot_results.py
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

METHODS = {
    "sliding_window": ("Sliding Window", "#4C72B0", "o", "--"),
    "summarization":  ("Summarization",  "#DD8452", "s", "--"),
    "secom_map":      ("SeCom-MAP",      "#55A868", "^", "-"),
    "secom_only":     ("SeCom-only (ablation)", "#C44E52", "v", ":"),
}

CATEGORIES = ["multi_hop", "temporal", "single_hop", "open_domain"]
CAT_LABELS  = ["Multi-hop", "Temporal", "Single-hop", "Open-domain"]


def load_all() -> dict:
    data = {}
    for key in METHODS:
        path = RESULTS_DIR / f"cliff_{key}.json"
        if not path.exists():
            print(f"[WARN] {path.name} not found, skipping")
            continue
        with open(path) as f:
            d = json.load(f)
        data[key] = d["points"]
    return data


# ── Figure 1: Pareto frontier ──────────────────────────────────────────────────

def plot_pareto(data: dict):
    fig, ax = plt.subplots(figsize=(8, 5))

    for key, (label, color, marker, linestyle) in METHODS.items():
        if key not in data:
            continue
        pts = data[key]
        xs = [p["token_reduction"] for p in pts]
        ys = [p["accuracy"] for p in pts]
        ax.plot(xs, ys, marker=marker, linestyle=linestyle, color=color,
                label=label, linewidth=1.8, markersize=7)
        # annotate param values
        for p in pts:
            ax.annotate(str(p["param"]),
                        (p["token_reduction"], p["accuracy"]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=6, color=color, alpha=0.7)

    ax.set_xlabel("Token Reduction (higher = more compressed)", fontsize=12)
    ax.set_ylabel("Accuracy (binary)", fontsize=12)
    ax.set_title("Pareto Frontier: Accuracy vs Token Reduction", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0.35, 0.95)
    ax.set_ylim(0.0, 0.55)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "pareto_frontier.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


# ── Figure 2: Per-category accuracy (secom_map vs summarization) ───────────────

def plot_by_category(data: dict):
    methods_to_plot = ["sliding_window", "summarization", "secom_map"]
    n_cats = len(CATEGORIES)
    fig, axes = plt.subplots(1, n_cats, figsize=(14, 4), sharey=True)

    for ci, (cat, cat_label) in enumerate(zip(CATEGORIES, CAT_LABELS)):
        ax = axes[ci]
        for key in methods_to_plot:
            if key not in data:
                continue
            label, color, marker, linestyle = METHODS[key]
            pts = data[key]
            xs = [p["token_reduction"] for p in pts]
            ys = [p["by_category"].get(cat, 0) for p in pts]
            ax.plot(xs, ys, marker=marker, linestyle=linestyle,
                    color=color, linewidth=1.6, markersize=6, label=label)
        ax.set_title(cat_label, fontsize=11)
        ax.set_xlabel("Token Reduction", fontsize=9)
        if ci == 0:
            ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_xlim(0.35, 0.95)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    handles = [mpatches.Patch(color=METHODS[k][1], label=METHODS[k][0])
               for k in methods_to_plot if k in data]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Per-Category Accuracy vs Token Reduction", fontsize=12, y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "by_category.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ── Figure 3: Summary table bar chart ─────────────────────────────────────────

def plot_summary_bar(data: dict):
    """Best accuracy for each method at ~50% and ~80% token reduction."""
    targets = [(0.80, "~80% reduction"), (0.50, "~50% reduction")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (target, title) in zip(axes, targets):
        accs, labels, colors = [], [], []
        for key, (label, color, _, _) in METHODS.items():
            if key not in data:
                continue
            pts = data[key]
            # find point closest to target reduction
            closest = min(pts, key=lambda p: abs(p["token_reduction"] - target))
            accs.append(closest["accuracy"])
            labels.append(label)
            colors.append(color)

        bars = ax.bar(labels, accs, color=colors, alpha=0.85, edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
        ax.set_title(f"Accuracy at {title}", fontsize=11)
        ax.set_ylim(0, 0.6)
        ax.set_ylabel("Accuracy")
        ax.tick_params(axis="x", labelrotation=15, labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "summary_bars.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    data = load_all()
    if not data:
        print("[ERROR] No result files found in results/")
        return

    print(f"Loaded methods: {list(data.keys())}")
    plot_pareto(data)
    plot_by_category(data)
    plot_summary_bar(data)
    print(f"\nAll figures saved to: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()

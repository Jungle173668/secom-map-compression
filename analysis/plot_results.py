"""
Generate tables and plots from results/*.json.

Usage:
    python analysis/plot_results.py

Outputs (saved to results/):
    - summary_table.csv       overall accuracy + token_reduction per method
    - category_table.csv      per-category breakdown
    - cliff_curve.png         token_reduction vs accuracy curves
    - ablation_table.csv      2x2 ablation (EP x QR)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found; skipping plots.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[WARN] pandas not found; printing to stdout only.")


METHOD_DISPLAY = {
    "sliding_window": "Sliding Window",
    "summarization": "Summarization",
    "secom_only": "SeCom-only",
    "secom_ep": "SeCom+EP",
    "secom_qr": "SeCom+QR",
    "secom_map": "SeCom-MAP (ours)",
}

ABLATION_METHODS = ["secom_only", "secom_ep", "secom_qr", "secom_map"]
CATEGORIES = ["single_hop", "multi_hop", "temporal", "open_domain"]


def load_results(methods=None) -> dict:
    if methods is None:
        methods = list(METHOD_DISPLAY.keys())
    data = {}
    for m in methods:
        path = RESULTS_DIR / f"{m}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data[m] = json.load(f)
    return data


def _mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def summary_stats(results: list) -> dict:
    return {
        "accuracy": _mean([r["correctness"] for r in results]),
        "token_reduction": _mean([r["token_reduction"] for r in results]),
        "n": len(results),
    }


def category_stats(results: list) -> dict:
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["correctness"])
    return {cat: _mean(scores) for cat, scores in by_cat.items()}


# ── Table 1: Summary ──────────────────────────────────────────────────────────

def build_summary_table(data: dict) -> list:
    rows = []
    for m, results in data.items():
        s = summary_stats(results)
        rows.append(
            {
                "Method": METHOD_DISPLAY.get(m, m),
                "Token Reduction": f"{s['token_reduction']:.1%}",
                "Overall Accuracy": f"{s['accuracy']:.1%}",
                "N": s["n"],
            }
        )
    return rows


# ── Table 2: Per-category ─────────────────────────────────────────────────────

def build_category_table(data: dict) -> list:
    rows = []
    for m, results in data.items():
        cat_acc = category_stats(results)
        row = {"Method": METHOD_DISPLAY.get(m, m)}
        for cat in CATEGORIES:
            val = cat_acc.get(cat, float("nan"))
            row[cat] = f"{val:.1%}" if val == val else "—"
        rows.append(row)
    return rows


# ── Table 3: Ablation ─────────────────────────────────────────────────────────

def build_ablation_table(data: dict) -> list:
    rows = []
    labels = {
        "secom_only": ("✗", "✗"),
        "secom_ep":   ("✓", "✗"),
        "secom_qr":   ("✗", "✓"),
        "secom_map":  ("✓", "✓"),
    }
    for m, (ep, qr) in labels.items():
        if m not in data:
            continue
        s = summary_stats(data[m])
        rows.append(
            {
                "Method": METHOD_DISPLAY.get(m, m),
                "Entity Protection": ep,
                "Query Rewriting": qr,
                "Accuracy": f"{s['accuracy']:.1%}",
                "Token Reduction": f"{s['token_reduction']:.1%}",
            }
        )
    return rows


# ── Plot: Pareto frontier (unified token_reduction x-axis) ───────────────────

def plot_cliff(cliff_path: Path):
    if not HAS_MPL:
        return
    if not cliff_path.exists():
        print(f"[WARN] {cliff_path} not found; skipping cliff plot.")
        return

    with open(cliff_path, encoding="utf-8") as f:
        cliff = json.load(f)

    colors  = {"sliding_window": "#e74c3c", "summarization": "#f39c12", "secom_map": "#2980b9"}
    markers = {"sliding_window": "o",       "summarization": "s",       "secom_map": "^"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for method_name, points in cliff.items():
        xs = [p[0] for p in points if p[1] is not None]
        ys = [p[1] for p in points if p[1] is not None]
        if not xs:
            continue
        label = METHOD_DISPLAY.get(method_name, method_name)
        ax.plot(xs, ys,
                marker=markers.get(method_name, "o"),
                label=label,
                color=colors.get(method_name, "gray"),
                linewidth=2, markersize=7)

    ax.set_xlabel("Token Reduction", fontsize=12)
    ax.set_ylabel("Answer Accuracy", fontsize=12)
    ax.set_title("Pareto Frontier: Token–Quality Trade-off", fontsize=13)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = RESULTS_DIR / "pareto_frontier.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── Plot: Internal parameter curves (one subplot per method) ──────────────────

INTERNAL_PARAM_LABELS = {
    "sliding_window": "N turns kept",
    "summarization":  "max_summary_tokens",
    "secom_map":      "token_budget",
}


def plot_internal_params(cliff_path: Path):
    """
    For each method, plot its own control parameter (x) vs accuracy (y).
    Shows each method's natural behaviour before the unified comparison.
    """
    if not HAS_MPL:
        return
    if not cliff_path.exists():
        print(f"[WARN] {cliff_path} not found; skipping internal param plot.")
        return

    with open(cliff_path, encoding="utf-8") as f:
        cliff = json.load(f)

    methods = [m for m in ["sliding_window", "summarization", "secom_map"] if m in cliff]
    if not methods:
        return

    colors  = {"sliding_window": "#e74c3c", "summarization": "#f39c12", "secom_map": "#2980b9"}
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]

    for ax, method_name in zip(axes, methods):
        points = [(p[2], p[1]) for p in cliff[method_name] if p[1] is not None]
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys,
                marker="o",
                color=colors.get(method_name, "gray"),
                linewidth=2, markersize=7)
        ax.set_xlabel(INTERNAL_PARAM_LABELS.get(method_name, "parameter"), fontsize=11)
        ax.set_ylabel("Answer Accuracy", fontsize=11)
        ax.set_title(METHOD_DISPLAY.get(method_name, method_name), fontsize=12)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(True, alpha=0.3)

    plt.suptitle("Per-Method: Control Parameter vs Accuracy", fontsize=13, y=1.02)
    plt.tight_layout()

    out = RESULTS_DIR / "internal_params.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Print helpers ─────────────────────────────────────────────────────────────

def _print_table(title: str, rows: list):
    if not rows:
        return
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    keys = list(rows[0].keys())
    col_w = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in keys}
    header = " | ".join(k.ljust(col_w[k]) for k in keys)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(str(row[k]).ljust(col_w[k]) for k in keys))


def _save_csv(rows: list, path: Path):
    if not rows:
        return
    if HAS_PANDAS:
        import pandas as pd
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    else:
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data = load_results()
    if not data:
        print(f"[ERROR] No result files found in {RESULTS_DIR}. Run: python eval/run_all.py")
        sys.exit(1)

    print(f"Loaded results for: {list(data.keys())}")
    RESULTS_DIR.mkdir(exist_ok=True)

    summary_rows = build_summary_table(data)
    _print_table("Table 1: Overall Results", summary_rows)
    _save_csv(summary_rows, RESULTS_DIR / "summary_table.csv")

    cat_rows = build_category_table(data)
    _print_table("Table 2: Per-Category Accuracy", cat_rows)
    _save_csv(cat_rows, RESULTS_DIR / "category_table.csv")

    ablation_data = {m: data[m] for m in ABLATION_METHODS if m in data}
    if ablation_data:
        abl_rows = build_ablation_table(ablation_data)
        _print_table("Table 3: Ablation (EP × QR)", abl_rows)
        _save_csv(abl_rows, RESULTS_DIR / "ablation_table.csv")

    cliff_path = RESULTS_DIR / "cliff_analysis.json"
    plot_cliff(cliff_path)
    plot_internal_params(cliff_path)

    # Find a failure case: secom_map wrong but reference available
    if "secom_map" in data:
        failures = [
            r for r in data["secom_map"]
            if r["correctness"] == 0 and r.get("category") in ("multi_hop", "temporal")
        ]
        if failures:
            ex = failures[0]
            print(f"\n{'='*60}")
            print(" Failure Case Example")
            print(f"{'='*60}")
            print(f"  ID:        {ex['id']}")
            print(f"  Category:  {ex['category']}")
            print(f"  Question:  {ex['question']}")
            print(f"  Reference: {ex['reference_answer']}")
            print(f"  Candidate: {ex['candidate_answer']}")
            print(f"  Rewritten query: {ex['extra'].get('rewritten_query', 'N/A')}")


if __name__ == "__main__":
    main()

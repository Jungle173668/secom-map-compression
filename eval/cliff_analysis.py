"""
Cliff point / Pareto frontier analysis.

Each method is swept over its own natural control parameter.
All results are reported with token_reduction, accuracy, and per-category breakdown.

Sliding Window : sweep N turns kept      → SW_N_TURNS
Summarization  : sweep total context budget → SUM_BUDGETS
SeCom-MAP      : sweep token_budget       → SECOM_BUDGETS
SeCom-only     : same budgets, no EP/QR   → SECOM_BUDGETS (ablation)

Optimisations:
  - Summarisation summary cached per conversation (compute once, reuse for all budgets)
  - Judge uses gpt-4o-mini (cheaper; binary 0/1 task doesn't need gpt-4o)
  - SeCom pipeline loaded from disk cache (no rebuild)

Usage:
    python eval/cliff_analysis.py
    python eval/cliff_analysis.py --limit 10
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLES_PATH = ROOT / "data" / "samples.json"
RESULTS_DIR = ROOT / "results"

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

from eval.judge import llm_judge
from utils.llm import llm_call
from utils.token_counter import count_tokens, count_turns_tokens

# ── Per-method sweep parameters ────────────────────────────────────────────────
SW_N_TURNS    = [50, 100, 150, 200, 250, 300]
SUM_BUDGETS   = [1500, 3000, 5000, 7000, 9000, 11000]
SECOM_BUDGETS = [2000, 3500, 5000, 7000, 9000, 11000]

ANSWER_PROMPT = """\
Answer the question based on the conversation history below.

{context}

Question: {question}

Answer:"""


def _ask_llm(context: str, question: str) -> str:
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    return llm_call(
        prompt,
        system="You are an assistant that answers questions based on conversation history.",
    )


# ── Per-method runners ─────────────────────────────────────────────────────────

def _run_sliding_window(sample: dict, n_turns: int) -> tuple[str, int]:
    from methods.sliding_window import sliding_window
    return sliding_window(sample["full_history"], n_turns)


# Summary cache: conv_id → summary text (reused across all budgets for same conv)
_sum_cache: dict = {}

def _run_summarization(sample: dict, budget: int) -> tuple[str, int]:
    from methods.summarization import _summarise, recursive_summarize
    from utils.token_counter import turns_to_text

    conv_id = sample["id"].split("_")[0]
    history = sample["full_history"]

    # Build/load summary for this conversation (same for all budgets)
    if conv_id not in _sum_cache:
        split = max(1, len(history) // 2)
        old_turns = history[:split]
        summary = _summarise(turns_to_text(old_turns))
        _sum_cache[conv_id] = summary

    summary = _sum_cache[conv_id]
    summary_tokens = count_tokens(summary)

    # Greedy fill remaining budget with verbatim recent turns
    candidate_recent = history[max(1, len(history) // 2):]
    remaining = max(0, budget - summary_tokens - 50)
    verbatim = []
    for turn in reversed(candidate_recent):
        from utils.token_counter import turns_to_text as t2t
        t = t2t([turn])
        t_tokens = count_tokens(t)
        if t_tokens <= remaining:
            verbatim.insert(0, t)
            remaining -= t_tokens
        else:
            break

    context_parts = [f"[Conversation Summary]\n{summary}"]
    if verbatim:
        context_parts.append("[Recent Turns]\n" + "\n".join(verbatim))
    context = "\n\n".join(context_parts)
    return context, count_tokens(context)


def _run_secom(sample: dict, token_budget: int, pipeline_cache: dict,
               use_ep: bool, use_qr: bool,
               use_dense: bool = False) -> tuple[str, int]:
    from methods.secom_map import SeComMAP
    conv_id = sample["id"].split("_")[0]
    suffix = ("_dense" if use_dense else "") + f"_ep{int(use_ep)}"
    cache_key = (conv_id, use_ep, use_dense)
    cache_path = RESULTS_DIR / "secom_cache" / f"{conv_id}{suffix}.json"

    if cache_key not in pipeline_cache:
        if cache_path.exists():
            p = SeComMAP.load_memory(str(cache_path), use_query_rewriting=use_qr,
                                     use_dense=use_dense)
            print(f"\n[cache] Loaded {cache_path.name}")
        else:
            p = SeComMAP(use_entity_protection=use_ep, use_query_rewriting=use_qr,
                         use_dense=use_dense)
            p.build_memory(sample["full_history"])
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            p.save_memory(str(cache_path))
            print(f"\n[cache] Saved {cache_path.name}")
        pipeline_cache[cache_key] = p

    pipeline = pipeline_cache[cache_key]
    answer, _, tokens_used = pipeline.answer(sample["question"], token_budget=token_budget)
    return answer, tokens_used


# ── Sweep helper ───────────────────────────────────────────────────────────────

def _sweep_method(method: str, param_values: list, samples: list,
                  secom_cache: dict) -> list:
    """
    Sweep one method over its parameter values.
    Returns list of dicts with overall + per-category accuracy/reduction.
    """
    points = []
    for param in param_values:
        label = f"{method}[{param}]"
        correct, reductions = 0, []
        cat_correct: dict = defaultdict(int)
        cat_total: dict = defaultdict(int)

        for sample in tqdm(samples, desc=label, leave=False):
            full_tokens = max(count_turns_tokens(sample["full_history"]), 1)
            cat = sample.get("category", "unknown")
            try:
                if method == "sliding_window":
                    context, tokens_used = _run_sliding_window(sample, param)
                    answer = _ask_llm(context, sample["question"])

                elif method == "summarization":
                    context, tokens_used = _run_summarization(sample, param)
                    answer = _ask_llm(context, sample["question"])

                elif method == "secom_map":
                    answer, tokens_used = _run_secom(
                        sample, param, secom_cache, use_ep=True, use_qr=True)

                elif method == "secom_only":
                    answer, tokens_used = _run_secom(
                        sample, param, secom_cache, use_ep=False, use_qr=False)

                elif method == "secom_dense":
                    answer, tokens_used = _run_secom(
                        sample, param, secom_cache, use_ep=True, use_qr=True,
                        use_dense=True)

                score = llm_judge(
                    sample["question"], sample["reference_answer"], answer,
                    model=JUDGE_MODEL,
                )
                correct += score
                cat_correct[cat] += score
                reductions.append(1.0 - min(tokens_used, full_tokens) / full_tokens)

            except Exception as e:
                print(f"\n[ERROR] {label} on {sample['id']}: {e}")
                reductions.append(0.0)
            finally:
                cat_total[cat] += 1

        n = len(samples)
        avg_reduction = sum(reductions) / n if n else 0.0
        accuracy = correct / n if n else 0.0
        by_category = {
            c: round(cat_correct[c] / cat_total[c], 4)
            for c in sorted(cat_total)
        }
        points.append({
            "param": param,
            "token_reduction": round(avg_reduction, 4),
            "accuracy": round(accuracy, 4),
            "by_category": by_category,
        })
        cat_str = "  ".join(f"{c}={v:.2f}" for c, v in by_category.items())
        print(f"  {label}: acc={accuracy:.3f}  reduction={avg_reduction:.3f}  [{cat_str}]")

    return points




ALL_METHODS = ["sliding_window", "summarization", "secom_map", "secom_only", "secom_dense"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS,
        help="Which methods to run (default: all)",
    )
    args = parser.parse_args()

    if not SAMPLES_PATH.exists():
        print(f"[ERROR] {SAMPLES_PATH} not found. Run: python data/load_locomo.py")
        sys.exit(1)

    with open(SAMPLES_PATH, encoding="utf-8") as f:
        samples = json.load(f)

    if args.limit:
        samples = samples[: args.limit]
        print(f"[INFO] Limited to {len(samples)} samples")

    print(f"Running cliff analysis on {len(samples)} samples")
    print(f"  Methods:    {args.methods}")
    print(f"  Judge model: {JUDGE_MODEL}")

    RESULTS_DIR.mkdir(exist_ok=True)

    METHOD_PARAMS = {
        "sliding_window": SW_N_TURNS,
        "summarization":  SUM_BUDGETS,
        "secom_map":      SECOM_BUDGETS,
        "secom_only":     SECOM_BUDGETS,
        "secom_dense":    SECOM_BUDGETS,
    }
    METHOD_TITLES = {
        "sliding_window": "Sliding Window (sweep N turns)",
        "summarization":  "Summarization (sweep context budget)",
        "secom_map":      "SeCom-MAP  (sweep token_budget)",
        "secom_only":     "SeCom-only (ablation: no EP, no QR)",
        "secom_dense":    "SeCom-Dense (dense retrieval, EP+QR)",
    }

    secom_cache: dict = {}
    results = {}

    for method in args.methods:
        print(f"\n=== {METHOD_TITLES[method]} ===")
        results[method] = _sweep_method(
            method, METHOD_PARAMS[method], samples, secom_cache
        )
        out_path = RESULTS_DIR / f"cliff_{method}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"method": method, "points": results[method]},
                      f, ensure_ascii=False, indent=2)
        print(f"  → Saved {out_path.name}")

    # Combined file (only if all methods ran)
    if set(args.methods) == set(ALL_METHODS):
        out_path = RESULTS_DIR / "cliff_analysis.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nCombined results saved to: {out_path}")


if __name__ == "__main__":
    main()

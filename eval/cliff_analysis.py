"""
Cliff point / Pareto frontier analysis.

Each method is swept over its own natural control parameter.
All results are reported as (token_reduction, accuracy, param_value) so they
can be plotted on the same (token_reduction, accuracy) axes.

Sliding Window : sweep N turns kept      → SW_N_TURNS
Summarization  : sweep max_summary_tokens → SUM_MAX_TOKENS
SeCom-MAP      : sweep token_budget       → SECOM_BUDGETS

Usage:
    python eval/cliff_analysis.py
    python eval/cliff_analysis.py --limit 10
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLES_PATH = ROOT / "data" / "samples.json"
RESULTS_DIR = ROOT / "results"

from eval.judge import llm_judge
from utils.llm import llm_call
from utils.token_counter import count_tokens, count_turns_tokens

# ── Per-method sweep parameters ────────────────────────────────────────────────
# Designed so all three methods cover roughly the same 40–90% reduction range,
# giving overlapping x-axis points on the Pareto frontier plot.
# Assumes avg conversation ~17,800 tokens, ~35 tokens/turn.

SW_N_TURNS = [50, 100, 150, 200, 250, 300]
# reductions ≈ 90%, 80%, 73%, 61%, 51%, 41%

SUM_MAX_TOKENS = [1500, 3000, 5000, 7000, 9000, 11000]
# summary + ~350 verbatim → reductions ≈ 90%, 82%, 71%, 61%, 51%, 40%

SECOM_BUDGETS = [2000, 3500, 5000, 7000, 9000, 11000]
# reductions ≈ 89%, 80%, 72%, 61%, 49%, 38%

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


def _run_summarization(sample: dict, max_summary_tokens: int) -> tuple[str, int]:
    from methods.summarization import recursive_summarize
    return recursive_summarize(sample["full_history"], max_summary_tokens)


def _run_secom_map(sample: dict, token_budget: int, pipeline_cache: dict) -> tuple[str, int]:
    key = sample["id"]
    if key not in pipeline_cache:
        from methods.secom_map import SeComMAP
        p = SeComMAP(use_entity_protection=True, use_query_rewriting=True)
        p.build_memory(sample["full_history"])
        pipeline_cache[key] = p
    pipeline = pipeline_cache[key]
    answer, _, tokens_used = pipeline.answer(sample["question"], token_budget=token_budget)
    return answer, tokens_used  # answer already generated


# ── Sweep helpers ──────────────────────────────────────────────────────────────

def _sweep_method(method: str, param_values: list, samples: list,
                  secom_cache: dict) -> list:
    """
    Sweep one method over its parameter values.
    Returns list of (avg_token_reduction, accuracy, param_value).
    """
    points = []
    for param in param_values:
        label = f"{method}[{param}]"
        correct, reductions = 0, []

        for sample in tqdm(samples, desc=label, leave=False):
            full_tokens = max(count_turns_tokens(sample["full_history"]), 1)
            try:
                if method == "sliding_window":
                    context, tokens_used = _run_sliding_window(sample, param)
                    answer = _ask_llm(context, sample["question"])

                elif method == "summarization":
                    context, tokens_used = _run_summarization(sample, param)
                    answer = _ask_llm(context, sample["question"])

                elif method == "secom_map":
                    answer, tokens_used = _run_secom_map(sample, param, secom_cache)

                score = llm_judge(sample["question"], sample["reference_answer"], answer)
                correct += score
                reductions.append(1.0 - min(tokens_used, full_tokens) / full_tokens)

            except Exception as e:
                print(f"\n[ERROR] {label} on {sample['id']}: {e}")
                reductions.append(0.0)

        n = len(samples)
        avg_reduction = sum(reductions) / n if n else 0.0
        accuracy = correct / n if n else 0.0
        points.append((round(avg_reduction, 4), round(accuracy, 4), param))
        print(f"  {label}: accuracy={accuracy:.3f}  token_reduction={avg_reduction:.3f}")

    return points


# ── Main sweep ─────────────────────────────────────────────────────────────────

def sweep(samples: list) -> dict:
    """
    Run all three methods over their respective parameter ranges.
    Returns {method: [(token_reduction, accuracy, param), ...]}
    """
    secom_cache: dict = {}  # SeComMAP pipelines built once, reused across budgets

    results = {}

    print("\n=== Sliding Window (sweep N turns) ===")
    results["sliding_window"] = _sweep_method(
        "sliding_window", SW_N_TURNS, samples, secom_cache
    )

    print("\n=== Summarization (sweep max_summary_tokens) ===")
    results["summarization"] = _sweep_method(
        "summarization", SUM_MAX_TOKENS, samples, secom_cache
    )

    print("\n=== SeCom-MAP (sweep token_budget) ===")
    results["secom_map"] = _sweep_method(
        "secom_map", SECOM_BUDGETS, samples, secom_cache
    )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
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
    print(f"  Sliding Window: {len(SW_N_TURNS)} N values")
    print(f"  Summarization:  {len(SUM_MAX_TOKENS)} max_summary_tokens values")
    print(f"  SeCom-MAP:      {len(SECOM_BUDGETS)} token_budget values")

    RESULTS_DIR.mkdir(exist_ok=True)
    results = sweep(samples)

    out_path = RESULTS_DIR / "cliff_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()

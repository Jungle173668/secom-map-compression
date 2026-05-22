"""
Run all methods on data/samples.json and save results to results/*.json.

Usage:
    python eval/run_all.py
    python eval/run_all.py --methods sliding_window summarization secom_map
    python eval/run_all.py --limit 5   # quick test on first 5 samples
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.judge import llm_judge
from utils.llm import llm_call
from utils.token_counter import count_tokens

SAMPLES_PATH = ROOT / "data" / "samples.json"
RESULTS_DIR = ROOT / "results"

ANSWER_PROMPT = """\
Answer the question based on the conversation history below. \
If the relevant information is not present, say "Cannot answer based on conversation history."

{context}

Question: {question}

Answer:"""


def _ask_llm(context: str, question: str) -> str:
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    return llm_call(prompt, system="You are an assistant that answers questions based on conversation history.")


# ── Method runners ────────────────────────────────────────────────────────────

def run_sliding_window(sample: dict, n_turns: int = 150) -> dict:
    from methods.sliding_window import run as sw_run
    context, tokens_used, full_tokens = sw_run(sample, n_turns)
    answer = _ask_llm(context, sample["question"])
    return {
        "answer": answer,
        "tokens_used": tokens_used,
        "full_tokens": full_tokens,
        "extra": {"n_turns": n_turns},
    }


def run_summarization(sample: dict, max_summary_tokens: int = 5000, summarize_every: int = 10) -> dict:
    from methods.summarization import run as sum_run
    context, tokens_used, full_tokens = sum_run(sample, max_summary_tokens, summarize_every)
    answer = _ask_llm(context, sample["question"])
    return {
        "answer": answer,
        "tokens_used": tokens_used,
        "full_tokens": full_tokens,
        "extra": {"max_summary_tokens": max_summary_tokens},
    }


_secom_cache: dict = {}  # (conv_id, use_ep) → SeComMAP pipeline


def run_secom_variant(
    sample: dict,
    use_ep: bool,
    use_qr: bool,
    compress_rate: float = 0.65,
) -> dict:
    from methods.secom_map import SeComMAP
    from utils.token_counter import count_turns_tokens

    conv_id = sample["id"].split("_")[0]
    cache_key = (conv_id, use_ep)

    if cache_key not in _secom_cache:
        pipeline = SeComMAP(
            use_entity_protection=use_ep,
            use_query_rewriting=use_qr,
            compress_rate=compress_rate,
        )
        pipeline.build_memory(sample["full_history"])
        _secom_cache[cache_key] = pipeline
    else:
        pipeline = _secom_cache[cache_key]

    pipeline.use_qr = use_qr  # answer() reads this; build_memory() already done
    full_tokens = count_turns_tokens(sample["full_history"])
    answer, rewritten, tokens_used = pipeline.answer(sample["question"])

    return {
        "answer": answer,
        "tokens_used": tokens_used,
        "full_tokens": full_tokens,
        "extra": {"rewritten_query": rewritten, "compress_rate": compress_rate},
    }


METHOD_REGISTRY = {
    "sliding_window": lambda s: run_sliding_window(s),   # n_turns=150 (~75% reduction)
    "summarization": lambda s: run_summarization(s),     # max_summary_tokens=5000 (~70% reduction)
    "secom_only": lambda s: run_secom_variant(s, use_ep=False, use_qr=False),
    "secom_ep": lambda s: run_secom_variant(s, use_ep=True, use_qr=False),
    "secom_qr": lambda s: run_secom_variant(s, use_ep=False, use_qr=True),
    "secom_map": lambda s: run_secom_variant(s, use_ep=True, use_qr=True),
}


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate_method(method_name: str, samples: list) -> list:
    runner = METHOD_REGISTRY[method_name]
    results = []
    for sample in tqdm(samples, desc=method_name):
        try:
            out = runner(sample)
            correctness = llm_judge(
                sample["question"],
                sample["reference_answer"],
                out["answer"],
            )
        except Exception as e:
            print(f"\n[ERROR] {method_name} on {sample['id']}: {e}")
            traceback.print_exc()
            out = {"answer": "", "tokens_used": 0, "full_tokens": 0, "extra": {}}
            correctness = 0

        full_tokens = out["full_tokens"] if out["full_tokens"] > 0 else 1
        token_reduction = 1.0 - out["tokens_used"] / full_tokens

        results.append(
            {
                "id": sample["id"],
                "category": sample["category"],
                "question": sample["question"],
                "reference_answer": sample["reference_answer"],
                "candidate_answer": out["answer"],
                "correctness": correctness,
                "tokens_used": out["tokens_used"],
                "full_tokens": out["full_tokens"],
                "token_reduction": round(token_reduction, 4),
                "extra": out.get("extra", {}),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_REGISTRY.keys()),
        choices=list(METHOD_REGISTRY.keys()),
        help="Which methods to run (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N samples (for quick testing)",
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

    print(f"Loaded {len(samples)} samples")
    RESULTS_DIR.mkdir(exist_ok=True)

    for method_name in args.methods:
        print(f"\n{'='*50}")
        print(f"Running: {method_name}")
        print(f"{'='*50}")
        results = evaluate_method(method_name, samples)

        out_path = RESULTS_DIR / f"{method_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        accuracy = sum(r["correctness"] for r in results) / len(results) if results else 0
        avg_reduction = sum(r["token_reduction"] for r in results) / len(results) if results else 0
        print(f"  accuracy={accuracy:.3f}  avg_token_reduction={avg_reduction:.3f}")
        print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()

"""
Load the LoCoMo dataset and produce data/samples.json.

Download LoCoMo first:
    git clone https://github.com/snap-research/locomo.git

Then run:
    python data/load_locomo.py

Actual data format (locomo10.json):
  - list of 10 conversation objects
  - each has: qa, conversation, sample_id, ...
  - conversation is a dict with keys: speaker_a, speaker_b,
      session_1, session_1_date_time, session_2, ...
  - each session is a list of turns: {speaker, dia_id, text}
  - qa[i]: {question, answer, evidence: ["D1:3"], category: int}

Category mapping (LoCoMo paper):
  1 -> single_hop
  2 -> temporal
  3 -> open_domain
  4 -> multi_hop
  5 -> adversarial
"""

import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
LOCOMO_PATH = os.getenv("LOCOMO_PATH", "locomo/data/locomo10.json")
OUTPUT_PATH = ROOT / "data" / "samples.json"

CATEGORY_MAP = {
    1: "single_hop",
    2: "temporal",
    3: "open_domain",
    4: "multi_hop",
    5: "adversarial",
}

SAMPLE_TARGETS = {
    "multi_hop":   15,
    "temporal":    15,
    "single_hop":  10,
    "open_domain":  5,
}
EXCLUDE_CATEGORIES = {"adversarial"}
RANDOM_SEED = 42


# ── Conversation parsing ──────────────────────────────────────────────────────

def _flatten_history(conv_dict: dict) -> list[dict]:
    """
    Flatten all sessions in order into a list of {speaker, text, dia_id} dicts.
    Session keys are session_1, session_2, ... (skips *_date_time keys).
    """
    all_turns = []
    session_num = 1
    while True:
        key = f"session_{session_num}"
        if key not in conv_dict:
            break
        session = conv_dict[key]
        if isinstance(session, list):
            for turn in session:
                if isinstance(turn, dict) and "text" in turn:
                    all_turns.append({
                        "speaker": turn.get("speaker", "Speaker"),
                        "text": turn["text"],
                        "dia_id": turn.get("dia_id", ""),
                    })
        session_num += 1
    return all_turns


def _parse_evidence(evidence: list) -> list[str]:
    """Return evidence dia_ids as-is (e.g. ["D1:3", "D3:7"])."""
    if not evidence:
        return []
    return [str(e) for e in evidence]


# ── Main loader ───────────────────────────────────────────────────────────────

def load_raw(path: str) -> list:
    full_path = ROOT / path
    if not full_path.exists():
        print(f"\n[ERROR] Dataset not found at: {full_path}")
        print("\nPlease download the LoCoMo dataset first:")
        print("    cd", ROOT)
        print("    git clone https://github.com/snap-research/locomo.git")
        print("\nThen re-run:  python data/load_locomo.py")
        sys.exit(1)

    with open(full_path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return raw
    elif isinstance(raw, dict):
        return list(raw.values())
    else:
        raise ValueError(f"Unexpected top-level type in {path}: {type(raw)}")


def build_samples(conversations: list) -> list:
    buckets: dict[str, list] = {k: [] for k in SAMPLE_TARGETS}

    for conv_idx, conv in enumerate(conversations):
        qa_list = conv.get("qa", [])
        conv_dict = conv.get("conversation", {})
        all_turns = _flatten_history(conv_dict)

        for qa_idx, qa in enumerate(qa_list):
            cat_int = qa.get("category")
            category = CATEGORY_MAP.get(cat_int, "unknown")

            if category in EXCLUDE_CATEGORIES:
                continue
            if category not in buckets:
                continue

            question = str(qa.get("question", "")).strip()
            answer = str(qa.get("answer", "")).strip()
            if not question or not answer:
                continue

            evidence_ids = _parse_evidence(qa.get("evidence", []))

            sample = {
                "id": f"conv{conv_idx}_q{qa_idx}",
                "question": question,
                "reference_answer": answer,
                "category": category,
                "full_history": all_turns,
                "evidence_turn_ids": evidence_ids,
            }
            buckets[category].append(sample)

    rng = random.Random(RANDOM_SEED)
    selected = []
    for cat, target_n in SAMPLE_TARGETS.items():
        pool = buckets[cat]
        n = min(target_n, len(pool))
        selected.extend(rng.sample(pool, n))
        print(f"  {cat}: {n}/{len(pool)} sampled (target {target_n})")

    rng.shuffle(selected)
    return selected


def main():
    print(f"Loading LoCoMo from: {LOCOMO_PATH}")
    conversations = load_raw(LOCOMO_PATH)
    print(f"Found {len(conversations)} conversations")

    print("\nBuilding samples...")
    samples = build_samples(conversations)
    print(f"\nTotal samples: {len(samples)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {OUTPUT_PATH}")

    from collections import Counter
    dist = Counter(s["category"] for s in samples)
    print("\nCategory distribution:")
    for cat, cnt in sorted(dist.items()):
        print(f"  {cat}: {cnt}")


if __name__ == "__main__":
    main()

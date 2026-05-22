"""LLM-as-Judge: binary correctness scoring (0 or 1)."""

from utils.llm import llm_call

JUDGE_PROMPT = """\
You are a strict answer evaluator.

Question: {question}
Reference answer: {reference}
Candidate answer: {candidate}

Does the candidate answer correctly answer the question?

Criteria:
- Core facts (names, dates, numbers, events) must match the reference answer
- Exact wording is not required
- If the candidate says "cannot answer" but the reference has a clear answer, score 0

Reply with only 0 (incorrect) or 1 (correct), no explanation:"""


def llm_judge(question: str, reference_answer: str, candidate_answer: str) -> int:
    """Return 0 (wrong) or 1 (correct)."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference=reference_answer,
        candidate=candidate_answer,
    )
    try:
        result = llm_call(
            prompt,
            system="You are an answer evaluation assistant. Output only 0 or 1.",
        ).strip()
        # Extract first digit
        for ch in result:
            if ch in ("0", "1"):
                return int(ch)
        return 0
    except Exception:
        return 0

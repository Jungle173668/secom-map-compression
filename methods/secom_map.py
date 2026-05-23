"""
SeCom-MAP: segment-based memory with entity-aware compression and query rewriting.

Handles all four ablation variants via flags:
  - use_entity_protection  (EP)
  - use_query_rewriting    (QR)
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from methods.context_map import ContextMap
from utils.llm import llm_call
from utils.token_counter import count_tokens, turns_to_text, count_turns_tokens

load_dotenv()

LLMLINGUA_MODEL = os.getenv("LLMLINGUA_MODEL", "openai-community/gpt2")

SEGMENT_PROMPT = """\
Below is a multi-turn conversation. Split it into topically coherent segments.
Each segment should cover a single topic; different segments should have a clear topic shift.

Conversation (each line formatted as "index|speaker: text"):
{numbered_turns}

Return a JSON array of segments, each with start and end (inclusive, 0-based turn indices).
Example: [{{"start": 0, "end": 4}}, {{"start": 5, "end": 11}}]

Return only the JSON array, nothing else:"""

ANSWER_PROMPT = """\
Answer the question based on the conversation history below. \
If the relevant information is not present, say "Cannot answer based on conversation history."

{context}

Question: {question}

Answer:"""


# ── LLMLingua wrapper ─────────────────────────────────────────────────────────

_compressor = None
_compressor_loaded = False


def _get_compressor():
    global _compressor, _compressor_loaded
    if _compressor_loaded:
        return _compressor
    _compressor_loaded = True
    try:
        from llmlingua import PromptCompressor
        print(f"[LLMLingua] Loading model: {LLMLINGUA_MODEL} (first run downloads ~500MB)")
        _compressor = PromptCompressor(
            model_name=LLMLINGUA_MODEL,
            use_llmlingua2=True,
            device_map="cpu",
        )
        print("[LLMLingua] Model loaded.")
    except ImportError:
        print("[LLMLingua] Package not installed. Run: pip install llmlingua")
    except Exception as e:
        print(f"[LLMLingua] Failed to load model: {e}")
    return _compressor


def compress_text(
    text: str,
    rate: float = 0.65,
    force_tokens: Optional[list] = None,
) -> str:
    """
    Compress text with LLMLingua-2 (microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank).
    LLMLingua-2 uses token-level binary classification for compression.
    Falls back to line-based subsampling if llmlingua unavailable.
    """
    compressor = _get_compressor()
    if compressor is None:
        return _fallback_compress(text, rate)

    try:
        result = compressor.compress_prompt(
            context=[text],
            rate=rate,
            force_tokens=force_tokens or [],
        )
        return result.get("compressed_prompt", text)
    except Exception as e:
        print(f"[LLMLingua] Compression error: {e}. Using fallback.")
        return _fallback_compress(text, rate)


def _fallback_compress(text: str, rate: float) -> str:
    """Simple line-level sub-sampling as a no-model fallback."""
    lines = [l for l in text.split("\n") if l.strip()]
    keep = max(1, int(len(lines) * (1 - rate)))
    step = max(1, len(lines) // keep)
    return "\n".join(lines[i] for i in range(0, len(lines), step))


# ── BM25 retriever ────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(self):
        self.segments: list[str] = []
        self._bm25 = None

    def add(self, segments: list[str]) -> None:
        self.segments.extend(segments)
        self._rebuild()

    def _rebuild(self) -> None:
        from rank_bm25 import BM25Okapi
        corpus = [_tokenize(s) for s in self.segments]
        self._bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        if not self.segments or self._bm25 is None:
            return self.segments[:top_k]
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.segments[i] for i in ranked[:top_k]]


# ── Dense retriever ───────────────────────────────────────────────────────────

class DenseIndex:
    """Semantic retrieval via sentence-transformers + cosine similarity."""

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        self.segments: list[str] = []
        self.embeddings = None   # np.ndarray (n, dim)
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    def add(self, segments: list[str]) -> None:
        import numpy as np
        embs = self._get_model().encode(segments, convert_to_numpy=True,
                                        show_progress_bar=False)
        self.segments.extend(segments)
        self.embeddings = (embs if self.embeddings is None
                           else np.vstack([self.embeddings, embs]))

    def restore(self, segments: list[str], embeddings_list: list) -> None:
        import numpy as np
        self.segments = segments
        self.embeddings = np.array(embeddings_list, dtype="float32")

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        import numpy as np
        if not self.segments or self.embeddings is None:
            return self.segments[:top_k]
        q = self._get_model().encode([query], convert_to_numpy=True)
        # Cosine similarity
        seg_norm = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        q_norm = np.linalg.norm(q) + 1e-8
        scores = ((self.embeddings / seg_norm) @ (q / q_norm).T).flatten()
        ranked = np.argsort(scores)[::-1]
        return [self.segments[i] for i in ranked[:top_k]]


# ── Session grouper ───────────────────────────────────────────────────────────

def _group_by_session(turns: list) -> list[list]:
    """Group turns by dia_id prefix (D1, D2, ...). Preserves chronological order."""
    buckets: dict[str, list] = {}
    for turn in turns:
        sid = turn.get("dia_id", "D0").split(":")[0]
        buckets.setdefault(sid, []).append(turn)
    sorted_keys = sorted(buckets.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    return [buckets[k] for k in sorted_keys]


# ── Topic segmenter (kept for reference / cliff_analysis) ─────────────────────

def segment_history(turns: list) -> list[list]:
    """
    Use LLM to split turns into topically coherent segments.
    Returns a list of segments, each segment being a list of turn dicts.
    """
    if not turns:
        return []

    if len(turns) <= 4:
        return [turns]

    numbered = "\n".join(
        f"{i}|{t.get('speaker','S')}: {t.get('text','')}"
        for i, t in enumerate(turns)
    )
    prompt = SEGMENT_PROMPT.format(numbered_turns=numbered)

    try:
        raw = llm_call(prompt, system="You are a dialogue structure analyst. Output JSON only.", json_mode=True)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"```[^\n]*\n?", "", raw).strip()
        boundaries = json.loads(raw)
    except Exception:
        # Fallback: every 5 turns = one segment
        boundaries = [
            {"start": i, "end": min(i + 4, len(turns) - 1)}
            for i in range(0, len(turns), 5)
        ]

    segments = []
    for b in boundaries:
        if not isinstance(b, dict):
            segments = []
            break
        start = int(b.get("start", 0))
        end = int(b.get("end", len(turns) - 1))
        seg = turns[start: end + 1]
        if seg:
            segments.append(seg)

    return segments if segments else [turns]


# ── SeComMAP ──────────────────────────────────────────────────────────────────

class SeComMAP:
    """
    SeCom-MAP pipeline.

    use_entity_protection: protect high-priority entities during LLMLingua compression
    use_query_rewriting:   rewrite query using context map before BM25 retrieval
    """

    def __init__(
        self,
        use_entity_protection: bool = True,
        use_query_rewriting: bool = True,
        compress_rate: float = 0.65,
        top_k: int = 5,
        map_token_budget: int = 300,
        use_dense: bool = False,
    ):
        self.use_ep = use_entity_protection
        self.use_qr = use_query_rewriting
        self.compress_rate = compress_rate
        self.top_k = top_k
        self.use_dense = use_dense
        self.context_map = ContextMap(token_budget=map_token_budget)
        self.index = DenseIndex() if use_dense else BM25Index()
        self._total_compressed_tokens = 0

    def build_memory(self, history: list) -> None:
        """Session-incremental build: update context map per session, then compress."""
        sessions = _group_by_session(history)
        for session in sessions:
            self.context_map.update(session)  # incremental: LLM sees ~20 turns at a time
            seg_text = turns_to_text(session)

            if self.use_ep:
                priority_ents = self.context_map.get_high_priority_entities(top_k=20)
                has_important = any(e in seg_text for e in priority_ents)
                rate = 0.85 if has_important else self.compress_rate
                compressed = compress_text(seg_text, rate=rate, force_tokens=priority_ents)
            else:
                compressed = compress_text(seg_text, rate=self.compress_rate)

            self.index.add([compressed])
            self._total_compressed_tokens += count_tokens(compressed)

    def save_memory(self, path: str) -> None:
        """Persist memory to JSON so build_memory can be skipped on future runs."""
        data = {
            "version": 2,
            "use_ep": self.use_ep,
            "use_dense": self.use_dense,
            "compress_rate": self.compress_rate,
            "context_map": {
                "entities": self.context_map.entities,
                "decisions": self.context_map.decisions,
                "timeline": self.context_map.timeline,
                "session_counter": self.context_map._session_counter,
                "token_budget": self.context_map.token_budget,
            },
            "segments": self.index.segments,
        }
        if self.use_dense and self.index.embeddings is not None:
            data["embeddings"] = self.index.embeddings.tolist()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_memory(cls, path: str, use_query_rewriting: bool = True,
                    use_dense: bool = False) -> "SeComMAP":
        """Load persisted memory. Dense index restored from saved embeddings (no re-encode)."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        use_dense = data.get("use_dense", use_dense)
        pipeline = cls(
            use_entity_protection=data["use_ep"],
            use_query_rewriting=use_query_rewriting,
            compress_rate=data["compress_rate"],
            use_dense=use_dense,
        )
        cm = data["context_map"]
        pipeline.context_map.entities = cm["entities"]
        pipeline.context_map.decisions = cm["decisions"]
        pipeline.context_map.timeline = cm["timeline"]
        pipeline.context_map._session_counter = cm["session_counter"]
        pipeline.context_map.token_budget = cm["token_budget"]
        if data["segments"]:
            if use_dense and "embeddings" in data:
                pipeline.index.restore(data["segments"], data["embeddings"])
            else:
                pipeline.index.add(data["segments"])
        return pipeline

    def answer(self, question: str, token_budget: int = None) -> tuple[str, str, int]:
        """
        Return (llm_answer, rewritten_query, tokens_in_prompt).

        If token_budget is set, greedily add ranked segments until the budget
        is reached (budget covers the retrieved history portion only).
        """
        if self.use_qr:
            rewritten = self.context_map.rewrite_query(question)
        else:
            rewritten = question

        if token_budget is not None:
            # Reserve space for context map and prompt overhead (~100 tokens)
            map_str = self.context_map.to_prompt_string()
            map_tokens = count_tokens(map_str) + 100
            remaining = max(0, token_budget - map_tokens)
            # Retrieve all segments ranked by relevance, then greedily fill budget
            all_ranked = self.index.retrieve(rewritten, top_k=len(self.index.segments))
            retrieved = []
            for seg in all_ranked:
                seg_tokens = count_tokens(seg)
                if seg_tokens <= remaining:
                    retrieved.append(seg)
                    remaining -= seg_tokens
                if remaining <= 0:
                    break
        else:
            retrieved = self.index.retrieve(rewritten, top_k=self.top_k)

        context_parts = []
        if self.context_map.to_prompt_string():
            context_parts.append(f"[Context Map]\n{self.context_map.to_prompt_string()}")
        if retrieved:
            context_parts.append("[Retrieved History]\n" + "\n---\n".join(retrieved))
        context = "\n\n".join(context_parts)

        prompt = ANSWER_PROMPT.format(context=context, question=question)
        tokens_in_prompt = count_tokens(prompt)

        answer = llm_call(prompt, system="You are an assistant that answers questions based on conversation history.")
        return answer, rewritten, tokens_in_prompt


def run(
    sample: dict,
    use_entity_protection: bool = True,
    use_query_rewriting: bool = True,
    compress_rate: float = 0.65,
) -> tuple[str, int, int, str]:
    """
    Run SeCom-MAP on a single sample.

    Returns (answer, tokens_used, full_tokens, rewritten_query).
    """
    full_tokens = count_turns_tokens(sample["full_history"])

    pipeline = SeComMAP(
        use_entity_protection=use_entity_protection,
        use_query_rewriting=use_query_rewriting,
        compress_rate=compress_rate,
    )
    pipeline.build_memory(sample["full_history"])
    answer, rewritten, tokens_used = pipeline.answer(sample["question"])

    return answer, tokens_used, full_tokens, rewritten

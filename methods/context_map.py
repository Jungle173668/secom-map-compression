"""Context Map: a fixed-size, incrementally maintained structured record of the conversation."""

import json
import math
from utils.llm import llm_call
from utils.token_counter import count_tokens, turns_to_text

EXTRACT_PROMPT = """\
Extract structured information from the following conversation.

Conversation:
{turns}

Return JSON with the following fields:
- entities: list of named entities, each with name (string) and type (person/place/date/org/number/other)
- decisions: list of confirmed facts or decisions (list of strings)
- timeline: list of time-stamped events, each with time (string) and event (string)

Example format:
{{"entities": [{{"name": "Alex", "type": "person"}}], "decisions": ["Alex decided to pursue a PhD in Beijing"], "timeline": [{{"time": "March 2022", "event": "Alex received the acceptance letter"}}]}}

Return only JSON, nothing else:"""

REWRITE_PROMPT = """\
You are a dialogue understanding assistant. Given a context map and a question, \
rewrite the question to be more explicit by resolving all pronouns (he/she/it/they) \
and implicit references, replacing them with specific entity names.

Context map:
{map_str}

Original question: {query}

Rewritten question (return as-is if already explicit; do not add information not present in the original):"""


class ContextMap:
    def __init__(self, token_budget: int = 300):
        self.token_budget = token_budget
        self.entities: dict[str, dict] = {}  # name -> {type, count, last_session}
        self.decisions: list[str] = []
        self.timeline: list[dict] = []
        self._session_counter = 0

    def update(self, turns: list) -> None:
        """Extract entities/decisions/timeline from turns and merge into the map."""
        if not turns:
            return

        text = turns_to_text(turns)
        prompt = EXTRACT_PROMPT.format(turns=text)

        try:
            raw = llm_call(prompt, system="You are an information extraction assistant. Output JSON only.", json_mode=True)
            data = json.loads(raw)
        except (json.JSONDecodeError, Exception):
            return

        self._session_counter += 1

        for ent in data.get("entities", []):
            name = ent.get("name", "").strip()
            if not name:
                continue
            if name in self.entities:
                self.entities[name]["count"] += 1
                self.entities[name]["last_session"] = self._session_counter
            else:
                self.entities[name] = {
                    "type": ent.get("type", "other"),
                    "count": 1,
                    "last_session": self._session_counter,
                }

        for dec in data.get("decisions", []):
            if dec and dec not in self.decisions:
                self.decisions.append(dec)

        for ev in data.get("timeline", []):
            if ev and ev not in self.timeline:
                self.timeline.append(ev)

        self._evict_if_needed()

    def _importance(self, name: str) -> float:
        """Importance = count × recency_weight (exponential decay by session age)."""
        ent = self.entities[name]
        age = self._session_counter - ent["last_session"]
        recency = math.exp(-0.1 * age)
        return ent["count"] * recency

    def _evict_if_needed(self) -> None:
        """Remove low-importance entities until the map fits in token_budget."""
        while count_tokens(self.to_prompt_string()) > self.token_budget:
            if not self.entities:
                break
            worst = min(self.entities, key=self._importance)
            del self.entities[worst]

            # Also trim oldest decisions/timeline entries if still too big
            if self.decisions:
                self.decisions.pop(0)
            if self.timeline:
                self.timeline.pop(0)

    def get_high_priority_entities(self, top_k: int = 20) -> list[str]:
        """Return entity names sorted by importance (highest first)."""
        sorted_ents = sorted(self.entities, key=self._importance, reverse=True)
        return sorted_ents[:top_k]

    def rewrite_query(self, query: str) -> str:
        """Use the map to resolve pronouns and ellipsis in the query."""
        if not self.entities and not self.decisions:
            return query
        map_str = self.to_prompt_string()
        prompt = REWRITE_PROMPT.format(map_str=map_str, query=query)
        try:
            return llm_call(prompt, system="You are a dialogue understanding assistant.")
        except Exception:
            return query

    def to_prompt_string(self) -> str:
        """Serialise the map into a compact text for prompt injection."""
        parts = []
        if self.entities:
            ent_lines = []
            for name, info in self.entities.items():
                ent_lines.append(f"{name}({info['type']})")
            parts.append("Entities: " + ", ".join(ent_lines))
        if self.decisions:
            parts.append("Confirmed facts: " + "; ".join(self.decisions[-5:]))
        if self.timeline:
            tl_lines = [f"{ev.get('time','')}: {ev.get('event','')}" for ev in self.timeline[-5:]]
            parts.append("Timeline: " + "; ".join(tl_lines))
        return "\n".join(parts)

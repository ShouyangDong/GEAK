import json
import os
import random
import re
from typing import List, Dict, Optional, Any, Iterable


def _tokenize_code(code: str) -> List[str]:
    # simple tokenizer: split on non-word chars, filter short tokens
    toks = re.split(r"\W+", code)
    return [t.lower() for t in toks if len(t) > 1]


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / (len(union) + 1e-12)


class ProgramDatabase:
    """
    Simple in-memory program DB with category support and few-shot retrieval.

    Each program entry is a dict:
      {
        "id": str,
        "program": str,
        "metrics": dict,
        "category": str,    # e.g., "softmax", "matmul", "conv2d"
        "meta": dict,       # optional metadata
    }
    """

    def __init__(self, programs: Optional[List[Dict[str, Any]]] = None):
        self.programs: List[Dict[str, Any]] = programs or []

    def add_program(
        self,
        program: str,
        metrics: Dict[str, Any],
        category: str,
        meta: Optional[Dict[str, Any]] = None,
        pid: Optional[str] = None,
    ):
        entry = {
            "id": pid or f"prog_{len(self.programs)}",
            "program": program,
            "metrics": metrics or {},
            "category": category or "unknown",
            "meta": meta or {},
        }
        self.programs.append(entry)
        return entry

    def load_from_jsonl(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    self.programs.append(json.loads(line))
                except Exception:
                    continue

    def save_to_jsonl(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for entry in self.programs:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _score(self, metrics: Dict[str, Any]) -> float:
        # composite score: prefer speedup then efficiency; fallback to 0
        speedup = float(metrics.get("speedup", 0) or 0)
        eff = float(metrics.get("efficiency", 0) or 0)
        # weight speedup higher; user can override by providing 'score'
        if "score" in metrics:
            return float(metrics["score"])
        return speedup * 10.0 + eff

    def _filter_by_category(self, category: Optional[str]) -> List[Dict[str, Any]]:
        if category is None:
            return list(self.programs)
        return [p for p in self.programs if p.get("category") == category]

    def select_parents(
        self,
        elite_count: int = 2,
        diverse_count: int = 3,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Select parents: top `elite_count` by score plus `diverse_count` chosen to maximize
        token-space diversity relative to elites (simple greedy heuristic).
        """
        candidates = self._filter_by_category(category)
        if not candidates:
            return []

        # rank by score
        ranked = sorted(
            candidates, key=lambda x: self._score(x.get("metrics", {})), reverse=True
        )
        elites = ranked[:elite_count]

        # build diversity set
        remaining = [r for r in ranked if r not in elites]
        examples = elites.copy()
        # tokens of selected set
        selected_tokens = []
        for e in elites:
            selected_tokens.extend(_tokenize_code(e["program"]))

        # greedy pick to increase diversity
        for _ in range(diverse_count):
            best = None
            best_score = None
            for cand in remaining:
                sim = _jaccard(selected_tokens, _tokenize_code(cand["program"]))
                # want lowest similarity
                if best is None or sim < best_score:
                    best = cand
                    best_score = sim
            if best is None:
                break
            examples.append(best)
            remaining.remove(best)
            selected_tokens.extend(_tokenize_code(best["program"]))
        return examples

    def get_few_shot_examples(
        self, category: Optional[str], k: int = 3, strategy: str = "elite"
    ) -> List[Dict[str, Any]]:
        """
        Return up to k few-shot examples for a category.
        strategy: 'elite' (top by score), 'diverse' (diverse set), 'similar' (code-similarity)
        """
        cand = self._filter_by_category(category)
        if not cand:
            return []

        if strategy == "elite":
            return sorted(
                cand, key=lambda x: self._score(x.get("metrics", {})), reverse=True
            )[:k]
        if strategy == "diverse":
            # use select_parents logic to get a diverse set
            return self.select_parents(
                elite_count=1, diverse_count=k - 1, category=category
            )[:k]
        # fallback to similarity-based top-k on category
        return cand[:k]

    def find_similar_by_code(
        self, new_code: str, k: int = 5, category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        tokens = _tokenize_code(new_code)
        cand = self._filter_by_category(category)
        scored = []
        for entry in cand:
            s = _jaccard(tokens, _tokenize_code(entry["program"]))
            scored.append((s, entry))
        # highest similarity first
        scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
        return [e for _, e in scored_sorted[:k]]

    def update(self, evaluated_variants, replace_if_better: bool = True) -> None:
        """
        Update the database with evaluated variants.
        - evaluated_variants: iterable of dicts or a single dict. Each item expected to contain:
            { "id"(optional), "program": str, "metrics": dict, "category"(optional), "meta"(optional) }
        - replace_if_better: if True, only replace existing entry when new score is higher.
        """
        if evaluated_variants is None:
            return
        # allow single dict
        if isinstance(evaluated_variants, dict):
            items = [evaluated_variants]
        else:
            try:
                items = list(evaluated_variants)
            except Exception:
                return

        for item in items:
            if not item:
                continue
            # normalize fields
            program = item.get("program") if isinstance(item, dict) else None
            metrics = item.get("metrics", {}) if isinstance(item, dict) else {}
            category = item.get("category", None) if isinstance(item, dict) else None
            meta = item.get("meta", {}) if isinstance(item, dict) else {}
            pid = item.get("id") if isinstance(item, dict) else None

            if not program:
                # if item is just a string, treat it as program text
                if isinstance(item, str):
                    program = item
                else:
                    continue

            # try find existing by id first, then by exact program match
            existing = None
            if pid is not None:
                for e in self.programs:
                    if e.get("id") == pid:
                        existing = e
                        break
            if existing is None:
                for e in self.programs:
                    if e.get("program") == program:
                        existing = e
                        break

            if existing is None:
                # add as new entry
                self.add_program(program=program, metrics=metrics, category=category or "unknown", meta=meta, pid=pid)
            else:
                # decide whether to replace/merge
                if not replace_if_better:
                    # merge fields conservatively
                    existing["metrics"].update(metrics or {})
                    existing["meta"].update(meta or {})
                    if category:
                        existing["category"] = category
                else:
                    old_score = self._score(existing.get("metrics", {}))
                    new_score = self._score(metrics or {})
                    if new_score >= old_score:
                        # replace fields
                        existing["program"] = program
                        existing["metrics"] = metrics or {}
                        existing["meta"] = meta or {}
                        if category:
                            existing["category"] = category
                        if pid:
                            existing["id"] = pid
                    else:
                        # keep existing, but update meta/aux fields
                        existing["meta"].update(meta or {})

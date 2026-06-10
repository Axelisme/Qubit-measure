"""MemoryStore — the file-backed lab notebook over ``agent_memory/<namespace>/``.

Two entry kinds, two trees (see the package docstring). Each entry is a markdown
file with a YAML frontmatter block; the agent never reads the files directly, it
calls the MCP tools which call these methods.

Lifecycle differs by kind:
  - records are episodic and append-mostly; ``record`` auto-suffixes on a same-day
    same-experiment collision rather than overwriting history.
  - solutions are semantic and self-improving; ``add_solution`` fails fast if the
    symptom slug already exists (refine via ``update_solution``), and confidence is
    promoted to ``confirmed`` once two records cite it.

Expected failures raise ``RuntimeError`` with an actionable message; the MCP stdio
loop turns that into a clean tool error.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml

_FM_DELIM = "---"
_VALID_CONFIDENCE = ("provisional", "confirmed")


def _slug(text: str) -> str:
    """A filesystem-safe slug. Keeps unicode word chars (so Chinese survives),
    folds whitespace/underscores and ``/`` to ``-``, drops other punctuation."""
    s = text.strip().lower()
    s = re.sub(r"[^\w\s/-]", "", s)
    s = s.replace("/", "-")
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "untitled"


def _parse(text: str) -> Tuple[Dict[str, Any], str]:
    """Split a frontmatter markdown file into (frontmatter mapping, body)."""
    lines = text.splitlines()
    if lines and lines[0].strip() == _FM_DELIM:
        for i in range(1, len(lines)):
            if lines[i].strip() == _FM_DELIM:
                data = yaml.safe_load("\n".join(lines[1:i])) or {}
                if not isinstance(data, dict):
                    raise RuntimeError("frontmatter is not a mapping")
                body = "\n".join(lines[i + 1 :]).lstrip("\n")
                return data, body
    return {}, text


def _dump(frontmatter: Dict[str, Any], body: str) -> str:
    fm = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    return f"{_FM_DELIM}\n{fm}\n{_FM_DELIM}\n\n{body.strip()}\n"


class MemoryStore:
    """File-backed store for one namespace under a memory root directory."""

    def __init__(self, root: Path, namespace: str) -> None:
        self._ns_dir = Path(root) / namespace
        self._records_dir = self._ns_dir / "records"
        self._solutions_dir = self._ns_dir / "solutions"

    # -- path / id helpers -------------------------------------------------

    def _safe_component(self, value: str, what: str) -> str:
        if not value or "/" in value or "\\" in value or value in (".", ".."):
            raise RuntimeError(f"invalid {what}: {value!r}")
        return value

    def _path_for_id(self, entry_id: str) -> Path:
        if not entry_id or entry_id.startswith("/") or ".." in entry_id.split("/"):
            raise RuntimeError(f"invalid entry id: {entry_id!r}")
        path = (self._ns_dir / (entry_id + ".md")).resolve()
        root = self._ns_dir.resolve()
        if root not in path.parents:
            raise RuntimeError(f"invalid entry id: {entry_id!r}")
        return path

    def _id_for_path(self, path: Path) -> str:
        rel = path.resolve().relative_to(self._ns_dir.resolve())
        return rel.with_suffix("").as_posix()

    def _exp_dir(self, exp_type: Optional[str]) -> str:
        if not exp_type or exp_type == "general":
            return "_general"
        segs = [_slug(s) for s in exp_type.split("/") if s.strip()]
        return "/".join(segs) or "_general"

    def _unique_id(self, rel_dir: str, name: str) -> str:
        candidate = f"{rel_dir}/{name}"
        if not self._path_for_id(candidate).exists():
            return candidate
        i = 2
        while self._path_for_id(f"{rel_dir}/{name}-{i}").exists():
            i += 1
        return f"{rel_dir}/{name}-{i}"

    # -- scanning ----------------------------------------------------------

    def _iter_entries(self) -> Iterator[Dict[str, Any]]:
        """Yield every parseable entry. A single malformed file is skipped (a
        corrupt hand-edit must not break a whole query); ``get`` of that specific
        id still fails fast."""
        for directory in (self._records_dir, self._solutions_dir):
            if not directory.exists():
                continue
            for f in sorted(directory.rglob("*.md")):
                try:
                    fm, body = _parse(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                entry = dict(fm)
                entry["id"] = self._id_for_path(f)
                entry["body"] = body
                yield entry

    # -- reads -------------------------------------------------------------

    def recall(
        self,
        chip: Optional[str] = None,
        qub: Optional[str] = None,
        exp_type: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Open the notebook to a qubit/experiment: recent records for this
        chip/qub plus solutions for this exp_type."""
        records: List[Dict[str, Any]] = []
        solutions: List[Dict[str, Any]] = []
        for e in self._iter_entries():
            kind = e.get("type")
            if kind == "record":
                if chip and e.get("chip") != chip:
                    continue
                if qub and e.get("qub") != qub:
                    continue
                if exp_type and exp_type not in (e.get("exp_type") or []):
                    continue
                records.append(e)
            elif kind == "solution":
                if exp_type and e.get("exp_type") != exp_type:
                    continue
                solutions.append(e)
        records.sort(key=lambda e: str(e.get("date", "")), reverse=True)
        solutions.sort(
            key=lambda e: (
                e.get("confidence") != "confirmed",
                str(e.get("symptom", "")),
            )
        )
        return {"records": records[:limit], "solutions": solutions[:limit]}

    def search(
        self,
        query: str,
        kind: Optional[str] = None,
        exp_type: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Keyword search over symptom + body (and exp_type/outcome), with optional
        kind / exp_type / category filters. Ranked by term-hit count."""
        terms = [t for t in re.split(r"\s+", query.strip().lower()) if t]
        results: List[Dict[str, Any]] = []
        for e in self._iter_entries():
            if kind and e.get("type") != kind:
                continue
            if exp_type and not _exp_matches(e.get("exp_type"), exp_type):
                continue
            if category and e.get("category") != category:
                continue
            blob = _searchable_text(e)
            score = sum(blob.count(t) for t in terms) if terms else 0
            if terms and score == 0:
                continue
            hit = dict(e)
            hit["score"] = score
            results.append(hit)
        results.sort(key=lambda e: e["score"], reverse=True)
        return {"results": results[:limit]}

    def get(self, entry_id: str) -> Dict[str, Any]:
        path = self._path_for_id(entry_id)
        if not path.exists():
            raise RuntimeError(f"no entry: {entry_id!r}")
        fm, body = _parse(path.read_text(encoding="utf-8"))
        out = dict(fm)
        out["id"] = entry_id
        out["body"] = body
        return out

    # -- writes ------------------------------------------------------------

    def record(
        self,
        chip: str,
        qub: str,
        date: str,
        exp_type: List[str],
        outcome: str,
        body: str,
        data_ref: Optional[str] = None,
        solutions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Append an episodic measurement record. Auto-suffixes on a same-day,
        same-experiment collision (history is never overwritten)."""
        if not isinstance(exp_type, list) or not exp_type:
            raise RuntimeError("exp_type must be a non-empty list of experiment types")
        chip = self._safe_component(chip, "chip")
        qub = self._safe_component(qub, "qub")
        date = self._safe_component(date, "date")
        fm: Dict[str, Any] = {
            "type": "record",
            "chip": chip,
            "qub": qub,
            "date": date,
            "exp_type": list(exp_type),
            "outcome": outcome,
        }
        if data_ref:
            fm["data_ref"] = data_ref
        if solutions:
            fm["solutions"] = list(solutions)
        rel_dir = f"records/{chip}/{qub}"
        name = f"{date}-{_slug('-'.join(exp_type))}"
        entry_id = self._unique_id(rel_dir, name)
        path = self._path_for_id(entry_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_dump(fm, body), encoding="utf-8")
        return {"id": entry_id, "created": True}

    def add_solution(
        self,
        exp_type: str,
        symptom: str,
        category: str,
        body: str,
        seen_in: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Add a context-free solution (starts ``provisional``). Fails fast if the
        symptom slug already exists under this exp_type — refine via
        ``update_solution`` rather than duplicating."""
        rel_dir = f"solutions/{self._exp_dir(exp_type)}"
        entry_id = f"{rel_dir}/{_slug(symptom)}"
        path = self._path_for_id(entry_id)
        if path.exists():
            raise RuntimeError(
                f"solution already exists at {entry_id!r}; refine it with "
                "memory_update_solution instead of adding a duplicate"
            )
        fm: Dict[str, Any] = {
            "type": "solution",
            "exp_type": exp_type or "general",
            "symptom": symptom,
            "category": category,
            "confidence": "provisional",
            "seen_in": list(seen_in) if seen_in else [],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_dump(fm, body), encoding="utf-8")
        return {"id": entry_id, "confidence": "provisional"}

    def update_solution(
        self,
        entry_id: str,
        body: Optional[str] = None,
        confidence: Optional[str] = None,
        add_seen_in: Optional[List[str]] = None,
        symptom: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Refine an existing solution: replace body/symptom, extend ``seen_in``
        (deduped), and/or set confidence. With no explicit confidence, reaching two
        cited records auto-promotes to ``confirmed``."""
        if confidence is not None and confidence not in _VALID_CONFIDENCE:
            raise RuntimeError(
                f"confidence must be one of {_VALID_CONFIDENCE}, got {confidence!r}"
            )
        path = self._path_for_id(entry_id)
        if not path.exists():
            raise RuntimeError(f"no solution: {entry_id!r}")
        fm, cur_body = _parse(path.read_text(encoding="utf-8"))
        if fm.get("type") != "solution":
            raise RuntimeError(f"{entry_id!r} is not a solution")
        if symptom is not None:
            fm["symptom"] = symptom
        if add_seen_in:
            seen = list(fm.get("seen_in") or [])
            for ref in add_seen_in:
                if ref not in seen:
                    seen.append(ref)
            fm["seen_in"] = seen
        if confidence is not None:
            fm["confidence"] = confidence
        elif len(fm.get("seen_in") or []) >= 2:
            fm["confidence"] = "confirmed"
        new_body = body if body is not None else cur_body
        path.write_text(_dump(fm, new_body), encoding="utf-8")
        return {
            "id": entry_id,
            "confidence": fm.get("confidence"),
            "seen_in_count": len(fm.get("seen_in") or []),
        }

    def delete(self, entry_id: str) -> Dict[str, Any]:
        path = self._path_for_id(entry_id)
        if not path.exists():
            raise RuntimeError(f"no entry: {entry_id!r}")
        path.unlink()
        return {"id": entry_id, "deleted": True}


def _exp_matches(stored: Any, wanted: str) -> bool:
    """A record stores exp_type as a list, a solution as a string."""
    if isinstance(stored, list):
        return wanted in stored
    return stored == wanted


def _searchable_text(entry: Dict[str, Any]) -> str:
    parts = [str(entry.get(k, "")) for k in ("symptom", "body", "outcome")]
    exp = entry.get("exp_type")
    parts.extend(exp if isinstance(exp, list) else [str(exp or "")])
    return " ".join(parts).lower()

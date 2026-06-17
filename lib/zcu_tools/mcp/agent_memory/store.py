"""MemoryStore — the file-backed lab notebook over ``agent_memory/<namespace>/``.

Three human-readable trees (see the package docstring), each a markdown file with
a YAML frontmatter block; the agent never reads the files directly, it calls the
MCP tools which call these methods. Files are hand-editable: the server parses the
same markdown a human would write.

  - ``records/<chip>/<qub>/<date>-<slug>/record.md`` — one measurement per *folder*
    (so figures can sit beside the note). Records are episodic and IMMUTABLE: a
    same-day same-experiment collision auto-suffixes the folder rather than
    overwriting history, and ``delete`` refuses to touch a record.
  - ``troubleshooting/<exp_slug>/<symptom_slug>.md`` — context-free problem -> fix
    (``type: solution``), self-improving: confidence promotes to ``confirmed`` once
    two records cite it.
  - ``checklists/<exp_slug>.md`` — one acceptance checklist per exp_type, items kept
    as a markdown bullet list in the BODY (hand-edit friendly), replaced wholesale.

Entry ids are namespace-relative paths without ``.md``. Path resolution dispatches
on the id PREFIX: a ``records/`` id is a folder holding ``record.md``; a
``troubleshooting/`` or ``checklists/`` id is a single ``<id>.md`` file.

Expected failures raise ``RuntimeError`` with an actionable message; the MCP stdio
loop turns that into a clean tool error.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml

_FM_DELIM = "---"
_VALID_CONFIDENCE = ("provisional", "confirmed")
_VALID_DECISION = ("accept", "reject")
_RECORD_FILENAME = "record.md"


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


def _items_to_body(items: List[str]) -> str:
    """Render checklist items as a markdown bullet list (the hand-editable form).
    Multi-line items would silently truncate on round-trip, so they are rejected fast."""
    for item in items:
        if "\n" in item:
            raise RuntimeError(
                f"checklist item must be single-line, got: {item!r}"
            )
    return "\n".join(f"- {item.strip()}" for item in items if item.strip())


def _body_to_items(body: str) -> List[str]:
    """Parse a markdown bullet list back into items. Accepts ``-``, ``*`` and ``+``
    bullets (so a human's hand-edit is read the same way the tool wrote it); any
    non-bullet line is ignored."""
    items: List[str] = []
    for line in body.splitlines():
        m = re.match(r"\s*[-*+]\s+(.*\S)\s*$", line)
        if m:
            items.append(m.group(1))
    return items


class MemoryStore:
    """File-backed store for one namespace under a memory root directory."""

    def __init__(self, root: Path, namespace: str) -> None:
        self._ns_dir = Path(root) / namespace
        self._records_dir = self._ns_dir / "records"
        self._troubleshooting_dir = self._ns_dir / "troubleshooting"
        self._checklists_dir = self._ns_dir / "checklists"

    # -- path / id helpers -------------------------------------------------

    def _safe_component(self, value: str, what: str) -> str:
        if not value or "/" in value or "\\" in value or value in (".", ".."):
            raise RuntimeError(f"invalid {what}: {value!r}")
        return value

    def _is_record_id(self, entry_id: str) -> bool:
        return entry_id == "records" or entry_id.startswith("records/")

    def _path_for_id(self, entry_id: str) -> Path:
        """Resolve an entry id to its on-disk file, dispatching on the id prefix:
        a ``records/`` id is a folder holding ``record.md``; everything else is a
        single ``<id>.md``. Rejects traversal / absolute ids."""
        if not entry_id or entry_id.startswith("/") or ".." in entry_id.split("/"):
            raise RuntimeError(f"invalid entry id: {entry_id!r}")
        rel = entry_id + "/" + _RECORD_FILENAME if self._is_record_id(entry_id) else entry_id + ".md"
        path = (self._ns_dir / rel).resolve()
        root = self._ns_dir.resolve()
        if root not in path.parents:
            raise RuntimeError(f"invalid entry id: {entry_id!r}")
        return path

    def _id_for_path(self, path: Path) -> str:
        """Invert ``_path_for_id``: only a file named ``record.md`` that lives under
        the ``records/`` tree maps back to its folder id; every other ``*.md`` (including
        a troubleshooting file whose symptom slug happens to be ``record``) maps to its
        stem id.  This is symmetric with ``_is_record_id`` which dispatches on the id
        prefix, not on the bare filename."""
        rel = path.resolve().relative_to(self._ns_dir.resolve())
        if rel.parts and rel.parts[0] == "records" and rel.name == _RECORD_FILENAME:
            return rel.parent.as_posix()
        return rel.with_suffix("").as_posix()

    def _exp_dir(self, exp_type: Optional[str]) -> str:
        """A nested exp_type sub-path (``reset/bath`` -> ``reset/bath``), used by
        troubleshooting so an exp family is a browsable subtree."""
        if not exp_type or exp_type == "general":
            return "_general"
        segs = [_slug(s) for s in exp_type.split("/") if s.strip()]
        return "/".join(segs) or "_general"

    def _checklist_slug(self, exp_type: Optional[str]) -> str:
        """A FLAT one-segment slug for the checklist filename (``reset/bath`` ->
        ``reset-bath``), so ``checklists/`` is one hand-browsable file per exp_type
        rather than a nested tree."""
        if not exp_type or exp_type == "general":
            return "_general"
        return _slug(exp_type)

    def _unique_record_id(self, rel_dir: str, name: str) -> str:
        """A record folder id that does not yet exist (auto-suffix on collision so
        history is never overwritten)."""
        candidate = f"{rel_dir}/{name}"
        if not self._path_for_id(candidate).parent.exists():
            return candidate
        i = 2
        while self._path_for_id(f"{rel_dir}/{name}-{i}").parent.exists():
            i += 1
        return f"{rel_dir}/{name}-{i}"

    # -- scanning ----------------------------------------------------------

    def _iter_entries(self) -> Iterator[Dict[str, Any]]:
        """Yield every parseable record + solution entry (checklists are not
        searchable entries — they have no symptom/body knowledge). A single
        malformed file is skipped (a corrupt hand-edit must not break a whole
        query); ``get`` of that specific id still fails fast."""
        if self._records_dir.exists():
            for f in sorted(self._records_dir.rglob(_RECORD_FILENAME)):
                entry = self._read_file(f)
                if entry is not None:
                    yield entry
        if self._troubleshooting_dir.exists():
            for f in sorted(self._troubleshooting_dir.rglob("*.md")):
                entry = self._read_file(f)
                if entry is not None:
                    yield entry

    def _read_file(self, f: Path) -> Optional[Dict[str, Any]]:
        try:
            fm, body = _parse(f.read_text(encoding="utf-8"))
        except Exception:
            return None
        entry = dict(fm)
        entry["id"] = self._id_for_path(f)
        entry["body"] = body
        return entry

    # -- reads -------------------------------------------------------------

    def recall(
        self,
        chip: Optional[str] = None,
        qub: Optional[str] = None,
        exp_type: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Open the notebook to a qubit/experiment. Three buckets:
          - ``checklist``: the acceptance items for ``exp_type`` (empty if none).
          - ``gotchas``:   solutions for ``exp_type`` (confirmed first).
          - ``recent``:    recent records for this chip/qub (newest first; NOT
            filtered by exp_type — opening a qubit's notebook shows its whole
            recent history)."""
        gotchas: List[Dict[str, Any]] = []
        recent: List[Dict[str, Any]] = []
        for e in self._iter_entries():
            kind = e.get("type")
            if kind == "record":
                if chip and e.get("chip") != chip:
                    continue
                if qub and e.get("qub") != qub:
                    continue
                recent.append(e)
            elif kind == "solution":
                if exp_type and e.get("exp_type") != exp_type:
                    continue
                gotchas.append(e)
        recent.sort(key=lambda e: str(e.get("date", "")), reverse=True)
        gotchas.sort(
            key=lambda e: (
                e.get("confidence") != "confirmed",
                str(e.get("symptom", "")),
            )
        )
        checklist = self.checklist_get(exp_type)["items"] if exp_type else []
        return {
            "checklist": checklist,
            "gotchas": gotchas[:limit],
            "recent": recent[:limit],
        }

    def search(
        self,
        query: str,
        kind: Optional[str] = None,
        exp_type: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Keyword search over symptom + body (and exp_type/reason), with optional
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

    # -- checklist (one acceptance list per exp_type) ----------------------

    def checklist_get(self, exp_type: str) -> Dict[str, Any]:
        """Return the acceptance items for ``exp_type`` as a list (parsed from the
        file's markdown bullet body). Missing file -> empty list, not an error."""
        entry_id = f"checklists/{self._checklist_slug(exp_type)}"
        path = self._path_for_id(entry_id)
        if not path.exists():
            return {"exp_type": exp_type, "items": []}
        _, body = _parse(path.read_text(encoding="utf-8"))
        return {"exp_type": exp_type, "items": _body_to_items(body)}

    def checklist_set(self, exp_type: str, items: List[str]) -> Dict[str, Any]:
        """Replace the acceptance checklist for ``exp_type`` wholesale. Items are
        stored as a markdown bullet list in the body (hand-editable)."""
        if not isinstance(items, list):
            raise RuntimeError("items must be a list of checklist strings")
        entry_id = f"checklists/{self._checklist_slug(exp_type)}"
        path = self._path_for_id(entry_id)
        fm: Dict[str, Any] = {"type": "checklist", "exp_type": exp_type or "general"}
        body = _items_to_body([str(i) for i in items])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_dump(fm, body), encoding="utf-8")
        return {"id": entry_id, "items": _body_to_items(body)}

    # -- record (episodic, folder-per-measurement, immutable) --------------

    def record_measurement(
        self,
        chip: str,
        qub: str,
        date: str,
        exp_type: List[str],
        decision: str,
        reason: str,
        body: str,
        figure_paths: Optional[List[str]] = None,
        data_ref: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a measurement record FOLDER (never overwrite — auto-suffix on a
        same-day same-experiment collision) and copy any ``figure_paths`` into it
        as ``figure.png`` / ``figure_2.png`` / ...  ``decision`` is accept|reject;
        ``reason`` is the one-line verdict, ``body`` the per-item pass/fail with
        evidence and the numbers."""
        if not isinstance(exp_type, list) or not exp_type:
            raise RuntimeError("exp_type must be a non-empty list of experiment types")
        if decision not in _VALID_DECISION:
            raise RuntimeError(
                f"decision must be one of {_VALID_DECISION}, got {decision!r}"
            )
        chip = self._safe_component(chip, "chip")
        qub = self._safe_component(qub, "qub")
        date = self._safe_component(date, "date")
        rel_dir = f"records/{chip}/{qub}"
        name = f"{date}-{_slug('-'.join(exp_type))}"
        entry_id = self._unique_record_id(rel_dir, name)
        record_path = self._path_for_id(entry_id)
        folder = record_path.parent
        # Validate all figure paths before creating any directory so that a bad path
        # never leaves an empty/partial folder that orphans future auto-suffix ids.
        self._validate_figure_paths(figure_paths or [])
        folder.mkdir(parents=True, exist_ok=True)
        figures = self._copy_figures(folder, figure_paths or [])
        fm: Dict[str, Any] = {
            "type": "record",
            "chip": chip,
            "qub": qub,
            "date": date,
            "exp_type": list(exp_type),
            "decision": decision,
        }
        if data_ref:
            fm["data_ref"] = data_ref
        if figures:
            fm["figures"] = figures
        # The reason is the verdict headline; body is the per-item evidence.
        full_body = f"{reason.strip()}\n\n{body.strip()}" if reason.strip() else body
        record_path.write_text(_dump(fm, full_body), encoding="utf-8")
        return {"id": entry_id, "created": True, "figures": figures}

    def _validate_figure_paths(self, figure_paths: List[str]) -> None:
        """Raise before any folder is created if any source path does not exist.
        Called ahead of mkdir so a bad path never leaves an orphan record folder."""
        for src in figure_paths:
            if not Path(src).is_file():
                raise RuntimeError(f"figure path does not exist: {src!r}")

    def _copy_figures(self, folder: Path, figure_paths: List[str]) -> List[str]:
        """Copy each source figure into ``folder`` as figure.png / figure_2.png / …
        Fast-fails if a given path does not exist (a bad path the agent passed is a
        bug, not something to silently drop — a record with NO figures is fine,
        just omit figure_paths)."""
        copied: List[str] = []
        for i, src in enumerate(figure_paths):
            src_path = Path(src)
            if not src_path.is_file():
                raise RuntimeError(f"figure path does not exist: {src!r}")
            dest_name = "figure.png" if i == 0 else f"figure_{i + 1}.png"
            shutil.copy(src_path, folder / dest_name)
            copied.append(dest_name)
        return copied

    # -- troubleshooting (context-free solution, mutable, self-improving) --

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
        rel_dir = f"troubleshooting/{self._exp_dir(exp_type)}"
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
        """Delete a single troubleshooting / checklist file. Records are IMMUTABLE
        — a ``records/`` id is refused (history is never deleted)."""
        if self._is_record_id(entry_id):
            raise RuntimeError(
                f"records are immutable and cannot be deleted: {entry_id!r}"
            )
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
    parts = [str(entry.get(k, "")) for k in ("symptom", "body", "reason")]
    exp = entry.get("exp_type")
    parts.extend(exp if isinstance(exp, list) else [str(exp or "")])
    return " ".join(parts).lower()

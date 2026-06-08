"""Tests for the agent-memory MemoryStore (file CRUD + search + self-improve)."""

import pytest
from zcu_tools.mcp.agent_memory.store import MemoryStore


def _store(tmp_path) -> MemoryStore:
    return MemoryStore(root=tmp_path, namespace="ns")


def test_record_creates_and_get_roundtrip(tmp_path):
    s = _store(tmp_path)
    res = s.record(
        chip="Q5_2D",
        qub="Q1",
        date="2026-06-08",
        exp_type=["reset/bath"],
        outcome="partial",
        body="κ≈0.5MHz, post_delay 拉到 1.6us 後對比度回升。",
    )
    assert res["id"] == "records/Q5_2D/Q1/2026-06-08-reset-bath"
    got = s.get(res["id"])
    assert got["type"] == "record"
    assert got["chip"] == "Q5_2D"
    assert got["exp_type"] == ["reset/bath"]
    assert "post_delay" in got["body"]
    assert "κ" in got["body"]  # unicode frontmatter+body round-trip


def test_record_collision_auto_suffixes(tmp_path):
    s = _store(tmp_path)
    a = s.record(
        chip="Q1",
        qub="Q1",
        date="2026-06-08",
        exp_type=["onetone/freq"],
        outcome="success",
        body="a",
    )
    b = s.record(
        chip="Q1",
        qub="Q1",
        date="2026-06-08",
        exp_type=["onetone/freq"],
        outcome="success",
        body="b",
    )
    assert a["id"] != b["id"]
    assert b["id"].endswith("-2")


def test_record_requires_nonempty_list_exp_type(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(RuntimeError):
        s.record(
            chip="Q1",
            qub="Q1",
            date="2026-06-08",
            exp_type="t1",
            outcome="success",
            body="x",
        )  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        s.record(
            chip="Q1",
            qub="Q1",
            date="2026-06-08",
            exp_type=[],
            outcome="success",
            body="x",
        )


def test_add_solution_then_duplicate_raises(tmp_path):
    s = _store(tmp_path)
    r = s.add_solution(
        exp_type="reset/bath",
        symptom="bath reset low contrast",
        category="failure-fix",
        body="post_delay >= 5/(2 pi kappa)",
    )
    assert r["confidence"] == "provisional"
    assert r["id"] == "solutions/reset/bath/bath-reset-low-contrast"
    with pytest.raises(RuntimeError):
        s.add_solution(
            exp_type="reset/bath",
            symptom="bath reset low contrast",
            category="failure-fix",
            body="dup",
        )


def test_update_solution_promotes_to_confirmed(tmp_path):
    s = _store(tmp_path)
    sid = s.add_solution(
        exp_type="reset/bath", symptom="low contrast", category="failure-fix", body="b"
    )["id"]
    u1 = s.update_solution(
        entry_id=sid, add_seen_in=["records/Q5_2D/Q1/2026-06-08-reset-bath"]
    )
    assert u1["confidence"] == "provisional"
    u2 = s.update_solution(
        entry_id=sid, add_seen_in=["records/Q3_2D/Q2/2026-05-20-reset-bath"]
    )
    assert u2["confidence"] == "confirmed"
    assert u2["seen_in_count"] == 2
    # dedup: re-adding the same record id does not inflate the count
    u3 = s.update_solution(
        entry_id=sid, add_seen_in=["records/Q3_2D/Q2/2026-05-20-reset-bath"]
    )
    assert u3["seen_in_count"] == 2


def test_update_solution_rejects_bad_confidence(tmp_path):
    s = _store(tmp_path)
    sid = s.add_solution(exp_type="t1", symptom="x", category="gotcha", body="b")["id"]
    with pytest.raises(RuntimeError):
        s.update_solution(entry_id=sid, confidence="maybe")


def test_recall_filters_and_orders(tmp_path):
    s = _store(tmp_path)
    s.record(
        chip="Q5_2D",
        qub="Q1",
        date="2026-06-07",
        exp_type=["onetone/freq"],
        outcome="success",
        body="r_f",
    )
    s.record(
        chip="Q5_2D",
        qub="Q1",
        date="2026-06-08",
        exp_type=["reset/bath"],
        outcome="partial",
        body="reset",
    )
    s.record(
        chip="OTHER",
        qub="Q9",
        date="2026-06-08",
        exp_type=["reset/bath"],
        outcome="success",
        body="elsewhere",
    )
    s.add_solution(
        exp_type="reset/bath", symptom="low contrast", category="failure-fix", body="x"
    )
    out = s.recall(chip="Q5_2D", qub="Q1", exp_type="reset/bath")
    rec_ids = [r["id"] for r in out["records"]]
    assert rec_ids and all("Q5_2D/Q1" in i for i in rec_ids)
    assert not any("OTHER" in i for i in rec_ids)
    assert out["records"][0]["date"] == "2026-06-08"  # newest first
    assert [sol["exp_type"] for sol in out["solutions"]] == ["reset/bath"]


def test_search_keyword_and_filters(tmp_path):
    s = _store(tmp_path)
    s.add_solution(
        exp_type="reset/bath",
        symptom="low contrast",
        category="failure-fix",
        body="residual photon; raise post_delay",
    )
    s.add_solution(
        exp_type="onetone/freq",
        symptom="asymmetric dip",
        category="analysis-heuristic",
        body="use hm model and fit_bg_slope",
    )
    s.record(
        chip="Q1",
        qub="Q1",
        date="2026-06-08",
        exp_type=["onetone/freq"],
        outcome="success",
        body="post_delay note in a record",
    )

    hits = s.search(query="post_delay")["results"]
    assert {h["type"] for h in hits} == {"solution", "record"}
    assert (
        s.search(query="post_delay", kind="solution")["results"][0]["symptom"]
        == "low contrast"
    )
    assert all(
        h["type"] == "record"
        for h in s.search(query="post_delay", kind="record")["results"]
    )
    cat = s.search(query="dip", category="analysis-heuristic")["results"]
    assert len(cat) == 1 and cat[0]["symptom"] == "asymmetric dip"
    assert s.search(query="nonexistent-token")["results"] == []


def test_get_missing_and_delete(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(RuntimeError):
        s.get("records/none/x")
    rid = s.record(
        chip="Q1",
        qub="Q1",
        date="2026-06-08",
        exp_type=["t1"],
        outcome="success",
        body="t1",
    )["id"]
    s.delete(rid)
    with pytest.raises(RuntimeError):
        s.get(rid)
    with pytest.raises(RuntimeError):
        s.delete(rid)


def test_path_traversal_rejected(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(RuntimeError):
        s.get("../escape")
    with pytest.raises(RuntimeError):
        s.get("/abs/escape")

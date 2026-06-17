"""Tests for the agent-memory MemoryStore — folder records, figure copy,
recall three buckets, checklist get/set, immutable records, id-prefix dispatch."""

import pytest
from zcu_tools.mcp.agent_memory.store import MemoryStore


def _store(tmp_path) -> MemoryStore:
    return MemoryStore(root=tmp_path, namespace="ns")


def _make_png(tmp_path, name: str):
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + name.encode())
    return p


# -- record (folder) ------------------------------------------------------


def test_record_creates_folder_and_get_roundtrip(tmp_path):
    s = _store(tmp_path)
    res = s.record_measurement(
        chip="Q5_2D",
        qub="Q1",
        date="2026-06-08",
        exp_type=["reset/bath"],
        decision="accept",
        reason="contrast recovered after raising post_delay",
        body="κ≈0.5MHz, post_delay 拉到 1.6us 後對比度回升。",
    )
    # The id is the FOLDER, not the .md file.
    assert res["id"] == "records/Q5_2D/Q1/2026-06-08-reset-bath"
    folder = tmp_path / "ns" / res["id"]
    assert (folder / "record.md").is_file()

    got = s.get(res["id"])
    assert got["type"] == "record"
    assert got["chip"] == "Q5_2D"
    assert got["exp_type"] == ["reset/bath"]
    assert got["decision"] == "accept"
    assert "post_delay" in got["body"]  # body keeps reason headline + detail
    assert "κ" in got["body"]  # unicode frontmatter + body round-trip


def test_record_rejects_bad_decision(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(RuntimeError):
        s.record_measurement(
            chip="Q1",
            qub="Q1",
            date="2026-06-08",
            exp_type=["t1"],
            decision="maybe",  # not accept|reject
            reason="r",
            body="b",
        )


def test_record_collision_auto_suffixes_folder(tmp_path):
    s = _store(tmp_path)
    a = s.record_measurement(
        chip="Q1",
        qub="Q1",
        date="2026-06-08",
        exp_type=["onetone/freq"],
        decision="accept",
        reason="r",
        body="a",
    )
    b = s.record_measurement(
        chip="Q1",
        qub="Q1",
        date="2026-06-08",
        exp_type=["onetone/freq"],
        decision="accept",
        reason="r",
        body="b",
    )
    assert a["id"] != b["id"]
    assert b["id"].endswith("-2")
    # the original folder's record.md is untouched (history never overwritten)
    assert s.get(a["id"])["body"].endswith("a")
    assert s.get(b["id"])["body"].endswith("b")


def test_record_requires_nonempty_list_exp_type(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(RuntimeError):
        s.record_measurement(
            chip="Q1",
            qub="Q1",
            date="2026-06-08",
            exp_type="t1",  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
            decision="accept",
            reason="r",
            body="x",
        )
    with pytest.raises(RuntimeError):
        s.record_measurement(
            chip="Q1",
            qub="Q1",
            date="2026-06-08",
            exp_type=[],
            decision="accept",
            reason="r",
            body="x",
        )


# -- figure copy ----------------------------------------------------------


def test_record_copies_figures_into_folder(tmp_path):
    s = _store(tmp_path)
    fig1 = _make_png(tmp_path, "plot_a.png")
    fig2 = _make_png(tmp_path, "plot_b.png")
    res = s.record_measurement(
        chip="Q1",
        qub="Q1",
        date="2026-06-08",
        exp_type=["onetone/freq"],
        decision="accept",
        reason="clean dip",
        body="r_f=5998 MHz",
        figure_paths=[str(fig1), str(fig2)],
    )
    folder = tmp_path / "ns" / res["id"]
    assert (folder / "figure.png").read_bytes() == fig1.read_bytes()
    assert (folder / "figure_2.png").read_bytes() == fig2.read_bytes()
    assert res["figures"] == ["figure.png", "figure_2.png"]
    assert s.get(res["id"])["figures"] == ["figure.png", "figure_2.png"]


def test_record_without_figures_is_fine(tmp_path):
    s = _store(tmp_path)
    res = s.record_measurement(
        chip="Q1",
        qub="Q1",
        date="2026-06-08",
        exp_type=["t1"],
        decision="reject",
        reason="pure noise",
        body="no decay visible",
    )
    assert res["figures"] == []
    folder = tmp_path / "ns" / res["id"]
    assert not (folder / "figure.png").exists()
    assert "figures" not in s.get(res["id"])  # omitted when none


def test_record_bad_figure_path_fast_fails(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(RuntimeError):
        s.record_measurement(
            chip="Q1",
            qub="Q1",
            date="2026-06-08",
            exp_type=["t1"],
            decision="accept",
            reason="r",
            body="b",
            figure_paths=[str(tmp_path / "does_not_exist.png")],
        )


# -- checklist ------------------------------------------------------------


def test_checklist_set_then_get_roundtrip(tmp_path):
    s = _store(tmp_path)
    items = ["dip is clean", "window not too wide", "SNR acceptable"]
    out = s.checklist_set(exp_type="onetone/freq", items=items)
    assert out["id"] == "checklists/onetone-freq"
    assert s.checklist_get(exp_type="onetone/freq")["items"] == items


def test_checklist_set_replaces_wholesale(tmp_path):
    s = _store(tmp_path)
    s.checklist_set(exp_type="t1", items=["a", "b", "c"])
    s.checklist_set(exp_type="t1", items=["only one"])
    assert s.checklist_get(exp_type="t1")["items"] == ["only one"]


def test_checklist_get_missing_is_empty(tmp_path):
    s = _store(tmp_path)
    assert s.checklist_get(exp_type="never/set")["items"] == []


def test_checklist_stored_as_markdown_bullets(tmp_path):
    s = _store(tmp_path)
    s.checklist_set(exp_type="reset/bath", items=["item one", "item two"])
    md = (tmp_path / "ns" / "checklists" / "reset-bath.md").read_text(encoding="utf-8")
    assert "- item one" in md and "- item two" in md
    assert "type: checklist" in md


def test_checklist_reads_hand_edited_file(tmp_path):
    # The server must read a checklist a human hand-wrote (mixed bullet styles).
    s = _store(tmp_path)
    path = tmp_path / "ns" / "checklists" / "manual.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\ntype: checklist\nexp_type: manual\n---\n\n"
        "- first\n* second\n+ third\nnot a bullet line\n",
        encoding="utf-8",
    )
    assert s.checklist_get(exp_type="manual")["items"] == ["first", "second", "third"]


# -- recall (three buckets) ----------------------------------------------


def test_recall_returns_three_buckets(tmp_path):
    s = _store(tmp_path)
    s.checklist_set(exp_type="reset/bath", items=["contrast > 0.5"])
    s.record_measurement(
        chip="Q5_2D",
        qub="Q1",
        date="2026-06-07",
        exp_type=["onetone/freq"],
        decision="accept",
        reason="r",
        body="r_f",
    )
    s.record_measurement(
        chip="Q5_2D",
        qub="Q1",
        date="2026-06-08",
        exp_type=["reset/bath"],
        decision="reject",
        reason="r",
        body="reset",
    )
    s.record_measurement(
        chip="OTHER",
        qub="Q9",
        date="2026-06-08",
        exp_type=["reset/bath"],
        decision="accept",
        reason="r",
        body="elsewhere",
    )
    s.add_solution(
        exp_type="reset/bath", symptom="low contrast", category="failure-fix", body="x"
    )

    out = s.recall(chip="Q5_2D", qub="Q1", exp_type="reset/bath")
    assert set(out) == {"checklist", "gotchas", "recent"}
    assert out["checklist"] == ["contrast > 0.5"]
    assert [sol["exp_type"] for sol in out["gotchas"]] == ["reset/bath"]
    rec_ids = [r["id"] for r in out["recent"]]
    assert rec_ids and all("Q5_2D/Q1" in i for i in rec_ids)
    assert not any("OTHER" in i for i in rec_ids)
    assert out["recent"][0]["date"] == "2026-06-08"  # newest first


def test_recall_recent_not_filtered_by_exp_type(tmp_path):
    # recent shows the qubit's whole recent history, not just this exp_type.
    s = _store(tmp_path)
    s.record_measurement(
        chip="Q1",
        qub="Q1",
        date="2026-06-07",
        exp_type=["onetone/freq"],
        decision="accept",
        reason="r",
        body="a",
    )
    s.record_measurement(
        chip="Q1",
        qub="Q1",
        date="2026-06-08",
        exp_type=["t1"],
        decision="accept",
        reason="r",
        body="b",
    )
    out = s.recall(chip="Q1", qub="Q1", exp_type="onetone/freq")
    exp_types = {tuple(r["exp_type"]) for r in out["recent"]}
    assert exp_types == {("onetone/freq",), ("t1",)}


def test_recall_without_exp_type_empty_checklist(tmp_path):
    s = _store(tmp_path)
    out = s.recall(chip="Q1", qub="Q1")
    assert out["checklist"] == []


# -- search ---------------------------------------------------------------


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
    s.record_measurement(
        chip="Q1",
        qub="Q1",
        date="2026-06-08",
        exp_type=["onetone/freq"],
        decision="accept",
        reason="r",
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


def test_solution_lives_under_troubleshooting(tmp_path):
    s = _store(tmp_path)
    r = s.add_solution(
        exp_type="reset/bath",
        symptom="bath reset low contrast",
        category="failure-fix",
        body="post_delay >= 5/(2 pi kappa)",
    )
    assert r["id"] == "troubleshooting/reset/bath/bath-reset-low-contrast"
    assert (
        tmp_path / "ns" / "troubleshooting" / "reset" / "bath"
        / "bath-reset-low-contrast.md"
    ).is_file()


def test_add_solution_then_duplicate_raises(tmp_path):
    s = _store(tmp_path)
    s.add_solution(
        exp_type="reset/bath",
        symptom="low contrast",
        category="failure-fix",
        body="b",
    )
    with pytest.raises(RuntimeError):
        s.add_solution(
            exp_type="reset/bath",
            symptom="low contrast",
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
    u3 = s.update_solution(
        entry_id=sid, add_seen_in=["records/Q3_2D/Q2/2026-05-20-reset-bath"]
    )
    assert u3["seen_in_count"] == 2  # dedup


def test_update_solution_rejects_bad_confidence(tmp_path):
    s = _store(tmp_path)
    sid = s.add_solution(exp_type="t1", symptom="x", category="gotcha", body="b")["id"]
    with pytest.raises(RuntimeError):
        s.update_solution(entry_id=sid, confidence="maybe")


# -- delete / immutability -----------------------------------------------


def test_delete_solution_then_missing(tmp_path):
    s = _store(tmp_path)
    sid = s.add_solution(exp_type="t1", symptom="x", category="gotcha", body="b")["id"]
    s.delete(sid)
    with pytest.raises(RuntimeError):
        s.get(sid)
    with pytest.raises(RuntimeError):
        s.delete(sid)


def test_delete_checklist(tmp_path):
    s = _store(tmp_path)
    s.checklist_set(exp_type="t1", items=["a"])
    s.delete("checklists/t1")
    assert s.checklist_get(exp_type="t1")["items"] == []


def test_record_cannot_be_deleted(tmp_path):
    s = _store(tmp_path)
    rid = s.record_measurement(
        chip="Q1",
        qub="Q1",
        date="2026-06-08",
        exp_type=["t1"],
        decision="accept",
        reason="r",
        body="t1",
    )["id"]
    with pytest.raises(RuntimeError):
        s.delete(rid)
    assert s.get(rid)["body"].endswith("t1")  # still there


# -- id-prefix path dispatch + traversal ----------------------------------


def test_get_missing_record(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(RuntimeError):
        s.get("records/none/x")


def test_id_prefix_dispatch_record_vs_single_file(tmp_path):
    s = _store(tmp_path)
    rid = s.record_measurement(
        chip="Q1",
        qub="Q1",
        date="2026-06-08",
        exp_type=["t1"],
        decision="accept",
        reason="r",
        body="rec",
    )["id"]
    sid = s.add_solution(exp_type="t1", symptom="x", category="gotcha", body="sol")[
        "id"
    ]
    # record id resolves to <folder>/record.md; solution id to <id>.md
    assert s.get(rid)["type"] == "record"
    assert s.get(sid)["type"] == "solution"


def test_path_traversal_rejected(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(RuntimeError):
        s.get("../escape")
    with pytest.raises(RuntimeError):
        s.get("/abs/escape")


# -- Fix 1: record.md alias bug (symptom slug == "record") ----------------


def test_solution_slug_record_roundtrips_without_alias(tmp_path):
    # A symptom named "Record" (slug -> "record") lives in troubleshooting/<exp>/record.md.
    # _id_for_path must NOT mistake it for the records/ folder prefix and must return
    # the stem id so that search -> get / update_solution / delete all hit the same entry.
    s = _store(tmp_path)
    for symptom in ("Record", "record"):
        exp_type = f"t1-alias-{symptom}"
        result = s.add_solution(
            exp_type=exp_type,
            symptom=symptom,
            category="gotcha",
            body="body content",
        )
        entry_id = result["id"]
        # Must be a stem id (troubleshooting/…/record), not a folder id.
        assert entry_id.startswith("troubleshooting/"), entry_id
        assert not entry_id.startswith("records/"), entry_id

        # search round-trip: id returned by search must be usable by get.
        hits = s.search(query="body content")["results"]
        hit_ids = [h["id"] for h in hits]
        assert entry_id in hit_ids, f"search did not return {entry_id!r}: {hit_ids}"
        assert s.get(entry_id)["body"] == "body content"

        # update_solution must find the entry.
        s.update_solution(entry_id=entry_id, body="updated body")
        assert s.get(entry_id)["body"] == "updated body"

        # delete must find and remove it.
        s.delete(entry_id)
        with pytest.raises(RuntimeError):
            s.get(entry_id)


# -- Fix 2: dirty-folder prevention (all-or-nothing figure validation) ----


def test_bad_figure_path_leaves_no_record_folder(tmp_path):
    # When figure_paths contains a missing file, record_measurement must raise
    # BEFORE creating any folder — no orphan directory may remain.
    s = _store(tmp_path)
    good = _make_png(tmp_path, "good.png")
    missing = str(tmp_path / "does_not_exist.png")
    with pytest.raises(RuntimeError):
        s.record_measurement(
            chip="Q1",
            qub="Q1",
            date="2026-06-17",
            exp_type=["t1"],
            decision="accept",
            reason="r",
            body="b",
            figure_paths=[str(good), missing],
        )
    # The record folder must not exist (all-or-nothing).
    records_root = tmp_path / "ns" / "records"
    assert not records_root.exists() or not any(records_root.rglob("*"))


# -- Fix 3: multi-line checklist item fast-fail ---------------------------


def test_checklist_multiline_item_raises(tmp_path):
    s = _store(tmp_path)
    with pytest.raises(RuntimeError, match="single-line"):
        s.checklist_set(exp_type="t1", items=["line A\nline B"])


# -- record_measurement traversal rejection --------------------------------


def test_record_measurement_rejects_traversal_in_chip_qub_date(tmp_path):
    s = _store(tmp_path)
    for bad_chip, bad_qub, bad_date in [
        ("../escape", "Q1", "2026-06-17"),
        ("Q1", "../escape", "2026-06-17"),
        ("Q1", "Q1", "../escape"),
    ]:
        with pytest.raises(RuntimeError):
            s.record_measurement(
                chip=bad_chip,
                qub=bad_qub,
                date=bad_date,
                exp_type=["t1"],
                decision="accept",
                reason="r",
                body="b",
            )

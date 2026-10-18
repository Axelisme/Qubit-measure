"""Fixed original indices, leaf provenance and historical recipe facts."""

from pathlib import Path

import pytest

from ._recipe_element_support import ANALYSIS_RECIPE, ElementGui, element_client


@pytest.mark.parametrize(
    "participants,indices", [(["Q3", "Q1"], [2, 0]), (["Q2"], [1])]
)
def test_indexed_overrides_capture_all_element_sources_and_original_names(
    tmp_path: Path, participants: list[str], indices: list[int]
) -> None:
    gui = ElementGui()
    with element_client(tmp_path, gui) as client:
        reply = client.call(
            "elements",
            {
                "participants": participants,
                "overrides": [
                    {
                        "path": ["tones", 1, "drive", "gain"],
                        "value": {"expr": "$qubit.freq / 10000"},
                    },
                    {"path": ["tones", 1, "sweep", "start"], "value": 1.5},
                ],
            },
        ).data
        assert reply["status"] == "finished", reply
        accepted = gui.accepted
        assert accepted is not None
        assert accepted.active_indices == indices
        assert [tone.drive.gain for tone in accepted.tones] == [0.1, 0.45, 0.6]
        expected = {
            "gain0": {
                "value": 0.1,
                "source": "default",
                "element": {"index": 0, "name": "Q1"},
            },
            "gain1": {
                "value": 0.45,
                "source": "explicit",
                "element": {"index": 1, "name": "Q2"},
            },
            "gain2": {
                "value": 0.6,
                "source": "derived",
                "element": {"index": 2, "name": "Q3"},
            },
            "start0": {
                "value": 1.0,
                "source": "default",
                "element": {"index": 0, "name": "Q1"},
            },
            "start1": {
                "value": 1.5,
                "source": "explicit",
                "element": {"index": 1, "name": "Q2"},
            },
            "start2": {
                "value": 1.0,
                "source": "default",
                "element": {"index": 2, "name": "Q3"},
            },
        }
        assert reply["actual"] == expected
        assert gui.owner.read_recipe_field(("tones", 1, "drive", "gain")).input == {
            "expr": "$qubit.freq / 10000"
        }
        gui.names = ["new3", "new2", "new1"]
        gui.md["$qubit.freq"] = 9000.0
        gui.library["new-source"] = {"gain": 0.9}
        gui.reset_form({"qubit": "Q2"})
        before = list(client.transport.sent)
        for detail in ("summary", "full"):
            assert (
                client.call(
                    "status", {"execution": reply["execution"], "detail": detail}
                )["actual"]
                == expected
            )
        assert (
            client.call("wait", {"execution": reply["execution"], "timeout": 0}).data[
                "actual"
            ]
            == expected
        )
        assert client.transport.sent == before


def test_estimates_and_candidates_keep_author_identity_after_source_reset(
    tmp_path: Path,
) -> None:
    gui = ElementGui()
    with element_client(tmp_path, gui, ANALYSIS_RECIPE) as client:
        reply = client.call(
            "elements", {"participants": ["Q3", "Q1"], "overrides": []}
        ).data
        assert reply["status"] == "awaiting_answer", reply
        pane = reply["analysis"]["primary"]
        assert pane["estimates"] == {
            "without_identity": {"value": 2.5, "stderr": None, "unit": None},
            "fit": {
                "value": 7.0,
                "stderr": 0.2,
                "unit": "us",
                "element": {"index": 2, "name": "Q3"},
            },
        }
        candidate = reply["writeback"]["stages"]["primary"][0]
        assert candidate["element"] == {"index": 2, "name": "Q3"}
        assert candidate["target"] == "trigger_offset"
        assert candidate["proposed"] == 0.24 and candidate["current"] == 0.1
        assert reply["writeback"]["destination"] == {
            "context": {"active_label": "sample"}
        }
        assert reply["writeback"]["receipts"] == ()
        assert reply["question_items"] == ("trigger_offset",)
        gui.names = ["Q7", "Q8", "Q9"]
        gui.native_summary.clear()
        gui.candidate_element = {"index": 0, "name": "later"}
        gui.reset_form({"qubit": "Q2"})
        before = list(client.transport.sent)
        retained = client.call("status", {"execution": reply["execution"]})
        assert retained["analysis"] == reply["analysis"]
        assert retained["actual"] == reply["actual"]
        assert retained["writeback"] == reply["writeback"]
        assert client.transport.sent == before
        final = client.call(
            "answer", {"recipe": reply["execution"], "decision": "skipped"}
        ).data
        assert final["status"] == "finished"
        assert final["writeback"] == reply["writeback"]
        assert all(
            method != "tab.writeback_write" for method, _ in client.transport.sent
        )


@pytest.mark.parametrize(
    "participants,override,diagnostic",
    [
        (
            ["Q3", "Q1"],
            {"path": ["tones", 1, "drive", "absent"], "value": 0.4},
            "field",
        ),
        (
            ["Q3", "Q1"],
            {"path": ["tones", "1", "drive", "gain"], "value": 0.4},
            "field",
        ),
        (["Q3", "Q1"], {"path": ["tones", 3, "drive", "gain"], "value": 0.4}, "field"),
        (
            ["Q3", "Q1"],
            {"path": ["tones", True, "drive", "gain"], "value": 0.4},
            "path",
        ),
        (["Q3", "Q1"], {"path": ["tones", 1, "drive", "gain"], "value": -1}, "gain"),
        ([], None, "Participants"),
        (["Q1", "Q1"], None, "participants"),
        (["unknown"], None, "participants"),
    ],
)
def test_invalid_element_inputs_fail_before_run(
    tmp_path, participants, override, diagnostic
):
    gui = ElementGui()
    with element_client(tmp_path, gui) as client:
        reply = client.call(
            "elements",
            {
                "participants": participants,
                "overrides": [] if override is None else [override],
            },
        )
        assert reply.is_error and reply.data["status"] == "failed"
        assert diagnostic.lower() in reply.data["error"]["message"].lower()
        assert gui.accepted is None and not gui.ran
        assert all(method != "tab.run_start" for method, _ in client.transport.sent)

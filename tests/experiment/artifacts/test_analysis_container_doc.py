"""The artifacts-owned analysis envelope is a generic container document."""

from dataclasses import replace
from pathlib import Path

from zcu_tools import datafile, experiment


def test_analysis_doc_roundtrips_through_container_without_domain_decoding(
    tmp_path: Path,
) -> None:
    document = experiment.AnalysisDocument(
        pane="analysis", parameters={"skip": 2}, summary={"accepted": True}
    )
    path = tmp_path / "analysis.h5"
    container = datafile.Container(
        {},
        {"analysis": document.to_doc()},
        {},
        datafile.ContainerAttrs("3.0", "synthetic", "analysis-result", {}),
    )
    datafile.save(path, container)
    loaded = datafile.load(path)
    assert loaded.docs["analysis"] == {
        "format": "zcu.analysis-result",
        "format_version": "1.0",
        "pane": "analysis",
        "parameters": {"skip": 2},
        "summary": {"accepted": True},
    }
    datafile.save(
        path,
        replace(loaded, docs={"analysis": {**loaded.docs["analysis"], "future": 7}}),
        replace=True,
    )
    assert datafile.load(path).docs["analysis"]["future"] == 7

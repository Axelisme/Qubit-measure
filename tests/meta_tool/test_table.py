from __future__ import annotations

from zcu_tools.meta_tool import SampleTable


def test_sample_table_accepts_path(tmp_path) -> None:
    path = tmp_path / "samples.csv"
    table = SampleTable(path)

    table.add_sample(qubit="Q1", flux=1.25)

    reloaded = SampleTable(path)
    samples = reloaded.get_samples()
    assert list(samples["qubit"]) == ["Q1"]
    assert list(samples["flux"]) == [1.25]

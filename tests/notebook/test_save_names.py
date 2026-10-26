"""Caller-owned notebook stems receive captured facts and control both formats."""

from dataclasses import replace
from pathlib import Path

from zcu_tools.device import DeviceManager
from zcu_tools.notebook import NotebookAdapter
from zcu_tools.plotting.plots import NonPresentingHost
from zcu_tools.resources.entry import ResultEntry, SaveContext

from tests.experiment._saving_support import SavingCfg, SavingCore


def test_caller_save_stem_controls_notebook_paths_and_receives_capture(
    tmp_path: Path,
) -> None:
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "results", database_root=tmp_path / "Database"
    )
    point = entry.new_point("sweet")

    contexts: list[SaveContext] = []

    def stem(context: SaveContext) -> str:
        contexts.append(context)
        return "notebook_Q1+Q2_exact"

    class CustomCore(SavingCore):
        AXES_SPEC = replace(SavingCore.AXES_SPEC, save_stem=stem)

    exp = NotebookAdapter(
        soc=object(),
        soccfg=object(),
        device_manager=DeviceManager(),
        host=NonPresentingHost(),
    )(CustomCore())
    source = exp.run(SavingCfg())
    saved = exp.save_run(point, source=source)
    again = saved.save()
    native, labber = again.outputs
    assert native.output.path.parent.name == "notebook_Q1+Q2_exact"
    assert labber.output.path.name == "notebook_Q1+Q2_exact_1.hdf5"
    assert contexts[-1].name == "test/saving"
    assert contexts[-1].run_id == saved.run_id
    assert contexts[-1].point == "sweet"
    assert contexts[-1].cfg == {"gain": 1.0, "dev": None}

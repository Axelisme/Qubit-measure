"""T2 record loading projects the caller-selected Run device."""

from pathlib import Path

from zcu_tools.notebook.analysis.t2_curve import load_t2_curve_context
from zcu_tools.resources.qubit_params import DispersiveFit, FluxDepFit, QubitParams
from zcu_tools.resources.record_table import JsonlRecordTable


def test_load_t2_records_uses_device_snapshot_coordinates(tmp_path: Path) -> None:
    params = QubitParams(tmp_path / "params.json")
    params.set_fluxdep_fit(
        FluxDepFit(EJ=4, EC=1, EL=1, flux_half=0.5, flux_int=0, flux_period=1)
    )
    params.set_dispersive_fit(DispersiveFit(g=0.1, bare_rf=7))
    JsonlRecordTable(tmp_path / "records.jsonl").append(
        run_id="run",
        devices={"bias": {"value": 0.25, "value_unit": "A"}},
        freq_mhz=4000,
        t1_us=40,
        t1_err_us=1,
        t2e_us=20,
        t2e_err_us=1,
    )
    context = load_t2_curve_context(result_dir=str(tmp_path), device_name="bias")
    assert context.samples_df["dev_value"].tolist() == [0.25]
    assert context.samples_df["dev_unit"].tolist() == ["A"]

"""Matching science adapters round-trip each native/Labber channel exactly."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from zcu_tools.datafile import DataVariable, ExperimentPayload, load_run_data, load_grouped_labber_data, save_run_data, write_labber
from zcu_tools.experiment.workflows import PreparedFluxdepExport

from tests._native_support import native_metadata
from zcu_lab.workflows import export as science

from ._export_inputs import RUN_ID
from ._fluxdep_export_inputs import synthetic_t2_run, synthetic_mist_run
from ._sim_inputs import Inputs


def read_exports(prepared: PreparedFluxdepExport, root: Path) -> tuple[ExperimentPayload, ExperimentPayload]:
    """Use public writers/readers, preserving the same science payload and cfg."""
    save_run_data(root / "merged.h5", prepared.payload, native_metadata(), cfg=prepared.cfg)
    write_labber(root / "merged-labber.h5", prepared.payload, cfg=prepared.cfg, grouped_layout="independent")
    native = load_run_data(root / "merged.h5")
    assert native.cfg == prepared.cfg
    assert native.metadata == native_metadata()
    labber = load_grouped_labber_data(str(root / "merged-labber.h5"), required_variables=tuple(str(key) for key in prepared.payload.variables))
    return native.payload, ExperimentPayload(variables={DataVariable(k):v for k,v in labber.variables.items()}, metadata=labber.metadata, representation="grouped")


@pytest.mark.parametrize("kind", ["echo", "ramsey"])
def test_t2_adapters_keep_unequal_delay_rows_rejection_and_missing_points(inputs: Inputs, tmp_path: Path, kind: Literal["echo", "ramsey"]) -> None:
    engine, plan, paths = synthetic_t2_run(inputs, tmp_path, kind)
    assert engine.execute(RUN_ID).lifecycle == "done"
    if kind == "echo":
        prepared = science.prepare_t2echo_fluxdep_export(plan=plan,data_root=paths.data_root,records=engine.records(RUN_ID))
    else:
        prepared = science.prepare_t2ramsey_fluxdep_export(plan=plan,data_root=paths.data_root,records=engine.records(RUN_ID))
    assert prepared.cfg.values == plan.model_dump(mode="json")
    for payload in read_exports(prepared,tmp_path):
        assert tuple(payload.variables) == ("signal","time",f"t2{kind}")
        signal = payload.variables[DataVariable("signal")]
        time = payload.variables[DataVariable("time")]
        lifetime = payload.variables[DataVariable(f"t2{kind}")]
        np.testing.assert_array_equal(signal.z, [[1+2j,2+3j,complex(np.nan,np.nan)],[4+5j,6+7j,complex(np.nan,np.nan)],[complex(np.nan,np.nan)]*3,[complex(np.nan,np.nan)]*3])
        np.testing.assert_array_equal(time.z, [[0,1e-6,2e-6],[0,3e-6,np.nan],[np.nan]*3,[np.nan]*3])
        np.testing.assert_array_equal(lifetime.z, [20e-6,np.nan,np.nan,np.nan])
        assert (signal.data.name,signal.data.unit)==("Signal","a.u.")
        assert (time.data.name,time.data.unit)==("Time","s")
        assert (lifetime.data.name,lifetime.data.unit)==(f"T2{kind}","s")
        for channel in (signal,time):
            assert [(a.name,a.unit) for a in channel.axes]==[("Time Index",""),("Flux","native")]
            np.testing.assert_array_equal(channel.axes[0].values,[0,1,2])
            np.testing.assert_array_equal(channel.axes[1].values,[1,1.1,1.2,1.3])
        assert [(a.name,a.unit) for a in lifetime.axes]==[("Flux","native")]
        np.testing.assert_array_equal(lifetime.axes[0].values,[1,1.1,1.2,1.3])


def test_mist_adapter_keeps_actual_gain_axis_and_success_only(inputs: Inputs,tmp_path: Path) -> None:
    engine,plan,paths=synthetic_mist_run(inputs,tmp_path)
    assert engine.execute(RUN_ID).lifecycle=="done"
    prepared=science.prepare_mist_fluxdep_export(plan=plan,data_root=paths.data_root,records=engine.records(RUN_ID))
    assert prepared.cfg.values==plan.model_dump(mode="json")
    for payload in read_exports(prepared,tmp_path):
        assert tuple(payload.variables)==("signal","success")
        signal=payload.variables[DataVariable("signal")]
        success=payload.variables[DataVariable("success")]
        np.testing.assert_array_equal(signal.z,[[1+2j,2+3j,3+4j],[2+2j,3+3j,4+4j],[complex(np.nan,np.nan)]*3])
        np.testing.assert_array_equal(success.z,[1.0,1.0,np.nan])
        assert (signal.data.name,signal.data.unit)==("Signal","a.u.")
        assert (success.data.name,success.data.unit)==("Success","")
        assert [(a.name,a.unit) for a in signal.axes]==[("Readout Gain","a.u."),("Flux","native")]
        np.testing.assert_array_equal(signal.axes[0].values,[0,.5,1])
        np.testing.assert_array_equal(signal.axes[1].values,[1,1.1,1.2])
        assert [(a.name,a.unit) for a in success.axes]==[("Flux","native")]
        np.testing.assert_array_equal(success.axes[0].values,[1,1.1,1.2])


def test_mist_adapter_refuses_a_point_with_a_different_gain_grid(inputs: Inputs,tmp_path: Path) -> None:
    engine,plan,paths=synthetic_mist_run(inputs,tmp_path,mismatch=True)
    assert engine.execute(RUN_ID).lifecycle=="done"
    with pytest.raises(ValueError, match="gain grid"):
        science.prepare_mist_fluxdep_export(plan=plan,data_root=paths.data_root,records=engine.records(RUN_ID))

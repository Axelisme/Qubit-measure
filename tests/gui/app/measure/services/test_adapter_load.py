"""Every registered value may attempt Load; incompatible files retain prior data."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, SessionEnv
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.guard import GuardService
from zcu_tools.gui.app.measure.services.load import LoadDataError, LoadService
from zcu_tools.gui.app.measure.state import Session, State
from zcu_tools.gui.form_adapter import Form, FormContext

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult
from tests.gui.app.measure._form_fakes import make_form
from tests.gui.app.measure._load_support import write_native_file
from tests.gui.app.measure._record_support import require_record


def saving_form(context: FormContext) -> Form[SavingCfg]:
    del context
    return Form(SavingCfg, {})


def saving_run(
    cfg: SavingCfg, *, context: RunContext
) -> RunRecord[SavingCfg, SavingResult]:
    return RunRecord(cfg=cfg, result=SavingCore().run(cfg, context=context))


def test_load_value_accepts_native_data_and_keeps_it_when_next_file_is_unsupported(
    tmp_path: Path,
) -> None:
    definition = Adapter(
        spec=SavingCore.AXES_SPEC,
        make_form=saving_form,
        run=saving_run,
        guide=None,
    )
    registry = Registry()
    registry.register("saving", definition)
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    state.add_tab(
        "tab",
        Session(
            adapter_name="saving",
            adapter=registry.create("saving"),
            cfg=make_form(saving_form(FormContext(None)), state=state),
        ),
    )
    service = LoadService(
        state,
        MagicMock(),
        context=MagicMock(),
        captured_parameters=lambda aliases: None,
    )
    supported = tmp_path / "supported.h5"
    metadata = write_native_file(supported, tag=SavingCore.AXES_SPEC.tag)
    GuardService(state).check_load("tab")
    outcome = service.load_result("tab", str(supported))
    retained = require_record(state.get_tab("tab").run)
    assert outcome.cfg_issue is None
    assert retained.metadata == metadata
    assert retained.cfg == SavingCfg()
    np.testing.assert_array_equal(retained.result.signal, [1 + 2j, 3 - 4j, 5 + 6j])

    unsupported = tmp_path / "unsupported.h5"
    write_native_file(unsupported, tag="other/experiment")
    GuardService(state).check_load("tab")
    with pytest.raises(LoadDataError, match="Expected experiment tag") as error:
        service.load_result("tab", str(unsupported))
    assert error.value.reason_code == "invalid_data_file"
    assert require_record(state.get_tab("tab").run) is retained

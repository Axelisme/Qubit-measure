"""GUI wiring for the single-carrier Raman raw-IQ scan."""

from pathlib import Path

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure import Adapter

from .core import RamanGFCfg, RamanGFExp, RamanGFResult
from .form import make_form


def _run(
    cfg: RamanGFCfg, *, context: RunContext
) -> RunRecord[RamanGFCfg, RamanGFResult]:
    return RunRecord(cfg=cfg, result=RamanGFExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=RamanGFExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
)

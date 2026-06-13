from qick.asm_v2 import QickParam
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.program.v2.utils import param2str, sweep2param


def test_sweep2param():
    cfg = SweepCfg(start=1.0, stop=5.0, expts=5, step=1.0)
    param = sweep2param("test_sweep", cfg)
    assert isinstance(param, QickParam)
    assert param.start == 1.0
    assert param.maxval() == 5.0
    assert param.minval() == 1.0


def test_param2str():
    # regular float
    assert param2str(3.14159) == "3.142"

    # integer
    assert param2str(10) == "10.000"

    # QickParam (not sweep)
    param_no_sweep = QickParam(start=2.5, spans={})
    assert param2str(param_no_sweep) == "2.500"

    # QickParam (sweep)
    cfg = SweepCfg(start=1.0, stop=5.0, expts=5, step=1.0)
    param_sweep = sweep2param("test_sweep", cfg)
    assert param2str(param_sweep) == "sweep(1.000, 5.000)"

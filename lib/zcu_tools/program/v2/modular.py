from copy import deepcopy
from typing import Mapping, Sequence

from typing_extensions import NotRequired

from qick import QickConfig

from ..base import SweepCfg
from .base import MyProgramV2, ProgramV2Cfg
from .modules import Module


class ModularProgramCfg(ProgramV2Cfg):
    sweep: NotRequired[Mapping[str, SweepCfg]]


class ModularProgramV2(MyProgramV2):
    """
    A class that allows custom behavior based on the provided modules.
    """

    def __init__(
        self,
        soccfg: QickConfig,
        cfg: ModularProgramCfg,
        modules: Sequence[Module],
        **kwargs,
    ) -> None:
        self.modules = modules

        # capture the relax delay from the cfg and set it to None
        # handle the relax delay in the _body() method
        # this can avoid the delay_auto() after the program body start from the readout window end
        # user may expect the relax delay start from the readout pulse end, which usually before the readout window end
        # for example, if the relax delay is 30 ns, readout pulse is 100 ns, readout trigger offset is 50 ns, readout window length is 100 ns,
        # default relax delay implemented will add 30 ns to readout window end, which is 180 ns from start of readout
        # for this implementation, it add relax delay smarter, for this case, it will delay to readout window end.
        # which is 150 ns from start of readout, is closer to user's expectation (130 ns)
        # TODO: non-hacky way to implement this?
        cfg = deepcopy(cfg)
        self._modular_program_relax_delay = cfg["relax_delay"]
        cfg["relax_delay"] = None

        super().__init__(soccfg, cfg, **kwargs)

    def _initialize(self, cfg: ModularProgramCfg) -> None:
        super()._initialize(cfg)

        # add v2 sweep loops
        for name, sweep in cfg.get("sweep", {}).items():
            self.add_loop(name, count=sweep["expts"])

        # initialize modules
        for module in self.modules:
            module.init(self)

    def _body(self, cfg: ModularProgramCfg) -> None:
        t = 0.0
        for module in self.modules:
            t = module.run(self, t)

        # handle relax delay here
        # add relax delay start from t, the end time of the last module
        self.delay(t=t + self._modular_program_relax_delay)
        self.delay_auto()  # force delay to end of all pulse and readout end


class BaseCustomProgramV2(ModularProgramV2):
    """
    A base class for custom programs to inherit.
    """

    def __init__(self, soccfg: QickConfig, cfg: ModularProgramCfg, **kwargs) -> None:
        super().__init__(soccfg, cfg, modules=self.make_modules(cfg), **kwargs)

    def make_modules(self, cfg: ModularProgramCfg) -> Sequence[Module]:
        return []

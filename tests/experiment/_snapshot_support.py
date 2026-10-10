"""Typed acquisition that changes a selected driver after Run admission."""

from zcu_tools.device import FakeDevice
from zcu_tools.experiment import AxesSpec, GroupedAxesSpec, RunRecord
from zcu_tools.experiment.context import RunContext

from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult


class SnapshotCore(SavingCore):
    """Acquire deterministic data, changing only the connected flux driver."""

    def run(self, config: SavingCfg, *, context: RunContext | None) -> SavingResult:
        """Write the connected Fake value, or acquire offline when context is None."""
        if context is not None:
            device = context.devices.get("flux")
            if isinstance(device, FakeDevice):
                device.set_value(0.75)
        return SavingCore().run(config, context=context)

    def persistence_spec(
        self,
    ) -> AxesSpec[SavingResult, SavingCfg] | GroupedAxesSpec[SavingResult, SavingCfg]:
        return SavingCore().persistence_spec()


class SnapshotRecordCore:
    """GUI record producer used behind the real BaseAdapter.run method."""

    def run(
        self, cfg: SavingCfg, *, context: RunContext
    ) -> RunRecord[SavingCfg, SavingResult]:
        return RunRecord(cfg, SnapshotCore().run(cfg, context=context))

    def persistence_spec(
        self,
    ) -> AxesSpec[SavingResult, SavingCfg] | GroupedAxesSpec[SavingResult, SavingCfg]:
        return SavingCore().persistence_spec()

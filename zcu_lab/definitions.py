"""Explicit user-owned measure catalog; importing it performs no registration."""

from zcu_tools.gui.app.measure.adapter import ExpAdapterProtocol
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.resources.entry.registry import ComponentRegistry

from zcu_lab.components import register_all as register_components
from zcu_lab.v2.fastflux.distortion.acc_phase.gui import AccPhaseAdapter
from zcu_lab.v2.fastflux.distortion.freq.gui import FastFluxFreqAdapter
from zcu_lab.v2.fastflux.distortion.phase.gui import PhaseAdapter
from zcu_lab.v2.fastflux.mist.gui import MistAdapter
from zcu_lab.v2.fastflux.t1.gui import T1Adapter as FastFluxT1Adapter
from zcu_lab.v2.fastflux.twotone.gui import TwoToneAdapter
from zcu_lab.v2.jpa.auto_optimize.gui import JpaAutoOptimizeAdapter
from zcu_lab.v2.jpa.check.gui import JpaCheckAdapter
from zcu_lab.v2.jpa.flux.gui import JpaFluxAdapter
from zcu_lab.v2.jpa.flux_onetone.gui import JpaFluxOneToneAdapter
from zcu_lab.v2.jpa.freq.gui import JpaFreqAdapter
from zcu_lab.v2.jpa.power.gui import JpaPowerAdapter
from zcu_lab.v2.lookback.gui import LookbackAdapter
from zcu_lab.v2.mist.flux_dep.gui import MistFluxDepAdapter
from zcu_lab.v2.mist.power_dep.drive_freq.gui import DriveFreqAdapter
from zcu_lab.v2.mist.power_dep.single_trace.gui import MistTraceAdapter
from zcu_lab.v2.mux.amp_rabi.gui import MuxAmpRabiAdapter
from zcu_lab.v2.mux.len_rabi.gui import MuxLenRabiAdapter
from zcu_lab.v2.mux.onetone.gui import MuxOnetoneAdapter
from zcu_lab.v2.mux.singleshot_ge.gui import MuxGEAdapter
from zcu_lab.v2.mux.t1.gui import MuxT1Adapter
from zcu_lab.v2.mux.t2echo.gui import MuxT2EchoAdapter
from zcu_lab.v2.mux.t2ramsey.gui import MuxT2RamseyAdapter
from zcu_lab.v2.mux.twotone_spectrum.gui import MuxFreqAdapter
from zcu_lab.v2.onetone.flux_dep.gui import OneToneFluxDepAdapter
from zcu_lab.v2.onetone.freq.gui import OneToneFreqAdapter
from zcu_lab.v2.onetone.power_dep.gui import OneTonePowerDepAdapter
from zcu_lab.v2.onetone.range_find.gui import RangeFindAdapter
from zcu_lab.v2.singleshot.ac_stark.gui import SsAcStarkAdapter
from zcu_lab.v2.singleshot.amp_rabi.gui import SsAmpRabiAdapter
from zcu_lab.v2.singleshot.check.gui import CheckAdapter
from zcu_lab.v2.singleshot.ge.gui import GEAdapter
from zcu_lab.v2.singleshot.len_rabi.gui import SsLenRabiAdapter
from zcu_lab.v2.singleshot.mist.freq.gui import MistFreqAdapter
from zcu_lab.v2.singleshot.mist.power.gui import MistPowerAdapter
from zcu_lab.v2.singleshot.mist.power_freq.gui import MistPowerFreqAdapter
from zcu_lab.v2.singleshot.mist.pre_freq.gui import MistPreFreqAdapter
from zcu_lab.v2.singleshot.reset_check.gui import SsResetCheckAdapter
from zcu_lab.v2.singleshot.t1.t1.gui import SsT1Adapter
from zcu_lab.v2.singleshot.t1.t1_with_tone.gui import SsT1ToneAdapter
from zcu_lab.v2.singleshot.t1.t1_with_tone_sweep.gui import (
    SsT1ToneSweepFreqAdapter,
    SsT1ToneSweepGainAdapter,
)
from zcu_lab.v2.two_qubit.chevron.gui import ChevronAdapter
from zcu_lab.v2.two_qubit.conditional_phase.gui import ConditionalPhaseAdapter
from zcu_lab.v2.two_qubit.conditional_spectroscopy.gui import SpectroscopyAdapter
from zcu_lab.v2.two_qubit.constructive_phase.gui import ConstructivePhaseAdapter
from zcu_lab.v2.two_qubit.cz_amplitude.gui import CZAmplitudeAdapter
from zcu_lab.v2.two_qubit.idle_zz.gui import IdleZZAdapter
from zcu_lab.v2.two_qubit.joint_readout.gui import JointReadoutAdapter
from zcu_lab.v2.two_qubit.local_z.gui import LocalZAdapter
from zcu_lab.v2.two_qubit.rb.gui import TwoQubitRBAdapter
from zcu_lab.v2.twotone.ac_stark.gui import AcStarkAdapter, AcStarkRamseyAdapter
from zcu_lab.v2.twotone.allxy.gui import AllXYAdapter
from zcu_lab.v2.twotone.ckp.gui import CKPAdapter
from zcu_lab.v2.twotone.dispersive.gui import DispersiveAdapter
from zcu_lab.v2.twotone.fluxdep.gui import FluxDepAdapter
from zcu_lab.v2.twotone.freq.gui import FreqAdapter
from zcu_lab.v2.twotone.irb.gui import IRBAdapter
from zcu_lab.v2.twotone.power_dep.gui import PowerDepAdapter
from zcu_lab.v2.twotone.rabi.amp_rabi.gui import AmpRabiAdapter
from zcu_lab.v2.twotone.rabi.len_rabi.gui import LenRabiAdapter
from zcu_lab.v2.twotone.rb.gui import RBAdapter
from zcu_lab.v2.twotone.reset.bath.freq.gui import BathFreqGainAdapter
from zcu_lab.v2.twotone.reset.bath.length.gui import BathLengthAdapter
from zcu_lab.v2.twotone.reset.bath.phase.gui import BathPhaseAdapter
from zcu_lab.v2.twotone.reset.dual_tone.freq.gui import DualToneFreqAdapter
from zcu_lab.v2.twotone.reset.dual_tone.length.gui import DualToneLengthAdapter
from zcu_lab.v2.twotone.reset.dual_tone.power.gui import DualTonePowerAdapter
from zcu_lab.v2.twotone.reset.rabi_check.gui import RabiCheckAdapter
from zcu_lab.v2.twotone.reset.single_tone.freq.gui import SingleToneFreqAdapter
from zcu_lab.v2.twotone.reset.single_tone.length.gui import SingleToneLengthAdapter
from zcu_lab.v2.twotone.ro_optimize.auto_optimize.gui import RoOptAutoAdapter
from zcu_lab.v2.twotone.ro_optimize.freq.gui import RoOptFreqAdapter
from zcu_lab.v2.twotone.ro_optimize.freq_gain.gui import RoOptFreqGainAdapter
from zcu_lab.v2.twotone.ro_optimize.length.gui import RoOptLengthAdapter
from zcu_lab.v2.twotone.ro_optimize.power.gui import RoOptPowerAdapter
from zcu_lab.v2.twotone.time_domain.cpmg.gui import CPMGAdapter
from zcu_lab.v2.twotone.time_domain.t1.gui import T1Adapter
from zcu_lab.v2.twotone.time_domain.t2echo.gui import T2EchoAdapter
from zcu_lab.v2.twotone.time_domain.t2ramsey.gui import T2RamseyAdapter
from zcu_lab.v2.twotone.zigzag.gui import ZigZagAdapter
from zcu_lab.v2.twotone.zigzag_sweep.gui import (
    ZigZagScanFreqAdapter,
    ZigZagScanGainAdapter,
)

ADAPTERS: dict[str, type[ExpAdapterProtocol]] = {
    "lookback": LookbackAdapter,
    "onetone/freq": OneToneFreqAdapter,
    "onetone/power_dep": OneTonePowerDepAdapter,
    "onetone/flux_dep": OneToneFluxDepAdapter,
    "twotone/freq": FreqAdapter,
    "twotone/flux_dep": FluxDepAdapter,
    "twotone/power_dep": PowerDepAdapter,
    "twotone/rabi/len_rabi": LenRabiAdapter,
    "twotone/rabi/amp_rabi": AmpRabiAdapter,
    "twotone/t1": T1Adapter,
    "twotone/t2ramsey": T2RamseyAdapter,
    "twotone/t2echo": T2EchoAdapter,
    "singleshot/ge": GEAdapter,
    "singleshot/check": CheckAdapter,
    "singleshot/len_rabi": SsLenRabiAdapter,
    "singleshot/amp_rabi": SsAmpRabiAdapter,
    "singleshot/t1": SsT1Adapter,
    "singleshot/ac_stark": SsAcStarkAdapter,
    "singleshot/reset_check": SsResetCheckAdapter,
    "singleshot/t1_tone": SsT1ToneAdapter,
    "singleshot/t1_tone_sweep_gain": SsT1ToneSweepGainAdapter,
    "singleshot/t1_tone_sweep_freq": SsT1ToneSweepFreqAdapter,
    "twotone/dispersive": DispersiveAdapter,
    "twotone/ac_stark": AcStarkAdapter,
    "twotone/ac_stark_ramsey": AcStarkRamseyAdapter,
    "twotone/ckp": CKPAdapter,
    "twotone/cpmg": CPMGAdapter,
    "twotone/allxy": AllXYAdapter,
    "twotone/zigzag": ZigZagAdapter,
    "twotone/zigzag_scan/gain": ZigZagScanGainAdapter,
    "twotone/zigzag_scan/freq": ZigZagScanFreqAdapter,
    "twotone/rb": RBAdapter,
    "twotone/irb": IRBAdapter,
    "twotone/ro_optimize/freq": RoOptFreqAdapter,
    "twotone/ro_optimize/power": RoOptPowerAdapter,
    "twotone/ro_optimize/length": RoOptLengthAdapter,
    "twotone/ro_optimize/freq_gain": RoOptFreqGainAdapter,
    "twotone/ro_optimize/auto": RoOptAutoAdapter,
    "twotone/reset/single_tone/freq": SingleToneFreqAdapter,
    "twotone/reset/single_tone/length": SingleToneLengthAdapter,
    "twotone/reset/dual_tone/freq": DualToneFreqAdapter,
    "twotone/reset/dual_tone/length": DualToneLengthAdapter,
    "twotone/reset/dual_tone/power": DualTonePowerAdapter,
    "twotone/reset/bath/freq_gain": BathFreqGainAdapter,
    "twotone/reset/bath/length": BathLengthAdapter,
    "twotone/reset/bath/phase": BathPhaseAdapter,
    "twotone/reset/check": RabiCheckAdapter,
    "mux/onetone": MuxOnetoneAdapter,
    "onetone/range_find": RangeFindAdapter,
    "mux/twotone_spectrum": MuxFreqAdapter,
    "mux/len_rabi": MuxLenRabiAdapter,
    "mux/amp_rabi": MuxAmpRabiAdapter,
    "mux/t1": MuxT1Adapter,
    "mux/t2ramsey": MuxT2RamseyAdapter,
    "mux/t2echo": MuxT2EchoAdapter,
    "mux/singleshot_ge": MuxGEAdapter,
    "jpa/flux_onetone": JpaFluxOneToneAdapter,
    "jpa/freq": JpaFreqAdapter,
    "jpa/flux": JpaFluxAdapter,
    "jpa/power": JpaPowerAdapter,
    "jpa/auto_optimize": JpaAutoOptimizeAdapter,
    "jpa/check": JpaCheckAdapter,
    "two_qubit/joint_readout": JointReadoutAdapter,
    "two_qubit/conditional_spectroscopy": SpectroscopyAdapter,
    "two_qubit/chevron": ChevronAdapter,
    "two_qubit/cz_amplitude": CZAmplitudeAdapter,
    "two_qubit/conditional_phase": ConditionalPhaseAdapter,
    "two_qubit/local_z": LocalZAdapter,
    "two_qubit/idle_zz": IdleZZAdapter,
    "two_qubit/constructive_phase": ConstructivePhaseAdapter,
    "two_qubit/rb": TwoQubitRBAdapter,
    "fastflux/twotone": TwoToneAdapter,
    "fastflux/t1": FastFluxT1Adapter,
    "fastflux/distortion/freq": FastFluxFreqAdapter,
    "fastflux/distortion/phase": PhaseAdapter,
    "fastflux/distortion/acc_phase": AccPhaseAdapter,
    "fastflux/mist": MistAdapter,
    "mist/flux_dep": MistFluxDepAdapter,
    "mist/power_dep/drive_freq": DriveFreqAdapter,
    "mist/power_dep/single_trace": MistTraceAdapter,
    "singleshot/mist/freq": MistFreqAdapter,
    "singleshot/mist/power": MistPowerAdapter,
    "singleshot/mist/power_freq": MistPowerFreqAdapter,
    "singleshot/mist/pre_freq": MistPreFreqAdapter,
}

# Core-derived forms of migrated adapters, registered with their adapter names.


def register_all(
    registry: Registry,
    *,
    components: ComponentRegistry | None = None,
) -> None:
    """Register adapters into the caller-owned registry.

    The registry already owns its injected persistent template definitions.
    Library creation reads ``registry.templates`` directly.
    Pass the shared ComponentRegistry as ``components`` at startup to register
    component models and roles. Omit this startup receiver on reload to
    preserve its identity. Duplicate adapters raise ValueError.
    Registration does not build forms or validate templates; Open/Reset does.
    """
    if components is not None:
        register_components(components)
    for name, cls in ADAPTERS.items():
        registry.register(name, cls)

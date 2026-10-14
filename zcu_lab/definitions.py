"""Explicit user-owned measure catalog; importing it performs no registration."""

from typing import Any

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.template_catalog import TemplateCatalog
from zcu_tools.gui.form_adapter import Form
from zcu_tools.resources.entry.registry import ComponentRegistry

from zcu_lab.components import register_all as register_components
from zcu_lab.v2.fake.freq.form import FORM as FAKE_FREQ_FORM
from zcu_lab.v2.fake.freq.gui import FakeFreqAdapter
from zcu_lab.v2.jpa.auto_optimize.form import FORM as JPA_AUTO_OPTIMIZE_FORM
from zcu_lab.v2.jpa.auto_optimize.gui import JpaAutoOptimizeAdapter
from zcu_lab.v2.jpa.check.form import FORM as JPA_CHECK_FORM
from zcu_lab.v2.jpa.check.gui import JpaCheckAdapter
from zcu_lab.v2.jpa.flux.form import FORM as JPA_FLUX_FORM
from zcu_lab.v2.jpa.flux.gui import JpaFluxAdapter
from zcu_lab.v2.jpa.flux_onetone.form import FORM as JPA_FLUX_ONETONE_FORM
from zcu_lab.v2.jpa.flux_onetone.gui import JpaFluxOneToneAdapter
from zcu_lab.v2.jpa.freq.form import FORM as JPA_FREQ_FORM
from zcu_lab.v2.jpa.freq.gui import JpaFreqAdapter
from zcu_lab.v2.jpa.power.form import FORM as JPA_POWER_FORM
from zcu_lab.v2.jpa.power.gui import JpaPowerAdapter
from zcu_lab.v2.lookback.form import FORM as LOOKBACK_FORM
from zcu_lab.v2.lookback.gui import LookbackAdapter
from zcu_lab.v2.onetone.flux_dep.form import FORM as ONETONE_FLUX_DEP_FORM
from zcu_lab.v2.onetone.flux_dep.gui import OneToneFluxDepAdapter
from zcu_lab.v2.onetone.freq.form import FORM as ONETONE_FREQ_FORM
from zcu_lab.v2.onetone.freq.gui import OneToneFreqAdapter
from zcu_lab.v2.onetone.power_dep.form import FORM as ONETONE_POWER_DEP_FORM
from zcu_lab.v2.onetone.power_dep.gui import OneTonePowerDepAdapter
from zcu_lab.v2.singleshot.ac_stark.gui import SsAcStarkAdapter
from zcu_lab.v2.singleshot.amp_rabi.gui import SsAmpRabiAdapter
from zcu_lab.v2.singleshot.check.gui import CheckAdapter
from zcu_lab.v2.singleshot.ge.gui import GEAdapter
from zcu_lab.v2.singleshot.len_rabi.gui import SsLenRabiAdapter
from zcu_lab.v2.singleshot.mist.freq.gui import MistFreqAdapter
from zcu_lab.v2.singleshot.mist.power.gui import MistPowerAdapter
from zcu_lab.v2.singleshot.mist.power_freq.gui import MistPowerFreqAdapter
from zcu_lab.v2.singleshot.reset_check.gui import SsResetCheckAdapter
from zcu_lab.v2.singleshot.t1.t1.gui import SsT1Adapter
from zcu_lab.v2.singleshot.t1.t1_with_tone.gui import SsT1ToneAdapter
from zcu_lab.v2.singleshot.t1.t1_with_tone_sweep.gui import (
    SsT1ToneSweepFreqAdapter,
    SsT1ToneSweepGainAdapter,
)
from zcu_lab.v2.twotone.allxy.gui import AllXYAdapter
from zcu_lab.v2.twotone.ckp.gui import CKPAdapter
from zcu_lab.v2.twotone.fluxdep.gui import FluxDepAdapter
from zcu_lab.v2.twotone.freq.form import FORM as TWOTONE_FREQ_FORM
from zcu_lab.v2.twotone.freq.gui import FreqAdapter
from zcu_lab.v2.twotone.irb.gui import IRBAdapter
from zcu_lab.v2.twotone.power_dep.gui import PowerDepAdapter
from zcu_lab.v2.twotone.rabi.amp_rabi.gui import AmpRabiAdapter
from zcu_lab.v2.twotone.rabi.len_rabi.gui import LenRabiAdapter
from zcu_lab.v2.twotone.rb.gui import RBAdapter
from zcu_lab.v2.twotone.reset.bath.freq.form import FORM as RESET_BATH_FREQ_FORM
from zcu_lab.v2.twotone.reset.bath.freq.gui import BathFreqGainAdapter
from zcu_lab.v2.twotone.reset.bath.length.form import FORM as RESET_BATH_LENGTH_FORM
from zcu_lab.v2.twotone.reset.bath.length.gui import BathLengthAdapter
from zcu_lab.v2.twotone.reset.bath.phase.form import FORM as RESET_BATH_PHASE_FORM
from zcu_lab.v2.twotone.reset.bath.phase.gui import BathPhaseAdapter
from zcu_lab.v2.twotone.reset.dual_tone.freq.form import (
    FORM as RESET_DUAL_TONE_FREQ_FORM,
)
from zcu_lab.v2.twotone.reset.dual_tone.freq.gui import DualToneFreqAdapter
from zcu_lab.v2.twotone.reset.dual_tone.length.form import (
    FORM as RESET_DUAL_TONE_LENGTH_FORM,
)
from zcu_lab.v2.twotone.reset.dual_tone.length.gui import DualToneLengthAdapter
from zcu_lab.v2.twotone.reset.dual_tone.power.form import (
    FORM as RESET_DUAL_TONE_POWER_FORM,
)
from zcu_lab.v2.twotone.reset.dual_tone.power.gui import DualTonePowerAdapter
from zcu_lab.v2.twotone.reset.rabi_check.form import FORM as RESET_RABI_CHECK_FORM
from zcu_lab.v2.twotone.reset.rabi_check.gui import RabiCheckAdapter
from zcu_lab.v2.twotone.reset.single_tone.freq.form import (
    FORM as RESET_SINGLE_TONE_FREQ_FORM,
)
from zcu_lab.v2.twotone.reset.single_tone.freq.gui import SingleToneFreqAdapter
from zcu_lab.v2.twotone.reset.single_tone.length.form import (
    FORM as RESET_SINGLE_TONE_LENGTH_FORM,
)
from zcu_lab.v2.twotone.reset.single_tone.length.gui import SingleToneLengthAdapter
from zcu_lab.v2.twotone.ro_optimize.auto_optimize.form import FORM as RO_OPT_AUTO_FORM
from zcu_lab.v2.twotone.ro_optimize.auto_optimize.gui import RoOptAutoAdapter
from zcu_lab.v2.twotone.ro_optimize.freq.form import FORM as RO_OPT_FREQ_FORM
from zcu_lab.v2.twotone.ro_optimize.freq.gui import RoOptFreqAdapter
from zcu_lab.v2.twotone.ro_optimize.freq_gain.form import FORM as RO_OPT_FREQ_GAIN_FORM
from zcu_lab.v2.twotone.ro_optimize.freq_gain.gui import RoOptFreqGainAdapter
from zcu_lab.v2.twotone.ro_optimize.length.form import FORM as RO_OPT_LENGTH_FORM
from zcu_lab.v2.twotone.ro_optimize.length.gui import RoOptLengthAdapter
from zcu_lab.v2.twotone.ro_optimize.power.form import FORM as RO_OPT_POWER_FORM
from zcu_lab.v2.twotone.ro_optimize.power.gui import RoOptPowerAdapter
from zcu_lab.v2.twotone.time_domain.t1.gui import T1Adapter
from zcu_lab.v2.twotone.time_domain.t2echo.gui import T2EchoAdapter
from zcu_lab.v2.twotone.time_domain.t2ramsey.gui import T2RamseyAdapter
from zcu_lab.v2.twotone.zigzag.gui import ZigZagAdapter
from zcu_lab.v2.twotone.zigzag_sweep.gui import (
    ZigZagScanFreqAdapter,
    ZigZagScanGainAdapter,
)

from . import ALIAS_PRESET as ALIAS_PRESET

ADAPTERS: dict[str, type[BaseAdapter[Any, Any, Any, Any]]] = {
    "lookback": LookbackAdapter,
    "fake/freq": FakeFreqAdapter,
    "onetone/freq": OneToneFreqAdapter,
    "onetone/power_dep": OneTonePowerDepAdapter,
    "onetone/flux_dep": OneToneFluxDepAdapter,
    "twotone/freq": FreqAdapter,
    "twotone/ckp": CKPAdapter,
    "twotone/power_dep": PowerDepAdapter,
    "twotone/flux_dep": FluxDepAdapter,
    "twotone/rabi/amp_rabi": AmpRabiAdapter,
    "twotone/rabi/len_rabi": LenRabiAdapter,
    "twotone/reset/single_tone/freq": SingleToneFreqAdapter,
    "twotone/reset/single_tone/length": SingleToneLengthAdapter,
    "twotone/reset/dual_tone/freq": DualToneFreqAdapter,
    "twotone/reset/dual_tone/length": DualToneLengthAdapter,
    "twotone/reset/dual_tone/power": DualTonePowerAdapter,
    "twotone/reset/bath/freq_gain": BathFreqGainAdapter,
    "twotone/reset/bath/length": BathLengthAdapter,
    "twotone/reset/bath/phase": BathPhaseAdapter,
    "twotone/reset/check": RabiCheckAdapter,
    "twotone/ro_optimize/freq": RoOptFreqAdapter,
    "twotone/ro_optimize/power": RoOptPowerAdapter,
    "twotone/ro_optimize/length": RoOptLengthAdapter,
    "twotone/ro_optimize/freq_gain": RoOptFreqGainAdapter,
    "twotone/ro_optimize/auto": RoOptAutoAdapter,
    "twotone/t1": T1Adapter,
    "twotone/t2ramsey": T2RamseyAdapter,
    "twotone/t2echo": T2EchoAdapter,
    "twotone/zigzag": ZigZagAdapter,
    "twotone/zigzag_scan/gain": ZigZagScanGainAdapter,
    "twotone/zigzag_scan/freq": ZigZagScanFreqAdapter,
    "twotone/allxy": AllXYAdapter,
    "twotone/rb": RBAdapter,
    "twotone/irb": IRBAdapter,
    "singleshot/ge": GEAdapter,
    "singleshot/check": CheckAdapter,
    "singleshot/reset_check": SsResetCheckAdapter,
    "singleshot/len_rabi": SsLenRabiAdapter,
    "singleshot/amp_rabi": SsAmpRabiAdapter,
    "singleshot/t1": SsT1Adapter,
    "singleshot/t1_tone": SsT1ToneAdapter,
    "singleshot/t1_tone_sweep_gain": SsT1ToneSweepGainAdapter,
    "singleshot/t1_tone_sweep_freq": SsT1ToneSweepFreqAdapter,
    "singleshot/ac_stark": SsAcStarkAdapter,
    "singleshot/mist/freq": MistFreqAdapter,
    "singleshot/mist/power": MistPowerAdapter,
    "singleshot/mist/power_freq": MistPowerFreqAdapter,
    "jpa/freq": JpaFreqAdapter,
    "jpa/flux": JpaFluxAdapter,
    "jpa/power": JpaPowerAdapter,
    "jpa/auto_optimize": JpaAutoOptimizeAdapter,
    "jpa/flux_onetone": JpaFluxOneToneAdapter,
    "jpa/check": JpaCheckAdapter,
}

# Core-derived forms of migrated adapters, registered with their adapter names.
FORMS: dict[str, Form[ConfigBase]] = {
    "lookback": LOOKBACK_FORM,
    "fake/freq": FAKE_FREQ_FORM,
    "onetone/freq": ONETONE_FREQ_FORM,
    "onetone/power_dep": ONETONE_POWER_DEP_FORM,
    "onetone/flux_dep": ONETONE_FLUX_DEP_FORM,
    "twotone/freq": TWOTONE_FREQ_FORM,
    "twotone/reset/single_tone/freq": RESET_SINGLE_TONE_FREQ_FORM,
    "twotone/reset/single_tone/length": RESET_SINGLE_TONE_LENGTH_FORM,
    "twotone/reset/dual_tone/freq": RESET_DUAL_TONE_FREQ_FORM,
    "twotone/reset/dual_tone/length": RESET_DUAL_TONE_LENGTH_FORM,
    "twotone/reset/dual_tone/power": RESET_DUAL_TONE_POWER_FORM,
    "twotone/reset/bath/freq_gain": RESET_BATH_FREQ_FORM,
    "twotone/reset/bath/length": RESET_BATH_LENGTH_FORM,
    "twotone/reset/bath/phase": RESET_BATH_PHASE_FORM,
    "twotone/reset/check": RESET_RABI_CHECK_FORM,
    "twotone/ro_optimize/freq": RO_OPT_FREQ_FORM,
    "twotone/ro_optimize/power": RO_OPT_POWER_FORM,
    "twotone/ro_optimize/length": RO_OPT_LENGTH_FORM,
    "twotone/ro_optimize/freq_gain": RO_OPT_FREQ_GAIN_FORM,
    "twotone/ro_optimize/auto": RO_OPT_AUTO_FORM,
    "jpa/freq": JPA_FREQ_FORM,
    "jpa/flux": JPA_FLUX_FORM,
    "jpa/power": JPA_POWER_FORM,
    "jpa/auto_optimize": JPA_AUTO_OPTIMIZE_FORM,
    "jpa/flux_onetone": JPA_FLUX_ONETONE_FORM,
    "jpa/check": JPA_CHECK_FORM,
}


def register_all(
    registry: Registry,
    *,
    template_catalog: TemplateCatalog | None = None,
    components: ComponentRegistry | None = None,
) -> None:
    """Register adapters into the caller-owned registry.

    The registry already owns its injected persistent template definitions.
    Pass their caller-owned TemplateCatalog receiver as ``template_catalog`` at
    startup to populate the library creation menu from ``registry.templates``.
    Pass the shared ComponentRegistry as ``components`` at startup to register
    component models and roles. Omit these startup receivers on reload to
    preserve identities. Duplicate entries and invalid definitions raise the
    receiving catalog's registration errors; duplicate adapters raise ValueError.
    Each FORMS entry is registered with its adapter, binding registry.templates;
    an unknown template id or incompatible seed raises ValueError naming them.
    """
    if components is not None:
        register_components(components)
    if template_catalog is not None:
        template_catalog.register_definitions(registry.templates)
    for name, cls in ADAPTERS.items():
        registry.register(name, cls, form=FORMS.get(name))

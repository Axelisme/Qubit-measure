"""Concrete legacy parameter and deferred module declarations for this lab.

Values stay in legacy working units. This profile explicitly creates R1, Q1 and
J1, with Q1.resonator pointing to R1; it does not infer topology from filenames.
Flux values require explicit unit keys in the same legacy JSON. No cfg is built
or evaluated, and no definitions are registered on import.
"""

from collections.abc import Mapping
from typing import Literal

from zcu_tools.datafile import VariableSchema
from zcu_tools.resources.storage_migration import KeyRule, MigrationMapping, ModuleRule

# Reviewed decisions: D120-D129, D133-D134. Changes to any rules, seeds,
# roles or native declarations require new revisions for both kind profiles.
# Never reuse a revision across profiles: resume compares this exact identity.
_KIND_REVISIONS = {"qubit/fluxonium": "1.2", "qubit/transmon": "1.3"}

_VALUE_PATHS = (
    ("r_f", "R1.freq"),
    ("rf_w", "R1.kappa"),
    ("res_ch", "R1.wiring.ch"),
    ("ro_ch", "R1.wiring.ro_ch"),
    ("timeFly", "R1.ext.time_of_fly"),
    ("res_edelay_calibration", "R1.ext.edelay"),
    ("theta0", "R1.ext.theta0"),
    ("res_probe_len", "R1.ext.default_probe_length"),
    ("q_f", "Q1.freq"),
    ("qf_w", "Q1.kappa"),
    ("t1", "Q1.t1"),
    ("t2r", "Q1.t2r"),
    ("t2e", "Q1.t2e"),
    ("rabi_f", "Q1.ext.rabi_freq"),
    ("t1_with_tone", "Q1.ext.t1_with_tone"),
    ("ac_stark_coeff", "Q1.ext.ac_stark_coeff"),
    ("cutoff", "Q1.ext.cutoff"),
    ("chi", "Q1.ext.chi"),
    ("best_ro_freq", "Q1.ext.best_ro_freq"),
    ("best_ro_gain", "Q1.ext.best_ro_gain"),
    ("best_ro_length", "Q1.ext.best_ro_length"),
    ("g_center", "Q1.ext.readout_cal.g_center"),
    ("e_center", "Q1.ext.readout_cal.e_center"),
    ("ge_radius", "Q1.ext.readout_cal.radius"),
    ("ge_s", "Q1.ext.readout_cal.sigma"),
    ("confusion_matrix", "Q1.ext.readout_cal.confusion_matrix"),
    ("fid", "Q1.ext.readout_cal.fidelity"),
    ("resetf_w", "Q1.ext.reset_fwhm"),
    ("best_jpa_freq", "J1.pump_freq"),
    ("best_jpa_power", "J1.pump_power"),
    ("flux_unit", "general.flux_unit"),
    ("local_flux_unit", "Q1.flux_unit"),
    ("jpa_flux_unit", "J1.flux_unit"),
)

_FLUX_PATHS = (
    ("cur_A", "general.flux_value", "flux_unit"),
    ("flx_half", "Q1.flux_half", "flux_unit"),
    ("flx_period", "Q1.flux_period", "flux_unit"),
    ("flx_int", "Q1.flux_int", "flux_unit"),
    ("flux_bias", "Q1.ext.flux_bias", "flux_unit"),
    ("best_jpa_flux", "J1.flux", "jpa_flux_unit"),
)

_CHANNEL_PATHS = (
    ("qub_0_1_ch", "Q1.wiring.L01"),
    ("qub_1_4_ch", "Q1.wiring.L14"),
    ("qub_4_5_ch", "Q1.wiring.L45"),
    ("qub_5_6_ch", "Q1.wiring.L56"),
    ("lo_flux_ch", "Q1.wiring.flux"),
)

_MODULE_KEY_PATHS = (
    ("pi_gain", "qubit.module.x180.gain", "D122"),
    ("pi_len", "qubit.module.x180.waveform.length", "D122"),
    ("pi2_gain", "qubit.module.x90.gain", "D122"),
    ("pi2_len", "qubit.module.x90.waveform.length", "D122"),
    ("reset_f", "qubit.module.sideband.pulse_cfg.freq", "D125"),
    ("reset_f1", "qubit.module.dual_tone.pulse1_cfg.freq", "D125"),
    ("reset_gain1", "qubit.module.dual_tone.pulse1_cfg.gain", "D125"),
    ("reset_f2", "qubit.module.dual_tone.pulse2_cfg.freq", "D125"),
    ("reset_gain2", "qubit.module.dual_tone.pulse2_cfg.gain", "D125"),
    (
        "bathreset_freq",
        "qubit.module.bath_g.cavity_tone_cfg.freq",
        "D125: also qubit.module.bath_e.cavity_tone_cfg.freq; split cfg in 4a",
    ),
    (
        "bathreset_gain",
        "qubit.module.bath_g.cavity_tone_cfg.gain",
        "D125: also qubit.module.bath_e.cavity_tone_cfg.gain; split cfg in 4a",
    ),
    ("bathreset_max_phase", "qubit.module.bath_g.pi2_cfg.phase", "D125"),
    ("bathreset_min_phase", "qubit.module.bath_e.pi2_cfg.phase", "D125"),
)

_REMOVED = (
    ("readout_f", "D121: use resonator.freq"),
    ("qub_ch", "D122: explicit band wiring replaces ambiguous channel"),
    ("flx_bias", "D123: obsolete alias"),
    ("mA_c", "D123: obsolete flux value"),
    ("mA_e", "D123: obsolete flux value"),
    ("resetf1_w", "D125: removed reset width"),
    ("cur_jpa_A", "D126: obsolete JPA alias"),
    ("fake_peak", "D127: fake adapter uses real fields/ext"),
    ("log_scale", "D127: analysis parameter, not a stored component value"),
)

# Candidates follow existing role_table Lib priority, not source YAML order.
# Each legacy name keeps its own destination. Bath cannot choose g/e solely
# from a flat name: the report leaves its D125 split to the 4a cfg owner.
_MODULE_NAMES = (
    ("qub_probe", "Q1.pulses.qub_probe", "Q1.module.probe"),
    ("res_probe", "R1.pulses.res_probe", "R1.module.probe"),
    ("pi_amp", "Q1.pulses.pi_amp", "Q1.module.x180"),
    ("pi_len", "Q1.pulses.pi_len", "Q1.module.x180"),
    ("pi2_amp", "Q1.pulses.pi2_amp", "Q1.module.x90"),
    ("pi2_len", "Q1.pulses.pi2_len", "Q1.module.x90"),
    ("readout_dpm", "R1.readout.readout_dpm", "Q1.module.readout"),
    ("readout_rf", "R1.readout.readout_rf", "Q1.module.readout"),
    ("readout", "R1.readout.readout", "Q1.module.readout"),
    ("res_readout", "R1.readout.res_readout", "Q1.module.readout"),
    ("readout_direct", "R1.readout.readout_direct", None),
    ("reset_10", "Q1.reset.reset_10", "Q1.module.sideband"),
    ("reset_120", "Q1.reset.reset_120", "Q1.module.dual_tone"),
    ("reset_none", "Q1.reset.reset_none", None),
    ("reset_bath", "Q1.reset.reset_bath", None),
)


def build_mapping(
    *,
    qubit_kind: Literal["qubit/fluxonium", "qubit/transmon"],
    data_schemas: Mapping[tuple[str, str], tuple[VariableSchema, ...]],
    native_tags: Mapping[tuple[str, str], str],
) -> MigrationMapping:
    """Return this lab's explicit R1/Q1/J1 conversion profile without any I/O.

    qubit_kind is the caller's declared historical kind, never inferred from
    chip/qubit names. data_schemas is supplied by the composition root's fixed
    native declarations. native_tags explicitly renames selected historical
    pairs for native metadata only; omitted pairs keep the historical tag.
    Seeds carry no guessed channels, flux units or pulse
    values. Keep complete readout_cal ext containers for dotted value writes.
    No entry registration or module cfg conversion occurs here. Unsupported
    kinds raise ValueError. Missing unit evidence is pending in the converter.
    """
    if qubit_kind not in _KIND_REVISIONS:
        raise ValueError(f"Unsupported migration qubit kind: {qubit_kind}")
    rules = (
        tuple(
            KeyRule(
                old_key=key,
                target_path=path,
                action="value",
                reason="D120-D129/D133: working units unchanged",
            )
            for key, path in _VALUE_PATHS
        )
        + tuple(
            KeyRule(
                old_key=key,
                target_path=path,
                action="value",
                requires_keys=(unit_key,),
                reason="D123/D126/D129: explicit flux unit required",
            )
            for key, path, unit_key in _FLUX_PATHS
        )
        + tuple(
            KeyRule(
                old_key=key,
                target_path=path,
                action="value",
                wrap_key="ch",
                reason="D123/D129: explicit band/flux channel",
            )
            for key, path in _CHANNEL_PATHS
        )
        + tuple(
            KeyRule(
                old_key=key,
                target_path=path,
                action="stderr",
                reason="D128: working-unit stderr",
            )
            for key, path in (
                ("t1err", "Q1.t1"),
                ("t2r_err", "Q1.t2r"),
                ("t2e_err", "Q1.t2e"),
            )
        )
        + tuple(
            KeyRule(old_key=key, target_path=path, action="module", reason=reason)
            for key, path, reason in _MODULE_KEY_PATHS
        )
        + tuple(
            KeyRule(old_key=key, target_path=None, action="remove", reason=reason)
            for key, reason in _REMOVED
        )
    )
    return MigrationMapping(
        mapping_version=_KIND_REVISIONS[qubit_kind],
        components={
            "R1": {"kind": "resonator"},
            "Q1": {"kind": qubit_kind, "resonator": "R1", "ext": {"readout_cal": {}}},
            "J1": {"kind": "amplifier/jpa"},
        },
        roles={"qubit": "Q1", "resonator": "R1"},
        rules=rules,
        data_schemas=data_schemas,
        native_tags=native_tags,
        module_rules=tuple(
            ModuleRule(
                old_name=name,
                target_path=target,
                reference_path=reference,
                reason="D156: named legacy candidate; D125 bath split deferred"
                if name == "reset_bath"
                else "D156: named legacy candidate",
            )
            for name, target, reference in _MODULE_NAMES
        ),
    )

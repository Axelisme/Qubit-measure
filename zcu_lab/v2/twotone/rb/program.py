"""Hardware program construction shared by reference and interleaved RB."""

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.v2.runtime.schedule import ProgramBuilder
from zcu_tools.program.v2 import (
    ComputedPulse,
    LoadValue,
    ModularProgramV2,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Repeat,
    Reset,
    ResetCfg,
    SweepCfg,
)


class RBModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    I_pulse: PulseCfg | None = None
    X90_pulse: PulseCfg
    X180_pulse: PulseCfg
    readout: ReadoutCfg


class RBSweepCfg(ConfigBase):
    depth: SweepCfg | list[int]


def build_rb_program(
    builder: ProgramBuilder[ModularProgramV2],
    modules: RBModuleCfg,
    tables: tuple[list[int], list[int], list[int], list[int]],
) -> ModularProgramV2:
    """Compile one seed/arm with a hardware depth sweep.

    Tables are emitted by build_seed_program_tables: physical gate IDs, prefix
    lengths, and two recovery slots. Pulses retain sequence order and virtual-Z
    frame conversion; adjacent pulses are never merged. Compilation errors
    propagate to the caller. Acquisition is left to the supplied builder.
    """
    (
        rand_gate_seq,
        prefix_len_by_depth,
        recovery_gate0_by_depth,
        recovery_gate1_by_depth,
    ) = tables
    max_rand_len = max(prefix_len_by_depth, default=0)

    Id_pulse = modules.I_pulse
    X90_pulse = modules.X90_pulse
    X180_pulse = modules.X180_pulse
    MX90_pulse = X90_pulse.with_updates(phase=X90_pulse.phase + 180.0)
    Y90_pulse = X90_pulse.with_updates(phase=X90_pulse.phase + 90.0)
    Y180_pulse = X180_pulse.with_updates(phase=X180_pulse.phase + 90.0)
    MY90_pulse = X90_pulse.with_updates(phase=X90_pulse.phase - 90.0)

    if Id_pulse is None:
        Id_pulse = X90_pulse.with_updates(gain=0.0)

    gate_pulses = [
        Id_pulse,
        X90_pulse,
        X180_pulse,
        MX90_pulse,
        Y90_pulse,
        Y180_pulse,
        MY90_pulse,
    ]

    return (
        builder.add(
            LoadValue(
                "load_rand_len",
                values=prefix_len_by_depth,
                idx_reg="depth_idx",
                val_reg="rand_len",
            ),
            LoadValue(
                "load_recovery_gate_0",
                values=recovery_gate0_by_depth,
                idx_reg="depth_idx",
                val_reg="recovery_gate_0",
            ),
            LoadValue(
                "load_recovery_gate_1",
                values=recovery_gate1_by_depth,
                idx_reg="depth_idx",
                val_reg="recovery_gate_1",
            ),
            Reset("reset", cfg=modules.reset),
            Repeat(
                "rand_gate_idx",
                "rand_len",
                range_hint=(0, max_rand_len),
            ).add_content(
                [
                    LoadValue(
                        "load_rand_gate",
                        values=rand_gate_seq,
                        idx_reg="rand_gate_idx",
                        val_reg="gate_idx",
                    ),
                    ComputedPulse(
                        "basic_gate",
                        val_reg="gate_idx",
                        pulses=gate_pulses,
                    ),
                ]
            ),
            # Both recovery slots must share the full
            # gate_pulses so their total_length (max over
            # candidates) is depth-independent.
            ComputedPulse(
                "recovery_gate_0",
                val_reg="recovery_gate_0",
                pulses=gate_pulses,
            ),
            ComputedPulse(
                "recovery_gate_1",
                val_reg="recovery_gate_1",
                pulses=gate_pulses,
            ),
            Readout("readout", cfg=modules.readout),
        )
        .declare_sweep("depth_idx", len(prefix_len_by_depth))
        .build()
    )

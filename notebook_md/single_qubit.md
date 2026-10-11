---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: zcu-tools
    language: python
    name: python3
---

# Single-qubit run with a result entry

This example uses a simulated SoC and synthetic parameters. It does not connect to hardware. All output goes to the caller-selected demo roots. Do not point these roots at real measurement data.

The flow opens or creates an entry, selects a point, resolves roles, edits its library, assembles a T1 cfg, and saves and accepts the result. Individual calibration, JPA, reset, readout optimization, coherence, singleshot, MIST, and fast-flux exploration cells from the former notebook are retired.

## Imports and definitions

```python
%matplotlib widget
from pathlib import Path

from zcu_lab.components import register_all
from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg, T1Exp
from zcu_tools.cfg_model import (
    ConstWaveformCfg,
    DirectReadoutCfg,
    PulseReadoutSpec,
    PulseSpec,
)
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.cfg_assembler import CfgEnv, make_cfg
from zcu_tools.experiment.utils import make_sweep
from zcu_tools.notebook import NotebookAdapter
from zcu_tools.program.v2 import make_mock_soc
from zcu_tools.program.v2.sim import SimParams
from zcu_tools.resources.entry import ResultEntry, component_registry
from zcu_tools.simulate.fluxonium import FluxoniumPredictor

register_all(component_registry)
```

Run the registration cell once per kernel. Restart the kernel before repeating it; duplicate registration fails rather than silently replacing definitions.

## Synthetic environment

SimParams frequencies are GHz; pulse and component frequencies are MHz. Coherence and pulse lengths use microseconds. The simulated operating value is zero, at half flux for this model. Bind that value explicitly with set_flux_source; an unbound mock uses reduced flux 1.0, not this operating point.

```python
sim = SimParams(
    EJ=8.0, EC=1.0, EL=0.5,
    flux_half=0.0, flux_period=1.0,
    T1=30.0, T2=40.0, T2_star=20.0,
    bare_rf=7.0, g=0.1, Ql=5000.0, Qi=10000.0,
    snr=100.0, pi_gain_len=0.1,
    poll_latency=0.0, seed=42,
)
soc, soccfg = make_mock_soc(n_gens=2, n_readouts=1, sim=sim)
soc.set_flux_source(lambda: 0.0)
device_manager = DeviceManager()
nb_adapter = NotebookAdapter(
    soc=soc, soccfg=soccfg, device_manager=device_manager,
)
predictor = FluxoniumPredictor(
    params=(sim.EJ, sim.EC, sim.EL),
    flux_half=sim.flux_half,
    flux_period=sim.flux_period,
    flux_bias=sim.flux_bias,
)
qubit_freq = predictor.predict_freq(0.0)
```

## Entry and setup

Use `create_entry=True` only for a new destination. Creation rejects existing entry directories. To revisit this demo, set it to False. Open requires a complete new-format entry and does not convert old data.

```python
result_root = Path("demo_storage/results")
database_root = Path("demo_storage/Database")
entry_name = "single_qubit_demo"
create_entry = True

if create_entry:
    entry = ResultEntry.create(
        entry_name, result_root=result_root, database_root=database_root,
    )
    entry.setup.add_component(
        "R1", kind="resonator", freq=sim.bare_rf * 1000.0,
        wiring={"ch": 0, "ro_ch": 0},
    )
    entry.setup.add_component(
        "Q1", kind="qubit/fluxonium", freq=qubit_freq, t1=sim.T1,
        resonator="R1", wiring={"L01": {"ch": 1}},
        flux_half=sim.flux_half, flux_period=sim.flux_period, flux_unit="A",
    )
else:
    entry = ResultEntry.open(
        entry_name, result_root=result_root, database_root=database_root,
    )
```

Setup seeds new points. Editing it does not change an existing point. Entry names and point labels are single path segments; they do not encode a required physical quantity.

## Point and roles

Set `create_point=False` when reopening an existing point. A new point copies setup and starts with an empty library. To copy an existing point and its library instead, use `entry.new_point("other_label", clone_from=point_label)`.

```python
point_label = "half_flux"
create_point = True

if create_point:
    point = entry.new_point(point_label, description="Synthetic T1 demo")
    with point.edit() as draft:
        draft.set("general.ext.flux_value", 0.0)
        draft.set("general.ext.flux_unit", "A")
else:
    point = entry.use_point(point_label)

roles = point.resolve(["qubit", "resonator"], focus="Q1")
roles.qubit.freq, roles.resonator.freq
```

Role names stay bound to this point. Reads use its cached snapshot; `point.refresh()` explicitly reloads it. Component and general flux values must use the same unit. The container does not convert units.

## Point-local library

Library keys identify a component, family, and item. Persistent Specs may contain inline child Specs or concrete keys; the library materializes them through a fixed snapshot. These demo pulses are synthetic, not hardware calibration.

```python
library = entry.module_library(point_label)

if create_point:
    pi_spec = PulseSpec(
        ch=roles.qubit.wiring.L01.ch, nqz=1,
        freq=roles.qubit.freq, gain=1.0,
        waveform=ConstWaveformCfg(length=0.1),
    )
    readout_spec = PulseReadoutSpec(
        pulse_cfg=PulseSpec(
            ch=roles.resonator.wiring.ch, nqz=2,
            freq=roles.resonator.freq, gain=0.05,
            waveform=ConstWaveformCfg(length=1.0),
        ),
        ro_cfg=DirectReadoutCfg(
            ro_ch=roles.resonator.wiring.ro_ch,
            ro_freq=roles.resonator.freq, ro_length=0.9, trig_offset=sim.timeFly,
        ),
    )
    with library.edit() as draft:
        draft.set_module("Q1.pulses.pi", pi_spec)
        draft.set_module("R1.readout.measure", readout_spec)

ml = library.snapshot()
ml.module_keys()
```

## Cfg and run

Capture a new library snapshot after edits or a point switch. CfgEnv does not refresh the library or evaluate parameter expressions. Read parameter values explicitly when building raw cfg. Omit the optional reset module to use T1Cfg's None default; make_cfg expects each supplied module to be a Spec or library key.

```python
env = CfgEnv(ml=ml, device_manager=device_manager)
cfg = make_cfg(
    {
        "modules": {
            "pi_pulse": "Q1.pulses.pi",
            "readout": "R1.readout.measure",
        },
        "sweep": {"length": make_sweep(0.0, 5.0 * roles.qubit.t1, 51)},
        "relax_delay": 5.0 * roles.qubit.t1,
    },
    T1Cfg, env, overrides={"reps": 100, "rounds": 1},
)
exp = nb_adapter(T1Exp())
record = exp.run(cfg)
analysis = exp.analyze(T1Exp.Options(dual_exp=False, skip=1), source=record)
analysis.result.t1, analysis.result.t1_err
```

## Save and accept

`save_run` requires the same record returned by this adapter's latest successful run. Loaded, hand-built, or stale records are not acquisition evidence. The point and roles must have the same storage origin.

```python
saved = exp.save_run(
    point, source=record, roles=roles, figures=analysis.figures,
    comment="Synthetic single-qubit T1 demo",
)
saved.run_id, saved.paths, saved.report
```

The run has one identity. Native data, figures, and analysis use fixed run paths; each save produces a new Labber file. Output failures raise PartialSaveError with its SavedRun handle. Ledger errors propagate separately. No automatic retry or rollback occurs.

Review the fit before executing accept. The tuple carries a working-unit value and stderr. Notebook acceptance uses ordinary role-prefixed paths, such as `qubit.t1`, not GUI `$alias` expressions.

```python
accepted_id = saved.accept(
    parameter={"qubit.t1": (analysis.result.t1, analysis.result.t1_err)},
)
point.refresh()
roles.qubit.t1, point.meta("Q1.t1"), entry.ledger.get(accepted_id)
```

Accept appends an event before committing values. Items commit separately and stop at the first error; earlier items and the event can remain. It does not change setup or another point. To repeat an experiment, keep the existing SavedRun if it is still needed and use the new run's record for the next save.

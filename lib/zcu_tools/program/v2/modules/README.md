# program/v2/modules — semantic program modules

**Last updated:** 2026-10-24

**讀者：** framework-dev。語意 program modules 的角色、組裝與邊界。

High-level cheat-sheet for `program/v2/modules/`. Read before touching this
package. Implementation detail belongs in code and tests; this file records module
roles and design boundaries.

## Purpose

`Module` objects are the semantic building blocks consumed by `ModularProgramV2`.
Each module owns a small piece of program setup in `init()` and time-ordered
runtime emission in `run()`. The module layer should preserve experiment meaning
without leaking hardware register choreography into experiment classes.

## Configuration names

In zcu_lab cores, import the runtime module and annotate settings with its
`Cfg` alias, for example `Pulse.Cfg`, `Readout.Cfg`, `Reset.Cfg`,
`PulseReadout.Cfg` or `MuxReadout.Cfg`. Concrete reset and waveform classes
also expose their paired settings as `Cfg`. These are type aliases, so Pydantic
fields and static annotations use the existing schemas without a second model.
The standalone `*Cfg` names remain available to framework code and tests.

`Readout(name, cfg)` and `Reset(name, cfg)` are factory classes. Their `__new__`
returns the concrete module from `cfg.build(name)`, not an instance of the
factory class. `Reset(name, None)` returns `None` (no reset). Module sequences
in `ModularProgramV2`, `ProgramBuilder.add`, `Repeat`, `Branch`, `Join` and
`ScanWith` skip `None` entries without emitting setup or timed operations.

## Dmem Tables

- `LoadValue` represents a scalar integer lookup table. It accepts non-negative
  values in the signed-positive int32 range and may bit-pack small values when
  `auto_compress=True`. Use it when later semantic modules interpret the loaded
  register as an ordinary scalar, such as non-uniform delay cycles.
- `LoadWord` represents raw uint32 register words. Its public values use the
  `0..2^32-1` word range; storage maps those words onto the signed int32 dmem
  representation while preserving the hardware bit pattern. It is intentionally
  uncompressed because consumers care about the exact word loaded into a register.

## Readout configuration

`PulseReadoutCfg` owns flat generator/ADC settings and a native waveform. One
semantic `freq` configures both channels. `PulseReadout` declares and schedules
the ADC itself, sends configuration at the module start, triggers at
`trig_offset`, and plays the generator at `pre_delay`. Blocking covers both the
ADC window and generator playback including `post_delay`, with clock rounding.

`Pulse` takes optional runtime `ro_ch` for QICK frequency matching. That channel
is not a cfg field. `PulseRegistry` includes it in hardware-definition identity,
so unpaired playback and ADC channels 0 and 1 cannot share a definition by
accident. Mixer declaration receives the channel only when mixing is configured.

`MuxReadoutCfg` reuses the numeric cfg_model schema. `MuxReadout` declares fixed
mixmux8 tones and explicit logical outputs on one static PFBv4, then plays one
all-tone const mask and triggers the outputs together. It protects successful
route declarations through `ModularProgramV2.protect_readout_routes`, including
when nested; unprotected routes retain QICK's declaration semantics.
`collect_iq` selects one standard averaged read using that program's actual
channel ordering and returns tone-ordered float64 I/Q. The experiment owns
complex conversion, interpretation and persistence. `collect_raw_iq(prog)` reads
that initialized program's last-round QICK buffers, normalizes each ADC's sums by
its rounded integration length, and keeps all loops, reads and paired shots in
tone order. A shared `read_index` optionally selects one read. It leaves DC
offsets and preparation meaning to the caller; unlike `collect_iq`, it does not
average reps or rounds. To retain every round, collect in the completed-round
hook. The method docstring specifies dtype, axes, units and failures. Mux has no
physical simulation support.

## Readout Runtime Words

- `PulseReadout` accepts optional `gain_val`, `freq_val`, and `ro_freq_val`
  register names when an experiment needs per-point hardware words that cannot be
  represented by a uniform `QickParam` sweep. The module consumes those registers
  explicitly and patches a scratch pulse/readout config before playback; fixed
  readout configs still use the ordinary `PulseReadoutCfg` path.
- Runtime word support is a consumer contract, not a cfg-field contract. The
  persistent readout cfg keeps semantic float/QickParam defaults that are valid as
  templates; raw word ownership remains with the `LoadWord` modules that populate
  the named registers.
- Frequency words supplied to `freq_val` and `ro_freq_val` are the final uint32
  hardware patterns applied to the wave frequency field. They already include
  generator mixer lowering and the readout downconversion sign; bare `freq2reg`
  outputs are not valid substitutes for this contract.
- Runtime pulse/readout playback reads its fixed template from wmem, applies the
  selected fields in `r_wave`, and sends that register bundle directly to the
  hardware port. The runtime path does not write and immediately re-read wmem;
  the template remains unchanged for the next sweep point.
- `TablePulseReadout` is the program-side arbitrary-frequency sweep path. It owns
  paired generator/readout uint32 tables, plays the current point through QICK's
  ordinary wmem-backed `PulseReadout`, and uses the sweep loop's exec-after hook
  to prepare the next point. Its dedicated templates prevent a table update from
  mutating a pulse shared through `PulseRegistry`; the rotated tables restore the
  first point before the next outer repetition.

## Swept pulse blocking timing

`Pulse` aligns a swept blocking duration with QICK's registered waveform
segments. Generator-cycle quantization and per-loop truncation precede
processor-clock rounding. Blocking start and each step round upward, so a
following operation cannot begin before playback ends. Overrun can accumulate
across points; it is not bounded to one processor tick for a whole sweep.
Scalar and pre/post-delay timing keep their existing rules.

## Runtime Pulse Length

- `TableLengthPulse` owns the arbitrary pulse-length hardware sweep for `const`
  and `flat_top`. It keeps one pulse template in wmem and loads both generator
  length cycles and total tProcessor duration from dmem on each hardware-loop
  point.
- A const pulse patches the template's single `w4` length field. A flat-top pulse
  reuses its fixed ramp entries and patches only the middle flat segment; its
  segment WPORT writes remain contiguous.
- Length values must be finite and strictly positive; zero fails before program
  build. Points below the generator's three-cycle minimum also fail fast. Runtime
  duration alignment places readout immediately after the selected pulse instead
  of padding every point to the maximum length.
- `Join` accepts one `TableLengthPulse` per parallel branch, with independent
  generators, a shared index register and equal table sizes. It emits all drives
  at one origin and loads one per-point maximum duration table for a common wait.
  The selected lengths, including pre/post delays, determine when readout starts;
  crossing length ranges do not cause sweep-wide padding. Mixed scalar/table
  branches and table-pulse sequences fail before build. Standalone table pulses
  retain their own wait. Hardware and semantic simulation use the same rounded
  duration tables.

## Design Boundaries

- Keep scalar values and raw hardware words as distinct module contracts. Expanding
  `LoadValue` to uint32 would make arithmetic/scalar users ambiguous and weaken
  compression assumptions.
- Dmem buffer ownership remains in `ModularProgramV2`; modules append their table
  and keep only the returned offset.
- A module that loads a register does not by itself define how a later pulse,
  readout, or simulator interprets that register. Add explicit consuming module
  contracts for those cases instead of relying on name conventions.

Readout 的跨 owner 契約見 [ADR-0074](../../../../../docs/adr/0074-flat-readout-module.md)。

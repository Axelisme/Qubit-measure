# program/v2/modules — semantic program modules

**Last updated:** 2026-10-31，fork／join 與 reference 推進守衛

**讀者：** lab-author、framework-dev、agent。語意 program modules 的宣告、發射與 owner 邊界速查。

High-level cheat-sheet for `program/v2/modules/`. Read before touching this
package. Implementation detail belongs in code and tests; this file records module
roles and design boundaries.

## Purpose

`Module` objects are declarations consumed by `TimelineProgramV2`, and semantic
building blocks consumed by `ModularProgramV2`. On the native path, authors call
`declare(module)` in `_initialize` and `emit(module, timeline=..., tag=...)` in
`_body`. Each owner supplies resource declaration and native lowering. The legacy
path still uses `init()` and time-ordered `run()`. The module layer should preserve experiment meaning
without leaking hardware register choreography into experiment classes.

## 原生寫法

Pulse、Delay、SoftDelay、DelayAuto、PulseReadout、Reset 系列、LoadValue、LoadWord、ScanWith、Branch、Join、ComputedPulse、TableLengthPulse、TablePulseReadout 與 MuxReadout 提供原生 lowering。
每次 `emit` 建立一個 `timeline.ModuleEmission`，保存該次原生指令與 tag。
Module 不是 QICK Macro，不能直接交給 `append_macro`。
同一宣告物件可重複發射，也可宣告進不同的原生 program 或 builder。
各 program 以 module 物件身分持有 pulse／waveform、runtime readout、table offset 與 loader 的宣告結果。
重新 compile 只重建該 program 的資源，不覆寫其他 program 的宣告。
`Pulse.pulse_name(prog)` 回傳指定 program 的 pulse 名稱，供原生 QICK 呼叫使用。
舊 module 的 tag 與 block_mode 不控制新路徑。
Legacy init／run 仍將狀態放在 module 上，不提供此跨 program 重用契約。

Pulse 與 PulseReadout 以實際 waveform／ADC 事件的最大結束點，加一次 post-delay，再向上取整到 tProc cycle。
Lowering 不用 `total_length` 重建時間，也不移動 reference。
省略 timeline 時選 default。其他名字先在 `_body` 用 `fork("default", into="drive")` 宣告。
同一 emission 內部事件共用起點，TwoPulseReset 的跨 channel 重疊不會被串成序列。
同 channel 的提前發射則在編譯期報錯，即使改用另一條 timeline 也一樣。
整數 register 起點也保留事件的分數尾段，直到加上 post-delay 才取整。
TwoPulseReset 合併各 pulse end 加各自 post-delay。BathReset 的 pi/2 從向上取整後的 cavity end 加 post-delay 開始。
BathReset 最後合併 qubit 與 pi/2 的實際結尾，包含各自 post-delay，不漏掉較長的 qubit。
Scalar 與可精確表示的整數仿射 sweep 保留靜態時間。
非仿射 sweep 由 program owner 以 dmem table 提供逐點精確 ceil，包含多 axis 與重複／nested emission。
Index 使用 QICK 原生 sweep 初始化、回邊與 reset，並連同所依賴的 loop counter 接受時間 producer 的全流寫入稽核。
表超過剩餘 dmem 容量時，編譯錯誤指出 emission 與所需 word 數。
Delay 與 SoftDelay 都只加 timeline 結尾。`sync()` 或 body 收尾才推進 reference。

DelayAuto 是唯一例外，未指定 timeline 時呼叫原生 delay_auto 或 delay_reg_auto。
它保留 gens／ros 選取，移動 reference 並扣減全部 timeline 結尾。
指定任何 timeline 名稱都在 emit 入口報錯，包含顯式 default。
Register cycles 必須非負。尚未被 native 部分完整消耗的 timeline，須以已知 producer bounds 扣除實際推進量。
Repeat 內仍拒絕這個 native register reference 操作，具名 timeline 的 register 時間則可用。

PulseReadout 的 runtime gain／generator frequency／ADC frequency words 沿用既有 consumer。
時間 guard 不限制這些非時間 words。Legacy init／run／block 保留供舊 program 使用。
未列出的 module 不因 legacy 可用就表示可直接 emit。

Module Branch 保留 selector 與 case 次序，lower 到 program 的 `case(index)`，None 是空 case。
Lowering 選擇 align_timeline=True，按 timeline 合併各 case 結尾，靜態取 max，runtime 取精確逐點 max。
Branch 只改結尾、不推進 reference，case 內的 native reference 操作與 Repeat 在 build 時報錯。
後續發射從該 timeline 的最大結尾開始，其他 timeline 不因這次結尾合併而延後。
Module Join 為每個 sequence fork 一條 timeline，依作者順序 emit，最後 join 合併實際 end。
它保留 sequence 及內部 Module 的作者次序，不重排。
Join 與 TwoPulseReset／BathReset 的 child 不得推進 reference。
Native delay／delay_auto、sync、register delay 與推進 reference 的 Repeat／ScanWith 在 build 時報錯，與 branch case 共用守衛及錯誤。
普通 Branch 不推進 reference，可以放進 Join。
原有 producer provenance 與翻譯後全流稽核照常執行。
同 generator 的 physical-end、全部 case 與 Repeat backedge guards 繼續檢查，不加 padding 或 reference rewind。
TableLengthPulse Join 保留獨立 generators、共用 index 與相同 table count，逐 index 合併 runtime maximum。
`prog.join` 只合併既有 timeline ends，不發射 children。

同一個宣告物件可在 body 重複使用。例如以下 `TimelineProgramV2` hooks 只累加等待結尾：

```python
from zcu_tools.program.v2.modules import Delay


def _initialize(self, cfg):
    self.gap = Delay("gap", 0.1)  # us
    self.declare(self.gap)


def _body(self, cfg):
    self.emit(self.gap, tag="gap_1")
    self.emit(self.gap, tag="gap_2")
    self.sync()  # Advance reference by the two rounded gaps.
```

兩次 emit 各有一個 emission 與 tag，不修改 module 的發射狀態。
Fork／join／Repeat 與原生 pulse 混用的範例見 [program README](../README.md#原生寫法與具名-timeline)。
設計理由與階段邊界見 [ADR-0083](../../../../../docs/adr/0083-program-macros-and-timelines.md)。

## Compiled timing 驗收

原生排程的驗收入口是 `ir.analyze_timing(ir.CompiledTimingInput(...))`。
傳入同一次編譯的 `prog_list`、`labels`、`timing_layout` 與 `compile_datamem()`。
Layout 提供 linked body boundaries 與 sweep axes，dmem 提供 table lookup 和 runtime dispatch 的實際輸入。
不從 jump 形狀推測 boundaries，也不以 legacy 輸出作正確性預期。

逐點核對 event 的實體 tProc port、cycle、`body_end_cycle` 與含 relax delay 的 `end_cycle`。
Wave events 包含 readout config 與 flat-top segments，不是一個 Module 對應一個 event。
分析結果保存 linked instruction address，不能把不同編譯的 address 當作穩定事件識別碼。

Timing analysis 只計算排程與 reference，不計算 waveform duration 或 ADC integration end。
驗收者須由輸入 waveform clocks、ADC window、post-delay 與時脈比例獨立算出結尾。
例如 384 MHz generator 的 39 clocks 是 430.08 MHz tProc 的 43.68 cycles。
起點 4 cycles 加 post-delay 0.0005 us 後，完整結尾是 `ceil(4 + 43.68 + 0.21504) = 48`。
先把 waveform 與 post 各自取整會得到 49，不能作為預期。
不同點的 end 必須逐點比較，不能用最長 entry 的 padding 隱藏 runtime 選擇或非仿射 ceil。

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

## Constructor overrides

Constructors accept `**overrides: Unpack[...]` using the exported `total=False`
TypedDicts `PulseOverrides`, `PulseReadoutOverrides`, `PulseResetOverrides`,
`TwoPulseResetOverrides` and `BathResetOverrides`. `ResetOverrides` lists the
factory's combined keys. Pyright checks keyword names and values. At runtime,
unknown or wrong-variant keys raise `TypeError`, even when their value is `None`.
Constructors copy cfg before applying overrides through its existing `set_param` routing. Numeric values and QICK sweep parameters use the same units
and waveform restrictions as `set_param`. `None` keeps the cfg value, including
zero-valued settings. The caller's cfg is unchanged. Registry hashing, read-axis
owners and lowering all use the effective cfg retained by the module.

| Constructor | Override keywords |
| --- | --- |
| `Pulse` / `PulseReset` | `freq`, `gain`, `phase`, `length` |
| `PulseReadout` / `Readout` with pulse cfg | `freq`, `gain`, `length`, `ro_length` |
| `TwoPulseReset` | `gain1`, `gain2`, `freq1`, `freq2`, `length` (both pulses) |
| `BathReset` | `qub_gain`, `qub_freq`, `qub_length`, `res_gain`, `res_freq`, `res_length`, `pi2_phase` |

`Reset` forwards the keywords for its selected variant. Frequencies are MHz,
lengths are us, phases are degrees and gains are normalized. `MuxReadout` does
not accept override keywords; `Readout` with mux cfg rejects non-None overrides.
Unsupported waveform length sweeps raise `ValueError` at construction.
Accepting a `ro_length` sweep value does not mean QICK can compile an ADC-window
sweep. That existing compile error is unchanged, just as with `set_param`.

`cfg.set_param(name, value)` remains available and mutates that cfg. Use a
caller-owned copy when deliberately updating cfg before constructing modules.

## Dmem Tables

- `LoadValue` represents a scalar integer lookup table. It accepts non-negative
  values in the signed-positive int32 range and may bit-pack small values when
  `auto_compress=True`. Use it when later semantic modules interpret the loaded
  register as an ordinary scalar, such as non-uniform delay cycles.
- `LoadWord` represents raw uint32 register words. Its public values use the
  `0..2^32-1` word range; storage maps those words onto the signed int32 dmem
  representation while preserving the hardware bit pattern. It is intentionally
  uncompressed because consumers care about the exact word loaded into a register.
- `ScanWith` supports native `declare`/`emit`. Each occurrence creates its own
  fixed Repeat counter, loads the scalar, then executes children in order.
  Repeated emissions reuse the declared table and destination register.
  Repeat-owned metadata and matching-writer audits certify the index bounds;
  native lowering does not pass the legacy `range_hint`.
- `LoadValue.lower_index` accepts the index for the current occurrence.
  `TimelineProgramV2.emission_name` allocates occurrence-local names only during
  lowering, so repeated or nested scans do not share a counter.
- Both loaders support native `declare`/`emit` without advancing timeline ends.
  `use_existed=True` leaves the destination declaration to the caller. An empty
  `LoadValue` still declares its destination unless reused; an empty `LoadWord`
  declares nothing. Neither empty loader allocates dmem or initializes a value.
- When a loaded value is consumed as time, the loader certifies its output range
  and the index range `0..len(values)-1`. The compiler checks all registered index
  writes, including later writes, Branch paths and loop backedges. A reset after
  an invalid lookup cannot hide it. Non-time consumers retain their scalar or
  raw-word contract without enabling the timing bounds checks.

## Readout configuration

Native conversion preserves the optional classifier on pulse readout and each
mux tone. The shared cfg_model schemas define circle/linear calibration in
normalized ADC coordinates without IQ-offset subtraction. Modules retain this
data but do not apply classification during declaration or playback.
`readout.circle_classifier(cfg, ro_ch=...)` returns a detached circle for one
ADC channel number, not a tone position. It accepts centers without radius or
matrix and rejects unknown ADCs, missing classifiers and linear mode with an
ADC-named ValueError. Its normalized raw ADC coordinates retain IQ offset.

`v2.shot_sums(prog, axes)` resolves compiled read labels for `Classify` and `IQ`
axes. Timeline labels use emit tags or untagged module names; modular labels
use module names. Mux axes select the cfg tone position. Each Classify axis
snapshots its explicit or cfg classifier. Circle emits g/e/other, linear g/e,
and centroid3 g/e/f. Circle requires a GE Post radius.
The processor sums ordered shot-vector outer products across reps, and
`acquire_summed` adds completed rounds without normalization.
`split_sums` converts a final IQ axis into counts and complex means.
Zero or unmeasured NaN counts give NaN means without divide warnings.

`PulseReadoutCfg` owns flat generator/ADC settings and a native waveform. One
semantic `freq` configures both channels. `PulseReadout` declares and schedules
the ADC itself, sends configuration at the module start, triggers at
`trig_offset`, and plays the generator at `pre_delay`. Blocking covers both the
ADC window and generator playback including `post_delay`, with clock rounding.
`PulseReadoutCfg.nominal_duration_us()` returns the scalar completion time without
clock rounding or program initialization. It covers the later ADC/generator end
for offline shot-time estimates; it does not replace runtime blocking timing.

`Pulse` takes optional runtime `ro_ch` for QICK frequency matching. That channel
is not a cfg field. `PulseRegistry` includes it in hardware-definition identity,
so unpaired playback and ADC channels 0 and 1 cannot share a definition by
accident. Mixer declaration receives the channel only when mixing is configured.

`MuxReadoutCfg` reuses the numeric cfg_model schema. `MuxReadout` declares fixed
mixmux8 tones and explicit logical outputs on one static PFBv4, then plays one
all-tone const mask and triggers the outputs together. Timeline programs share
identical ADC/generator resources and reject differences with both module objects
and field names. Emission delays and classifier/description metadata do not affect
resource identity. Static Mux does not switch runtime readout configurations.
Legacy `ModularProgramV2.protect_readout_routes` protects successful declarations,
including when nested; its unprotected routes retain QICK overwrite semantics.
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

`centroid3_classifier(cfg, ro_ch=...)` 由 native Pulse 或 Mux cfg 取得 detached g/e/f 校準。
它依 ADC 編號查詢，不使用 tone 位置或 live library。Missing、錯型與未知 ADC 都 raise ValueError。
Centroid3 的第三類是 f，不是 circle 的 other。需要 Other 的 consumer 須明確要求 circle。

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

## Legacy swept pulse blocking timing

`Pulse` aligns a swept blocking duration with QICK's registered waveform
segments. Generator-cycle quantization and per-loop truncation precede
processor-clock rounding. Blocking start and each step round upward, so a
following operation cannot begin before playback ends. Overrun can accumulate
across points; it is not bounded to one processor tick for a whole sweep.
Scalar and pre/post-delay timing keep their existing rules.

## Runtime Pulse Length

- `TableLengthPulse` owns the arbitrary pulse-length hardware sweep for `const`
  and `flat_top`. Native declaration keeps one template and a generator-length
  table. Native playback resolves the actual selected waveform end, adds pre/post
  once and ceils the aggregate once. Only legacy init/run uses a separate total
  tProcessor duration table.
- A const pulse patches the template's single `w4` length field. A flat-top pulse
  reuses its fixed ramp entries and patches only the middle flat segment; its
  segment WPORT writes remain contiguous.
- Length values must be finite and strictly positive; zero fails before program
  build. Points below the generator's three-cycle minimum also fail fast. Runtime
  duration alignment places readout immediately after the selected pulse instead
  of padding every point to the maximum length.
- Legacy `Join` accepts one `TableLengthPulse` per parallel branch, with independent
  generators, a shared index register and equal table sizes. It emits all drives
  at one origin and loads one per-point maximum duration table for a common wait.
  The selected lengths, including pre/post delays, determine when readout starts;
  crossing length ranges do not cause sweep-wide padding. Mixed scalar/table
  branches and table-pulse sequences fail before build. Standalone table pulses
  retain their own wait. Hardware and semantic simulation use the same rounded
  duration tables.

Native `ComputedPulse` selects each candidate waveform together with its own
post-delay. Fractional playback ends and selected posts stay separate until the
program owner ceils their aggregate. Candidate posts may differ or use native
sweeps. Neither selected owner pads every point to the longest entry.
Negative posts may put the aggregate before its origin. The physical channel end
still guards subsequent playback. Internal offsets use signed32 cycles; register
time consumers still require nonnegative cycles. Program owner compares raw
selected ends and paired aggregate ends to retain a coverage debt. Known positive
timeline advances consume that debt, so a compensating delay restores legal
same-channel placement without changing the scheduled end or reference.

`LoadValue.lower_selected` captures the index at the lookup position and returns
the declared table relation. Timeline playback checks the actual address/length
producer against that relation. Later author selector changes do not change the
already selected playback or its post. Every write to the playback register and
captured index must belong to that mapping, including Branch cases, backedges
and resets. An in-range constant or different-table write still fails with the
register and instruction position. Ordinary scalar and uint32 word consumers
keep their existing contracts.

## Design Boundaries

Legacy programs and composite Modules call `Module.emit(prog, t)` after `init`.
It runs the existing `run` behavior inside `prog.source_scope(module.name)`.
Each author-provided child emission gets its own nested diagnostic source, even
when pulse registration shares a waveform or `PulseCfg`. Ending a child restores
its parent source. Internal helpers run directly under their semantic owner's
source. `ScanWith` keeps its loop and loader under the scan owner while authored
children retain their own scopes. Anonymous `Join` scheduling keeps an empty
source, and its table-pulse children retain their author names without separate
blocking waits. These annotations add no machine instructions. The snapshot and
optimizer-input distinction is defined in [IR compile snapshot](../ir/README.md#pre-optimization-compile-snapshot).

- Keep scalar values and raw hardware words as distinct module contracts. Expanding
  `LoadValue` to uint32 would make arithmetic/scalar users ambiguous and weaken
  compression assumptions.
- `ProgramResourcesMixin` owns the dmem buffer shared by `TimelineProgramV2`
  and `ModularProgramV2`; modules append their table and keep only the returned offset.
- A module that loads a register does not by itself define how a later pulse,
  readout, or simulator interprets that register. Add explicit consuming module
  contracts for those cases instead of relying on name conventions.

Readout 的跨 owner 契約見 [ADR-0074](../../../../../docs/adr/0074-flat-readout-module.md)。

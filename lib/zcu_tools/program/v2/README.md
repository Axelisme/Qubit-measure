# README - program/v2

**Last updated:** 2026-10-31，軸分量、Abs 範圍證明與 runtime recipe

**讀者：** lab-author、framework-dev、agent。原生 V2 program 寫法與 module、readout、acquisition 契約速查。

## Timeline rounding

Timeline event starts now ceil to tProc cycles instead of nearest rounding.
Each operation quantizes its result once. Ceil treats values within 1e-10 tProc
cycles of the nearest integer as that integer to absorb floating-point conversion
noise. For example, 119/430.08 us remains 119 cycles; positive tails outside that
tolerance round up. Static waveform and ADC lengths use rational clock conversion
before the aggregate ceil. Author sweeps retain QICK
native start rounding and truncated steps; they do not allocate pointwise time
tables. Runtime cross-grid generator lengths use k=8 fixed-point upper conversion.
The compiler chooses W to minimize canonical signed-digit terms, breaking ties
with the smaller W. For every integer length up to 500 us, ceil(n*W/256) is never
earlier than the physical end and is at most 1 us late. The same W applies beyond
500 us, but the 1 us error bound does not. Each operation adds its post before the
final ceil. Static lengths and integer-affine tProc lengths stay exact.

Conversion uses shifts of at most 15 bits and addition/subtraction, without ARITH,
DIV or time tables. A clock ratio of one needs no conversion instructions.
The compiler rejects signed32 overflow in the product or any intermediate at build.
The existing native length sweeps and selected waveform/length tables supply
runtime lengths. Timeline length-register playback still requires selection
metadata. This does not add an arbitrary author length-register or ADC length API.
The conversion rule also gives W=359 for ratio7/5 and W=2872 for ratio56/5, but
ADC/PFB runtime length consumers are not exposed by the current API.

Known scalar and affine time getters return actual quantized values. Runtime-only
and non-affine time getters raise RuntimeError. Author-declared data tables and IR
dispatch jump-address tables remain supported.

## Testing & Type Checking Conventions

- **Type Narrowing**: When a mock or abstract base class type is too generic in tests (e.g., retrieving an item from an `IRNode.insts` list where you expect a `RegWriteInst`), use an explicit assertion (`assert isinstance(item, RegWriteInst)`) rather than `cast()`. This ensures Pyright can perform type narrowing safely while maintaining a runtime guarantee.
- **Mock Overrides**: When testing edge cases that intentionally return unexpected types to trigger errors, prefer a narrow helper/protocol or an explicit runtime assertion at the boundary. Do not widen signatures to `Any` or use `cast()` just to silence the type checker unless there is a reviewed reason.
- **Mock Interfaces**: Use type-compatible fakes or an existing typed seam when passing test collaborators to functions. Do not add type suppressions or `cast()` to silence an incompatible interface.
- **Program Emission Tests**: Unit tests that assert emitted program actions use a typed `ProgramTrace` adapter rather than loose `MagicMock` program objects. The trace records semantic program intents (readout declarations, triggers, pulses, loops, delays, register writes, and jumps) without acting as a second compiler or IR parser.
- **MagicMock Scope**: Keep `MagicMock` for non-program collaborators, child-module spy hooks, and deliberate error-injection tests. Do not use it as the default stand-in for a program object when the test is asserting emitted program behavior.
- **Offline acquisition**: `make_mock_soccfg(n_gens, n_readouts)` builds only a compile profile. Explicit environments use `SimulationSoc` and `SimulationBinding`; ordinary acquisition tests may inject a deterministic transport. `ProgramTrace` records emission and does not replace real acquire coverage.
- **QICK External Types**: `QickParam` runtime objects expose methods/attributes (`is_sweep`, `minval`, `maxval`, `start`, `spans`, `to_array`) that can be missing from third-party stubs. Program code narrows through `program.v2.utils.is_qick_param()` / `QickParamLike` instead of local `cast()` calls, so every use has a runtime guard.
- **Cfg Contracts**: `AbsModuleCfg` and `AbsWaveformCfg` are abstract contracts. Concrete cfg classes implement both `build()` and `set_param()`; unknown parameter names raise `ValueError` rather than being ignored. Pulse, PulseReadout and reset constructors also accept keyword-only scalar/sweep overrides on a detached effective cfg; `None` keeps the cfg value. Registry identity and lowering use that effective cfg. `set_param` remains supported and mutates the cfg on which it is called. See the [constructor parameter table](modules/README.md#constructor-overrides). Accepting `ro_length` sweeps does not add QICK ADC-window sweep support; both setter and constructor paths retain the existing compile error. [`tests/cfg_model/test_set_param_contract.py`](../../../../tests/cfg_model/test_set_param_contract.py) is the cross-family audit for this fail-fast rule.

## Gain sweeps

`GainSweep` annotates a `SweepCfg` whose start, stop and computed last point must be in [-1, 1]. Values are DAC full-scale fractions in a.u. The same validator runs on Form coordinate representations after their endpoint projection. For example, start=-1, stop=1.1, step=0.5 is accepted when the actual last point is 1.0. Range + Expts retains its stop, so stop=1.1 is rejected. Ordinary `SweepCfg` remains unit-independent.

## Fixed offline profile

`make_mock_soccfg(n_gens=2, n_readouts=1)` accepts nonnegative channel counts.
It builds no hardware, acquisition source or noise fallback. The profile uses
ZCU216-shaped metadata, tProc v2 at 430.08 MHz and the installed QICK assembly
revision. Generators are `axis_signal_gen_v6`, with 32-bit DDS/phase,
12288 MHz DDS, 384 MHz fabric, 16 envelope samples/clock, max gain word 32766
and 65536 envelope samples. ADCs are `axis_readout_v2`, with 32-bit DDS,
2457.6 MHz DDS, 307.2 MHz output and 8192-sample accumulated/decimated buffers.
Generator `tproc_ch` and ADC `tproc_ctrl` occupy distinct wave-output ports.
ADC `tproc_ch` indexes readback inputs, a separate namespace.
Ordinary ADCs use distinct bits on their shared trigger output port.
The ordinary profile does not offer mixers, mux channels or weighted readouts.
`make_mux_mock_soccfg(n_gens, n_readouts, gen_ch=..., ro_chs=...)` separately
selects one fixed mixmux8 generator and 1..8 unique static PFBv4 output routes.
Its generator DDS is 1536 MHz and ADC output is 76.8 MHz. Other channels keep
the ordinary profile. Neither helper permits arbitrary clock/firmware overrides. These are virtual
compile capabilities, not a claim about an attached board. Clocks, DDS, memory
limits and channel types are not override parameters. Explicit `(gen_ch, ro_ch)`
wiring belongs to `SimulationSoc`; actually using an unwired channel fails.

See [simulation](simulation/README.md) for the shared numeric contract, trusted
Python builder, ownership, seeded white noise and the supported minimal readout.

## IR System & Hardware Alignment

See the [tProc v2 ASM reference](../../../../docs/reference/tprocv2-asm-notes.md)
for instructions, registers, timing and waveform memory limits.

### Packed data-memory lookups

tProc v2's ALU uses only the low four bits of a shift-count operand, including
register operands. A single shift cannot extract slots beginning at bit 16 or
higher. `LoadValue` retains full 32-bit packing and decomposes extraction into
shifts of at most 15 bits. Lookup regression tests execute compiled register and
data-memory operations using this hardware limit; semantic physical simulation
alone does not validate the emitted extraction instructions.

### Instruction Address Increment (`addr_inc`)

In tProc v2, most instructions occupy 1 program memory word (`addr_inc=1`). However, some instructions are macros or expanded directives that occupy multiple words.

- **WAIT**: Occupies **2** words (`addr_inc=2`). This is because it is a directive that expands to a comparison and a conditional jump in hardware.
- **Labels & Jump Addresses**: When calculating label addresses (linking), we MUST account for `addr_inc`. If a `WAIT` instruction is at address `N`, the following instruction will be at `N+2`.

### IR Hierarchical Structure

The IR is divided into three layers to decouple high-level program structure from low-level hardware instructions:

1. **Structural IR (`IRNode`)**: Represents high-level constructs like loops (`IRLoop`), branches (`IRBranch`), and dispatch tables (`IRDispatch`), as well as structural containers (`BlockNode`).
2. **Basic Blocks (`BasicBlockNode`)**: A leaf `IRNode` representing a straight-line sequence of atomic instructions. It holds `labels` for entry points, `insts` for linear execution, and an optional terminal `branch` for control flow.
3. **Linear Instructions (`Instruction`)**: Represents atomic hardware operations (e.g., `RegWriteInst`, `WaitInst`) or meta markers (`MetaInst`, `LabelInst`). **Note: `Instruction` does NOT inherit from `IRNode`.**

### IR Pipeline & Linking

- **`IRPipeLine`**: A U-shape single-pass optimizer (see `ir/README.md` for the full flow): lex → ChunkList opt → parse → IR-tree opt → unparse → strip structural meta → ChunkList opt → flatten. IR optimization is **optional** — the macro/module layer already emits runnable ASM, and `disable_all_opt` is a clean bypass.
- **`IRLinker.link()`**: Takes a flattened instruction stream, assigns physical addresses (`P_ADDR`) to each instruction based on `addr_inc`, and resolves label addresses. **Note: `ADDR_INC` is used only during linking and is NOT emitted in the final dictionary to maintain QICK compatibility.**
- **`IRLinker.unlink()`**: Reconstructs the logical instruction stream (including label markers and meta structural markers) from QICK's flattened `prog_list`.

## Frequency word decoding

The scalar, paired and array decoders in `utils` take `cfg=PulseReadoutCfg`.
They borrow its generator/ADC channels, mixer frequency and Nyquist zone directly;
they do not change the cfg or use its waveform, gain or readout-window settings.
Encoding with `readout_freq_words` still takes those four explicit settings.
`generator_frequencies_from_words` accepts an already-rounded physical mixer
frequency and generator channel, without constructing a readout cfg.
`readout_frequencies_from_words` accepts a declared ADC channel and an explicit,
same-shaped physical reference array. It selects the nearest ADC-DDS image.
The compiled request builder supplies references from actual output effects.
The cfg-based APIs delegate to these same word decoders.

## Architecture

- `IRCompileMixin.optimize_asm()` runs the default IR pipeline (`make_default_pipeline`); the concrete pass set lives in `ir/passes/` and is documented in `ir/README.md`.
- `Nop` is preserved as an explicit instruction and is not treated as dead-code noise.
- `ir/passes/` is split by concern into sub-packages: `loop/`, `dataflow/`, `control_flow/`, `timeline/`, with a shared `base.py`.

- `MyProgramV2`: Base class for experiments, wraps QICK's `AveragerProgramV2`.
- `ModularProgramV2`: Supports pulse blocks and reusable modules.
- Program 不持有 experiment cfg。建構時以具名參數傳 `reps=1`、
  `initial_delay=1.0`、`relax_delay=1.0`，兩個 delay 的單位為 us。
- `acquire(..., rounds=1)` 與 `acquire_decimated(..., rounds=1)` 接收軟體平均次數。
  `reps` 與 `rounds` 必須為正整數，bool 不算整數。無效值在編譯或 acquire admission 前 raise ValueError。
- `acquire_summed` 以單一 processor 處理所有 ADC 的正規化 IQ，跨 round 只加總。
  `shot_sums(prog, axes)` 在 compile 後以讀出 label 解析 `Classify` 與 `IQ` 軸。
  Timeline 以 emit tag 或未標記 module 名稱識別讀出，modular 以 module 名稱識別。
  外積總和與取消契約見 [program](../README.md#summed-acquisition)。
- `BaseCustomProgramV2.make_modules()` 回傳 modules。作者自行保存組裝 modules 所需的輸入，
  program 的控制參數仍由建構子接收。
- `ir/`: Contains the Intermediate Representation for tProc v2 instructions, allowing for optimization passes and cross-module label resolution.

## 原生寫法與具名 timeline

`timeline.TimelineProgramV2` 繼承 `MyProgramV2` 加 `resources.ProgramResourcesMixin`，不經 `ModularProgramV2`。
跨 owner 分工、設計理由與階段邊界見 [ADR-0083](../../../../docs/adr/0083-program-macros-and-timelines.md)。
作者在 `_initialize(cfg)` 宣告通道、pulse、readout config、module 與 loops。
在 `_body(cfg)` 發射原生 QICK 操作或 `emit(module)`。
建構子使用 `reps`、`initial_delay`、`relax_delay`。
建構子沿用 `MyProgramV2` 的型別契約，不承接 `AveragerProgramV2` 的全部選項。
`initial_delay` 的型別為 `float`，不接受 `None`。
Shot 外殼為 `final_wait=0`、`final_delay=relax_delay`。
全部輸出共用既有 IR compiler。Boundary metadata 可能改變指令排列，不要求新舊逐位相同。
`TimelineBuilder` 從 package root 匯出，只記錄動作，`build()` 產生並編譯 `TimelineProgramV2`。
它不接收 SoC，不取數，也不依賴 experiment runtime。
Builder 依物件身分在第一次 emit 時記錄 module 宣告。
同一 module 物件可跨 builder 與原生 program 重用，不要在錄製或編譯後修改它。
每個 program 依 module 物件身分保存宣告結果，compile 時重新配置自己的資源。
Pulse 在各 program 內按定義去重，同名不同 cfg 的 Pulse 使用不同 waveform 名稱。
原生 QICK 呼叫需要 pulse 名稱時，用 `pulse.pulse_name(prog)` 取得該 program 的宣告結果。
Legacy `ModularProgramV2` 仍將宣告狀態保存在 module 上，不適用跨 program 重用。
同一物件重複 emit 沿用宣告，build 後所有錄製與 build 呼叫都 raise ValueError。
Builder 與 program 的 `add_sweep(name, spec)` 宣告一個 top-level loop，回傳均勻掃描的 QICK 參數。
Builder 的原生布林控制選項只接受 keyword，其餘參數沿用同名 program 方法。
Builder 的 `fork(source, into=...)` 與 `join(*sources, into=...)` 是普通記帳方法。
`with builder.repeat(...)` 與 `with builder.branch(by=..., align_timeline=...)` 錄製巢狀動作。
Branch 回傳公開的 `BuilderBranchCases` handle，用 `with branch.case(index):` 錄製指定 sweep 點的動作。
Index 是 zero-based，可不按順序宣告。未宣告的點不發射動作，重複或超界 index 報錯。
在 case 內仍呼叫原 builder 的 emit、delay_timeline 或其他不推進 reference 的 body 方法。
關閉全部錄製 context 後才能 build，否則 raise ValueError，builder 仍可繼續錄製。
Build 重播 program 的 context，沿用其 timeline 回饋、局部名字生命週期與錯誤訊息。
`SimulationSoc` acquisition 由 compiled frontend 解讀兩種 program 的最佳化前 stream。
既有 acquire 擴充仍由共同基底提供。

`TimelineProgramV2` 的 module 與原生直接 generator／ADC 宣告沿用 QICK 規則。
後宣告覆蓋先宣告，由作者負責設定是否一致。
需要宣告資訊時，直接讀 QICK 的 `gen_chs`／`ro_chs`。
tProc readout 用各自具名 config 切換 freq、phase、outsel 與 generator 頻率匹配，每次 trigger 使用當次設定。
原生 `add_readoutconfig` 保留 QICK 的 template 契約，不限制跨 template 的 mode、phrst 或 length。
靜態 readout 不支援 runtime 切換。
每 shot trigger 次數仍由 compiled 總數契約檢查，不按個別 module 的 emit 次數拒絕共用。
Timeline module 不自動保護 route。作者明確呼叫繼承的 `protect_readout_routes` 後，該 route 不可重新宣告。
不同 owner 重複 explicit protect 仍報錯，失敗不保留部分 protection。
Register 名稱與地址仍歸屬到宣告物件，衝突訊息列出兩個 module 名稱與物件。
明示 register 借用與 Pulse registry 的 template 共用規則不變。
`ModularProgramV2` 的宣告與 route protection 規則不變。
Body 以 `fork` 建立各 child timeline，依作者順序 `emit`，最後 `join` 合併結尾。

每條 timeline 只保存相對 reference 的結尾。
Default 是普通 timeline，不是 reference。其他名字先在 `_body` 用 `fork(source, into=...)` 宣告。
Source 必須存在，into 接受一個新名字或名字 list。未知來源、重名與未宣告使用都報錯。
帶 timeline 的 pulse／trigger 在結尾加 t 發射，然後更新結尾。
同 channel 的 pulse 提前開始時編譯報錯，不能用另一條 timeline 繞過。
Config 是瞬時事件，結尾更新到發射時間 E+t，不加 config length 或其他 ADC window。
未進入 timeline 範圍且未帶 timeline 的原生指令保留 QICK 行為。

`with builder.timeline(name)` 與 `with self.timeline(name)` 選擇範圍內的預設 timeline。
省略 `timeline=` 或傳入 `None` 時，下列入口都使用此名字：

- `emit`、`pulse`、`trigger`。
- `send_readoutconfig`、`send_readoutconfig_from_regs`。
- `pulse_from_regs`、`pulse_by_reg`、`pulse_from_length_reg`。
- `delay_timeline`、`sync` 與 `repeat` 的母 timeline。

`pulse_by_reg` 與 `pulse_from_length_reg` 選到 timeline 時仍須提供配對的 `selection` metadata。
範圍外省略 timeline 的原生輸出保留 QICK 時間行為，不改選 `"default"`。
顯式名字優先，包含 `"default"`。巢狀範圍覆蓋外層，離開後還原。
名字必須是 default 或已 fork 的 timeline，空範圍也會在編譯時驗證名字。
範圍不 fork、不 join，也不移動 reference。
`branch` 必填 keyword-only `by` 與 `align_timeline`。
`timeline=None` 選目前範圍作為母 timeline，case 另建自己的預設名字。
原生 `delay` 與 `delay_auto` 仍只推進 reference，不受範圍影響。

```python
builder.fork("default", into="drive")
with builder.timeline("drive"):
    builder.emit(drive)
    builder.delay_timeline(0.1)
    builder.sync()
    with builder.repeat("pairs", 2):
        builder.emit(drive)
        builder.sync()
    builder.sync()
builder.emit(drive)  # Uses default again.
```

`delay_timeline(t, timeline="drive")` 只將結尾加 t，不移動 reference。
`sync(timeline="drive")` 才將 reference 移到該結尾，並扣減全部 timeline，最低為零。
原生 delay／delay_auto／resync 也經公開 timestamp decrement 扣減全部結尾。
可靜態決定的結尾保留 scalar／`QickParam`。跨零 floor、交叉 max 與 path-selected end 才物化成 runtime cycle register。
Runtime 算術只用無 flag 的 ABS／ASR／加減。End 限於 `0..2**31-1`，中間值也須符合 signed 32-bit range。
每個 sweep／shot 重新建立所需狀態。`delay_timeline("duration", ...)` 讀取非負 tProc cycle register，仍不推進 reference。
作者可用 `write_reg_op` 或 `read_dmem` 自行準備時間，不必先登記 producer bounds。
`range_hint=(lo, hi)` 可選，單位是整數 tProc cycles，表示作者保證的非負閉區間，不增加硬體 range check。
沒有可用 bounds 時，編譯器在 runtime 計算 max，不證明該段同 channel 不重疊，溢位也由作者負責。
原生常數、sweep 與 framework table／長度 producer 保留既有 bounds 證明與稽核。
未知時間的 tag getter 明確 raise RuntimeError，不回傳配置指令時使用的暫存值或 bounds 上界。

編譯器以軸 ID 對齊時間分量。同 loop 的 sweep、選表與定點換算共用 loop 名稱作 ID。
已知分量是一維 numpy 陣列。只有範圍的分量是獨立未知軸的係數，常數使用大小為 1 的固定軸。
作者 register 的每次寫入、量測結果選表與取整誤差各自獨立，range_hint 不建立 loop 相關性。
加減先合併同軸分量並消去零項，再加總各軸的界。證明不建立多維網格。
同單一已知軸的非線性運算逐點計算。跨軸 ceil 與定點換算保留原式，另加有界取整誤差。
跨軸 max/min 使用線性部分加有理係數的 Abs 項，允許負係數及巢狀 Abs。
正規化合併同類項，消去零係數，並在已知符號時化簡 Abs。
取界依共用軸分連通塊，在塊內分支定界並剪枝。每塊超過 20 個未定號 Abs 項時退回區間界。
退回只影響證明，不改執行值，也不把保守 slack 當成實際差距來拒絕 relax_delay。
不可變節點快取範圍，支配判斷以正規化相減式快取結果。
Runtime 結尾、支配、重疊與 overflow 都從運算式取界，不另存結尾 min/max。
Runtime recipe 保留原整數候選與捕獲值，每個 join 只物化一次實際 max。
後續操作讀取捕獲的 register，不從 Abs 證明式重建 max 或重讀已覆寫的來源。
編譯期分量與誤差不配置時間 DMEM。

Framework owner 用 `time_producer(register, CycleBounds(...), dependencies=...)` 登記自己的 lowering writes。
Table owner 以 `dependency_ranges` 登記合法 index 範圍，mapping keys 自動納入稽核。
只有時間 consumer 啟用這些前置條件。證明涵蓋全部已登記 writes，不能用 lookup 後的合法 index 覆蓋先前非法讀取。
翻譯後稽核 source、依賴與 compiler 時間 storage 的全流寫入，錯誤指出 register 與 instruction 位置；無關 raw words 不受影響。
原生 delay／delay_auto／wait_auto／resync 不增加 timeline keyword。

Module 先明確 declare，每次 emit 建立獨立的 `ModuleEmission` 與 tag。
Module 本身不是 QICK Macro，不能直接 append。
Pulse／PulseReadout 的 emit 內部事件共用所選 timeline 的結尾作為起點。
結尾取實際 waveform／ADC event end 的最大值，加一次 post-delay，整段向上取整一次。
Runtime 整數 cycle 起點也保留分數 event tail 到加 post-delay 後才取整，不先各自取整。
Reset 保留每個 child 的 post-delay。BathReset 的 pi/2 等待 cavity end 加 post-delay 向上取整後才開始，最後也計入 qubit 尾段。
這些 owner 不從 `total_length` 重建 duration。Scalar 與可精確表示的整數仿射 sweep 保留靜態時間。
作者 sweep 使用 QICK 原生 start round 與逐圈 step 截斷，不配置逐點時間 table。
Runtime 跨 generator 格點的長度使用 k=8 定點換算，aggregate 加 post 後再 ceil。
作者明確宣告的值表與 waveform／length selection tables 保留。IR dispatch 的 jump-address table 也保留，不是逐點時間 table。
ComputedPulse 與 TableLengthPulse 使用捕捉的任意 selector 選擇實際 playback，並保留相配的 post 到 aggregate 才取整。Flat-top 的全部 segment 地址由 playback owner 在最後準備步驟產生，時間 storage 的全流稽核也涵蓋這些衍生地址。
未進入 timeline 範圍時，省略 timeline 選 default，但不移動 reference。
以下是 `TimelineProgramV2` 子類別的 hooks。`drive_cfg` 是呼叫端提供、使用 channel 0 的 `Pulse.Cfg`。
時間參數單位為 us，另一個 channel 1 的 native pulse 由原生 API 宣告。

```python
from zcu_tools.program.v2.modules import Pulse
from zcu_tools.program.v2.timeline import TimelineProgramV2


def _initialize(self, cfg):
    self.drive = Pulse("drive", drive_cfg)
    self.declare(self.drive)
    self.declare_gen(1, nqz=1)
    self.add_pulse(1, "native", style="const", freq=1000,
                   phase=0, gain=0.5, length=0.1)

def _body(self, cfg):
    self.emit(self.drive)
    self.pulse(1, "native", t=0)  # Relative to the original reference.
    self.sync()
    self.pulse(1, "native", t=0)  # Relative to the drive end.
    self.fork("default", into="drive")
    self.emit(self.drive, timeline="drive")
```

DelayAuto 是唯一例外。在 timeline 範圍外省略 timeline 時，它包裝原生 delay_auto／delay_reg_auto，保留 gens／ros。
任何 timeline 名稱都在 emit 入口報錯，包含顯式 default 與範圍選出的名字。
在 timeline 範圍內需要推進 reference 時，直接呼叫原生 `delay_auto`。
原生 delay_reg_auto 先消耗選定 native channels，再推進 register cycles。尚未消耗的 timeline 以已知 producer bounds 扣除實際推進量。
若 native 部分已完整消耗全部 timeline，register 部分沿用原生的非負時間契約。
PulseReadout 的 gain／frequency runtime words 不受這項時間限制排除。

Join 用 `join(*sources, into=target)` 合併既有結尾，保留來源與較晚的原目標。
它只記帳，不移 reference，也不要求每條 fork 都 join 回來。
交叉或 runtime source 以逐點 maximum 合併，不把所有掃描點補成同一個長度。

`with self.branch(by="scan", align_timeline=True, timeline="drive") as branch:` 依 sweep selector 選 case。
`timeline` 指定母 timeline，省略或傳入 None 時選目前範圍，範圍外選 default。
`with branch.case(index) as name:` 取得隱式 case timeline 的名字 `"scan[index]"`。
Case 從母 timeline 的結尾 fork，並以自己的名字屏蔽範圍預設。
Case 內只能使用這條 timeline 與其局部 fork。使用母 timeline 或其他外部名字會在 build 時報錯。
Index 是 `scan` 的 zero-based 點索引，可只宣告部分點，也可完全不宣告。
未宣告的點執行空 case，不要求作者填入空動作。重複、非整數或超界 index 報 ValueError。

`align_timeline` 是必填 keyword-only bool。每個 case 保留 branch 入口的 reference。
True 將各 case 結尾 join 回母 timeline。全靜態時取靜態 max，有 runtime 值時取精確逐點 max。
True 需要每個 case 的結尾都能從 branch 入口狀態算出，不執行未選 case 的動作。
False 將母 timeline 設為實際執行 case 的結尾。其他外部 timeline 在兩種模式都不變。
兩種都不推進 reference、不自動 sync，後續可在母 timeline 結尾繼續發射。
未使用的外部 long pulse 可以橫跨 Branch。
Case 內的 native delay、delay_auto 與 sync 在 build 時報錯，等待改用 delay_timeline。
Case 內可使用 `repeat(...)`。它的 body 同樣不得推進 reference，包括巢狀 Repeat 內的操作。
Repeat 內放 branch 仍合法，兩種 align_timeline 都使用上述規則。
Branch 區塊內、case 外的 emit、fork、join、delay_timeline 等動作在 build 時報錯，無條件動作寫在 with 區塊前或後。
隱式 case 名字與 case 內的 fork 名字在 case 結束後失效，即使其他 case 使用同名也一樣。
沒有 join 的局部結尾直接丟棄，不合回祖先、不影響其他 timeline，也不報錯。
跨 Branch 的 timeline 須在入口前宣告。Register 用量沒有保證，超限保留 QICK 原錯誤。

`repeat(name, n, *, timeline=None, range_hint=None)` 使用逐圈累積的 timeline。
`timeline=None` 選目前範圍，範圍外選 default。範圍內以循環 timeline 屏蔽預設名字。
只可使用母 timeline 與其局部 fork，離開後局部名字失效。
Repeat 不在入口或回邊自動推進 reference。作者可在 body 明寫 `sync()`、原生 `delay`／`delay_auto` 或巢狀 Repeat。
Branch case 內仍禁止任何經 owner 推進 reference 的操作，包括 Repeat body 裡的操作。
`n=0` 保留母 timeline，不執行或宣告 body 內容，也不自行同步入口。

編譯器只按已錄製的 body 結構選 lowering，不分析定點，也不重播 preprocess。
當 body 頂層最後一個操作是對循環 timeline 的 `sync()`，且入口結尾為常數 0，
不攜帶循環尾端 register。固定圈長直接編成每圈的常數 `TIME inc_ref`。
其他情況使用 shot-local register 累加器，入口以母 timeline 結尾初始化，每圈存回相對 reference 的最晚結尾。
中途 sync、只在巢狀區塊末尾 sync、同步其他 timeline 或沒有 sync 都走累加器。
若 body 以循環 timeline 的 sync 結尾，但入口結尾不為常數 0，會發 UserWarning 提示先在循環前加 sync。
編譯器不自行插入它。非定點的 register lowering 保留既有 IR 展開與 branch-address DMEM 成本，尚未改成 s14 指針。

`n` 是非負整數或既有 count register 名字，不能與 counter 的 `name` 相同。
Register 次數可提供作者保證的非負閉區間 `range_hint=(minimum, maximum)`，不增加硬體檢查。
省略 hint 時使用實際 count，不證明回邊的同 channel overlap。固定次數不接受 hint。
每 shot／sweep 點重新初始化循環狀態，局部 counter 不加入 QICK top-level sweep metadata。
Repeat owner 以公開 `fixed_loop_indices` 提供固定 counter 的名稱、點數與 writer metadata。
Producer audit 仍只接受相符的初始化、回邊與重設寫入。
累加器內事件的時間可能只在 runtime 可知，getter 會明確報 RuntimeError，不回上界。
編譯期可知的累積結尾仍可讀取實際量化值。

外部 long pulse 可以橫跨 Repeat，但不能在 body 內使用其 timeline。
沒有 reference 操作時，其他外部結尾不變。
作者明寫 reference 操作時，外部結尾逐圈按實際推進量 debit，再做 floor 0。
固定且可靜態表示的總推進量保留靜態記帳。Runtime 次數或路徑需要時使用 storage，未消耗的來源保持 wait-only。
Wait-only 來源可供 sync、join 或 fork 使用，不能直接放 pulse、trigger、config 或 delay_timeline。
Join 目標與 fork 的新名字從實際 remaining end 建立普通排程點。完全消耗來源後，它也恢復 placement。
原生 `delay_reg_auto` 可在 Repeat 內使用，仍須遵守其 producer bounds 與未消耗外部結尾的原生契約。

以下 body 明寫入口與每圈同步。每圈先合併局部結尾，再移動 reference：

```python
def _body(self, cfg):
    self.sync()
    with self.repeat("pair", 2):
        self.fork("default", into="local")
        self.emit(self.drive, timeline="local", tag="repeated")
        self.delay_timeline(0.05, timeline="local")
        self.join("local", into="default")
        self.sync()
    # Parent default is zero. The local name has expired.
```

Register 次數可改寫成 `with self.repeat("pair", "n_pair", range_hint=(2, 4)):`。
`n_pair` 須先準備，執行值必須符合作者保證的範圍。Register 超限保留 QICK 原錯誤。

ScanWith 先同步母 timeline，再載入 scalar、依序執行 children，每個 point 結尾明寫 sync。
同一宣告可重複 emit，每次使用獨立 fixed counter，沿用原 table 與 destination。
`LoadValue.lower_index` 接受本次 index，`emission_name` 只在 module lower 期間配置本次名稱。

Module Branch lower 到原生 branch/case。Module Join 為各 sequence fork timeline，依作者順序發射，再 join 回母 timeline。
各 sequence 從共同 origin 開始，不重排，與只記帳的 `prog.join` 不同。
Join 與 TwoPulseReset／BathReset 的 child 必須保留 reference。
原生 delay、delay_auto、sync、register delay 與內含這些操作的 Repeat／ScanWith 在 build 時報錯。
普通 Branch 只合併結尾，可以放進 Join。
同 generator 的 physical-end 與全部 case、Repeat backedge guards 保留，不加 padding 或 rewind。

守衛只涵蓋 timeline／module API 的 reference 操作。
Raw `asm_inst` 或自訂 Macro.translate 推進 reference 時，不更新 timeline 記帳，也不受 Join／Reset 或 case 守衛保護，結果由作者負責。
它們不是已分析 instruction effects 的安全入口。

Body 收尾先同步全部 timeline，再放 body_end marker，然後進 shot 外殼。
`sync` 保持每點 runtime maximum；shell WAIT 才使用編譯期常數上界。
Program／timeline owner 保留已知 loop／table 的逐點來源關係與 native channel 結尾式子，供收尾 WAIT 計算使用。
未知 ADC tail 經同步可證消耗時，收尾常數 WAIT 使用 0。
若 ADC tail 仍未知，build 報錯並列出 ADC，提示提供 range_hint 或在結尾前明寫同步。收尾不使用 variable WAIT，也不假造上界。
這些式子只作編譯期記帳，不是第二個 scheduler，也不改 runtime register 或指令。
能證明所有合法點的剩餘時間相同時，WAIT 精確且不警告。
已知點的 `slack=max-min` 大於 0 時會 warning。
已知 slack 超過 `relax_delay` 時 raise ValueError，等於時不拒絕；真正未知的 slack 只警告上界，不以該上界拒絕。
來源相關性、精確 min／max 與未知值政策見 [ADR-0083](../../../../docs/adr/0083-program-macros-and-timelines.md#收尾-wait-與-slack-護欄)。
每次 compile 重置宣告與結尾。公開 `timing_layout` 由真 boundary metadata 取得 linked addresses 與 loop 座標。
將它與 prog_list／labels 交給 `ir.analyze_timing`，可觀察逐點事件、body_end 與含外殼的 shot_end。
Register Repeat 的編譯結果可能包含 DMEM dispatch，分析時以 `CompiledTimingInput.dmem`
傳入公開 `compile_datamem()` 的結果，不在 caller 重建 table。
Resync 讀取硬體 s11，analysis 會明確拒絕，不宣稱其後精確時序。

`resources.ProgramResourcesMixin` 集中 dmem 配置與 readout route 保護。
TimelineProgramV2 與 ModularProgramV2 共用它，傳輸時由 MyProgramV2 將 dmem 轉為 list[int]。

## Macro Layer

### Runtime and table-backed waveform updates

Runtime pulse/readout words use dedicated macros that load a single-wave wmem
template into `r_wave`, patch selected fields, and send `r_wave` directly to the
target port. They do not persist the patched bundle to wmem. This keeps runtime
updates atomic at playback and leaves the fixed template stable across points.

`TablePulseReadout` uses a different contract for arbitrary frequency sweeps: the
body plays ordinary wmem-backed templates, while the program attaches dmem
lookup + wmem patch macros to QICK's loop exec-after hook. The update therefore
lands after the shot timing and before the loop test/jump, matching native
`QickParam` sweep placement without enabling DDS phase reset.

### `MetaMacro.regs` — Register Name Resolution

`MetaMacro` accepts an optional `regs: dict[str, str]` field. Each entry maps an `info` key to a logical register name (e.g. a QICK loop name like `"reset_sel"`), an ASM address or a QICK alias. At `translate()` time, the macro passes its declaration to `IRCompileMixin.record_meta()`. The compiler copies `info`, resolves the register references through QICK and records the metadata at the current instruction address without emitting ASM. Resolved addresses overwrite matching `info` keys without mutating the inputs. Resolution errors propagate without recording a marker.

**Design constraint:** `MetaMacro.translate()` does not go through QICK's normal macro `expand()` path, so loop names stored in `info` are never resolved automatically.  Any caller that passes a loop name (or other logical register alias) as a register reference in `info` must use `regs` to trigger resolution — otherwise the IR layer receives an unrecognisable string and the assembler raises `Source Data not Recognized`.

**Usage pattern:**

```python
prog.meta_macro(
    type="BRANCH_START",
    name=self.name,
    regs={"compare_reg": self.compare_reg},  # resolved at translate time
)
```

## ComputedPulse — DMEM Lookup 機制

新路徑在 `declare` 宣告候選 waveform 與 wmem-index table。每次 `emit` 先捕捉 selector，再由同次選擇決定 playback 與該候選的 post-delay。Program owner 保留原始 waveform 分數尾段，加入所選 post 後取整一次，不使用候選最大 duration。實際 address producer、捕捉的 selector 與其依賴都接受全流寫入稽核。

Legacy `ComputedPulse` 在 `init()` 時收集每個候選 pulse 的起始 wmem index，透過 `prog.add_dmem(base_idxs)` 寫入 dmem table 並記錄 `dmem_offset`。`run()` 時透過 `read_dmem` 動態查找：

```python
addr_reg = val_reg + dmem_offset   # address of table[val_reg]
wmem_reg = dmem[addr_reg]          # starting wmem index for this candidate
pulse_by_reg(ch, wmem_reg, ...)
```

**dmem table 設計約束**：candidate waveform 的 wmem index 不保證連續，因為 `PulseRegistry` 會去重。dmem table 直接存儲每個候選的起始 index，無連續性假設。

**flat_top**：table 中存的是每個 flat_top 候選的**第一個** wmem entry（ramp_up），`pulse_by_reg(flat_top_pulse=True)` 從此 index 自動連讀 3 個 entry（ramp_up / flat / ramp_down）。

**dmem 共用**：`ProgramResourcesMixin` 擁有全程式共享的順序 buffer，`add_dmem()` 回傳的 `offset` 是 slice 起點，`compile_datamem()` 最後轉為 `np.int32` 陣列。

**dmem 傳輸邊界**：`compile_datamem()` 維持 QICK 規定的 `np.int32` array contract；`MyProgramV2` 在完整 binary program 建立後，將送往 remote SoC 的 dmem 正規化為一維 built-in `list[int]`。跨 Python／NumPy 版本的傳輸使用 built-in 容器，保留 signed int32 與 two's-complement bit pattern。

## Big-Jump Threshold

Big-jump 判斷集中在 `ir/hw_semantics.py::needs_big_jump(pmem_size)` 回傳 `pmem_size > 2048`。小 pmem（≤ 2048）用 label-mode JUMP；大 pmem 用 `WriteLabel(s15) + JUMP ADDR=s15` 兩步。`macro/loop.py` 與 `modules/control.py` 皆從 `ir.hw_semantics` import，確保閾值一致。`ir/dispatch.py` 也從 `hw_semantics` import 此函式（非定義處）。`UnpackIRBranchPass` 使用 `PipeLineConfig.pmem_capacity` 做相同判斷：`pmem_capacity > 2048` 時在 case 間插入 `RegWriteInst(dst=s15, src=SrcKeyword.LABEL)` + `JumpInst(addr=s15)`，否則直接插入 label-mode `JumpInst`。

## LoadValue（dmem 值表）

`LoadValue` 儲存整數 lookup table 於 dmem，支援可選的 bit-packing 壓縮（`auto_compress=True` 且值個數 ≥ 30）。

**空表短路**：`values=[]` 僅用於上層已確定不會執行 lookup 的路徑（例如 loop 次數為 0）。`init()` 仍依 `use_existed` 宣告 destination register，讓編譯器能解析未執行路徑內的 consumer；它不配置 dmem，也不填入預設值。`run()` 不發出讀寫指令，並原樣回傳傳入時間。這不支援在執行期讀取空表。

**值域限制：0 ≤ v ≤ 2³¹−1**（有號 32-bit 正數範圍）。此限制只針對**未壓縮**路徑——直接以 `read_dmem` 將 word 讀進 register，若 bit 31 被設定讀回為負數會造成語意錯誤。

**壓縮路徑的 packed word 可合法設定 bit 31**：`_pack_values()` 在打包後若 word > INT32_MAX，以 two's complement 轉換（`word -= 2^32`）存成負的 int32，讓 `np.array(..., dtype=np.int32)` 不溢位。硬體 bit pattern 完全相同；`run()` 結尾的 `SR` + `AND value_mask` 提取路徑在 arithmetic 或 logical shift 模式下都能還原正確值（AND 會清掉 SR 帶上來的符號擴展位元）。

**壓縮條件**：`bits_per_value` 圓整至 2 的冪次後，`values_per_word = 32 // bits`；若 `values_per_word < 2` 則不壓縮。max value > 65535（bits 圓整至 32）時 `values_per_word = 1`，不壓縮。

## Repeat（control module）n=0 短路

`Repeat` 在 `n` 為整數 `0` 時，`init()` 直接返回。`run()` 只執行開頭的 `delay(t)`／`delay_auto(0)` 對齊時間基準後返回。此路徑使用零個 counter register，子模組與 loop macro 保持未初始化及未展開。

## ArbWaveform reference time axis

Native ArbWaveformCfg 保存邏輯 data key 與必填本機絕對 path。Length 是 path 指定 NPZ 的 time[-1]，不存另一個播放長度。ModuleCfgFactory.from_core 使用明示 context.assets.path_for 轉換 nested arb，成品不持有 provider。

Program 與 sim 從 path 讀 bytes，交 core.waveform_format 解碼 samples，不查 global DB 或解讀 formula。ArbWaveform.make_iqdata 在完整 reference time axis 插值到硬體 timing，不伸縮或裁切。覆寫同 path 會改變之後的播放內容。

# README - program/v2

**Last updated:** 2026-10-30，逐點精確 ceil、selected playback 與 paired post

**讀者：** lab-author、framework-dev、agent。原生 V2 program 寫法與 module、readout、acquisition 契約速查。

## Testing & Type Checking Conventions

- **Type Narrowing**: When a mock or abstract base class type is too generic in tests (e.g., retrieving an item from an `IRNode.insts` list where you expect a `RegWriteInst`), use an explicit assertion (`assert isinstance(item, RegWriteInst)`) rather than `cast()`. This ensures Pyright can perform type narrowing safely while maintaining a runtime guarantee.
- **Mock Overrides**: When testing edge cases that intentionally return unexpected types to trigger errors, prefer a narrow helper/protocol or an explicit runtime assertion at the boundary. Do not widen signatures to `Any` or use `cast()` just to silence the type checker unless there is a reviewed reason.
- **Mock Interfaces**: Use type-compatible fakes or an existing typed seam when passing test collaborators to functions. Do not add type suppressions or `cast()` to silence an incompatible interface.
- **Program Emission Tests**: Unit tests that assert emitted program actions use a typed `ProgramTrace` adapter rather than loose `MagicMock` program objects. The trace records semantic program intents (readout declarations, triggers, pulses, loops, delays, register writes, and jumps) without acting as a second compiler or IR parser.
- **MagicMock Scope**: Keep `MagicMock` for non-program collaborators, child-module spy hooks, and deliberate error-injection tests. Do not use it as the default stand-in for a program object when the test is asserting emitted program behavior.
- **Offline acquisition**: `make_mock_soccfg(n_gens, n_readouts)` builds only a compile profile. Explicit environments use `SimulationSoc` and `SimulationBinding`; ordinary acquisition tests may inject a deterministic transport. `ProgramTrace` records emission and does not replace real acquire coverage.
- **QICK External Types**: `QickParam` runtime objects expose methods/attributes (`is_sweep`, `minval`, `maxval`, `start`, `spans`, `to_array`) that can be missing from third-party stubs. Program code narrows through `program.v2.utils.is_qick_param()` / `QickParamLike` instead of local `cast()` calls, so every use has a runtime guard.
- **Cfg Contracts**: `AbsModuleCfg` and `AbsWaveformCfg` are abstract contracts. Concrete cfg classes implement both `build()` and `set_param()`; unknown parameter names raise `ValueError` rather than being ignored. [`tests/cfg_model/test_set_param_contract.py`](../../../../tests/cfg_model/test_set_param_contract.py) is the cross-family audit for this fail-fast rule.

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
- `BaseCustomProgramV2.make_modules()` 回傳 modules。作者自行保存組裝 modules 所需的輸入，
  program 的控制參數仍由建構子接收。
- `ir/`: Contains the Intermediate Representation for tProc v2 instructions, allowing for optimization passes and cross-module label resolution.

## 原生寫法與具名 timeline

`timeline.TimelineProgramV2` 繼承 `MyProgramV2` 加 `resources.ProgramResourcesMixin`，不經 `ModularProgramV2`。
跨 owner 分工、設計理由與階段邊界見 [ADR-0083](../../../../docs/adr/0083-program-macros-and-timelines.md)。
作者在 `_initialize(cfg)` 宣告通道、pulse、readout config、module 與 loops。
在 `_body(cfg)` 發射原生 QICK 操作或 `emit(module)`。
建構子使用 `reps`、`initial_delay`、`relax_delay`。
Shot 外殼為 `final_wait=0`、`final_delay=relax_delay`。
全部輸出共用既有 IR compiler。Boundary metadata 可能改變指令排列，不要求新舊逐位相同。
Builder 與模擬 frontend 尚未切換，既有 acquire 擴充仍由共同基底提供。

每條 timeline 只保存相對 reference 的結尾。
Default 是普通 timeline，不是 reference。其他名字先在 `_body` 用 `fork(source, into=...)` 宣告。
Source 必須存在，into 接受一個新名字或名字 list。未知來源、重名與未宣告使用都報錯。
帶 timeline 的 pulse／trigger 在結尾加 t 發射，然後更新結尾。
同 channel 的 pulse 提前開始時編譯報錯，不能用另一條 timeline 繞過。
Config 是瞬時事件，結尾更新到發射時間 E+t，不加 config length 或其他 ADC window。
未帶 timeline 的原生指令保留 QICK 行為。

`delay_timeline(t, timeline="drive")` 只將結尾加 t，不移動 reference。
`sync(timeline="drive")` 才將 reference 移到該結尾，並扣減全部 timeline，最低為零。
原生 delay／delay_auto／resync 也經公開 timestamp decrement 扣減全部結尾。
可靜態決定的結尾保留 scalar／`QickParam`。跨零 floor、交叉 max 與 path-selected end 才物化成 runtime cycle register。
Runtime 算術只用無 flag 的 ABS／ASR／加減。End 限於 `0..2**31-1`，中間值也須能證明符合 signed 32-bit range，否則編譯報錯。
每個 sweep／shot 重新建立所需狀態。`delay_timeline("duration", ...)` 讀取非負 tProc cycle register，仍不推進 reference。
Register 時間須有原生常數、sweep 或 framework table／長度 producer 的 bounds，不能把任意 custom Macro 初值當成證明。
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
編譯期已知的非仿射 sweep 由 program／timeline owner 以 dmem table 提供逐點精確 aggregate ceil，不加 padding。
ComputedPulse 與 TableLengthPulse 使用捕捉的任意 selector 選擇實際 playback，並保留相配的 post 到 aggregate 才取整。Flat-top 的全部 segment 地址由 playback owner 在最後準備步驟產生，時間 storage 的全流稽核也涵蓋這些衍生地址。
省略 timeline 選 default，但不移動 reference。
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

DelayAuto 是唯一例外。未指定 timeline 時包裝原生 delay_auto／delay_reg_auto，保留 gens／ros。
指定任何 timeline 名稱都在 emit 入口報錯，包含顯式 default。
原生 delay_reg_auto 先消耗選定 native channels，再推進 register cycles。尚未消耗的 timeline 以已知 producer bounds 扣除實際推進量。
若 native 部分已完整消耗全部 timeline，register 部分沿用原生的非負時間契約。
PulseReadout 的 gain／frequency runtime words 不受這項時間限制排除。

Join 用 `join(*sources, into=target)` 合併既有結尾，保留來源與較晚的原目標。
它只記帳，不移 reference，也不要求每條 fork 都 join 回來。
交叉或 runtime source 以逐點 maximum 合併，不把所有掃描點補成同一個長度。

`with self.branch(by="scan") as branch:` 依 sweep selector 選 case，每個點必須有一個 `with branch.case():`。
每個 case 只同步自身使用的 timeline 與局部 fork，不補齊其他 case 的長度。未使用的外部 long pulse 可以橫跨 Branch。
Case 內的 fork 名字在 case 結束後失效，即使其他 case 使用同名也一樣。跨 Branch 的 timeline 須在入口前宣告。
入口前既有 timeline 的 path ends 不同時，保留實際選中 end 並標為 wait-only。

`with self.repeat(name, n, timeline="default", range_hint=None)` 先同步母 timeline，
每圈再同步母 timeline 與局部結尾。內部只能用母 timeline，以及從母或局部 timeline
fork 的名字。退出後母 timeline 為0，局部名字失效，可以重新 fork 同名。
其他外部長 pulse 可以橫跨 Repeat，但不能在迴圈內使用它的 timeline。
固定 n 是非負整數，退出時精確扣減外部結尾；n=0 仍同步入口，但不執行 body。
Repeat owner 以公開 `fixed_loop_indices` 提供固定 counter 的名稱、點數與 writer metadata。
時間 producer 的 bounds 只接受相符的初始化、回邊與重設寫入，不把 nested loops 加入 QICK 的 top-level sweep metadata。

ScanWith 先載入 scalar，再依序執行 children。同一宣告可重複 emit，每次使用獨立 fixed counter，沿用原 table 與 destination。
其 native lowering 使用 Repeat metadata，不傳 legacy `range_hint`。
`LoadValue.lower_index` 接受本次 index，`emission_name` 只在 module lower 期間配置本次名稱。

Register n 使用既有 register 名字，必須提供作者保證的非負整數閉區間
`range_hint=(minimum, maximum)`。Int n 不接受 range_hint。
Register 名字不得與以 name 命名的 counter 相同。編譯器不從 register 初值推定 range。
未知 count 或 case advance 以 runtime storage 逐圈扣除實際執行路徑，並對每個 external end 做 floor 0。
相同且可靜態表示的 Branch paths 保留 static，不因出現 Branch 就物化。
原 timeline 禁止繼續發射或 delay_timeline，直到 sync 或原生 reference 推進完整消耗該 end。
Join 目標與 fork 新名字是普通 timeline，從實際 remaining end 建立排程點，不採 minimum count 的等待上界。
原 wait-only timeline 上的兩種 readout config 入口也拒絕放置，包含 t=0。
Join／fork 的普通結果，或 sync／原生推進完整消耗後的來源，可照常放置 config。
不帶 timeline 的 config 呼叫保留原生行為。
例如沿用上面的宣告，以下 `_body` 先安排跨 channel 重疊，再重複發射 drive：

```python
def _body(self, cfg):
    self.fork("default", into="parallel")
    self.emit(self.drive, tag="first")  # Channel 0, ordinary default end.
    self.pulse(1, "native", t=0, timeline="parallel")
    self.join("parallel", into="default")  # Accounting only.
    self.sync()  # Both channels have finished before the next group.
    with self.repeat("pair", 2):
        self.fork("default", into="local")
        self.emit(self.drive, timeline="local", tag="repeated")
        self.delay_timeline(0.05, timeline="local")
    # Parent default is zero. The local name is out of scope.
```

每圈結尾自動對齊 local 的 module 結尾加 0.05 us，不必在迴圈內手動 sync。
Register 次數改傳既有 register 名字，例如
`with self.repeat("pair", "n_pair", range_hint=(2, 4)):`，body 可沿用上例。
`n_pair` 須先由作者用公開 register API 準備，執行值必須符合作者保證的閉區間。

Repeat body 內支援具名 timeline 的 register 時間與各 case 不同的 Branch advance。
各 case 分別累積 pulse guard，回邊使用全路徑 envelope 證明不重疊。Envelope 只用於證明，不改實際 reference 或 body／shot end。
Repeat body 內的 DelayAuto(register) 與 delay_reg_auto 仍報錯。迴圈外 consumer 不受這項限制。

Body 收尾先同步全部 timeline，再放 body_end marker，然後進 shot 外殼。
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

# README - program/v2

**Last updated:** 2026-06-08

## Testing & Type Checking Conventions

- **Type Narrowing**: When a mock or abstract base class type is too generic in tests (e.g., retrieving an item from an `IRNode.insts` list where you expect a `RegWriteInst`), use an explicit assertion (`assert isinstance(item, RegWriteInst)`) rather than `cast()`. This ensures Pyright can perform type narrowing safely while maintaining a runtime guarantee.
- **Mock Overrides**: When testing edge cases that intentionally return unexpected types to trigger errors (like testing how `Branch` handles a `QickParam` duration), expand the overridden function's return signature (e.g. `-> Any`) rather than using `cast()` to silence the type checker.
- **Mock Interfaces**: For simple fake objects passed to functions (like `_FakeTracker` to `snr_as_signal`), if it structurally satisfies the requirements for the test, a single `# type: ignore[arg-type]` at the injection site is preferred over verbose `cast(list[Interface], ...)` syntax to keep the testing code clean.

## IR System & Hardware Alignment

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

## Architecture

- `IRCompileMixin.optimize_asm()` runs the default IR pipeline (`make_default_pipeline`); the concrete pass set lives in `ir/passes/` and is documented in `ir/README.md`.
- `Nop` is preserved as an explicit instruction and is not treated as dead-code noise.
- `ir/passes/` is split by concern into sub-packages: `loop/`, `dataflow/`, `control_flow/`, `timeline/`, with a shared `base.py`.

- `MyProgramV2`: Base class for experiments, wraps QICK's `AveragerProgramV2`.
- `ModularProgramV2`: Supports pulse blocks and reusable modules.
- `ir/`: Contains the Intermediate Representation for tProc v2 instructions, allowing for optimization passes and cross-module label resolution.

## Macro Layer

### `MetaMacro.regs` — Register Name Resolution

`MetaMacro` accepts an optional `regs: dict[str, str]` field.  Each entry maps an `info` key to a logical register name (e.g. a QICK loop name like `"reset_sel"`).  At `translate()` time, each name is resolved to its hardware ASM address via `prog._get_reg()` and written into `resolved_info` before `_add_meta()` is called.

**Why needed:** `MetaMacro.translate()` does not go through QICK's normal macro `expand()` path, so loop names stored in `info` are never resolved automatically.  Any caller that passes a loop name (or other logical register alias) as a register reference in `info` must use `regs` to trigger resolution — otherwise the IR layer receives an unrecognisable string and the assembler raises `Source Data not Recognized`.

**Usage pattern:**

```python
prog.meta_macro(
    type="BRANCH_START",
    name=self.name,
    regs={"compare_reg": self.compare_reg},  # resolved at translate time
)
```

## ComputedPulse — DMEM Lookup 機制

`ComputedPulse` 在 `init()` 時收集每個候選 pulse 的起始 wmem index，透過 `prog.add_dmem(base_idxs)` 寫入 dmem table 並記錄 `dmem_offset`。`run()` 時透過 `read_dmem` 動態查找：

```python
addr_reg = val_reg + dmem_offset   # address of table[val_reg]
wmem_reg = dmem[addr_reg]          # starting wmem index for this candidate
pulse_by_reg(ch, wmem_reg, ...)
```

**為何改用 dmem 方案**：舊方案要求 candidate waveform 的 wmem index 必須連續，與 `PulseRegistry` 去重機制潛在衝突。dmem table 直接存儲每個候選的起始 index，無連續性假設。

**flat_top**：table 中存的是每個 flat_top 候選的**第一個** wmem entry（ramp_up），`pulse_by_reg(flat_top_pulse=True)` 從此 index 自動連讀 3 個 entry（ramp_up / flat / ramp_down）。

**dmem 共用**：`ModularProgramV2._dmem_buffer` 是全程式共享的順序 buffer，`add_dmem()` 回傳的 `offset` 是 slice 起點，`compile_datamem()` 最後轉為 `np.int32` 陣列。

## Big-Jump Threshold

Big-jump 判斷集中在 `ir/hw_semantics.py::needs_big_jump(pmem_size)` 回傳 `pmem_size > 2048`。小 pmem（≤ 2048）用 label-mode JUMP；大 pmem 用 `WriteLabel(s15) + JUMP ADDR=s15` 兩步。`macro/loop.py` 與 `modules/control.py` 皆從 `ir.hw_semantics` import，確保閾值一致。`ir/dispatch.py` 也從 `hw_semantics` import 此函式（非定義處）。`UnpackIRBranchPass` 使用 `PipeLineConfig.pmem_capacity` 做相同判斷：`pmem_capacity > 2048` 時在 case 間插入 `RegWriteInst(dst=s15, src=SrcKeyword.LABEL)` + `JumpInst(addr=s15)`，否則直接插入 label-mode `JumpInst`。

## LoadValue（dmem 值表）

`LoadValue` 儲存整數 lookup table 於 dmem，支援可選的 bit-packing 壓縮（`auto_compress=True` 且值個數 ≥ 30）。

**空表短路**：`values=[]` 允許，`init()` / `run()` 皆短路為 no-op（不配置 dmem，不發出讀取指令）。用於上層已確定不會執行該 lookup 的路徑（例如 loop 次數為 0）。

**值域限制：0 ≤ v ≤ 2³¹−1**（有號 32-bit 正數範圍）。此限制只針對**未壓縮**路徑——直接以 `read_dmem` 將 word 讀進 register，若 bit 31 被設定讀回為負數會造成語意錯誤。

**壓縮路徑的 packed word 可合法設定 bit 31**：`_pack_values()` 在打包後若 word > INT32_MAX，以 two's complement 轉換（`word -= 2^32`）存成負的 int32，讓 `np.array(..., dtype=np.int32)` 不溢位。硬體 bit pattern 完全相同；`run()` 結尾的 `SR` + `AND value_mask` 提取路徑在 arithmetic 或 logical shift 模式下都能還原正確值（AND 會清掉 SR 帶上來的符號擴展位元）。

**壓縮條件**：`bits_per_value` 圓整至 2 的冪次後，`values_per_word = 32 // bits`；若 `values_per_word < 2` 則不壓縮。max value > 65535（bits 圓整至 32）時 `values_per_word = 1`，不壓縮。

## Repeat（control module）n=0 短路

`Repeat` 在 `n` 為整數 `0` 時，`init()` 與 `run()` 都會短路：不分配 counter register、不初始化子模組，也不展開 loop macro。`run()` 仍保留開頭的 `delay(t)`/`delay_auto(0)` 對齊時間基準，然後直接返回。

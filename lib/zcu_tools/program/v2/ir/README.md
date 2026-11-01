# IR architecture guide

**Last updated:** 2026-11-01，s14 output offsets and scheduling windows

**讀者：** framework-dev。QICK ASM V2 的 IR 分層、分析與 lowering 契約。

本模塊負責把 QICK ASM V2 程式表示成可分析、可優化、可再降階回 assembler 輸入的 IR。

## Core layers and owners

1. `IRLinker` ([linker.py](linker.py))
   - 負責 `list[Instruction]` 與 `prog_list + labels + meta_infos` 的互轉。
   - `position()` 統一計算 word address、assembler line 與 end cursor，拒絕重複 label。它回傳保留 input reference 的 `PositionedInstruction` views，不建立 snapshot。
   - `link()` 序列化這些位置，是唯一會寫入 `P_ADDR` 與 `LINE` 的地方。Compile snapshot 直接讀取位置 views，再自行 defensive-copy payload。
2. `IRLexer` ([factory.py](factory.py))
   - 負責 `list[Instruction] ↔ list[BasicBlockNode | MetaInst]`。
   - `MetaInst` 在 chunked stream 中獨立於 `BasicBlockNode` 存在。
3. `IRParser` ([factory.py](factory.py))
   - `parse()` 把 `list[BasicBlockNode | MetaInst]` 建構為 `BlockNode` 根節點。
   - `unparse(root: IRNode)` 是唯一的降階入口，接受 tree pass 替換後的任意合法 IR 根節點，輸出含結構 MetaInst 的 chunked stream。原有 `BlockNode` tree 與 `parse()` 保持 round-trip。`IRDispatch` 葉節點由 pass 生成。
4. [dispatch.py](dispatch.py)
   - 提供 loop / branch 共用的 dispatch-table lowering 原語。

[operands.py](operands.py) 擁有 operand 型別與 parse factories。
[instructions.py](instructions.py) 擁有指令基底、compute／control／time 指令與中央 QICK dispatcher。
[io_instructions.py](io_instructions.py) 擁有完整的 memory／port 指令及其 QICK dictionary 互轉。
[instruction_fields.py](instruction_fields.py) 擁有兩組指令共用的有限值契約、QICK 欄位解析與 label reference remap。
[node.py](node.py) 擁有 IR tree 節點、`clone_renamed` 與唯讀共用 interface
`collect_subtree_names`。Clone、parser 與 passes 共用此 interface 收集子樹定義的 label names
及 structure names，名稱配置仍由 [labels.py](labels.py) 擁有。
[labels.py](labels.py) 擁有 label value objects 與名稱配置。
硬體角色性常量集中在 [hw_semantics.py](hw_semantics.py)，避免 magic string 散落各處。
[analysis.py](analysis.py) 提供 cost/size 估算 helper 與 liveness 工具，pass 端優先使用。

## Pre-optimization compile snapshot

`program.compiled_stream` returns a `CompiledStream` after successful compilation
on `TimelineProgramV2`. Its `records` tuple preserves
logical instructions, labels and metadata before the optimizer consumes them.
Each `CompiledRecord.p_addr` is a pre-optimization program-memory word offset,
not an optimized address. Metadata and labels can share positions.

`record.instruction` returns a detached typed IR deep copy on each access.
Changing that copy cannot change the snapshot, program or optimizer. Consumers
should read each record once per request build and retain the copies for
pre-decoding, rather than reading the property per lane or loop iteration.
Saved snapshots survive recompilation. QICK cached `compile()` calls retain the
current snapshot. Set public `binprog = None` before `compile()` to rebuild;
the accessor raises `RuntimeError` while no compiled binary is available.
This compiler artifact contains no simulation state or frontend dependency.

The snapshot includes zero-word source scopes and native QICK loop annotations.
`SOURCE_START` / `SOURCE_END` carry a Module diagnostic name or a native pulse
source in `name`. Scopes nest; the innermost active scope labels each output,
and its end restores the enclosing source. Native pulse sources use the explicit
`tag`, or `native:pulse:<ordinal>` with a one-based call ordinal. Tagged native
calls also advance the ordinal. Module lowering uses `source_scope(name)` and
its internal pulse calls neither allocate native sources nor advance ordinals.
Each actual recompile resets these scopes and ordinals.

`MarkedOpenLoop` / `MarkedCloseLoop` preserve the four `LOOP_*` records. Their
`info["native"] is True` declares that existing QICK instructions own the shell,
including waveform/register sweep increments and resets. `LOOP_START` also
records resolved `counter_reg`, trip count `n`, and `range_hint=None`.
Its `sweeps` dict has `registers` and `waves` maps. Registers use canonical
addresses; waves use zero-based WMEM word addresses and field indexes 0..5
(freq, phase, env, gain, length, conf). Each field contains integer `start` and
per-loop-iteration `step` words from resolved QICK definitions. Inner IR loops
emit empty maps. Native `tables` maps WMEM word/field indexes to DMEM word
offsets for declared deterministic hooks. The tables contain rotated exec-after
values: point zero retains initial WMEM, later points read offset + index - 1,
and the last table word restores the initial field. The waveform macro owns
target resolution. Loop producers collect only the resolved numeric definitions.
`LOOP_BODY_START` follows native counter initialization; `LOOP_BODY_END` precedes
native sweep updates/backedge; `LOOP_END` follows the complete closing shell.
The compiler snapshots the full logical stream, then gives the optimizer a new
sequence excluding exactly `SOURCE_START` / `SOURCE_END` and these four native
loop annotations. It keeps all instructions and labels. Unflagged structured
loops and branches still use the existing IR parser/lowering. Unknown metadata
still fails there. `TIMING_BOUNDARY` continues through the anchor/link path;
`body_start` / `body_end` exclude final QICK shot wait/delay operations.

## Compiled timing analysis

`analyze_timing(CompiledTimingInput(prog_list, labels, layout))` reads linked
assembly without changing it. `TimingLayout` gives final body_start/body_end
instruction addresses and outer-to-inner `TimingLoop(name, count)` axes,
including reps. Axes label shot ordinals; assembly register updates and jumps
determine execution.

Producers emit `TIMING_BOUNDARY` meta records named `body_start` and
`body_end`. The pipeline carries these through non-removable label anchors,
then restores the metadata before final linking. `TimingLayout.from_metadata`
reads the final addresses. Producers must not infer boundaries from opcodes or
instruction counts. Boundaries act as optimizer barriers, so adding them can
change instruction layout without changing the scheduled outputs.

Results report every wave-port write, data-port mask and trigger edge, with its
physical tProc port, linked instruction address and absolute scheduled cycle.
Wave-port events include pulse segments and readout configs, not one event per
module. Port numbers are not logical generator or ADC channel numbers. Events
remain in execution order, which can differ from scheduled-time order.

Reference starts at cycle zero. TIME inc_ref advances reference independently
of the s14 output-offset register. A shot starts before body_start executes.
Its end is reference at the next body_start, or program termination for the last
shot. The shell final_wait/final_delay belongs to that interval. WAIT time does
not advance reference. The result also reports reference at body_end and outputs
before the first shot.

Pass `CompiledTimingInput(..., dmem=prog.compile_datamem())` for programs with
read-only data-memory loads, including compiler-generated dispatch tables.
`dmem` accepts the public one-dimensional NumPy int32 array or its transported
Python integer sequence. Addresses are zero-based word offsets, not byte offsets.
None or empty data supplies no words. Analysis reads known in-range addresses
and follows known register jumps without modifying assembly or memory.
Missing data, malformed words, unknown/out-of-range addresses, writes and read
side effects raise errors; memory is never implicitly zero-filled.

Known register arithmetic, sweep updates/resets, literal/register event times
and known loop branches are evaluated from assembly. Waveform contents are not
modeled; unknown wave data propagates until it affects time or control.
Unsupported instructions, unknown timing/control values and inconsistent layouts
raise errors instead of returning partial or guessed timing. Native Resync reads
hardware s_usr_time/s11 and raises an error naming that source. The analysis
does not model CPU latency, waveform duration, hardware completion or measurement
data. It does not replace gain/frequency consumer tests.

## Pipeline

IR 優化是可選層：macro/module 層自身已能生成可運行 ASM，`IRPipeLine` 是純優化器。`disable_all_opt=True` 時 pipeline 原樣回傳輸入。

[pipeline.py](pipeline.py) 的 `IRPipeLine` 使用三種 pass 介面：

- `AbsChunkPass`：ChunkList 層的 per-block 轉換，正確性僅依賴單一 block 內容。
- `AbsChunkListPass`：ChunkList 層的全局轉換，需要看完整 chunk list（如跳轉引用、block 鄰接關係）。
- `AbsIRTreePass`：IR tree 層的 post-order 轉換。
  - `None` → pass 未變動該節點，pipeline 試下一個 pass
  - `IRNode` → 替換子樹，pipeline re-recurse 進新節點（內含的新結構節點也會被處理）
  - 遞歸由 pipeline 統一掌控（post-order）：`transform` 被呼叫時子節點已收斂；pass 若需未展開子樹的資訊（如 unroll 估算 body size），在 `transform` 開頭、回傳前自行分析。

### Pipeline 執行流程

```text
list[Instruction]
  → IRLexer.lex            → ChunkList
  → ChunkList 優化          （ChunkPass + ChunkListPass 同組迭代至收斂/上限）
  → IRParser.parse         → IR tree
  → IR tree 優化            （_optimize_tree：AbsIRTreePass post-order）
  → IRParser.unparse       → ChunkList（含結構 MetaInst）
  → strip 結構 MetaInst     （DISABLE_OPT 資訊已在 BasicBlockNode.disable_opt 屬性）
  → ChunkList 優化          （ChunkPass + ChunkListPass 同組迭代至收斂/上限）
  → IRLexer.flatten        → list[Instruction]
```

[passes/](passes/) 按 `control_flow`、`dataflow`、`timeline`、`loop` 分目錄。

## Output offsets and scheduling windows

TIME updates the reference clock without reading or writing s14. Untimed port
writes read s14 as their output offset. A preceding s14 write must survive across
TIME and remain before the port write.

Loop scheduling windows include literal `s14 = s14 + #N` increments as well as
literal `TIME inc_ref #N`. Register-driven increments have an unknown window and
do not trigger time-margin unrolling. Each unrolled copy retains its increment.
The dataflow pipeline removes unconditional pure register self-copies only when
UF and other effects are absent. Conditional copies and flag updates remain.
Instruction counts describe the post-IR stream, not QICK's temporary self-copies.

## Invariants that passes rely on

- Operand 的讀/寫語義不在 operand 層決定：所有 `Operand` 子類只實作 `regs() -> frozenset[str]`（中性語義，「這個值涉及哪些暫存器」）。讀還是寫由持有該 operand 的 instruction 自行決定，在 `reg_read` / `reg_write` 屬性中區分。
- `MetaInst` 只能存在於 flat / chunked structural layer，不能進入 `BasicBlockNode.insts`。
- `disable_opt=True` 的兩個來源：（1）dispatch-table stub blocks，pmem word 數與 object identity 不可改變；（2）包含 QICK pseudo-label（`HERE`/`NEXT`/`SKIP`）的 macro 展開，pseudo-label 在 assembler 階段才解析為 `P_ADDR ± offset`，IR pass 若插入、移除或替換 block 會讓 offset 計算錯位。
- `IRLoop.body` 必須視為完整 iteration 的語義單位，不能假設 counter update 一定位於最後一條指令。
- `IRBranch.cases` 保持泛化 `IRNode` contract；analysis helper 必須能估算 direct `BasicBlockNode`、`BlockNode` 與 nested structure case，不能 cast 成 `BlockNode`。
- `IRParser._check_sese()` 會禁止從任一結構 region 外部跳入或透過 label-address instruction 引用該 region 的 internal label，包含 parent region 內部跳入 nested child region 的情況。
- 所有帶 `if_cond` 的 `JumpInst` 不能與 `-op` 寫在同一行。所有的寫法都必須拆分為先 `TEST (op)` 更新 flag，再 `JUMP (if_cond)` 判斷跳轉。
- `ADDR_REG` 是 lowering scratch register，不可作為 dispatch selector 或 branch compare register。`IRBranch.compare_reg`、`IRDispatch.value_reg` 與 high-level `modules.control.Branch(compare_by=...)` 會拒絕 canonical name 為 `s15` 的 register（含 `s_addr` alias），避免 big-PMEM jump setup 覆寫 selector。

## Serialization boundary

`from_dict` 是接收 QICK `prog_list` 的外部序列化邊界。缺少必要 key 時拋出 `KeyError`，parse 失敗時拋出 `ValueError`，讓呼叫端在反序列化時取得明確錯誤。

- IR 中的 `Instruction`（`BaseInst` 的子類，如 `RegWriteInst`, `PortWriteInst`, `WmemWriteInst` 等）的 `from_dict` 與 `to_dict` 實作必須嚴格對應 QICK `tprocv2_assembler.py` 裡的欄位。
- 序列化只使用底層 QICK 支援的欄位，並保留精確欄名，例如 `DST`。`WW`、`WP` 等 side-effects 也必須完整序列化，讓 `prog_list` 符合 QICK 格式。

# Task Plan: IR / Macro / Modules

## Goal

1. 維持 `program/v2/ir` 的正確性與 strict parsing 邊界。
2. 維持 IR 作為「可選優化層」的 U-shape pipeline 架構。
3. 把後續工作集中在真正未完成的 coverage / 文件 / 新功能項目。

## Current State

- IR pipeline 已完成 U-shape 重構。
- dispatch 已改為 dmem-backed 方案（IR 開啟時）。
- Phase 16 所有修復已合入（commit `90eb9f65`）：SimplifyDispatchPass bug 、BlockMergePass、pipeline 重構、LoopConditionMergePass 刪除。
- `task_plan.md` / `findings.md` / `progress.md` 保持精簡，供後續新增項目使用。

## Architecture Baseline

```text
list[Instruction]
  -> IRLexer.lex
  -> ChunkList optimization
  -> IRParser.parse
  -> IR-tree optimization
  -> IRParser.unparse
  -> strip structural MetaInst
  -> ChunkList optimization
  -> IRLexer.flatten
```

### 關鍵決策

- IR 優化是可選層；macro/module 層本身必須能生成可執行 ASM。
- tree pass 介面固定為 `transform(node) -> Optional[IRNode]`。
- post-order 遞歸由 pipeline 掌控，不由各 pass 自行遞歸。
- `DISABLE_OPT` 靠 `BasicBlockNode.disable_opt` 傳遞，不靠 meta 保留。
- `IRBranch` 正常路徑由 `UnpackIRBranchPass` 展開；fallback `parse(unparse(...))` 仍必須回到乾淨 AST。
- `PipeLineContext.allocated_names` 是單次 pipeline run 的共享名稱池，供 tree pass 做全域唯一 label 配置。

## Completed Milestones

- Phase 1-3：review 初始 bug 修復（TimedMerge / strict parsing / WaitInst / module consistency）
- Phase 4-5：U-shape pipeline 重構
- Phase 6：文件與收尾修正
- Phase 7：dmem-backed dispatch
- Phase 8：IR 測試補強
- Phase 9：最近 20 個 commit review
- Phase 10：review follow-up 修復（IR-8 / IR-9 / IR-10）
- Phase 11：補完 `instructions.py` 測試覆蓋率（`78% → 94%`，324 passed）
- Phase 12：補完 `program/v2` 其餘 coverage 缺口（整體 `91% → 97%`）；涵 蓋 `waveform.py`、`linker.py`、`node.py`、`operands.py`、`macro/debug.py`、`sweep.py`、`mocksoc.py`、`modules/util.py`、`utils.py`
- Phase 13：實作 `ComputedPulse` DMEM Lookup 機制——以 dmem 表取代連續 wmem 假設，runtime 動態查 wave index（commit `dc97b80b`）
- Phase 14：Code Review 修復——消除 `_needs_big_jump` 重複、`LoadValue.run()` 改用 `SR`、`TimedMergePass` docstring 修正、刪除死屬性、魔術數字抽常 數（673 passed）
- Phase 15：程式碼品質與測試覆蓋修正——`LoadValue` 值域限制改為 0–2³¹-1、`computed_pulse.py` generator expression、補測試覆蓋缺口（758 passed，整體 97%）

## Remaining / Next Phases

### Phase 16：IR Pipeline 正確性與重構修正

**來源**：`2026-05-18` 第二輪 review 發現的問題

#### 16.1 [Bug] 修復 `SimplifyDispatchPass` — 兩個分支都必須顯式 jump

- **問題**：`ir/passes/control_flow/simplify_dispatch.py` 將 2-target `IRDispatch` 替換為單一 `BasicBlockNode`，Z 路徑（value_reg == 0）依賴 fallthrough 到 `target_labels[0]`。但 pass 只看到 `IRDispatch` 本身，不知道其 後的 block 是誰，fallthrough 假設不穩固。
- **修復**：回傳 `BlockNode([cond_bb, fallthrough_bb])`，兩個分支都顯式 jump：
  - 小 PMEM：`cond_bb` 條件跳 target1；`fallthrough_bb` 無條件跳 target0 （`BranchEliminationPass` 在 target0 緊接時自動消除此 jump）
  - 大 PMEM：兩個分支都用 `REG_WR s15 label; JUMP [s15]`（間接 jump 不被 `BranchEliminationPass` 消除，永遠保留，多 2 words）
- **受影響檔案**：
  - [x] `lib/zcu_tools/program/v2/ir/passes/control_flow/simplify_dispatch.py`
  - [x] `tests/program/v2/ir/test_ir_validation.py` — 更新 3 個 SimplifyDispatch 測試（result 型別從 `BasicBlockNode` 改為 `BlockNode`，並驗證兩個 jump）

#### 16.2 [Cleanup] 移除 `LoopConditionMergePass`

- **問題**：此 pass 的目標 pattern（`REG_WR r1 op r1-#1` + `JUMP -if(NZ) -op(r1-#0)`）在當前 Module / Macro 層不會產生。`CloseInnerLoop` 使用 `S`-condition（`counter < n`），不是 `NZ`。Pass 存在但永遠不會被觸發。
- **修復**：刪除 pass 及其測試，從 pipeline 移除。
- **受影響檔案**：
  - [x] 刪除 `lib/zcu_tools/program/v2/ir/passes/loop/condition_merge.py`
  - [x] 刪除 `tests/program/v2/ir/test_ir_passes_loop_merge.py`
  - [x] `lib/zcu_tools/program/v2/ir/passes/loop/__init__.py`
  - [x] `lib/zcu_tools/program/v2/ir/passes/__init__.py`
  - [x] `lib/zcu_tools/program/v2/ir/pipeline.py` — 移除 import 與 `make_default_pipeline` 引用
  - [x] `tests/program/v2/ir/test_ir_passes_optimization.py` — 移除相關 test case

#### 16.3 [Refactor] 合併 `_run_chunk_passes` / `_run_chunk_list_passes`

- **問題**：`ir/pipeline.py` 中 `_run_chunk_passes`（line 149）與 `_run_chunk_list_passes`（line 171）兩函數主體完全相同，只差型別標注。
- **修復**：新增統一 helper `_run_passes(passes, chunks, ctx)`，接受 `list[Union[AbsChunkPass, AbsChunkListPass]]`；`_run_chunklist_opt` 對兩個 pass 列表分別呼叫 `_run_passes`（維持分組語意）；刪除兩個舊函數。
- **受影響檔案**：
  - [x] `lib/zcu_tools/program/v2/ir/pipeline.py`

#### 16.4 [Fix] `BlockMergePass` — 每次迭代重算 `referenced`

- **問題**：`_merge_pass` 的 while 迴圈外只算一次 `referenced = collect_referenced_labels(chunks)`。合併後有些 label 可能已不再被引用，但 set 未更新，導致本可繼續合併的 block 被保留（missed optimization）。
- **修復**：將 `referenced = collect_referenced_labels(chunks)` 移入 while 迴圈內，每次迭代重算。
- **受影響檔案**：
  - [x] `lib/zcu_tools/program/v2/ir/passes/control_flow/block_merge.py`

#### 16.5 驗證

```bash
.venv/bin/python -m pytest tests/program/v2/ -q
pyright lib/zcu_tools/program/v2/ir/
ruff check --fix lib/zcu_tools/program/v2/ir/ tests/program/v2/ir/
ruff format lib/zcu_tools/program/v2/ir/ tests/program/v2/ir/
```

- **結果**：`673 passed`；pyright 0 errors；ruff 通過
- **覆蓋率**：整體 97%，`simplify_dispatch.py` 100%，`block_merge.py` 100%
- **Status:** complete（commit `90eb9f65`）

---

### Phase 17：IR 模塊責任最小化與介面整理

**來源**：`2026-05-18` 對 `ir/` 模塊責任分析，目標是讓每個模塊只做自己該 負責的事，介面與型別本身就能描述行為。

#### 17.1 [Refactor] `Label.__str__` 改為回傳裸名（不帶 `&`）

- **問題**：`Label.__str__` 回傳 `&{name}`，與 `MemAddr.__str__` 的 `&{n}` 格式相同，容易混淆；
  `&` 前綴是 QICK serialization 格式，屬於呼叫端關注，不應內聚於 Label identity 層。
- **影響分析**：
  - `factory.py:147` — `str(lbl.name)` 與 `factory.py:162` — `str(jump.label)` 兩側同步改變，比較仍一致。
  - `linker.py:38` 的 `f"&{p_addr}"` 是 address reference 格式，與 label name 無關，不受影響。
  - 目前沒有任何序列化路徑期待 `str(label)` 帶 `&`（QICK dict 都走 `.as_label().name`）。
- **附帶簡化（17.2 必須同步）**：改後 `str(LabelRef)` 的語義剛好等於 QICK LABEL 欄位的序列化形式：
  - 普通 label → `{name}`
  - pseudo-label → `"HERE"` / `"NEXT"` / …
- **受影響檔案**：
  - [x] `lib/zcu_tools/program/v2/ir/labels.py` — `Label.__str__` 改為 `return self.name`

#### 17.2 [Refactor] 統一 `to_dict()` 中的 LabelRef 序列化

- **問題**：`JumpInst`、`RegWriteInst`、`DmemReadInst`、`CallInst` 的 `to_dict()` 各自有相同的 4 行模板：

  ```python
  if self.label is None:
      label_name = None
  elif not self.label.is_pseudo():
      label_name = self.label.as_label().name
  else:
      label_name = self.label.target
  ```

  完成 17.1 後，`str(LabelRef)` 的語義即為序列化形式，可縮成 1 行：

  ```python
  label_name = str(self.label) if self.label is not None else None
  ```

- **受影響檔案**：
  - [x] `lib/zcu_tools/program/v2/ir/instructions.py` — 4 處 `to_dict()` 改用 `str(self.label)`
  - [x] `lib/zcu_tools/program/v2/ir/instructions.py` — `LabelInst.to_dict()` 中的 `self.name.name` 改為 `str(self.name)`

#### 17.3 [Refactor] `collect_referenced_labels` 移至 `analysis.py`

- **問題**：`labels.py` 內的 `collect_referenced_labels(chunks)` 以懶 import 方式載入
  `instructions.BaseInst` 和 `node.BasicBlockNode`，用延遲 import 迴避循 環依賴。
  「標籤字串表示」層不應知道指令型別；`analysis.py` 本就負責靜態分析且已 可見 instruction/node 型別。
- **受影響檔案**：
  - [x] `lib/zcu_tools/program/v2/ir/analysis.py` — 新增 `collect_referenced_labels`
  - [x] `lib/zcu_tools/program/v2/ir/labels.py` — 移除函數（更新 `__all__`）
  - [x] 所有 import `collect_referenced_labels` from `labels` 的地方改為 從 `analysis` import：
    `passes/control_flow/dead_label.py`、`passes/control_flow/block_merge.py`

#### 17.4 [Refactor] `needs_big_jump` 移至 `hw_semantics.py`

- **問題**：`needs_big_jump(pmem_size)` 是純硬體閾值謂詞（`pmem_size > 2048`），與 dispatch table
  建構邏輯無關，應與 `TIMED_BASE_REG`、`ADDR_REG` 等常量同住 `hw_semantics.py`。
- **受影響檔案**：
  - [x] `lib/zcu_tools/program/v2/ir/hw_semantics.py` — 新增 `needs_big_jump`
  - [x] `lib/zcu_tools/program/v2/ir/dispatch.py` — 移除函數，改從 `hw_semantics` import
  - [x] `lib/zcu_tools/program/v2/macro/loop.py` — import 路徑改為 `from ..ir.hw_semantics import needs_big_jump`
  - [x] 其餘所有 `from .dispatch import needs_big_jump` → `from .hw_semantics import needs_big_jump`

#### 17.5 [Refactor] `BaseInst.remap_labels()` 介面，解耦 `node.py` 對指 令型別的依賴

- **問題**：`node.py` 的 `_remap_inst()` 明確列舉 `JumpInst, RegWriteInst, DmemReadInst, CallInst`
  並對 `AluExpr.rhs` 是 `DmemAddr` 的情況做特殊處理——這是 node 層對 instruction 層的深入耦合。
  未來新增任何持有 `LabelRef` 的指令型別時，必須同時修改 `node.py`。
- **修復**：在 `BaseInst` 加 `remap_labels(remap: dict[str, Label]) -> BaseInst`（預設 `return self`）；
  各需 override 的指令實作自己的重映射邏輯；`_remap_inst` 退化為 `inst.remap_labels(label_remap)`。
- **受影響檔案**：
  - [x] `lib/zcu_tools/program/v2/ir/instructions.py` — `BaseInst` 加 `remap_labels` 預設實作；
    `JumpInst`、`RegWriteInst`、`DmemReadInst`、`CallInst` 各自 override
  - [x] `lib/zcu_tools/program/v2/ir/node.py` — `_remap_inst` / `_remap_op` 合併為一行呼叫

#### 17.6 驗證

```bash
.venv/bin/python -m pytest tests/program/v2/ -q
pyright lib/zcu_tools/program/v2/ir/
ruff check --fix lib/zcu_tools/program/v2/ir/ tests/program/v2/ir/
ruff format lib/zcu_tools/program/v2/ir/ tests/program/v2/ir/
```

- **結果**：`673 passed`；pyright 0 errors；ruff 通過
- **Status:** complete（commit `ea7af7d1`）

---

### Phase 18：修復 inner-loop 條件 JUMP 缺少 UF=1 的 bug

**來源**：`2026-05-18` 實際執行 zigzag / RB 實驗發現的 hang bug

#### 問題

tProc v2 ISA 的 `JUMP -if(S/Z) -op(A - B)` 只有在同時設置 `UF: '1'` 時， 才會先計算 OP 並更新 condition flags，然後根據更新後的 flag 跳轉。沒有 `UF: '1'` 時 OP 雖計算但 **flags 不更新**，`IF` 條件測試的是 stale flag。

QICK 的外層 `CloseLoop` 使用 `TEST counter - (n-1)`（assembler 強制 TEST 為 UF=1），在大多數外層迭代留下 S=1（負值）。我們的 inner-loop back-edge `JUMP -if(S)` 遇到 stale S=1 → 永遠跳回 → 無窮迴圈 → `acquire()` hang。

#### 症狀

- **zigzag**：有 times 外層迴圈，第一個 CloseLoop TEST 設 S=1，進入第二次 times 迭代即 hang（任何 reps 值）
- **RB**：reps=1 時初始 S=0（無 stale 問題）可正常結束；reps≥2 時 reps CloseLoop TEST 設 S=1，第二個 reps 迭代 hang

#### 修復（Part 1）— disable_opt=True 路徑

- [x] `lib/zcu_tools/program/v2/macro/loop.py` — `_emit_cond_jump()` 所有 AsmInst 加 `'UF': '1'`
- [x] `lib/zcu_tools/program/v2/ir/factory.py` — `_loop_prologue()` guard JumpInst、`_loop_epilogue()` back-edge JumpInst 加 `uf=True`

#### 修復（Part 2）— disable_opt=False 路徑（IR passes 自行生成的 JUMP）

IR 優化開啟時，各個 pass 自行生成 `JumpInst`，全部也遺漏 `uf=True`：

- [x] `lib/zcu_tools/program/v2/ir/passes/loop/unroll.py` — `_unroll_partial` back-edge、`_maybe_build_jump_table` guard + remainder check + back-edge（共 8 處）
- [x] `lib/zcu_tools/program/v2/ir/passes/control_flow/simplify_dispatch.py` — `NZ` 條件 JUMP（2 處）
- [x] `lib/zcu_tools/program/v2/ir/passes/control_flow/dmem_dispatch.py` — `NS` out-of-range guard（2 處）
- [x] `lib/zcu_tools/program/v2/ir/factory.py` — `_lower_dispatch` fallback guard（2 處）

#### 驗證

- **結果**：`673 passed`；pyright 0 errors；ruff 通過
- **Status:** complete（commit `dd66fc31` Part 1 + `d3780a73` Part 2）

---

### Phase 19：RB 改為「每個 seed 一次 acquire + depth 內掃描」架構

**來源**：`2026-05-18` RB 效能/一致性分析後的重構提案

#### 19.1 目標

- 目前 `RB_Exp.run()` 對每個 `(seed, depth)` 都建立/快取一個 program 並執行一次 `measure_fn`，rounds 由 `Task.repeat("rounds", rounds)` 在 host 端實作。
- 新方案改為：**每個 seed 只建一個 program**，在 FPGA 端 sweep `depth_idx` 一次拿到全部 depth 的結果，rounds 回到 `cfg.rounds` 的原生 acquire 流程。

#### 19.2 可行性評估（結論：可行）

- 現有模組已支援此模式（參考 `twotone/allxy.py`）：
  - `sweep=[("depth_idx", len(depths))]`
  - 以 `LoadValue(values=..., idx_reg="depth_idx", val_reg=...)` 做每個 sweep 點的參數查表
  - `Repeat(name, n=<register>)` 支援 register-driven loop count
  - `ComputedPulse` 可由 gate id register 動態選 pulse
- 因此 RB 可改為「前段 random 序列 + 最後 recovery pulse」兩段式執行，無 需為每個 depth 重建程式。

#### 19.3 設計草案（seed 內單程式）

對每個 seed 預先產生：

1. `rand_gate_seq`: 最大深度對應的完整 random 基礎 gate 序列（不含 recovery）
2. `prefix_len_by_depth`: 每個 depth 要執行的 random gate 數量（對應 prefix 長度）
3. `recovery_gate_by_depth`: 每個 depth 對應的 recovery gate id

Program 內流程：

1. sweep 變數：`depth_idx`（長度 `len(depths)`）
2. `LoadValue(prefix_len_by_depth, idx_reg="depth_idx", val_reg="rand_len")`
3. `LoadValue(recovery_gate_by_depth, idx_reg="depth_idx", val_reg="recovery_gate")`
4. `Repeat("rand_gate_idx", n="rand_len", range_hint=(0, max_rand_len))` ：
   - `LoadValue(rand_gate_seq, idx_reg="rand_gate_idx", val_reg="gate_idx")`
   - `ComputedPulse("basic_gate", val_reg="gate_idx", pulses=...)`
5. `ComputedPulse("recovery_gate", val_reg="recovery_gate", pulses=...)`
6. `Readout(...)`

#### 19.4 dmem 規模與資料佈局

- 主要 dmem 需求為：`len(rand_gate_seq) + 2*len(depths)`（不是 `2*len(rand_seq)`）。
- 其中 `rand_gate_seq` 可能大於 `max_depth`（Clifford 經 reduce 後每步 pulse 數不固定），需在 seed 建表時精確計算。

#### 19.5 需同步調整的 RB host 端流程

- `RB_Exp.run()`：
  - 移除 `cfg.rounds = 1` 與 `Task.repeat("rounds", rounds)`。
  - `Task` 改為只掃 seed（每個 seed 一次 acquire，輸出 shape=`(n_depths,)`）。
  - `prog_cache` key 由 `(seed, depth)` 改為 `seed`。
- `average_signals()` 與 live plot update 需要改成新資料維度（不再是 `n_seeds x rounds x n_depths`）。

#### 19.6 風險與注意事項

- `prefix_len_by_depth` 必須對齊 `reduce_gate_seq` 後的實際 pulse 數；不 可直接用 clifford depth 當 gate 次數。
- register-driven `Repeat` 的 `range_hint` 應提供合理上限，避免 optimizer 估算過度保守。
- rounds 交回 acquire 後，若要保留「每輪異常值遮罩」策略，需要重新定義 NaN 聚合行為。

#### 19.7 實作子任務（完成）

- [x] 在 `rb.py` 抽出「seed→(rand_gate_seq, prefix_len_by_depth, recovery_gate_by_depth)」前處理 helper。
- [x] 重寫 `measure_fn` 為 seed 級單程式 + `depth_idx` sweep。
- [x] 調整 `run_task` 掃描結構與 `average_signals`。
- [x] 補測試：確認新舊流程在固定 seed/depth 下 gate 序列與 recovery 對齊 。
- [x] 驗證：`.venv/bin/python -m pytest tests/experiment/v2/twotone/test_rb.py -q`、`pyright lib/zcu_tools/experiment/v2/twotone/rb.py tests/experiment/v2/twotone/test_rb.py`、`ruff check --fix ...`、`ruff format ...`。

#### 19.8 偏好對齊（完成）

- [x] `RB_Exp` 改為直接在 `ModularProgramV2(..., modules=[...])` 內列出 module list（不使用中間 `program_modules` 變數）。
- [x] random 段 `Repeat` 固定使用 `n="rand_len"`（不使用 `repeat_n` 變數 做 `n=0` 分支）。
- [x] `LoadValue` 支援 `values=[]` 短路 no-op（`init`/`run` 不配置 dmem、不發指令），讓 RB 不需要 `rand_gate_values` sentinel。
- **Status:** complete（commit `8a3cf83b`）

---

### Phase 20：修復 LoadValue 壓縮路徑 OverflowError

**來源**：`2026-05-19` 使用 LoadValue 時 `compile_datamem()` 拋出 `OverflowError: Python int too large to convert to C long`

#### 問題

`_pack_values()` 將多個小整數打包進單一 32-bit word（e.g. 4 × 8-bit 值，slot 3 在 bit 24 → max `0xFF000000 > INT32_MAX`）。`compile_datamem()` 使 用 `np.array(..., dtype=np.int32)`，超出 INT32_MAX 的 Python int 觸發 OverflowError。

#### 分析

bit 31 的限制只對**未壓縮**路徑成立（直接 `read_dmem` 當整數用）。壓縮路 徑結尾執行 `SR` + `AND value_mask`，無論是 arithmetic 或 logical shift，AND 都能清掉符號擴展位元，值還原正確。

#### 修復

`_pack_values()` 在打包後加一行 two's complement 轉換：

```python
if word > _INT32_MAX:
    word -= 1 << 32
```

等同 bit pattern 不變，讓 `np.int32` 接受，`compile_datamem()` 不動。

- [x] `lib/zcu_tools/program/v2/modules/dmem.py` — `_pack_values()` 加 two's complement 轉換；更新 `_INT32_MAX` comment
- [x] `tests/program/v2/modules/test_dmem.py` — 更新 16-bit 壓縮測試；新 增 `test_packed_words_fit_in_int32`

- **結果**：22 passed
- **Status:** complete（commit `1ab698cc`）

---

### Phase 21：修復 Repeat 與 Optimizer JUMP 邏輯的底層 Flag 評估 Bug

**來源**：`2026-05-19` Zigzag Sweep 測量異常行為除錯時發現。

#### 21.1 問題描述

在 tProc v2 架構中，帶有 `if_cond` 的 `JUMP` 指令若同時包含 `op` 並設置了 `-uf`（`UF=1`），其行為是**並行**評估與計算：它會使用**執行前的陳舊 Condition Flags** 來判斷是否跳轉，並同時計算新的 `op` 結果寫入下一週期的 Flags。
因此，`JUMP -if(S) -op(r1 - r0) -uf` 這樣的寫法是無效的，它判斷的不是當前 `r1 - r0` 的正負，而是上一條更新 Flag 的指令所留下的狀態。在 `Repeat` 迴圈以及 Optimizer 生成的 Jump Table 中，這導致了迴圈的 back-edge JUMP 讀到的是迴圈啟動前（或是外部迴圈）所遺留的 Flags（通常 S=0），造成條件永遠不成立（S=0 認為大於等於），迴圈永遠只會執行精確的 1 次（發生 fall-through）。

#### 21.2 修復計畫

- [x] **Macro 層修復 (`loop.py`)**：修改 `_emit_cond_jump` 函式，將帶有 `-if` 條件與 `-op` 的組合拆分為兩條指令：
  1. `TEST` 指令（或類似的算術指令）計算 `op`，專門負責將計算結果的 Flag 寫入狀態暫存器。
  2. 單純的 `JUMP -if(...)` 指令，不帶 `op`，負責在下一時鐘週期評估前一指令正確更新的 Flag 並執行跳轉。
- [x] **IR Optimizer 層修復**：直接在所有生成 `JumpInst` (含 `if_cond` 和 `op`) 的階段，拆解為 `TestInst` 加上 `JumpInst`，影響範圍包含：
  - `lib/zcu_tools/program/v2/ir/factory.py` (迴圈 epilogue 及 dispatch fallback)
  - `lib/zcu_tools/program/v2/ir/passes/control_flow/simplify_dispatch.py` (分支化簡 jump)
  - `lib/zcu_tools/program/v2/ir/passes/control_flow/dmem_dispatch.py` (越界檢查 guard jump)
  - `lib/zcu_tools/program/v2/ir/passes/loop/unroll.py` (迴圈展開的 back-edge 及 jump table 的 guard/back-edge)
- [x] **測試與驗證**：
  - 更新相關 `IR` 指令的測試，確保新的拆分邏輯能正確執行。
  - 確保含有 `Repeat` 的程式在真實硬體或模擬環境中能夠真正執行超過一 次，不再只有迴圈 1 次的 bug。
- **Status:** complete（commit `394f7b1b`）

## Default Verification Checklist

```bash
pytest -q
uv run pyright
uv run ruff check --fix
uv run ruff format
```

## Notes

- planning 檔案預設不進 git。
- `AI_NOTE.md` 只保留高層概念、架構、重要設計決策，不寫實作細節。

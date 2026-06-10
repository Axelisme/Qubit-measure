# Findings & Decisions

## Scope

- Review 範圍：`program/v2/ir`、`program/v2/macro`、`program/v2/modules`
- 主要對照：QICK `asm_v2.py`、`tprocv2_assembler.py`

## Important Historical Findings

### 已修復的高價值問題

- `IR-1`：`TimedMergePass` 不可吸收 `WAIT time @T`；`WaitInst` 必須是 flush barrier。
- `IR-2`：`DivInst.from_dict()` 不可 silent fallback。
- `IR-3`：`parse_register()` 必須只接受合法 alias / 裸 `sN/rN/wN`。
- `IR-4`：`WaitInst.addr` 只能是 `s15`。
- `IR-8`：`IRBranch` 的 fallback `parse(unparse(...))` 必須回到乾淨 AST ，不能把 synthetic jump 汙染進 case body。
- `IR-9`：`UnpackIRBranchPass` 生成的 label 必須保證整棵 tree 唯一，不可只做 subtree-local 去重。
- `IR-10`：`DportWriteInst.from_dict()` 必須在 IR parse 階段拒絕非法 `DATA` 字串。

### 已定案的架構決策

- IR 優化改為可選層；macro/module 層是 baseline ASM 生成者。
- pipeline 採單次 U-shape 流程。
- tree pass 遞歸由 pipeline 統一掌控。
- `IRBranch` 在正常優化路徑中由 `UnpackIRBranchPass` 展開成 `IRDispatch + cases`。
- dispatch 在 IR 開啟時採 dmem-backed 方案；fallback 仍保留 pmem island 路徑。
- `DmemAddr` 採「引用」而非「已分配資源」模型，resolve 在所有 clone 完成後才進行。

## Current Technical Baseline

### Branch / Dispatch 契約

- `IRBranch.cases` 只承載 case body，不承載 dispatch skeleton。
- `parse(unparse(IRBranch))` 必須維持 round-trip 純度。
- `UnpackIRBranchPass` 可沿用既有 case-entry label；若需生成新 label，必須走全域名稱池。
- `IRDispatch` 是 tree node；case bodies 是 sibling，不在節點內部。

### Strict Parsing 契約

- `from_dict()` 失敗要盡量在 IR 邊界直接 `raise ValueError`。
- 不接受把未知字串包成假 register 再交給 assembler 爆炸的做法。

## Remaining Risks / Future Attention

- `program/v2/AI_NOTE.md` 的部分架構描述仍可能過時，之後宜統一整理。
- dmem dispatch 依賴 dmem 容量；大量 dispatch 的程式仍需注意 dmem 使用量。
- 若未來新增會 clone chunk/subtree 的 pass，必須維持「resolve dmem dispatch 在 clone 之後」這個不變量。
- 若未來新增直接生成 `IRBranch` / `IRDispatch` 的 pass，必須遵守目前的 round-trip 與 label uniqueness 契約。
- **`ComputedPulse` 與 `PulseRegistry` 的衝突風險（嚴重）**：`ComputedPulse` 依賴在 runtime 透過連續的 waveform memory (wmem) index 進行動態索引切換（`base_wmem + gate_idx * stride`）。然而，`PulseRegistry` 預設會進行 deduplication，若 candidate pulses 剛好與其他註冊過的脈衝重複，會被分配到非連續的 index，進而導致 `ComputedPulse.init()` 的連續性檢查失敗（拋出 `ValueError`）。**需要透過顯式傳入 `pulse_id` 給內部的 `Pulse`，來繞過 registry 的自動複用機制。**

## Phase 14 Review Findings（2026-05-18）

以下問題由全面 review `ir/`、`macro/`、`modules/` 後發現，待 Phase 14 修復。

| ID | 嚴重性 | 位置 | 問題摘要 |
|----|--------|------|----------|
| R14-1 | High | `macro/loop.py:14` | `_needs_big_jump()` 私自複製 `ir/dispatch.py::needs_big_jump()`，不同閾值會靜默分歧 |
| R14-2 | Medium | `modules/dmem.py:110` | `ASR`（算術右移）用於無號 packed value，語意應為 `SR`（邏輯右移），AND mask 雖修正結果但閱讀性差 |
| R14-3 | Medium | `ir/passes/timeline/timed_merge.py` docstring | 把 timed port/wmem inst 統稱「transparent（不 flush）」，實際上無 `@T` 時讀 `TIMED_BASE_REG` 仍 flush |
| R14-4 | Low | `modules/dmem.py:42-43` | 死屬性 `self.addr_reg = ""` / `self.word_reg = ""`，從未被讀取 |
| R14-5 | Low | `modules/dmem.py:138` | 壓縮 threshold `30` 是魔術數字，應抽成常數並加說明 |

### 確認無問題的項目（原本被懷疑）

- `IncRegMergePass` ↔ `TimedMergePass` 振盪：**無問題**，literal `TIME inc_ref #N` 對 IncRegMerge 是 barrier，兩 pass 方向相反不會互相觸發。
- `_unroll_partial` back-edge `counter - n`：**正確**，do-while 語意，counter 從 0 開始，`counter < n` 時跳回。
- `UnreachableEliminationPass` 保留死區 `MetaInst`：**正確**，structural marker 必須存活。
- `WaitInst` 被 `TimedMergePass` 排除：**正確**，其 `@T` 是絕對時間比較，非 `s14` offset。

## Useful References

- QICK assembler: `.venv/lib/python3.9/site-packages/qick/tprocv2_assembler.py`
- QICK asm_v2: `.venv/lib/python3.9/site-packages/qick/asm_v2.py`
- IR architecture note: `lib/zcu_tools/program/v2/ir/AI_NOTE.md`
- Project-level note: `lib/zcu_tools/program/v2/AI_NOTE.md`
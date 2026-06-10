# Progress Log

## Summary

- `2026-05-16`：完成 `ir` / `macro` / `modules` 三模塊 review，建立 planning 檔。
- `2026-05-17`：完成架構對齊、U-shape pipeline 重構、dmem dispatch、IR 測試補強、最近 20 個 commit review、Phase 10 follow-up 修復。
- `2026-05-18`：完成第二輪全面 review，發現 5 個問題（1 High / 2 Medium / 2 Low），已記錄為 Phase 14，待修復。

## Milestones

### Review 與初始修復

- 找出初始 review issues，並完成最優先的正確性 / strict parsing 修復。
- 代表 commit：
  - `def454d1`
  - `113ec2a0`

### U-shape Pipeline

- 完成 IR pipeline 重構為單次 U-shape 流程。
- 完成 `Optional[IRNode]` tree-pass 介面與 pipeline-managed recursion。
- 代表 commit：
  - `e8defa08`

### Dmem Dispatch

- dispatch 改為 dmem-backed 方案。
- `IRBranch` 經 `UnpackIRBranchPass` 進入 tree-pass 路徑。
- 代表 commit：
  - `eb6269ce`

### Review Follow-up

- 修復：
  - `IRBranch` fallback round-trip 汙染 case body
  - `UnpackIRBranchPass` 全域 label uniqueness
  - `DPORT_WR.DATA` strict parsing
- 代表 commit：
  - `c7c5256e`

## Latest Validation Snapshot

- `.venv/bin/python -m pytest tests/program/v2/ -q` → `633 passed`
- `.venv/bin/python -m pytest tests/program/v2/ --cov=lib/zcu_tools/program/v2 --cov-report=term-missing -q`
  - total coverage: `94%`
  - `modules/waveform.py`: `99%`
  - `ir/linker.py`: `100%`
  - `ir/node.py`: `99%`
  - `ir/operands.py`: `80%`
  - `macro/debug.py`: `34%`
- `uv run basedpyright tests/program/v2/modules/test_waveform.py tests/program/v2/ir/test_ir_linker.py tests/program/v2/ir/test_ir_node.py` → `0 errors, 0 warnings, 0 notes`
- `uv run ruff check tests/program/v2/modules/test_waveform.py tests/program/v2/ir/test_ir_linker.py tests/program/v2/ir/test_ir_node.py` → pass
- `uv run ruff format tests/program/v2/modules/test_waveform.py tests/program/v2/ir/test_ir_linker.py tests/program/v2/ir/test_ir_node.py` → pass

## Latest Work

- Phase 12 已完成第一批 coverage 補強：
  - `tests/program/v2/modules/test_waveform.py`
  - `tests/program/v2/ir/test_ir_linker.py`
  - `tests/program/v2/ir/test_ir_node.py`
- coverage 變化：
  - `program/v2` overall: `91% -> 94%`
  - `modules/waveform.py`: `70% -> 99%`
  - `ir/linker.py`: `78% -> 100%`
  - `ir/node.py`: `77% -> 99%`
- 目前剩餘明顯缺口集中在：
  - `ir/operands.py` `80%`
  - `ir/base.py` `69%`
  - `macro/debug.py` `34%`
  - `sweep.py` `68%`
  - `mocksoc.py` `69%`

## Current Status

- 核心 IR 重構與 follow-up 修復已完成。
- `instructions.py` coverage 補強已達成階段性目標。
- Phase 12 第一優先中的 `waveform.py`、`ir/linker.py`、`ir/node.py` 已完成。
- Phase 14（review 問題修復）已完成：
  - R14-1：`macro/loop.py` 改用 `ir.dispatch.needs_big_jump`，刪除私有複製
  - R14-2：`modules/dmem.py` 改 `ASR` → `SR` 於無號值 shift
  - R14-3：`ir/passes/timeline/timed_merge.py` docstring 補充正確的 flush/transparent 說明
  - R14-4：刪除 `LoadValue.__init__` 中的死屬性 `self.addr_reg` / `self.word_reg`
  - R14-5：抽出 `_COMPRESS_MIN_VALUES = 30` 常數並加說明 comment
  - 驗證：`673 passed`；`basedpyright` 0 errors；`ruff` 通過
- planning 檔已精簡，可直接加入新的 phase / finding / progress 項目。

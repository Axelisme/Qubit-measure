# QICK Note for `program`

**Last updated:** 2026-06-08

這份筆記整理 `lib/zcu_tools/program` 對 QICK 的實際依賴，目的是讓後續開發能快速定位「應該看哪個 QICK 類別/方法」，而不用每次從頭追。

## 主要依賴面

- `ImproveAcquireMixin` 直接擴充 `qick.qick_asm.AcquireMixin`，是 `program` 層最核心的 QICK 接點。
- 主要加值分成四類：
  - 型別友善封裝（`TypedAcquireMixin`）：補強 `acquire()` / `acquire_decimated()` / `get_raw()` 等回傳型別。
  - 統計追蹤（`TrackerMixin`）：在 `finish_round()` 內注入 tracker 更新流程。
  - 提前停止（`EarlyStopMixin`）：透過 `stop_checkers`（callable list）在 `finish_round()` 內中止後續 rounds。
  - 每輪 callback（`RoundHookMixin`）與 single-shot population/threshold（`SingleShotMixin`）。

## `AcquireMixin` 行為對照（你目前程式最常踩到）

- 生命週期：
  - `acquire()`/`acquire_decimated()` 先建立 `acquire_params`，再進入 `prepare_round()` + `finish_round()` 迴圈。
  - `finish_acquire()` 負責 rounds 聚合（accumulated 或 decimated）。
- 你的擴充掛點：
  - `TrackerMixin.finish_round()`：利用原生 `self.acc_buf`、`self.ro_chs`、`self.avg_level` 更新 tracker。
  - `RoundHookMixin.finish_round()`：每 round 呼叫 `round_hook`，輸入增量摘要資料。
  - `SingleShotMixin._process_accumulated()`：覆寫原生 threshold 路徑，新增 population radius 分類。
- 相容性關鍵：
  - 你保留 `AcquireMixin` 的 `extra_args` 傳遞模式，避免破壞原生 acquire 參數流。
  - 若非 `accumulated` 或有 threshold 時，部分統計路徑會顯式 `NotImplementedError`，這是刻意防呆。

## soccfg 摘要顯示（`describe_soc`）

- `describe_soc(soccfg)`（`soc_summary.py`）是 `print(soccfg)`（QICK `QickConfig.description()`，很長）之外的精簡替代：generator / readout 各一張表，每列一個 channel，欄位為 type / 物理 port label / sample rate / max pulse 或 buffer length。
- port label（如 `0_230`）採 ZCU216 RF-DC 慣例 `block_(tile+228)`（DAC）/ `block_(tile+224)`（ADC），是 board-specific 假設，故 **只支援 ZCU216**，其他 board 直接 `raise NotImplementedError`（fast fail），不靜默產生誤導標籤。

## 快速查表（需求 -> 優先看哪裡）

- 想改「每回合後做什麼」：先看 `ImproveAcquireMixin` 各 `finish_round()`，再對照 `AcquireMixin.finish_round()`。
- 想改「accumulated 輸出格式」：看 `SingleShotMixin._process_accumulated()` 與 `AcquireMixin._average_buf()`。
- 想改「時間軸或原始資料格式」：看 `AcquireMixin.get_time_axis()`、`get_raw()`。
- 想改「執行控制（early stop）」：看 `EarlyStopMixin.acquire()` 的 `stop_checkers` 參數與 `finish_round()`。

## QICK ASM dict 注意事項

- 在 QICK asm v2，label definition 和 label reference 都可能使用 `LABEL` key，但需要靠 `CMD` 區分：
  - label definition 通常是只有 `LABEL`、沒有 `CMD` 的 dict。
  - `jump()` / `cond_jump()` 會產生 `{"CMD": "JUMP", "LABEL": ...}`。
  - `call()` 會產生 `{"CMD": "CALL", "LABEL": ...}`。
  - 大 program memory 模式可能先用 `REG_WR` + `SRC: "label"` / `LABEL` 寫入 `s15`，再用 `ADDR: "s15"` 跳轉。
- 因此 IR parser 不應只用 `"LABEL" in dict` 判斷為 label definition；有 `CMD` 時應保留為 instruction 並把 `LABEL` 視為 reference。

## IR pass 維護注意事項

- `IRBranch` 只有 `cases: list[IRNode]`，不含 `insts`；有 `insts` 的是 `BasicBlockNode` / `BlockNode`。
- `UnrollLoopPass` 依 `IRLoop.n: Union[int, Register]` 決定是否可展開；`n` 為 `Register` 時無法靜態展開，不要從 QICK loop control 指令猜 iteration count，除非後續已建立可靠的 typed instruction model。
- `ZeroDelayDCEPass` 只刪除 lower-level `TIME C_OP=inc_ref LIT=#0` 這類 reference-time zero increment，不刪 tagged、register-driven、`set_ref` 或 `NOP`。
- `TimedMergePass` 將 literal `TIME inc_ref #N` 吸收入下游指令的 `@T` 欄位（`TimeOffset += pending_lit`）。`PortWriteInst`/`DportWriteInst`/`WmemWriteInst` 等帶 `@T` 的硬體寫入指令可吸收；不帶 `@T` 但不讀寫 `s14` 的同類指令則透明通過（不中斷累積）；只有讀寫 `s14` 或 `WaitInst` 才會強制 flush。

## 原始碼出處

### 你專案中的對應實作

- `lib/zcu_tools/program/base/improve_acquire.py`
- `lib/zcu_tools/program/base/__init__.py`
- `lib/zcu_tools/program/__init__.py`

### QICK upstream（.venv 來源）

- `/.venv/lib/python3.13/site-packages/qick/qick_asm.py`
  - `class AcquireMixin`
  - `acquire()`, `acquire_decimated()`, `prepare_round()`, `finish_round()`, `finish_acquire()`
  - `_average_buf()`, `_apply_threshold()`, `get_raw()`, `get_time_axis()`
- `/.venv/lib/python3.13/site-packages/qick/__init__.py`
  - `QickConfig`, `AveragerProgram`/`RAveragerProgram`/`NDAveragerProgram` 匯出入口

## 維護建議（給未來自己）

- 若 QICK 升版後 `AcquireMixin` 介面變動，優先比對：
  - `acquire_params` 欄位名稱
  - `finish_round()` 回傳語意（是否仍為「還有下一 round」）
  - `_average_buf()` 的輸入 shape 假設
- 你目前 mixin 疊加順序是有語意的（`RoundHookMixin, TrackerMixin, SingleShotMixin, EarlyStopMixin`），改順序前請先檢查 MRO 對 `finish_round()`/`acquire()` 的影響。

---

## 更新紀錄

| 日期 | Codebase commit | 說明 |
|------|-----------------|------|
| （未知） | — | 初始建立，尚未追蹤更新歷程；下次修改時請補上對應 commit |
| 2026-04-26 | `cd0bc869` | 初次建立更新紀錄（本次全面審閱，內容與 codebase 相符） |
| 2026-04-27 | `5e09cf1c` | 修正 Markdown 結構：合併重複的「更新紀錄」區塊。 |
| 2026-04-27 | `3f9bb55f` | 對齊現況：`StatisticMixin/CallbackMixin` 更名為 `TrackerMixin/RoundHookMixin`，並更新 mixin 疊加順序。 |
| 2026-05-01 | `1f23b1a7` | 補充 QICK asm v2 label definition/reference dict 形狀，避免 IR parser 誤判 jump/call。 |
| 2026-05-01 | `0e195063` | 補充 IR pass 保守語意：branch case metadata normalize、explicit loop unroll、hoist/peephole/timing pass 限制。 |
| 2026-05-01 | `48d54c3b` | 補充 timeline passes 保守語意：zero-delay DCE 與 adjacent positive `TIME inc_ref` merge 限制。 |
| 2026-06-08 | `e26b200a` | 新增 `describe_soc`（soccfg 精簡 channel 表格，ZCU216-only port label）。 |

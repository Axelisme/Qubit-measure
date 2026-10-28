---
date: 2026-10-29
status: active
superseded_by: null
related:
  - lib/zcu_tools/cfg_model/module.py
  - lib/zcu_tools/program/acquisition.py
  - zcu_lab/v2/singleshot/ge/gui.py
  - zcu_lab/v2/mux/singleshot_ge/gui.py
  - docs/adr/0074-flat-readout-module.md
  - docs/discussions/2026-10-29-heralding-post-selection.md
basis:
  - "repo@c30bf2bbd"
  - "事實調查為未入庫的工作檔（2026-10-29），結論與必要事實寫在正文"
---

# Singleshot 分類參數放進 readout cfg

## 結論

1. `PulseReadout` cfg 與 mux 的每個 `MuxReadoutTone` 各加可選欄位 `classifier`（預設 None，GUI 放 Advanced 群組）。兩種模式：`circle`（g/e 中心、radius、可選 confusion matrix）與 `linear`（angle、threshold、可選 confusion matrix）。單位是正規化 ADC 值：原始積分值除以讀出長度，不扣 offset。Module 本身不使用這個欄位。
2. 數值由 library 的 readout module 擁有。singleshot GE（單 qubit 與 mux）寫回 module；point 參數 `$qubit.ext.readout_cal.*` 不再寫入也不再讀取，舊值不搬，改版後重跑 singleshot GE。
3. `acquire` 的分類參數由實驗作者明確傳入。readout module 提供工具，依自己的 tones 與發射順序算出型別化規格；`acquire` 不讀 program 的登記表。
4. Library YAML 的複數沿用 point 參數的 `{__complex__: [re, im]}` 格式。
5. Singleshot GE 維持兩段 writeback：Primary 寫只有 g/e 中心的 classifier，Post 再寫一次完整的 classifier，套用順序不影響結果。fidelity 與 sigma 沒有 reader，不再存。
6. 讀出設定改變後讓舊 classifier 失效的機制，這次不做。

## 問題

使用者提出：「關於原本singleshot的ge與confusion matrix，目前是透過隱式注入到實驗cfg達成，但mux實驗如果要支援就很麻煩，我考慮將center/radius/confusion matrix等資訊直接放到readout cfg裡的可選欄位中，大概在advance的子樹裡，也支援一般的線性區分模式，module本身不利用，但可以自然綁定到對應的readout module上，實驗cfg也不用bypass了」。

現況是約 15 個實驗的 Form 用 `Expr("$qubit.ext.readout_cal.g_center")` 一類的表達式，把 point 參數注入自己的 cfg 欄位，再以 `g_center=…, e_center=…, ge_radius=…` 傳給 `acquire`。`SingleShotMixin` 對所有 ADC 套同一組純量，mux 每個 tone 各自的校準無法表達。雙 qubit 序列另外用 `GECenters`。

## 考慮過的選項與比較

數值放在哪裡、誰寫入：

- A：library 的 readout module 擁有，singleshot GE 寫回 module。校準與讀出設定綁在一起，移植 point 時一起帶走。
- B：數值留在 point 參數，library 的 readout module 以表達式引用。需要先確認 library module 支援表達式。

使用者選 A。既有 `readout_cal` 數值：A 不搬、重跑 GE；B 寫一次性遷移。使用者選 A。順序：先做 classifier，heralding 接在後面，使用者同意。

`acquire` 的分類參數該不該由實驗作者傳入（使用者的提問）：

| 選項 | 寫法 | 評估 |
| --- | --- | --- |
| A 作者自取 cfg 傳入 | `prog.acquire(soc, classify=cfg.readout.classifier)` | 顯式；mux 的 tone 對應與讀出索引要手寫，容易錯位 |
| B `acquire` 自動掃描 readout module | `prog.acquire(soc, herald=True)` | 作者最省事；`acquire` 的結果會因 module cfg 默默改變，與「module 本身不利用」和「框架不能有默默依賴慣例的機制」相衝 |
| C readout module 提供工具，作者明確傳入 | `prog.acquire(soc, herald=mux.herald_reads(prog))` | 呼叫端看得到傳了什麼；tone 對應與讀出索引由唯一知道它們的 module 算出 |

使用者：「我也傾向C」。

實作細節的三個問題：

- QC1 複數格式：tag dict、字串或兩個 float。使用者採建議的 tag dict。
- QC2 GE writeback：A 由 Post 一次寫完整 classifier；B 維持兩段。使用者選 B：「後者全部再寫一遍應該就可以了？」。Post 寫完整 classifier 就讓順序無關。
- QC3 讀出設定改變後的失效機制：A 不做；B 在改設定的 writeback 清掉 classifier；C classifier 記下校準時的讀出設定，取用時比對。advisor 建議 C。使用者：「我認為分類參數比對需求還不確定，先A」。

## 被否決的選項與理由

- 數值留在 point 參數、由 module 以表達式引用：校準與讀出設定分離，且依賴 library 表達式能力。
- `acquire` 自動掃描 readout module：隱藏依賴，與框架原則相衝。
- 搬遷舊 `readout_cal` 數值：GE 重跑只需幾分鐘；不加相容邏輯。
- Primary 不寫 classifier、只由 Post 寫：使用者保留兩段 writeback。

## 留下的問題

- 讀出設定改變後的 classifier 失效機制（QC3），待有需求再議。
- 這一版沒有產生 linear classifier 的 writeback；linear 可在表單手動設定與使用。
- Mux 的 N 個 tone 合成一個 module writeback，draft 中無法逐 tone 取捨。
- 寫了 `classifier` 的 library，被舊版程式讀到時會因 `extra="forbid"` 視為壞條目；版本一致依使用者紀律。

## 依據

- `lib/zcu_tools/cfg_model/module.py`（readout Spec／Cfg 與 `MuxReadoutTone`）、`lib/zcu_tools/cfg_model/materialization.py`（持久化白名單，原本拒絕 Python complex）。
- `lib/zcu_tools/program/acquisition.py`（`SingleShotMixin.acquire` 的 `g_center/e_center/ge_radius`）。
- `zcu_lab/v2/singleshot/ge/gui.py`、`zcu_lab/v2/mux/singleshot_ge/gui.py`（現行 writeback 寫 `readout_cal`）。
- `lib/zcu_tools/core/complex_codec.py`（point 參數的複數格式）。

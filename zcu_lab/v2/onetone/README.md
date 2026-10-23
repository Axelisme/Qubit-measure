# One-tone 掃頻初值

**讀者：** lab-author。查閱 one-tone 表單的 readout timing 初值與共用來源。

**Last updated:** 2026-10-23，mux fine 初值來源

`freq`、`power_dep`、`flux_dep`、`range_find` 的 `make_form` 共用
[`_form_shared.py`](_form_shared.py) 的 `initial_readout_timing`。
`freq` 與 `range_find` 另共用 `initial_spectrum_tree`，集中 readout、linear sweep 與平均次數的初值。
[`_shared.py`](_shared.py) 保存不依賴 GUI 的 cfg、頻率格點與繪圖 buffer。
兩個 core 的 `run` 各自列出 reset → PulseReadout → declare_sweep → acquire 序列。
[`range_find`](range_find/guide.md) 掃大範圍絕對 MHz、找多個凹陷並提議各 resonator 的 `freq`。
Open／Reset 從固定 capture 讀取 resonator 校準，並以 `readout` Template 建立初值。

- `kappa` 使用 MHz，時間使用 us。TOF 讀 resonator 頂層 `$resonator.time_of_fly`，不從 ext 讀取。缺失或 null 時，trigger 與 pulse length 留空必填。
- 未設 `kappa`（缺 leaf 或 null）時，`trig_offset = time_of_fly`。
  已設時，`trig_offset = time_of_fly + 5 / (2 * pi * kappa)`。
  `kappa = 1 MHz` 時，等待增量約為 `0.795775 us`。
- `ro_length` 初值為 Template 原 pulse 長度減 `0.01 us`，目前為 `0.99 us`。
- `pulse length = max(Template 原長度, trig_offset + ro_length)`。
  Pulse 至少涵蓋 ADC 窗口，原長度較大時保留原長度。
- Open／Reset 產生 detached、可編輯的 timing。操作員之後的編輯保留於 cfg 輸出。
  Reset 重新捕捉來源並產生初值。
- 缺少 capture，或已設 kappa 不是數值、非有限／非正時，Open／Reset 明確報錯。
  Alias 與來源故障由來源 owner 回報。

## Mux 初值

[`mux/onetone_fine`](../mux/onetone_fine/form.py) 啟用 `mux_readout_initial` 的 ringup 初值政策。
有 captured readout Slot 時，副本保留來源的 `ro_length`，在原 `trig_offset` 上增加 ringup 等待。
缺 slot 時，從完整 group 校準推導 mixer 與 timing 候選，不沿用 blank Template 的 offset 或積分窗。
兩條路徑的公式、缺值條件與無效來源錯誤以 [fine guide](../mux/onetone_fine/guide.md#無-readout-slot-的候選來源) 為準。

Tone 槽位沿 group 順序產生。Timing 在 Open／Reset 後保持可編輯，cfg 輸出保留操作員的編輯。
候選未經裝置確認，Run 前須核對 mixer、接線、功率與 timing。

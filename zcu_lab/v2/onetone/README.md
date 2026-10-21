# One-tone 掃頻初值

**讀者：** lab-author。查閱 one-tone 表單的 readout timing 初值與共用來源。

**Last updated:** 2026-10-21，未設 kappa 的 timing 初值

`freq`、`power_dep`、`flux_dep` 的 `make_form` 共用
[`_shared.py`](_shared.py) 的 `initial_readout_timing`。
Open／Reset 從固定 capture 讀取 resonator 校準，並以 `readout` Template 建立初值。

- `kappa` 使用 MHz，時間使用 us。
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
共同等待為 `5 / (2 * pi * min(已設 group kappas)) us`，忽略缺 leaf 或 null 的成員。
全未設時不加等待，`trig_offset` 保留來源的 time-of-flight offset。

有 captured readout Slot 時，以 detached copy 的原 `trig_offset` 加上共同等待。
使用 `readout/mux:blank` Template 時，以其 `0 us` offset 加上共同等待。
兩條來源路徑都保留原 `ro_length`，並以
`length = max(原長度, 新 trig_offset + ro_length)` 覆蓋 ADC 窗口。
Tone 槽位沿 group 順序產生，scalar 初值沿所選來源保留。

每個已設 group member 的 kappa 都須為有限正數。任一已設值無效時，Open／Reset 明確報錯。
Mux 的 timing 同樣在 Open／Reset 後保持可編輯，cfg 輸出保留操作員的編輯。

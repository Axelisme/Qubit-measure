# ADR-0055：Resonator electrical delay 使用 route-scoped calibration prior

**狀態：** accepted

## Context

Nonuniform one-tone frequency grid 可用 adjacent-phase coherence 分辨 electrical-delay
branch，但有限搜尋範圍的 optimum 可能落在邊界。直接接受邊界解會把截斷結果當成 fit，
固定使用極大範圍則讓每次分析都付出不必要成本，且可能超過 candidate resource guard。

Uniform grid 只能識別 modulo `1/|Δf|` 的 local alias，不能單靠該次資料決定唯一物理
cable delay。另一方面，成功辨識的 delay 只對相同 generator/readout route 可信，不能在
不同 channel 間無條件共用。MetaDict 的內容寫入仍遵循 [[0006]] 的單一 write authority。

## Decision

- Core branch search 保留既有 initial radius；caller 額外提供 `max_search_radius` 時，
  boundary-limited search 以二倍幾何級數擴張，直到找到 interior optimum、到達 cap，或
  觸發既有 candidate resource guard。cap 只限制擴張，不縮小 initial search。
- One-tone GUI 的預設最大擴張半徑為 `100.0`，單位是 frequency 軸的反單位
  （frequency 為 MHz 時即 μs），並作為 analyze parameter 暴露。
- Analyze 提供 `auto`、`calibrated`、`manual` 三種 electrical-delay mode：
  - `auto` 優先使用 route-matched prior，否則執行 adaptive global search。
  - `calibrated` 必須取得 route-matched prior，缺失、無效或 route mismatch 時 fast-fail。
  - `manual` 必須取得 finite manual seed。
- prior/manual 值是 branch seed；circle-loss local refinement 仍會調整它。它們不是固定
  `edelay`，也不改變 explicit `edelay` 的 authoritative bypass contract。
- MetaDict 使用 `res_edelay`、`res_edelay_res_ch`、`res_edelay_ro_ch` 表達 calibration
  與 route identity。worker 只讀取這三個值，成功結果仍透過 analyze writeback proposal
  交給 `ContextService` 寫入。
- 只有 run snapshot 可辨識 pulse generator/readout route，且本次使用可信 prior/manual
  seed，或 frequency grid 為 nonuniform 時，才提出 route-scoped prior writeback。
  Uniform grid 在沒有可信 seed 時只保留 local canonical alias，不持久化為 calibration。
- Analyze parameter schema 是 remote wire contract；加入 mode/manual/max radius 時同步遞增
  measure-gui `WIRE_VERSION` 與 `GUI_VERSION`。

## Consequences

- 首次可辨識的 nonuniform fit 能安全跨越原本的搜尋邊界，成功後可把 delay 提案保存，
  後續相同 route 的 linear 或 homophasal fit 直接從同一 branch 做局部精修。
- 搜尋仍有 deterministic cap 與 candidate resource guard；真正超界時會明確失敗，不會
  無限擴張或接受截斷 optimum。
- 切換 generator/readout channel 不會沿用舊 cable-delay calibration；`auto` 退回 global
  search，`calibrated` 則要求 operator 先建立相符 prior。
- 從缺少 cfg snapshot 的舊資料載入仍可分析，但不會產生無 route identity 的 prior。

## Rejected alternatives

- **永遠使用極大固定 radius：** 每次 fit 成本增加，且不能取代 resource guard。
- **持久化 uniform-grid local alias：** alias 不代表唯一物理 cable delay，後續 nonuniform
  fit 可能被帶到錯誤 branch。
- **只存一個不含 route 的 global edelay：** 更換 generator/readout channel 後 calibration
  不再可信。
- **background worker 直接修改 MetaDict：** 破壞 [[0006]] 的單一寫入權威與 writeback
  preview/apply 流程。

---
date: 2026-11-22
status: active
superseded_by: null
related:
  - 2026-11-22-wall-time-fixture-flux-routing.md
  - 2026-11-22-ftf-projection-weight-floor.md
  - 2026-11-22-isotropic-receiver-noise.md
basis:
  decision: "0200"
  preset_commit: b1308a924ff9b650abb9837c55aaac52b8b41298
  dispatch_commit: c9586cd616d1805a3754031d8b8a4269528b284f
  external_sources: none
---

# FTF 標準 preset 與牆鐘工具的輸入

決策 0200 取代[保留工具自持 fixture 並將十列記為未量](2026-11-22-wall-time-fixture-flux-routing.md)的做法。使用者要求補齊標準預設，不在工具另養一份設定。FTF 只保留 [`default.yaml`](../../zcu_lab/simulation_envs/FTF/default.yaml) 作為標準 preset，不再以 `low-snr.yaml` 提供第二套雜訊。標準 preset 的接收雜訊以 half-flux 聯合工作點、各 tone 的 GE SNR 約為 2 為目標，不靠降低讀出 gain 或縮短 ADC 積分達標。

[牆鐘工具](../../tools/README.md#實驗-run-牆鐘)從標準 preset 取半磁通偏壓、快磁通路由、共用 coil、receiver PSD 與 projection floor。工具重用 GUI/MCP 的 entry 建立流程，從標準 template 和 module library 為每列建立新的 `ftf-demo/initial`。工具只停用 pacing（`time_scale: 0`），維持原板型和 acquire telemetry。標準 demo 不移植舊工作點的讀出 classifiers，也不需要使用者的既有 project 或 `ftf.yaml`。另一份場景會讓表單和計時工具各自維護偏壓、模組、路由和雜訊，即使名稱相同也可能不再是同一輸入。

原缺少快磁通或共用 coil 的十列，在 `c9586cd616d1805a3754031d8b8a4269528b284f`、seed 123、CPU 三 workers 的離線核對中均到達 Form Ready，且 Controller 發出 Run Started。這不是完整計時。`fastflux/distortion/freq` 在派送後遇到 flux kernel 路由拒絕；`fastflux/t1` 在派送後遇到板型的 pulse-length 上限。工具將這兩列記為 `run_failed`，不記為缺預設或未派送。

## 已知風險

- 工具原設定的 projection floor 是 `1e-9`（`201b3358fe6f7ece723d719aa9610efa76fd6f9f`）。工具現在採標準 preset 的 `1e-5`（`b1308a924ff9b650abb9837c55aaac52b8b41298`），可能跳過更多低權重投影並改變耗時。不要將舊 fixture 與新版的時間差當成演化效能變動。
- Step 12 的既有診斷在 `9888a2df3`、task-local FTF 同一 half-flux 偏壓及相同 mux GE 設定下，只宣告快磁通線、未打快磁通 pulse。Q2 的 e cluster 沿原 g/e 軸移動約 0.2%。重啟、版本與 RNG 也有變動，不能把位移單獨歸因給路由。
- `BL-20261122T090237904736Z-ftf-flux-reset-4-5` 記錄 `9888a2df3` 之後的 GUI 診斷：task-local FTF、Q1 smooth flat-top flux reset gain 0.32、hold 20/58/96 µs、各 12 shots，加入 reset 後整個 Run 約慢 4.5 倍。這是既有成本風險，不是本次 dispatch 核對結果。

---
date: 2026-11-16
status: active
superseded_by:
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
basis:
  - repo: c028a186328437798407d0e618bee3b0d4f925e8
---

# 成對 Engine 計時的 process、身份與統計

## 結論

唯一成對入口是 `tools.paired_benchmark`。Controller 直接管理兩側 child。
Warm 每側保留一個 process。First 與 cold 每筆建立新 process。
每筆 first 只呼叫一次既有 `SampleRunner(mode="first")`，不套 Pareto 的慢 family 規則。
Session、history 與 clock 仍由共同量測 owner 管理。
現行責任見 [ADR-0072](../adr/0072-module-boundaries.md#開發工具的數值交接)。

兩側先核輸入格式、快取格式、工具協定與逐筆 record 身份。
數值演算法版本只作 provenance，否則無法比較合法的數值優化。
格式 owner 見 [ADR-0078](../adr/0078-evolution-module-boundary.md#凍結數值輸入與參考解交接)。

## 問題

長駐 process 內只換 Engine/session，不等於新 process 第一次 evolve。
先前載入的 Numba signatures 會影響下一筆首次成本。
兩個版本若用不同 disk-cache 狀態，編譯或載入成本就會混進版本差異。
計時工具也不能因記錄 family CPU 的開銷而讓兩側用不同 accounting 模式。

## 考慮過的選項與比較

Controller 直接建立 measurement child，只需管理一層 process 生命週期。
Warm 保留 child，first/cold 則在每筆後關閉。
First 先在每側跑同 workload 的丟棄暖機 child，讓之後各筆載入同側既有 namespace。
Cold 每筆用新的空 Numba cache parent，child 結束後清理。
Process 啟動與 import 都不進 evolve timer。

ABBA 區塊以兩筆 candidate 總秒數除以兩筆 baseline 總秒數。
Wall 與 process 分開計算 median 和線性 IQR。
兩側各自執行 off/off 與 off/on 區塊，四個 overhead median 都在相應 noise IQR 內才共同啟用 telemetry。
超出時兩側都關，另跑同座標、空 history 診斷。
這些診斷不解釋成 telemetry-off 原樣本的 family 分解。

2026-11-16 的設計裁定採上述 process 邊界，並將 first 收窄為每 child 一筆。
它不增加共同 owner 的 API，也不改 Pareto 原有慢 family 協定。
這是 designer 的介面裁定，不記為使用者追認。

## 被否決的選項與理由

- 長駐 child 再轉送短命 measurement process 會多一層轉送與清理，controller 已能直接管理。
- 長駐 process 內重建 session 不能對齊新 process 的 Numba 載入狀態。
- 把 Pareto 的慢 family 補 warm 規則搬到成對 consumer 會複製 owner 的門檻，且需要擴大原本單筆 first 的介面。
- 用數值演算法版本當格式相等條件，會拒絕原本要量的數值優化候選。
- 校準與最終比較使用不同估計量，會讓 noise floor 與速度比值不再指同一件事。
- 兩側各選 telemetry 或只看 wall，會讓 process CPU 比值混入不對稱觀測開銷。

## 留下的問題

數值驗收仍需兩側含共同 committed 工具，以同版本量 noise floor 和同次獨占診斷。
Synthetic 協定測試不替代這項驗收。實際 CLI、K 與成本要在執行前凍結並取得資源授權。
這篇紀錄不宣稱完成任何吞吐量或冷快取重建驗收。

## 依據

- [共用計時 owner](../../tools/simulator_measurement.py) 與 [成對工具](../../tools/paired_benchmark.py)。
- [工具用法與 timer 邊界](../../tools/README.md#paired-engine-benchmark)。
- [Numba cache namespace](../../lib/zcu_tools/simulate/README.md#numba-machine-code-cache)。
- [共用模擬器計時討論](2026-11-16-shared-simulator-measurement.md)。遵循 prepared session 的職責分界。
- [Frozen inputs 討論](2026-11-16-frozen-simulator-inputs.md)。遵循 codec 擁有格式與 solver-free cache 的分界。

原始設計工作檔未入庫。本篇保存採用與否決的理由，現行契約由上述 ADR 和公開 API 擁有。

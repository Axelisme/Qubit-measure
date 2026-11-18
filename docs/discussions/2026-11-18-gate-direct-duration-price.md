---
date: 2026-11-18
status: active
superseded_by: null
related:
  - ../adr/0078-evolution-module-boundary.md
  - ../../lib/zcu_tools/simulate/evolution/README.md#decision-graph
  - 2026-11-18-short-dop853-dp54.md
  - 2026-11-16-floquet-kept-price.md
basis:
  repo_commit: 77dfb3450de2ee204f86d3c0d0f7667a98ba64c0
  external_sources: none
---

# Full-drive gate 的 CPU 競品估價

## 結論

使用者選擇以 row 的 duration 調整 `multitone_gate` 建構前比較的 CPU `dop853` 競品估價。
Duration 是價格的代理量，不是 frame 或 tableau 的選擇規則。
[選路說明](../../lib/zcu_tools/simulate/evolution/README.md#decision-graph)記錄係數與適用範圍。
這次調價不改 gate 建構和播放、RWA 比價、split 判準或 fallback。

## 問題

短 row 改用 [DP5(4)](2026-11-18-short-dop853-dp54.md) 後，原固定係數 0.15 在六組成形雙 tone 訓練案例上給出的估價／實測比為 0.383 至 1.984。
三組低於 0.5。六組一律使用相同的預算 1e-3、CPU pinned `dop853` 的 DP5(4) 端點，以及各組的 apply thread CPU 秒數。
估價偏差會改變是否先付 gate 建構成本，不代表求解精度有變。

## 考慮過的選項與比較

將係數限制在 0.12 到 0.24，以 25 至 100 ns 的 duration 線性內插。
這讓較長 row 的估價相對原比例上升，較短 row 的估價下降。
同一公式適用各種包絡和 frame。先用原實測 CPU 秒數重算訓練案例，不重解。

[ADR-0078](../adr/0078-evolution-module-boundary.md#內部允許實質耦合dispatch-擁有決策知識)將這項比較留在 dispatch。
這次沿用 [Floquet 價格重校](2026-11-16-floquet-kept-price.md) 的做法，只改價格，不改 guard。
這批證據比較的是估價和既有 apply CPU 秒數，並未重跑端到端速度量測。

## 被否決的選項與理由

固定係數無法使六組估價／實測比都落在 0.5 至 2。
Gaussian pair 要求係數至少為 0.195980，flat-top ramp 卻要求係數最多為 0.151229。
不按包絡或 frame 分開定係數；這次使用者選擇共用 duration 代理。

## 留下的問題

DRAG lab 與原 flat-top lab 連段是留出案例，尚未量測。
兩者各自的估價／實測比都須落在 0.5 至 2 才繼續預篩；失敗不拿留出資料重擬係數。
訓練比值不證明這兩組會通過，也不證明完整序列的速度或精度。

## 依據

原始訓練收據與工作檔未入庫。上文的比值及固定係數界線來自同六組 CPU DP5(4) 收據。

## 後記：加入 DRAG lab 訓練點

原 0.12／0.24 端點在 DRAG lab 留出窗得到 0.474 倍估／實比，低於 0.5。原 DRAG train
窗的 apply 為 2.533 CPU s，lab 連段的 apply 為 3.279 CPU s。同一 key 在兩個窗的成本
相差約 30%。這不是只改變入口態的對照實驗，不能據此隔離原因。

使用者將這個 DRAG 窗改列訓練點，保留原六組實測。新版取 25 ns 與 100 ns 各自可行係數
區間的幾何中心，分別為 0.0891 與 0.412，維持區間內線性內插及兩端 clamp。不按包絡或
frame 加係數。原 flat-top lab 連段是唯一未量的留出案例。若估／實比不在 0.5 至 2，即停止，
不再擬合。

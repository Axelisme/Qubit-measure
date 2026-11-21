---
date: 2026-11-21
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - ../../tools/README.md
  - 2026-11-15-simulator-test-measurement-layering.md
basis:
  repo_commit: 666eb48ac584d73141e232610cc67485b7e1eef3
  external_sources: none
---

# 實驗 Run 牆鐘工具

## 結論

使用者採用整個 Run 牆鐘作為 3 分鐘目標的指標，並要求量測工具入庫。這是 [ADR-0078](../adr/0078-evolution-module-boundary.md#模擬器的測試與量測分層不做-e2e) 的限定例外，不是恢復端到端測試。

## 問題

既有成對工具量 `Engine.evolve`，不含編譯、讀出或 Controller 交付結果的等待。先前定位發現 heralded 實驗約 168–211 秒的執行時間中，只有約 47–53 秒花在演化。只看演化會錯估操作員的等待。一次性腳本又各自手寫設定，無法用同一份輸入重跑耗時表。

## 考慮過的選項與比較

設計採用 offscreen Qt 的 `Controller(view=None)`。它沿用 GUI 已有的 Open、表單預設與 Run，不必為單一工具擴充框架。Lab 透過 assembly 註冊元件與實驗。工具保有 FTF 設定、board profile 和校準 point 的副本。組裝用公開 board builder，不另做 board 或元件註冊實作。

這遵循 [ADR-0072](../adr/0072-module-boundaries.md#使用者空間的依賴原則) 的 assembly 邊界及固定環境原則。與凍結 kernel 場景不同，實驗和表單正是牆鐘工具要觀察的對象，工具不把它們複製成自己的場景。kernel 比較仍遵循原先的分層裁決。

## 被否決的選項與理由

- 不保留只有 scratch 腳本的做法。新增實驗或調整模擬器時，維護者需要用同一個入口及固定輸入重跑。
- 不複製各實驗的核心 pulse 序列或手寫 cfg。這會繞過實際表單和 Run，而兩者的成本屬於量測範圍。
- 不新增不使用 Qt 的組裝接縫。只有一個新使用者，還要另行證明計時涵蓋既有 Run 路徑。
- 不沿用 lab 的可變 FTF 環境或呼叫者的 project。它們日後可能改變而未記入工具輸入。

## 留下的問題

首版只證明一個 adapter 能從 Open 跑到報表。全量覆蓋、覆寫與家族選擇不在這次交付內。非排他的 tracer 秒數不能用於正式比較。

## 依據

使用者於 2026-11-21 對整個 Run 作為指標、工具入庫但不當 gate，以及 kernel 比較仍量 `Engine.evolve` 的建議回答「都採用建議」。組裝方式是後續代裁。校準值取自 2026-11-20 的 FTF CZ 模擬校準；原始工作檔未入庫。執行方式見 [Tools README](../../tools/README.md#實驗-run-牆鐘)。

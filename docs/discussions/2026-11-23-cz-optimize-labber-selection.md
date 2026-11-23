---
date: 2026-11-23
status: active
superseded_by: null
related:
  - 2026-11-23-labber-scalar-index-axis.md
basis:
  repo: 37c2bbac59a004668e08a50df906daed5d49994d
  external: null
---

# CZ 最佳化的 Labber 選集

## 結論

CZ 最佳化的 Native 檔保留所有 Result 變數，包含起點時長 `duration`。Labber 各用一個 log 匯出 `params` 和 `peak`，保留各自的軸。`sums` 和 `duration` 只存在 Native 檔。Labber writer 恢復拒絕無軸變數。

2026-11-23，使用者說：「理論上RunResult都應該要能存檔與讀取還原」。對原則只管 Native、CZ 保留時長、撤回索引軸、Labber 只匯出候選參數與波形峰值的提議，使用者答：「都對」。

## 問題

起點時長在 Run 讀取波形時確定。Native cfg 只保存 arb 資產的路徑與身分，不保存時長。同一路徑覆寫後可能指向不同時長的波形，原檔移除後也不能從該路徑讀取。實作前以合成檔查核這兩種情況，原始查核紀錄未入庫。

Labber 的索引軸做法沒有修好這個實驗的匯出。即使略過無軸的 `duration`，分類計數、參數和峰值仍不共用網格。

## 選項與取捨

- 採用：保留完整的 Native 結果；以既有選集匯出 `params` 和 `peak`，用 independent layout 保存不同的軸。
- 移除 Native 時長並改從 cfg 讀取：讀到的是現存資產的時長，不是 Run 當時的時長。把數值補進 arb cfg 又會改變所有 arb 的 cfg 格式。
- 新增略過整份 Labber 的選項，或繼續讓匯出失敗：前者需加新機制，後者仍會讓 Save All 只完成部分輸出。

## 留下的問題

如果往後定出專供 Run 時計算的固定值保存的 metadata 欄位，再單獨評估是否搬移 `duration`。這次不改 Native 格式。

## 依據

沿用 [ADR-0063](../adr/0063-persistence-ownership.md#資料表示與完整性) 的 Native 正本／Labber 衍生分工，以及 [ADR-0069](../adr/0069-experiment-core-and-frontends.md#native-可選變數與-labber-投影) 的 Labber 選集。選集由 [CZ 最佳化的 AXES_SPEC](../../zcu_lab/v2/two_qubit/cz_optimize/core.py) 宣告。撤回的是[原索引軸處置](2026-11-23-labber-scalar-index-axis.md)，不改 reader。

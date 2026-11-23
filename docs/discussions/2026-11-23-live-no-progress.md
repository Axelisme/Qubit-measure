---
date: 2026-11-23
status: active
superseded_by: null
related:
  - 2026-11-22-presentation-rules.md
  - ../../zcu_lab/v2/figure-style.md#r8圖型對應資料
  - ../../zcu_lab/v2/experiment-authoring.md
basis:
  repo:
    main: 2b8f628c31a181dfd9fe187859408c9f4d236c7e
  external: []
---

# Live 圖不顯示進度計數

## 結論

2026-11-23，使用者決定：「全面拿掉計數」。所有 `zcu_lab/v2` 實驗的 Live 圖移除輪次、累積 shots、完成比和目前階段序號。執行進度由 GUI progress bar 顯示。[實驗呈現指南 R8](../../zcu_lab/v2/figure-style.md#r8圖型對應資料)和[實驗撰寫指南](../../zcu_lab/v2/experiment-authoring.md)記錄此規則。

## 問題

實驗曾在 Live 標題寫 `Averaging round 3 / 10`，有些也列累積 shots 或第幾個 train。使用者指出：「我原本要說liveplot顯示進度是多餘的，因為gui自帶progress bar。」圖上的計數重複報告執行進度，卻不說明曲線代表哪個量。

## 考慮過的選項與比較

1. **保留 Live 例外。** 同日稍早，使用者對「圖放理論關注的重點」中的 Live 進度例外答「2對」。這延續[前一日的呈現規範](2026-11-22-presentation-rules.md)，讓圖本身可顯示進度。缺點是與 GUI progress bar 重複。
2. **只移除輪次和 shot 計數。** 階段序號仍可說明執行到哪一步，但它仍然是執行進度，而非圖上資料的意義。
3. **全面移除進度計數。** GUI 負責進度，圖保留資料與判讀資訊。使用者採用此項，取代同日稍早的 Live 例外。

## 被否決的選項與理由

保留例外會繼續在兩處報告同一件事；只刪輪次和 shots 則留下功能相同的階段計數。兩者都不符合使用者要求的「全面拿掉計數」。

## 留下的問題

此裁決不改取數、圖的更新頻率、存檔或 Analyze，也不改 GUI progress bar。圖仍需說明正在畫什麼。Lookback 的 `Full ADC window per shot` 說明每筆資料的時間範圍；JPA 搜尋階段色階標示各測點從哪類搜尋取得，不報目前跑到哪個階段；目前最佳分數用於判讀搜尋結果，也不是進度。這些資訊可以留在圖上。

## 依據

當時的實作基底為 frontmatter 所列 commit。上文保留 2026-11-23 確認的使用者原話及取捨；原始問答未入庫。未修改前一日的討論快照。

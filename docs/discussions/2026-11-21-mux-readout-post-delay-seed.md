---
date: 2026-11-21
status: active
superseded_by: null
related:
  - zcu_lab/v2/support/measure/mux_forms.py
  - zcu_lab/v2/mux/onetone/guide.md
  - zcu_lab/v2/two_qubit/guide.md
  - docs/adr/0073-measure-form-derivation.md
  - docs/adr/0074-flat-readout-module.md
  - docs/discussions/2026-11-21-herald-wait-readout-post-delay.md
basis:
  - "repo@3d73c92af：實作前的 main"
  - "2026-11-21 使用者覆核與 FTF 調查工作檔未入庫，正文保留比較數字與裁決"
---

# Mux readout 的 post_delay 從五個光子衰減時間起步

## 結論

2026-11-21，使用者選擇由 guide 說明起點，並由表單給初值。初值是 `5 / (2π · κ)` µs，κ 是以 MHz 記的半高全寬。多個腔共用讀出時，用最小的 κ。操作者在 herald 後打名目 Xπ，比較此等待與更長等待的終端 e 比例，再選擇自己的設定。

初值在 [mux 表單的 Open／Reset](../../zcu_lab/v2/support/measure/mux_forms.py) 產生，沿用 trigger 已用的線寬來源。有 readout 副本時保留較長的原值；沒有副本時只在線寬齊全時推導。寫回沿用 Run 的 module，不覆蓋手動編輯。既有 library 不改，herald 和終端讀出仍共用同一個欄位。

## 問題

[前次討論](2026-11-21-herald-wait-readout-post-delay.md) 比較了 FTF 中等待 0 µs 與 2 µs 的結果，沒有從線寬推導 2 µs。使用者指出他的經驗是五個光子衰減時間可能不到 100 ns，問這兩種時間為何相差一個數量級。FTF 的 R2 線寬約 1.2 MHz，五個光子衰減時間約 0.66 µs。當時的 2 µs 相當於約 15 個光子衰減時間，並非從量測得到的最短等待。

## 考慮過的選項與比較

- 只改 guide，操作者依線寬自行填欄位。這能說明 2 µs 的由來，但由範本開始時仍須手算。
- 改 guide，並在 Open／Reset 重用 ring-up 公式。操作者可在 Run 前檢查或改寫初值。
- 改 guide，等 FIT 後才由寫回代填同一公式。FIT 有更新的線寬，但此處代填會覆蓋本次 Run 所用的設定，操作者不能在 Run 前確認這個值。

## 被否決的選項與理由

不沿用只列 0／2 µs 的說法。它容易讓線寬較大的操作者把微秒等級的等待當成通用值。使用者選擇 guide 與初值一起改，實作放在 Open／Reset，不在 FIT 或寫回後自動更動。

## 留下的問題

FTF 沒有比較 0.66 µs 和 2 µs 的翻轉比例，故不能用該例證明五個光子衰減時間足夠。各讀出條件仍須用名目 Xπ 和更長等待比較。

## 依據

- 遵循 [ADR-0073 的表單初值原則](../adr/0073-measure-form-derivation.md#自動推導點到為止)，重用固定公式，不依裝置調整。缺 κ 時沿用已有設定，不新增量測或隱式預設。
- 遵循 [ADR-0074 的共用 readout 契約](../adr/0074-flat-readout-module.md#herald-reset-與逐-shot-遮罩)。本次只改表單起點，沒有新增 herald 專屬設定。
- 延續[前次等待比較](2026-11-21-herald-wait-readout-post-delay.md)的驗證方法。那次從場衰減時間估算，本次以使用者指定的光子數衰減時間給起點，不把兩種時間常數互換。

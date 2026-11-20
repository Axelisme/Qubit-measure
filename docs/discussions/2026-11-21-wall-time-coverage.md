---
date: 2026-11-21
status: active
superseded_by: null
related:
  - ../adr/0067-gui-application.md
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - ../../tools/README.md
  - 2026-11-21-experiment-run-wall-time.md
basis:
  repo_commit: f8c537af1f3424d62cb7ee3cb47f65acfdfb8bfd
  external_sources: none
---

# 牆鐘報表的 guide 與輸入

## 結論

Adapter 交付原先宣告的 guide 路徑。工具用該路徑核對 operator guide 盤點，沿用 [ADR-0067](../adr/0067-gui-application.md#adapter-capability-與事件) 的宣告歸屬。校準 point、FTF 設定和表單覆寫各自留檔。工具只走到 Run 終態，不執行 Analyze 或 Save。它仍是 [ADR-0078](../adr/0078-evolution-module-boundary.md#模擬器的測試與量測分層不做-e2e) 的限定例外。

## 問題

單一 tracer 的 guide 路徑是手填的。增加到全 adapter 後，單靠 `guide()` 的文字與目錄無法區分同一個目錄的 `guide.md` 和 `guide_heralded.md`。另外，載入 point 只更新表單來源，不會設定虛擬電流。起始電流若維持 0 A，量測就會在不同工作點使用校準過的 pulse 和 carrier。

## 考慮過的選項與比較

Adapter 增加一個唯讀屬性，取用原有的宣告值。這遵循 [ADR-0072](../adr/0072-module-boundaries.md#使用者空間的依賴原則)：工具由 assembly 取得 lab 宣告，不重組 lab 的內部配置。既有 `guide()` 仍負責讀取呈現用的 Markdown。FTF 的工具副本採用校準所用的 156.63、83.34、0 µA 初始電流，不改模型係數。工具沿用 tracer 的 Controller Run 路徑，讓操作員等待的這段時間仍是同一個指標。

## 被否決的選項與理由

- 依 adapter key 猜 guide 路徑並維護例外表。盤點曾找到 18 個不符合直接路徑慣例的 adapter；兩份宣告會在搬檔後不同步。
- 讀 Markdown 內容反查檔名。相同內容不能辨認是哪一個檔案，Run 本身也不需要打開這些文件。
- 在終態後再量 Analyze，或留一欄固定寫「未執行」。GUI 把 Analyze 視為另一個指令，固定空欄不提供資訊。計時範圍寫在[工具說明](../../tools/README.md#實驗-run-牆鐘)。

## 留下的問題

這張表不回答 Analyze 要多久。若現有 Run 紀錄或操作員回報顯示 Analyze 在等待路徑上，另行量測，不把它併入 Run 秒數。首版 T2 衰減估值來自短窗診斷，不是精度驗收或持久校準。

## 依據

使用者 2026-11-21 對整個 Run 作為優化指標、kernel 比較維持 `Engine.evolve` 的建議答「都採用建議」。指南路徑與是否觸發 Analyze 是後續代裁。校準採用 2026-11-20 FTF CZ 模擬採集的資料，原始工作檔未入庫；可核對的版本與覆寫依據留在[工具輸入](../../tools/data/experiment_wall_time/)。

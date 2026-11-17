---
date: 2026-11-16
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - ../../lib/zcu_tools/simulators/README.md
  - ../../tools/README.md
basis:
  repo: b3c479d2694adf317ce3c6f79ec8d42f87b6349f
  external: none
---

# Acquire wall 的分組與未量測資料

## 結論

三個 environment 都已提供真實的 compile、evolve、readout wall 與 binding 成敗。
共用 `AcquireTelemetry` 的三段型別因此收緊為 `float`，未進入的段填 `0.0`。
Analyzer 加入 `wall_by_complete`，只依行內的 `complete` 分組。
`true` 是正常返回，`false` 是拋錯，缺 key 或 null 是 unknown。
正常返回包含 Stop 後返回的 partial 結果，不代表要求的 rounds 全做完。

耗時只計入 `wall_s` 與三段四個俱為 JSON 數字的行。
每組同時報計入與排除筆數，未量測行仍計入 family 統計。
有值卻型別錯的行沿用壞行政策，警告檔名與行號，計 skipped，family 也不計。
現行欄位與分析用法由 [simulators README](../../lib/zcu_tools/simulators/README.md#acquire-telemetry)
及 [tools README](../../tools/README.md#routing-telemetry-analysis) 擁有。

## 問題

原 analyzer 只讀 family CPU，無法區分編譯、演化、讀出與 acquire 其餘時間。
既有檔案可能缺少三段欄位，過渡檔案可能把它們寫成 null。
把未量測當零會把整段 wall 算進餘數，讓 preparation 與 pacing 看起來過重。
舊行也可能沒有 complete，不能把它們解釋成成功或失敗。

三個 environment 預設寫到同一個 project-root `.sim_telemetry`。
行內沒有模擬器身分，family 名稱、檔名或 backend version 也不能提供這個身分。

## 考慮過的選項與比較

- 只報一個總和。容易實作，但正常返回與拋錯前的部分工作混在一起。
- 按模擬器分組。需要新增身分欄位，或猜測資料來源。這次不擴張 producer 的八欄契約。
- 只按 complete 分組。沿用 binding 已產生的事實，不把 partial Stop 當失敗。採用此案。

對缺值採 unknown，而非替舊行推測成敗。
四個 wall 的完整性規則也適用 unknown，因此缺 complete 的行仍可提供已量測耗時。
每組 included 加 excluded，跨組合計等於 records，讓讀者能核對分析覆蓋範圍。
只列實際出現的組，避免空組看起來像觀察到的資料。

遵循 [ADR-0072](../adr/0072-module-boundaries.md) 的定位，writer 留在 simulators，
analyzer 留在 tools 並只讀 JSONL，不 import runtime。
遵循 [ADR-0078](../adr/0078-evolution-module-boundary.md) 的 Engine 黑盒邊界，
family 統計沿用公開診斷的 CPU 與成本語意。
沿用 [共用 writer 討論](2026-11-16-shared-acquire-telemetry.md) 的八欄型別與明列投影，
結束三段 nullable 的過渡，不在 child TypedDict 重宣告。

## 被否決的選項與理由

不從路徑、filename、family 或 backend version 猜模擬器。
要分開看，操作者為各 environment 設不同的 telemetry directory，再各跑一次 analyzer。
一次傳入多個目錄仍合併成一份結果。

不把缺少的耗時補零，也不讓缺值阻止有效 family 工作進入統計。
不讓型別轉換把字串秒數、bool 秒數或數字 complete 當成有效量測。
不新增格式版本或 legacy adapter，只有缺值與已宣告欄位型別的輸入規則。

## 留下的問題

預設共用目錄的歷史行無法事後按模擬器拆開。
本次不加入選路錄製、debug capture 或數值重播。

## 依據

以 basis commit 的三個 environment producer、共用 writer 與 analyzer 為準。
公開行為測試放在 [analyzer tests](../../tests/tools/test_analyze_routing_telemetry.py)，
從同 process 的 `main(argv)` 讀合成 JSONL，核對輸出與壞行警告。
原始工作裁決未入庫，本篇保留缺值、型別、分組與目錄操作的結論。

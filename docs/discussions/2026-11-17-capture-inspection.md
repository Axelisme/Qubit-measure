---
date: 2026-11-17
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - ../../tools/inspect_simulator_capture.py
  - ../../lib/zcu_tools/simulate/evolution/_capture.py
basis:
  repo_commit: 880785b3e72d2fda62f8e314a210b5b0f66495fe
  external: []
---

# Capture 的唯讀體檢與 typed metadata

## 結論

體檢工具只讀 frozen manifest 與 telemetry JSONL。它保留每輪實錄，分開呈現 caller 延後與 worker 解決，不重新呼叫 guard 或 Engine。Prefix 採用者引用代表 batch 的 run，不冒充第二次執行。誤差寫「未提供」，因為錄製資料沒有獨立 reference。

工具沿用 `read_frozen_manifest` 讀容器，再透過 evolution 公開的 `read_capture_metadata` 取得既有 `CaptureMetadata`。公開 reader 只驗證 JSON 形狀。它不改 generic codec，不驗證跨欄位關係，不重建模型。依賴方向遵循 [ADR-0072](../adr/0072-module-boundaries.md#開發工具的數值交接) 與 [ADR-0078](../adr/0078-evolution-module-boundary.md#黑盒邊界是-simulateevolution)。

## 問題

Generic `FrozenRecord.metadata` 是 opaque object。工具若自行宣告同形 schema，producer 與 reader 就有兩個欄位 owner。只加型別斷言則不能識別外部錄製檔的錯型別、漏欄位或多欄位。

同一 acquire 可以包含多次 evolve。Metadata 的 complete 只表示該次 evolve 正常返回，不能代替 acquire 的 complete。耗時也不能從路徑、family 或相近時間猜測關聯。

## 考慮過的選項與比較

### 由 metadata owner 提供公開 reader

沿用既有 `CaptureMetadata` TypedDict，以 module 層 `TypeAdapter` 嚴格讀取。不是 dict、缺 key、任一層型別或 Literal 錯誤，以及任一層額外 key，都傳出原始 `pydantic.ValidationError` 與欄位路徑。不供預設、不做兼容。這讓 producer 與 consumer 共用同一 schema，generic manifest 仍能裝其他 caller 的 metadata。

此選擇遵循 [凍結輸入先例](2026-11-16-frozen-simulator-inputs.md) 的 owner 分工。區別在於 capture metadata 本來就由 evolution 定義，reader 因此也留在 evolution，而不是把工具專屬場景 metadata 移進 codec。

### 用明確檔名關聯 acquire

沿用 [capture 發布先例](2026-11-17-capture-publication.md) 的 `capture_file`。工具只按 manifest basename 找 telemetry。沒有同行就不列 wall，也不推測原因。多行同名是壞資料，錯誤訊息指出檔名。Acquire complete 與各筆 evolve complete 分開輸出。

### 對照條件只產生標記

五個可省略的 CLI 選項對照 budget、families、source identity 與兩個版本。沒有明示的條件保持原值。差異標成「新條件下重新評估」，下列路由仍明說是錄製實錄、未重算。不新增跨 evolve 或跨檔案的對應政策。

這遵循 [逐輪事件先例](2026-11-17-engine-capture-events.md) 的實錄語意。工具不把新條件標記稱為舊次重現，也不以 guard accepted 當作數值誤差。

## 被否決的選項與理由

工具端鏡像 metadata schema 會複製 owner。Cast 會跳過 runtime 驗證。修改 generic manifest reader 會把 capture 專屬語意放進 opaque 容器。這三種方式都沒有採用。

重新執行 dispatch、guard 或 evolve 會產生另一輪計算，不能解釋舊次實際走過的路。工具因此不載入 simulator、program、tests 或 reference solver，也不讀數值 cache。

## 留下的問題

錄製沒有 guard 數值證書與 reference，工具無法補出被拒絕的數值原因或誤差。未來若增加這些資料，應先在資料 owner 定義契約，不在報告端推測。

## 依據

- [工具用法與輸出語意](../../tools/README.md#simulator-capture-inspection)。
- [Capture metadata 與公開讀取方式](../../lib/zcu_tools/simulate/evolution/README.md#engine-capture)。
- [Generic frozen record 與 manifest codec](../../lib/zcu_tools/simulate/evolution/frozen.py)。

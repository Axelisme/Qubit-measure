---
date: 2026-10-23
status: active
superseded_by: null
related:
  - docs/adr/0063-persistence-ownership.md
  - docs/adr/0067-gui-application.md
  - docs/adr/0068-remote-transport.md
  - tests/README.md
  - lib/zcu_tools/gui/remote/control_service.py
basis:
  - "repo@e837af2a014518320649dbc09646204274e9eede，已包含 library cleanup 的批次 provenance 查詢，尚未包含 writeback 分組提交、本次增量讀及已驗證快照快取"
  - "writeback 分組提交當時已裁定，後來由 base 02 實作並收集至 repo@fbd681a42dfa1f8e8d7bde31156841445cd50718，不屬於上述 e837 基準已實作的功能"
  - "2026-10-23 寫回延遲調查、使用者裁定與後續依賴存在性檢查、ledger 失效邊界的細化，原始工作檔未入庫；必要事實與判斷保留於本文，未引用外部來源"
---

# Store 讀取效率與寫入慣例

## 結論

使用者問「怎麼在架構上或慣例上根本解決這種效率問題」，於 2026-10-23 對 store 層改善、寫入與測試慣例、延遲 warning 及本篇討論快照回覆「認同」。當時決定保留已裁定的 writeback 分組提交方案，再減少 store 層對已讀內容的重複 parse 與驗證，不把 I/O 搬離 owner thread。分組提交後來由 base 02 實作，收集至 `fbd681a42dfa1f8e8d7bde31156841445cd50718`。

寫入邊界由 [ADR-0063](../adr/0063-persistence-ownership.md#資料表示與完整性) 擁有。一次使用者動作對每份文件提交一筆交易，owner API 接受該文件的一組變更，單項是長度 1 的批次。這不提供跨文件 rollback，也不改既有成功前綴與錯誤契約。

效率回歸由 [tests/README](../../tests/README.md#測試成本) 的工作量測試保護。現場延遲另由 [共用 remote dispatcher](../../lib/zcu_tools/gui/remote/control_service.py) 記錄。Owner handler 執行超過 500 ms 時，不論成功或拋錯，都向 GUI log 記一行含 method 與毫秒耗時的 warning。計時只含 handler 呼叫，不含 owner 排隊、權限檢查與回覆處理；off-main handler 不在此 warning 範圍內，既有 10 s request timeout 不變。

## 問題

當時的延遲調查使用含 100 筆歷史的合成專案，比較單項 `set_module` 與多項寫回。調查指出，同一次寫回反覆開啟與驗證文件，library 的本地來源驗證又會查 ledger。逐項提交使這些工作重複執行，資料增加後，單項寫入也付出更多讀取成本。

前一輪已裁定把同份文件的變更分組。`e837af2a014518320649dbc09646204274e9eede` 基準已讓 library cleanup 批次查詢 provenance，但尚未實作分組提交。該基準也會在 bytes 未變時重用 parsed YAML，但仍重跑 Pydantic、cleanup 與完整來源驗證，ledger 查詢仍會重新 parse 已讀的歷史。批次查詢減少了部分重複工作，卻沒有消除單次讀取的完整驗證成本。

## 考慮過的選項與比較

| 選項 | 能減少的工作 | 成本與限制 | 當時選擇 |
| --- | --- | --- | --- |
| 只保留已裁定的分組提交方案與既有批次查詢 | 同一次動作內的重複開檔與提交 | 單次讀取仍重做 parse 與驗證 | 保留，但不足以單獨解決問題 |
| Ledger 增量讀 | 對已讀歷史重複做 JSON 與 schema parse | 需保存 byte offset 與 event_id 索引，處理檔案失效 | 採用 |
| DocumentStore 以內容 hash 重用已驗證快照 | 內容未變時重做 YAML、Pydantic、cleanup 與完整 validate | 仍讀完整 bytes 並計算 hash，library 另查外部依賴存在性 | 採用 |
| 把 I/O 與驗證搬到 worker | 減少 owner thread 的直接佔用 | 增加併發、發布與衝突處理，需重新設計 owner 邊界 | 本次不做 |

增量 ledger 以 append-only 歷史為契約，記住已 parse 的 byte offset 與 event_id 索引。後續細化把失效檢查限定為前 4096 bytes、檔案的 st_dev 與 st_ino、檔案是否縮短、同大小時的 mtime_ns，以及上次 offset 前最後一行的 bytes。任一不符就整份重讀，其餘只 parse 新增尾段。這避免每次查詢都 hash 整份歷史。

同 inode、同 prefix、同前次最後一行下，中段原地改寫又同時增長的情況不在偵測範圍。外部需要完整驗證時，必須使用強制整份重讀的公開入口，不能把增量查詢當成整份歷史校驗。入口名稱、查詢順序、附檔與錯誤傳遞由 [RecordsLedger](../../lib/zcu_tools/resources/entry/ledger.py) 的公開契約擁有。

文件快取以內容是否相同為依據。讀取先 stat 並對完整 bytes 計算 hash；不同則重新驗證與清理。Library bytes 相同時，不重 parse YAML、不重跑 Pydantic、cleanup 或完整 validate，但不能直接沿用快照。Library owner 先檢查快照內全部 provenance source，以 ledger event 索引查事件存在性，再對 attachment 做 stat。這只查依賴存在，不重驗未變 library 或附檔內容。

後續細化保留了當次讀取就偵測附檔缺失的行為。全部依賴存在才重用已驗證快照；任一依賴缺失就進完整驗證、清理、備份與錯誤傳遞路徑。不存在的本地事件仍使所屬 root 移除，attachment 缺失仍在當次讀取立即失敗，不等 YAML 改變。[DocumentStore](../../lib/zcu_tools/resources/document_store.py) 的 lock 範圍不變，完整失敗契約由 ADR-0063 擁有。

## 被否決的選項與理由

- 只調大 handler timeout。這會讓 caller 等更久，沒有減少 parse、驗證或 owner 佔用，也無法指出哪個 method 變慢。
- 以秒數斷言 store 效率。CPU 負載、檔案快取與平行測試會改變耗時，不能用來判斷是否重讀舊行。決定使用 synthetic 規模與工作量觀察，真實耗時另做 old／new 量測。
- 同 hash 就完全不查外部依賴。Library bytes 不變時，外部仍可能刪除 ledger event 或 attachment。直接沿用快照會延後 root 清理或附檔錯誤，破壞既有當次讀取契約。
- 每次對整份 ledger 歷史計算 hash。這會讓 I/O 隨歷史大小增加，違背增量讀的目的。決定使用有限 prefix 與前次尾行比對，公開未涵蓋的改寫情況，另提供強制完整重讀。
- 同 hash 仍完整重驗。這保留舊行為，卻讓每次讀取重新執行完整來源驗證。決定以索引與 attachment stat 查存在性，缺失才退回完整路徑；不以快取吞掉 ledger schema、附檔或 I/O 錯誤。
- 本次引入 worker 與另一份可寫快照。[ADR-0067](../adr/0067-gui-application.md#session狀態與協作) 與 [ADR-0068](../adr/0068-remote-transport.md#分層與連線) 的單 owner 模型不因此改變；延遲 warning 只提供診斷，不取消、重試或回滾 handler。

## 留下的問題

當時把 100 筆歷史與乾淨專案的單項 `set_module` 耗時差小於 2 倍列為驗收目標，並要求 24 項寫回的 old／new 記錄。兩邊應用同一 harness、相同資源配置，在同一輪背靠背執行。本篇記錄設計依據，不宣稱該量測已通過。

內容 hash 仍隨文件 bytes 大小付出讀取成本，增量索引也占用記憶體。Library 的依賴檢查仍需對全部 sources 做索引查詢與 attachment stat，成本隨來源數量增加。工作量測試能防止重做舊內容的 parse，不能證明所有專案規模下的延遲都相同。若實際 warning 仍顯示長時間 owner 佔用，需另行評估剩餘成本，不能直接把這次裁定當成改併發模型的授權。

## 依據

- [ADR-0063](../adr/0063-persistence-ownership.md) 定義 ledger 讀取、文件交易、library 清理與失敗邊界。
- [ADR-0067](../adr/0067-gui-application.md) 與 [ADR-0068](../adr/0068-remote-transport.md) 定義單 owner 排程與 remote 責任。
- [Library store](../../lib/zcu_tools/resources/context/library_store.py)、[RecordsLedger](../../lib/zcu_tools/resources/entry/ledger.py) 與 [DocumentStore](../../lib/zcu_tools/resources/document_store.py) 是當時調查的 store 接縫。
- [測試慣例](../../tests/README.md) 區分工作量回歸與離線耗時量測。

**讀者：** operator、lab-author、framework-dev、agent。辨識退役的模組來源欄位與 library 的讀寫、複製方式。

# 移除模組來源

2026-10-28 移除 `module_cfg.yaml` 的 per-leaf `provenance`。Library 容器維持 `format: zcu.module-library` 與 `format_version: '2.0'`。Point 與 setup 的 opaque `metadata` 不受影響。

## 舊資料

舊 library 的頂層 `provenance` 以葉路徑記錄 source、kind、run_id、at、stderr 與 cloned_from。Source 曾引用本 entry 的 ledger 事件，跨 entry clone 曾追加 import 事件並複製附檔。

讀取現在忽略整個舊欄位，不驗其形狀，也不查事件或附件。缺事件不會使模組被刪除。無效 Spec 從讀取投影略過，公開 `snapshot.invalid_entries()` 回報 root 與驗證原因，其他模組仍可用。Header、容器及 I/O 錯誤仍傳遞。

讀取及空 edit 不改原檔、不建立 `.bak.yaml`。Strict refresh 仍拒絕無效 Spec，並保留原快照。非空寫入以目前模型序列化，自然移除舊 provenance 與已略過的無效 Spec。沒有批次清除工具，也不需要手動改檔。

## 新寫入與 clone

GUI、MCP、Notebook、writeback 與 migration 的 library 寫入只保存 Spec，不產生模組來源或 per-leaf stderr。Writeback 仍先追加 accepted 稽核事件，再提交參數、library 與 records。Record 的 source 仍引用 accepted 事件。

Clone 複製 point 參數、opaque metadata、合法 library Specs 與來源 point 的 alias。來源 bytes 保留。跨 entry 不匯入事件、不複製附檔，也不寫 cloned_from。歷史 ledger 的 import 行仍可讀，沒有新的 import producer。

`context.snapshot` 不再回傳 `library.provenance`。同批將 `WIRE_VERSION` 由 127 升至 128，`GUI_VERSION` 由 160 升至 161。其餘 snapshot 欄位維持原契約。

## 原因與依據

使用者於 2026-10-28 說明：「我其實沒有要知道來源的需求，這功能純屬於錦上添花」「來源我一直以為只是metadata，沒想要解析」。模組來源不再限制移植後參數的使用，也不再觸發讀取時刪資料。決策見 [ADR-0063](../adr/0063-persistence-ownership.md#資料表示與完整性) 與 [ADR-0067](../adr/0067-gui-application.md#session狀態與協作)。

---
date: 2026-11-16
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - ../../tools/README.md
  - ../../tools/family_pareto.py
  - ../../tools/family_pareto_reference.py
  - ../../tools/family_pareto_results.py
basis:
  repo_commit: 53b368d71ac6fbce49e7afa0add02646433f8bce
  external_sources: none
---

# Family 量表改為整批 cache-only

## 結論

量表的 reference consumer 不再求解。`run_table` 先載入整批參考解，全部成功才執行 policy。缺失時一次列出所有名稱與 key。相同 physics、不同 model identity 的輸入共用同一個 reference 物件。`--check-references` 共用這個載入步驟，寫出名稱、key、收斂誤差與載入時間後結束，不開始 Engine 量測。

`--scenario NAME` 必須恰好選到一筆。零筆列合法名稱，多筆列來源，兩者都在量測前失敗。它可以與 `--manifest` 合用。摘要發布與跨次比較沒有改變。

刪除 consumer 的 `--prepare-references`、`--freeze-inputs`、`--include-synthetic`、`--reference-cache-only` 與 `run_table(reference_solver=...)`。保留 `--reference-cache DIR`。參考解由 tests 側 producer 生產，求解仍須成本授權。決策 owner 見 [模組定位](../adr/0072-module-boundaries.md#開發工具的數值交接)。

## 問題

舊工具同時建場景、求解參考解與量測。一筆輸入的 cache miss 可以在量測流程內啟動獨立 solver；後一筆 miss 也可能發生在前一筆已開始計時之後。這讓前置檢查無法保證整批已準備好。

結果型別與 dominance 原先也在 runner。Report 要取得它們就得載入執行入口。它們現在由 `family_pareto_results` 持有，誤差與資格規則保持不變。

## 考慮過的選項與比較

保留 `prepare` 名稱但改成只讀 cache，可以減少命令列修改，卻會把原本允許求解的命令改成另一種行為。採用新名稱 `check-references`，並刪除舊旗標，讓讀者分得清生產與檢查。

逐筆載入再逐筆量測需要較少的前置記憶體，但後續 miss 會留下已計時的前半批。採用整批 preflight，選取失敗或 cache 不完整都先結束。量測仍在自己的入口重做檢查，不把先前 check 的成功當成免查許可。

沿用 [數值 codec 的 key 契約](../adr/0078-evolution-module-boundary.md#凍結數值輸入與參考解交接)，不另加 solver identity，不改 NPZ 欄位。這遵循既有 cache 可重用的要求。

## 被否決的選項與理由

沒有保留只接受舊命令的相容旗標，也沒有在 miss 後換 solver。工具缺資料就失敗，由 caller 在另一次核准的 producer 工作中補足。

沒有為來源待定發布一套暫時預設或新的 report 身份格式。那會讓後續存放裁決還要支援一個短命契約。現有來源與舊 archive 行為只集中到同一接縫，尚未完成替換。

## 留下的問題

使用者於 2026-11-16 要求所有凍結場景不入庫，Git 只留名稱與 hash 小清單。存放位置、預設來源、report 身份與既有大檔歷史處置仍待定。既有 gzip bytes 未改，舊 `freeze_manifest` 的內部 caller 尚待遷移。這次交付只完成已備來源的 consumer seam，不代表預設 CLI 已完成凍結讀取。

測試側 exclusion estimator 的 helper 與反向 import 尚待歸屬裁決。新的共用計時 owner 與 consumer cutover 也不在這次修改內，原 mode transition 與 clock 邊界保持不變。

## 依據

- [Tools README](../../tools/README.md#inventory-wide-manifest-workflow) 記 check 後量測的命令順序與遷移限制。
- [數值交接 owner](../adr/0078-evolution-module-boundary.md#凍結數值輸入與參考解交接) 記公開 codec 與 tests 側 producer 的分工。
- [原單場景討論](2026-11-14-family-pareto-remeasurement.md) 保留當時的求解與建構流程；本篇只取代 consumer 求解介面，不重定摘要 schema。

原始工作紀錄未入庫。採用的行為、被否決的替代方案與未完成邊界已寫在本文，不需要讀取本機工作檔才能理解。

## 2026-11-16 接縫與身份補充

後續裁決將 report 身份固定為每筆 `{name, physical_sha256, oracle_sha256, metadata_sha256}`，加既有 HEAD／tree／dirty provenance。
Metadata hash 只用公開 `simulate.evolution.digest(record.metadata)`。
`family_pareto.select_inputs` 集中凍結來源、嚴格 scenario 選取與這組身份，目前暫讀現行 tools/data，只供 integration。
存放遷移將替換接縫來源與 README，不改這組身份，也不要求成對 adapter 解析存放版面。

本次因此退休 `freeze_manifest`、其 caller 與量測 `main` 的 `inputs` keyword，不保留舊 archive。
CLI 測試經臨時凍結 manifest 與同一公開接縫提供來源。Producer 的 `main(argv, inputs=...)` 不受此變更影響。
三個原 producer 可重現 cases 改用公開 seam 的兩次全新建構，比較 codec 提供的 physical／oracle hash。
Pytest 涵蓋兩種 random gate 與一個固定 carrier 場景，不執行整批重建。其餘場景由存放 prepare 的清單比對把關。

Exclusion estimator 與其唯一測試、反向 import 正式交由後續工作承接，原歸屬裁決與不執行的限制仍在。
前文的 partial 停點不再阻擋本次 consumer 收斂，但存放與大檔歷史處理仍必須在 landing 前完成。
量測結果與摘要 schema、mode transition 及 clock 不變；摘要舊欄名 `manifest_sha256` 改記有序 record 身份的 digest。

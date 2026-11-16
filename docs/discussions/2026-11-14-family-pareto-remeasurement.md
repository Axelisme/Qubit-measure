---
date: 2026-11-14
status: active
superseded_by: null
related:
  - ../../tools/README.md
  - ../../tools/family_pareto.py
  - ../../tools/family_pareto_reference.py
  - ../../tools/family_pareto_summary.py
basis:
  repo_commit: b95321f78c61640dc1060f11475507341c804744
  external_sources: none
---

# Family 量表的單場景重量與摘要

## 結論

當時採用單場景命令、共用本機參考解及入庫摘要。使用者於 2026-11-14 的回覆是「甲」「2可以」。完整輸入、樣本與 journal 留在本機；每個場景的一次量測摘要才入庫。操作與格式由 [tools README](../../tools/README.md#single-scenario-remeasurement) 說明。

`--scenario <Input.name>` 在建構模型前選定 builder，再於同一命令凍結與量測。量測預設 cache-only。缺少參考解時停止，先取得求解成本核准再重建，不把 miss 當成求解許可。

## 問題

分開凍結與量測時，native thread 設定不同會改變部分模型值的最後浮點位元，造成輸入或快取指紋不同。單場景流程因此在同一命令凍結，並要求準備和量測都經重計算執行器。這不改參考解 key 或容差。

參考解與完整量表需要跨 worktree 重用，也不能隨短期工作紀錄清除。完整矩陣含原始樣本與模型資料，不適合作為每次 review 都要讀的摘要。不同時段的耗時還受機器負載影響，不能拿新舊摘要直接裁定哪個 family 較快。

## 考慮過的選項與比較

參考解與完整矩陣入庫能讓換機後直接使用，但會讓版本庫承擔完整數值資料。採用的方案只將摘要入庫。參考解、完整 matrix 與 journal 放在主 repo 的 `.agent_state/benchmark/family_pareto/`，由 Git common dir 定位。各 worktree 共用這個位置，工具會建立缺少的目錄。清空或換機後仍須先重解。

摘要保留 pinned、default、fixed 同組的狀態、最終 density 誤差、各歷史模式耗時、樣本數與未量原因。白名單、伴隨及歸屬讓讀者知道耗時屬於整個 group，不是 target 單獨成本。每份摘要只代表一個 run，不逐 family 拼接不同來源。

狀態判定集中在 `FamilyResult`，包含五種結果與 2e-3 門檻。先判伴隨被擋，再讀公開拒絕分類。報表與摘要共用這個結果。摘要 owner 集中 schema、建置、JSON/Markdown 輸出與比較，避免命令與報表各維護一套格式。命令在結果型別初始化後才載入摘要 owner，因此不需要移動狀態判定。

Provenance 使用工具自己的 UTC 開始時間與命令旗標宣告的 exclusive。這個旗標不驗證獨佔鎖。工具不讀執行器收據，也不根據 CPU 數推定 exclusive。HEAD、tree、dirty 與輸入指紋保留在摘要，供 review 核對量測版本。

比對只描述新舊值。輸入指紋不同時標不可比較，部分 family 只比交集，另列任一方缺少的 family。耗時列差值，不裁定跨時段快慢。比對失敗不阻止寫入當次摘要。

## 被否決的選項與理由

沒有加入舊矩陣轉存或格式相容讀取。歷史表留作歷史資料，新摘要由新版命令實測產生，避免把舊容差或舊程式結果改稱新版量測。

沒有用 exclusive、clean tree 或全 family 作摘要寫入門禁。這些資訊是 provenance，review 可以辨識非正式或部分量測，不需要阻止保存。沒有保存支配結論，後續彙整從最終誤差與同表耗時重算。

沒有改 production guard、演化演算法或參考精度。單場景命令沿用原本的三種歷史模式、慢 family 協定與 target/伴隨路徑核對。

## 留下的問題

工具與格式不代表已完成新版正式量表。正式數值仍須在指定版本、clean commit、一次 exclusive、cache-only 與全 family 條件下另批成本量測。本文不提供數值執行額度。

## 依據

- [量表命令](../../tools/family_pareto.py) 擁有場景選擇、凍結、量測及開始時間。
- [參考解 owner](../../tools/family_pareto_reference.py) 擁有既有 cache key、求解與 cache-only miss。
- [摘要 owner](../../tools/family_pareto_summary.py) 擁有目前 schema、輸出與比對。
- [Evolution 黑盒邊界](../adr/0078-evolution-module-boundary.md) 規定量表只透過公開引擎接縫與診斷使用演化能力。

原始工作紀錄未入庫。操作契約與取捨已寫在本文及 tools README，不需要讀取那些紀錄才能使用工具。

## 2026-11-16 後記

Consumer 的求解介面與逐筆載入流程已由[整批 cache-only 遷移](2026-11-16-family-pareto-cache-only.md)取代。
工具不再接受 prepare 或 freeze 命令。預設來源、report 身份與凍結場景存放仍待定，本文記錄的同命令建構與凍結不是新的來源終態。
摘要 schema、比較與 annotation 的取捨保留。

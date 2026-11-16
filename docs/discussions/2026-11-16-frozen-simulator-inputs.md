---
date: 2026-11-16
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
basis:
  repo_commit: 2a537164567bb157550bfc82c043a5545e85c9ee
---

# 模擬器場景凍結與參考解的數值交接

## 結論

固定場景先編譯或直接宣告成數值，再透過 `simulate.evolution` 的 codec 凍結入庫。
模型和 pulse 控制由場景 producer 擁有，codec 只處理通用數值。
獨立 solver 留在 tests 側，參考解經既有 NPZ cache 交給讀端。
職責與 key 契約見 [ADR-0072](../adr/0072-module-boundaries.md) 與 [ADR-0078](../adr/0078-evolution-module-boundary.md)。

## 問題

原量表工具直接 import accuracy 案例表與 tests 側 solver。
前端、具體模型或案例表改動時，量測輸入也會跟著改。
跨版本的耗時差因此可能包含場景差，而不只是引擎差。

使用者於 2026-11-15 說：「我會預期模擬器測試腳本只會依賴模擬器模塊，後端與前端分開測，kernel與dispatch也分開測，不該有e2e的測試需求」。
同日也說：「研究的话就没差，都可以用。但是如果你要持久的工具的话，麻烦请不要依赖于作者可能会改到的东西。」

## 考慮過的選項與比較

保留工具對 tests 的依賴，改動最少，但無法固定模型與案例表的邊界。
每次從固定參數重建並比 hash，可以拒絕不同輸入，卻不能讓舊場景脫離新前端。
把編譯完成的 batch、初態和逐 level 模型凍結入庫，讀端才不必重建裝置或編譯程式。

參考解另建一套 bundle，會產生第二份場景清單與 key。
沿用原 key 和 NPZ 格式，能直接讀取既有資料。
Key 保留完整 identity-free 物理輸入、兩組 tolerance、version 1 與 convergence limit 2e-6。
它不加入 solver identity，因為目前每類場景只有一個指定 solver。

場景型別使用 tools 自持的具名 `Scene`。
工具測試也建構 `Scene`，不另加只服務舊 fixture 的 Protocol。
共享的 single/integer 編譯與資源生命週期放進獨立 builder owner，場景檔保留控制宣告。
這個拆分依職責進行，不按行數切半。

## 被否決的選項與理由

Solver 不搬進 `simulate`，因為精度參考必須獨立於 production 演化實作。
讀端不在 cache miss 時重解，因為量測前置檢查不能混入未核准的求解。
凍結資料不使用使用者的環境 YAML，也不 import lab 實驗。

現有唯一共用 cache 的 20 個 NPZ 沒有涵蓋 116 個舊 synthetic 場景。
不能新解一份 synthetic 資料冒充歷史 cache。
因此 synthetic 驗原 key 一致，real 另外驗既有 cache-only 命中。
Key 比對在乾淨基準版本執行原函式，再把完整 canonical physical 數值交給新 codec。
這避免把已改接同一個 key 函式的兩個入口拿來互相比較。

## 無損容器與 solver 選擇

19 個完整數值場景產生 52,152,956 bytes 的 compact JSON。
長 shaped 序列在每個 track 保存完整包絡，重複內容佔大部分空間。
入庫改用 stdlib gzip，mtime 固定為 0，不放檔名，不為壓縮而去重或改 array schema。
這保留每筆 physical 的 canonical bytes、point order、兩個 hash 與 reference key。

容器的 version、唯一 name、hash 計算與核對由 `simulate` 持有。
Tools 的 reason、category、families、routes 和 companions 移入 opaque JSON metadata。
新舊外層容器的 bytes 因此不同，不要求全檔相同。
要求的是每筆 physics 不變，以及新容器對同輸入寫兩次得到相同 bytes。
舊 raw 直接壓縮再解壓的 byte identity 只用來證明 gzip 無損。

Synthetic 與 real 分成兩份 manifest。
Tests 側 producer 的呼叫端明示每份要用的 solver，不從 metadata 或名稱推測。
它不在失敗後換 solver。容器也不增加 reference method 欄位。

Report 旁的舊 `inputs.json` 歸檔只在 reader 遷移期間保留，不加新舊容器的相容 reader。
Reader 遷移完成時，舊 helper 也要移除，歸檔改用公開 writer，或只記 manifest 名稱、record 名稱與兩個 hash。
這避免兩邊各自持有一份容器版面。

## 留下的問題

新增或改變場景仍須重新凍結，並另行核准參考解準備的成本。
Kernel 單段與完整序列是不同場景，不能用裁切後的段冒充原完整序列。
數值 codec 不保存任意 Python builder、裝置佈局或 mutable routing history。

## 依據

當時的工作紀錄與原始補驗證據未入庫。
本篇保留取捨與必要事實，現行用法由 [tools README](../../tools/README.md) 與 [evolution README](../../lib/zcu_tools/simulate/evolution/README.md) 擁有。
固定模型參數與場景控制由 tracked producer 持有。

## 2026-11-16 存放補充

後續使用者要求所有凍結場景不入庫，Git 只留名稱與 hash 小清單。
上文的 gzip 入庫與 report 歸檔選項不再是終態；存放、預設來源與 report 身份仍待裁定。
既有兩份 gzip 與歷史尚未遷移，不依原大小上限新增資料。
Reference consumer 的已放行變更與未完成邊界見[整批 cache-only 遷移](2026-11-16-family-pareto-cache-only.md)。
同日後續裁決已固定每筆 record 的 name、physical_sha256、oracle_sha256 與 metadata_sha256 身份，退休舊 archive 與 helper。
存放遷移只換來源，不再改這組身份。

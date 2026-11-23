---
date: 2026-11-23
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - 2026-11-16-frozen-simulator-inputs.md
  - 2026-11-23-evolution-full-collapse-operators.md
basis:
  repo_commit: 91421a8d176b796718a4b37245b2bb61cb6cd383
---

# 凍結數值升到 physical v2

## 結論

當時將凍結場景和 debug capture 升到 physical v2，仍用 version 1 的場景索引與根內佈局。沒有顯式 jump 時省略欄位，保留不變的物理身份與 reference key。決策見 [ADR-0078](../adr/0078-evolution-module-boundary.md#凍結數值輸入與參考解交接)。

## 問題

在 v1 加欄位，舊 reader 可能驗過 hash 卻丟掉 L。若所有無 jump 模型都寫空列表，雖然物理不變，canonical record、pins 與參考 key 仍會改。新容器若又沿用舊場景目錄，無 jump 的同名檔會撞到 v1 payload。

## 考慮過的選項與比較

選定 v2 只對非空集合保存完整且有序的複數矩陣。缺欄位代表零個 jump，顯式空值和非法矩陣則拒絕。新場景根使用 `scenes/v1-physical-v2`，capture 用 `simulator-debug/v2`。顯式絕對 scene 路徑仍原樣傳遞。這延續[既有共用快取的分工](2026-11-16-frozen-scene-storage.md)：Evolution 管內容和編解碼，tools 選位置，tests 提供獨立參考。

## 被否決的選項與理由

沒有把完整 L 放回 v1，也沒有把空值寫進每筆記錄再從 hash 中剝掉。前者會讓舊讀者漏算；後者會讓完整記錄與其驗證讀到不同內容。沒有提高未改的索引版次。未替新版 inspector 加歷史 reader；用匹配的舊 checkout 已能查舊檔。

## 留下的問題

新版拒絕 v1，不移動或覆寫舊資料。場景 producer 待另行授權後在新根重建並核原 pin；不拿搬運舊 gzip 代替驗證。參考 producer 只在 miss 時請已選定的 solver 求解。既有 secular solver 不支援完整 L，遇到這類 miss 就拒絕，不能發布漏掉 L 的解。

## 依據

2026-11-22 由 designer 代裁，使用者後覆核「其他两个是接受。」原調查未入庫。版本、重建命令與錯誤契約見 [Evolution README](../../lib/zcu_tools/simulate/evolution/README.md#frozen-numeric-inputs-and-references) 和 [tools README](../../tools/README.md#freeze-numeric-scenes)。

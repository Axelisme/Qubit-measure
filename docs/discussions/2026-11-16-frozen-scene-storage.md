---
date: 2026-11-16
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - 2026-11-16-frozen-simulator-inputs.md
  - 2026-11-14-family-pareto-remeasurement.md
  - ../../tools/README.md
  - ../../tools/data/family_pareto/README.md
basis:
  repo_commit: c82f24522e8009defb7567a17abffe677a85821e
  external_sources: none
---

# 凍結場景改為共用快取，Git 只留名稱與 hash 清單

## 結論

凍結場景數值不入庫，每筆 record 一個 gzip，放 repo 共用快取。Git 只留 synthetic 與 real 兩份名稱、physical hash、oracle hash、metadata hash 清單。Producer 重建與釘住身份不符就停，只有明確更新清單才能改所選列。使用者對存放提案答「R」，生效責任與原話由 [ADR-0072](../adr/0072-module-boundaries.md#使用者空間的依賴原則) 擁有。

位置、檔名與重產指令見 [Tools README](../../tools/README.md#freeze-numeric-scenes)。格式讀寫歸 `simulate`，只收顯式路徑；Git 共用位置由 tools 解析。量測 consumer 不建場景，也不解參考解。

## 問題

先前 19 個場景分成兩個 gzip。Real 有 18 筆、10,374,107 bytes，synthetic 有一筆、730 bytes，合計 10,374,837 bytes。解壓後 physical 共 51,819,897 bytes，分別為 real 51,818,118 與 synthetic 1,779 bytes。Real metadata 為 328,967 bytes，最大單筆 45,185 bytes；synthetic metadata 為 79 bytes。19 筆的 physical hash 與 oracle hash 各自全不相同。

兩個 gzip 由同一個、同時改 28 檔的 commit 帶入。若直接進 main，刪掉 tip 檔案仍不能移除歷史內容。因此存放要在落地前決定，不能只延後到快取清理。

2026-11-16，使用者要求「檔案大就請不要入庫，用類似快取的方式來減少重複計算，記得考慮跨worktree也能復用的方案」。對範圍與釘住方式答「1a 2a」，包括 730 bytes 的 synthetic 也不入庫，重建不符就停，只有明確更新清單才算數。

## 考慮過的選項與比較

| 選項 | 作法 | 代價 |
| --- | --- | --- |
| K，繼續入庫 | 保留兩個 gzip | 違反不入庫的限制，歷史繼續增長 |
| F，去重後入庫 | physical 改用共用包絡表 | 改格式、hash 與 reference key，參考解須重算 |
| N，內容定址 | physical 與 metadata 分成兩種 blob | 同物理不同名可共用，但現有 19 筆沒有共用收益 |
| C，整包快取 | 將現有 bundle 移到快取 | 改一筆要重寫整包，不同場景集合不能共用 |
| B，使用者層級快取 | 跨 clone 共用 | 與 repo 參考解快取分家，刪 repo 也不刪它 |
| D，量測缺檔時自建 | Consumer 呼叫 producer | 重新引入 program／simulators 依賴與建構成本 |
| R，每筆單筆 gzip | 沿用既有容器，每筆獨立發布 | 改名或 metadata 會重存該筆 physical |

挑戰提出 R 後，採用它取代 N。現有 physical 各不相同，拆兩種 blob 沒有已觀察到的收益。R 可沿用容器，當時估計增加約 50–80 行與測試，比 N 少約一半。改名與 metadata 變更當時罕見，接受重存單筆的成本。

| 先例 | 立場 |
| --- | --- |
| Repo 共用 reference 與 ratchet 快取 | 遵循 Git common-dir 定位、顯式覆寫、原子發布與 miss 失敗 |
| Numba 編譯快取與 pytest 隔離 | 區別。編譯產物要隔離，場景需要跨 worktree 共用 |
| 量測不 import program、simulators 或 tests | 遵循，缺檔提示 producer 指令，不自建 |
| 原 gzip 入庫與大小上限 | 由使用者的新限制取代，只保留容器 owner、metadata 不透明與 solver 顯式對應 |
| [單場景重測](2026-11-14-family-pareto-remeasurement.md)的 native thread 觀察 | 遵循浮點末位會變的事實。區別同命令建構與量測，現在由 producer 重建核對 |
| Tracked 工具可建立 git 外快取 | 遵循，builder 入庫，快取不依賴會隨任務刪除的工作目錄 |

## 被否決的選項與理由

K 與 F 都違反使用者的不入庫限制。F 只延後檔案增長，還會改 key。N 多一種格式卻沒有現存去重收益。C 要整包重寫。B 超過跨 worktree 的需求，還分散快取位置。D 讓量測重新依賴場景建構，不能保證計時前只讀資料。

首次遷移因此直接使用既有 payload，保留 physical、oracle、metadata hash、point order 與 reference key，不重建模型或重算 reference。舊 gzip 不進落地歷史，後續 checkout 接乾淨 base。這不等於驗過冷快取重建。

## 留下的問題

完整冷重建尚未驗，需要另取運算授權。先前一次 19 場景建構約 8.9 秒，使用 4 CPU、四個 native thread 變數皆為 1，但那是舊單一 manifest 與 dirty tree，不能當現行兩份清單的重現證據。

Native thread 會改浮點末位。快取遺失且無法逐位元重現時，只能明確更新清單，再重算受影響的 reference。若將來大量場景同物理不同名，再比較 N；若需要跨 clone 共用，再比較 B。

## 依據

- [場景資料 README](../../tools/data/family_pareto/README.md) 與 [Tools README](../../tools/README.md#freeze-numeric-scenes) 擁有目前的清單、原子發布及操作契約。
- [ADR-0078](../adr/0078-evolution-module-boundary.md#凍結數值輸入與參考解交接) 擁有 codec 與 reference key。
- 原始角度調查、提案與挑戰工作檔未入庫。上列容量、當時估算、使用者原話與選項理由已寫入本文。

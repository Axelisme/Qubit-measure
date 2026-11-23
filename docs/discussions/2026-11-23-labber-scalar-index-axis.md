---
date: 2026-11-23
status: superseded
superseded_by: 2026-11-23-cz-optimize-labber-selection.md
related:
  - 2026-10-14-artifact-persistence.md
basis:
  repo: 5fff715cd7acca2a099f4152a53cf5f42fefbfba
  external: null
---

# Labber scalar 匯出的索引軸

## 結論

Labber 匯出把沒有軸的 scalar 寫為一個樣本，放在無單位的索引軸上，座標為 0。每個加軸的變數記一筆警告。既有 reader 讀回這個樣本和索引軸。Native 結果仍是無軸 scalar。

## 問題

CZ 最佳化的 `duration` 是 scalar。Save All 寫出 `data.h5` 後，Labber writer 因為沒有 step axis 而拒絕匯出，保存回報為部分成功。原始保存回報未入庫。

## 選項與取捨

- 在 Labber writer 加索引軸。它沿用已有的單點網格，不用改 reader，也不替結果捏造物理座標。
- 只保存 native scalar，或在 Labber 匯出時略過 scalar。這會保留 native 正本，但 Log Browser 仍看不到 operator 要匯出的值。
- 放寬 reader 接受無軸檔案。這會改變讀檔契約，也不能讓既有 Labber 工具讀到 step axis。

後兩個選項未採用。Common layout 仍要求共用網格，不會為混合網格自動切換 layout。

## 留下的問題

本次只用 repo 的公開 Labber 讀取入口驗證，未操作外部 Log Browser。

## 依據

沿用 [ADR-0063](../adr/0063-persistence-ownership.md#設定快照與-acquired-事實) 的 native 正本／Labber 衍生輸出分工，及 [ADR-0080](../adr/0080-datafile-generic-container.md#介面) 的 codec 邊界。索引軸只在 [Labber 匯出](../../lib/zcu_tools/datafile/README.md#labber) 加入，不寫進原容器。

## 後記

2026-11-23，使用者改採 [CZ 最佳化的 Labber 選集](2026-11-23-cz-optimize-labber-selection.md)。`duration` 留在 Native，Labber 只匯出 `params` 與 `peak`。Writer 恢復拒絕無軸變數，不加索引軸。

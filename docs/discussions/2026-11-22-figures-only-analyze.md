---
date: 2026-11-22
status: active
superseded_by: null
related:
  - ../../zcu_lab/v2/gui-contracts.md#capability-宣告與實作
  - ../adr/0069-experiment-core-and-frontends.md
  - 2026-11-22-shared-presentation-facts.md
basis:
  repo_commit: a90cb66bbaa7e5269572d375fef321936dbeb3cc
  external_sources: none
---

# Live-only 實驗的 figures-only Analyze

## 結論

`onetone/power_dep`、`twotone/power_dep`、`twotone/zigzag` 與 `jpa/flux_onetone` 可用 Analyze 從明確的 Run 或 Load 來源重畫具名圖。Load 不自動畫圖。來源 cfg 存在時發布 Reps × Rounds，沒有數值 Analysis 或寫回。這是四個既有入口的改動，不要求所有 adapters 提供 Analyze。

## 問題

這四種 Result 早已有重畫需要的 IQ 與座標。保存的原生資料可 Load，但舊入口不能從同次來源重新看圖；使用者沒有保存舊 PNG 時，例如無法重看功率掃描的共振響應。

## 選項與取捨

沿用現有 figures-only publication，像 `jpa/check` 一樣不創造空的結果型別。首圖用原末格同類的投影與座標，未採點仍缺值。只從原 cfg 取摘要；不讀目前表單或重播 Run 進度。JPA flux 的結果仍是 device-native a.u.，不把操作裝置的 A／V 欄位套到舊結果。

## 未採用

維持 live-only 最省程式，卻留下可重畫的資料不能看。改保存末格 PNG 需要另一條圖與摘要的發佈路徑，而且補不了沒有舊圖的檔案。從零重做 Run 與 Analyze renderer 也超出本次需要。

## 留下的問題

離線 JPA 合成資料只驗讀檔、圖與接線，不能驗代表實驗的 pump-to-readout 行為或 R0。

## 依據

採用 [ADR-0069](../adr/0069-experiment-core-and-frontends.md) 的 explicit source 邊界與 [具名圖規則](../../zcu_lab/v2/figure-style.md)。原始 probe 資料未入庫；本次用各 leaf 的公開 core stage、註冊 adapter 與至少一個 application publication 分別核對圖、來源摘要與無寫回。

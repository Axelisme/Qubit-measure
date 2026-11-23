---
date: 2026-11-15
status: active
superseded_by: null
related:
  - docs/adr/0078-evolution-module-boundary.md
  - lib/zcu_tools/simulate/evolution/README.md
  - docs/discussions/2026-11-14-t6-direct-strategy.md
  - docs/discussions/2026-11-13-family-dominance-within-tolerance.md
basis:
  - "repo main@a836e1353b786217eb22190c0db6c2a3c7c06ca9"
  - "使用者 2026-11-15 覆核，選項 2B"
---

# t6 只保留 direct

## 結論

使用者於 2026-11-15 對第二批選項答「2B」，選擇移除 `t6` 的 channel 做法與價格分支。
`t6` 本身及 direct 做法保留。
現行規則由 [ADR-0078](../adr/0078-evolution-module-boundary.md#t6-只保留-direct) 擁有。

## 問題

[前一輪討論](2026-11-14-t6-direct-strategy.md) 保留了 channel，並依完整 acquire 的 key 重用量選擇做法。
後續調查發現，channel 的問題是沒有工作負載選用，不是精度不足。

當時的調查量到 channel 對 direct 的最大 density 差為 `1.6e-5`。
21 能階模型要讓同一 key 累積超過約 `1.5e4` 個載波週期，channel 才開始划算。
20、50、100 ns 的輸入，direct 每列便宜 4 至 21 倍。
四個代表性場景與 benchmark 都沒有選到 channel。
N-long 的 343 列有 196 種相位，類 CPMG 的 1,616 個 π pulse 有 1,565 種相位。
相位包含 `2πf·start`，很難重用同一 phase 的結果。
當時量過的 N-long、類 CPMG 與 RB 單一 acquire 最多 25.6 µs，沒有到門檻。

這些數字來自當時未入庫的調查工作檔。
本文保留裁決所需的數字，不要求讀者取得原始工作檔。

## 考慮過的選項與比較

- 保留 channel，補每列 apply 價格。它保留高重用輸入的加速機會，但代表性工作負載仍不會選到。
- 移除 channel，保留 direct。它刪掉未使用的建置、插值與價格比較。代價是高重用的合成輸入會變慢。
- 移除整個 `t6` 不在選項內。Direct 仍是 plateau 的 terminal candidate。

採用了第二案，沒有改變 direct 的容差或候選順序。
`floquet_period` 和 two-tone Floquet 的 channel 屬於其他算法，不在這次刪除範圍。

## 被否決的選項與理由

保留並補價格的建議曾引用「快速算法先修好，不輕易放棄」。
覆核時區別了兩者。該原則處理有誤差問題的算法，channel 已算得準，卻沒有服務現有瓶頸。

本次遵循 [容許範圍內只比耗時](2026-11-13-family-dominance-within-tolerance.md) 的精度與速度取捨。
也遵循使用者先前同意的刪除先例，刪掉代表性工作負載上未被選用、只增加維護量的做法。
這不表示只憑局部誤差或單次耗時就能刪除其他 family。

## 留下的問題

日後若有高重用輸入，要重新評估或從 Git 歷史恢復 channel。
例如同一 key 重用兩萬個載波週期的合成輸入，改走 direct 後每列會較慢。
這類輸入仍須驗證完整 density 誤差不超過 `2e-3`。

## 依據

- [Evolution README](../../lib/zcu_tools/simulate/evolution/README.md) 擁有現行 kernel 總表與數值契約。
- [t6 Kernel contract](../../tests/simulate/evolution/kernels/test_t6.py) 驗證高重用 direct 路徑及完整 density。
- [精度矩陣](../../tests/README.md#evolution-精度矩陣) 擁有跨 family 的最終誤差驗收方式。

## 後記

現行 `t6` 局部選路規則已搬到 [Evolution README〈t6 只保留 direct〉](../../lib/zcu_tools/simulate/evolution/README.md#t6-只保留-direct)。[ADR-0078](../adr/0078-evolution-module-boundary.md#t6-只保留-direct) 保留指標。

---
date: 2026-11-13
status: active
superseded_by: null
related:
  - ../adr/0078-evolution-module-boundary.md
  - ../../tools/README.md
  - 2026-11-14-family-pareto-remeasurement.md
basis:
  base_commit: 3d6eaedffdf7a15e1b122f3fe30cc39ad4d809fa
  repo_commit: 54175b3aab9d9dca13067cb13b3b75e48548f449
  external_sources: none
---

# Family 支配的定義：容許範圍內只比耗時

## 結論

當時確認以完整 pulse sequence 的最終誤差判定合格。
合格 family 視為一樣準，只比較耗時；另一個合格 family 快超過 10% 才算支配。
獨占適用域與各適用域最後一道候選不參加比較。
刪除較準但較慢的 family 前，仍須有同型態長序列的最終精度證據，並由使用者看過數據後點頭。
這項原則由 [ADR-0078](../adr/0078-evolution-module-boundary.md#family-去留合格只比耗時) 擁有，工具的判定機制見 [tools README](../../tools/README.md#engine-family-pareto-table)。

## 問題

原定義要求支配者誤差不較大，且快超過 10%。
S 的 Gaussian gate 序列上，三個 family 都遠低於 `2e-3`，但較慢的兩個只因更準就不被支配。

當時討論使用的 S 表數字如下。這些是歷史範例，不是後續十三張新版表的數字。

| Family | 完整序列最終誤差 | 耗時 |
| --- | --- | --- |
| shaped_sambe | 1.45e-4 | 0.27 秒 |
| period_table | 2.96e-5 | 9.0 秒 |
| gate_cache | 1.10e-5 | 54 秒 |

若只看誤差是否更小，就會為了容許範圍內一個數量級的精度差，留下慢得多的 family。

## 考慮過的選項與比較

1. 沿用原定義，誤差不較大且快超過 10% 才支配。它對應使用者 2026-11-12 的「每個算法至少是帕累託最優」，保留每個速度與誤差取捨點。
2. 把誤差 ≤ `2e-4` 視為一樣準。這是 agent 的中間讀法，試圖在原定義與速度優先之間另設一條線。
3. 只要最終誤差合格就視為一樣準，只比耗時，刪除前須有同型態長序列證據。使用者於 2026-11-13 對整理後的規則答「確認」。

既有先例是 2026-11-12 允許在容許誤差內用一致性換速度的裁決，見 ADR-0078 的歷史控制與精度驗收。
本次遵循它，並用後續原話釐清「帕累託最優」在合格精度範圍內的含義。
使用者說：「我通常是要求在给定最终精度容许范围内尽可能加速，所以通常这种精度差距极小的，我会优先选速度。」
使用者也澄清：「你要注意哦，这个精度是最终精度，也就是整个 post sequence 模拟最后的精度，而不是单个 kernel 的精度哦。」
因此合格證據取自完整序列最後的 density，不以單一 kernel 或每段 `budget` 代替。

## 被否決的選項與理由

原定義會為了遠低於門檻的精度差留下 family，與「在容許範圍內盡可能加速」相反。
中間的 `2e-4` 界線沒有使用者依據，未採用。
採用合格只比耗時，不代表短序列合格就能刪除其他 family；刪除前仍須補同型態長序列的實測。

## 套用結果

後續十三張表的結論引用了 [入庫摘要](../../tools/README.md#recorded-scenario-summaries)。
它們的量測版本是 `3545520dd8d5e8784be642246605f263654ce72b`，tree 為 `c9f484253fae2cc0db5a7d80dc1dd21bbe808d77`。
各表獨立比較，未用跨表或跨時段耗時判定快慢。

- `gate_cache` 在 A、S、S-long 已量且合格的格子被支配。A 的 clear-history 未量，不將缺值當支配證據。
- `period_table` 在 O2 不被支配。
- `shaped_sambe` 在 A、S、S-long 有不被支配的已量合格格子。
- `floquet_period` 在 F、I、F-bias045 與 N-bias045 有不被支配的已量合格格子。

這些結論沒有刪除 family。
逐 family 的場景與限制見 [kernel 證據表](../../lib/zcu_tools/simulate/evolution/README.md#recorded-real-workload-evidence)。
摘要的保存與比對取捨見 [單場景重量與摘要討論](2026-11-14-family-pareto-remeasurement.md)。

## 留下的問題

- 刪除仍等使用者看過數據後點頭。
- A 型 gain sweep 與 L 型變長 plateau 的相關長序列證據仍缺。
- C、P、X 型態還沒量。

## 依據

- [ADR-0078](../adr/0078-evolution-module-boundary.md) 保存取捨原則與最終精度邊界。
- [量表工具說明](../../tools/README.md#engine-family-pareto-table) 保存比較方式、測量模式與摘要入口。
- [Kernel 證據表](../../lib/zcu_tools/simulate/evolution/README.md#recorded-real-workload-evidence) 保存新版十三表的 family 結論與限制。

當時的原始討論與 S 表工作檔未入庫。必要原話、數字與選項已寫在本文，不需讀取那些工作檔。

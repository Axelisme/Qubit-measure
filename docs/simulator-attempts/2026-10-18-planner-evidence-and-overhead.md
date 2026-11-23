---
date: 2026-10-18
status: rejected
commit: 53b22f6dd
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-10-18-planner-evidence-and-overhead.md
superseded_by: null
---

# Whole-acquire planner 的 evidence admission

**讀者：** framework-dev、agent。查找這次優化的做法、結果與證據。

## 核心思路

先替整次 acquire 排出候選路徑，再用實測決定是否啟用，而不是只看各 stage 的估價。

## 實測

[原討論](../discussions/2026-10-18-planner-evidence-and-overhead.md#20-fit16-held-out-與完整-acquire-gate)記下驗收時的 main 與 calibration lane 起點，沒有標明被測候選版的確切執行 commit。最後修正後也未重跑。本文保留結論，不移錄 A/B 數字。

## 為何採用或否決

完整 acquire 的固定成本抵銷了 stage 內的收益，未啟用 production profile。後來以 `46647a3a5` 移除未啟用的 planner。

## 留下的線索

重試前先用當時的預設路徑做[完整 acquire 的配對比較](../discussions/2026-10-18-planner-evidence-and-overhead.md#後續需要的證據)，不要用 stage 校準通過代替。

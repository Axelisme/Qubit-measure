---
date: 2026-11-22
status: active
superseded_by: null
related:
  - 2026-11-22-presentation-rules.md
  - 2026-11-22-amp-rabi-shot-moments.md
basis:
  repo:
    before_task: 85e48d392961e9d990797febc27b761ea4b30fca
    verified_candidate: 18214ab7cce221ee88bab411a83a67f48b7131c7
---

# Analyze 摘要與逐點誤差棒的責任分工

## 結論

2026-11-22，使用者決定 Analyze 摘要只補本次 cfg 的 Reps×Rounds 和實驗已回傳的實際施加值。GUI 和 MCP 顯示 adapter 提供的同一份行。對數值大小敏感的一維資料，使用者選 shot SEM 作為逐點誤差棒。Designer 將估計器放進 analysis，把標題與圖例留在各實驗 core。這兩項分工是代裁，使用者未覆核。規則見 [ADR-0069〈呈現規範〉](../adr/0069-experiment-core-and-frontends.md#呈現規範)。

## 問題

RunMetadata 沒有完成輪數、提前停、實際 shot 數或施加值。Schedule 知道各次取數的進度，但同一 run 可能含多次 acquire 和 retry。另一方面，T2 Ramsey 在 Result 中已有實際施加的 detune，最初卻沒有共同入口讓 GUI 和 MCP 顯示它。重開沒有保存 detune 的結果時，舊 cfg 也不能證明施加的值。

繪圖 helper 已能畫指定的誤差範圍。缺的是計算所需的資料與算法。傳統 Rabi 只保存平均 IQ，無法由平均值逆推出每發散布；已有分類計數或 IQ 矩的實驗則已有部分輸入。RB 的跨 seed 標準差和擬合參數誤差回答的是不同問題。

## 考慮過的選項與比較

- 保留完整採集事實，得先定義多次 acquire 或 retry 如何合併，再決定是否寫進 RunMetadata。只在當次 session 顯示全部事實可以省下磁碟變更，但仍要增加 runtime 到 Analyze 的通道，重開也不再有這些值。
- 共享 adapter 提供的行不需要補新的 runtime 事實。T2 Ramsey 可用保存的 cfg 與已有的 Result。使用者答「1a, 存檔看有沒有需求」，接受這個範圍，也把回存實際值的決定留給各實驗。
- 逐發資料的 sample SD 除以保留 shot 數的平方根，估計該點平均值的誤差。留每輪平均則量到輪間變異，含漂移。使用者選「A」，並說「像我其实更想要设计的，其实应该是平均标准差，也就是这个点的预期误差。」
- 標題 formatter 能統一措辭，卻無法自行判斷分支名稱或被畫的物理量。調查掃描 110 個 core source，同檔 call graph 在 61 個 live 路徑沒找到明確的 title 或 legend 呼叫。這不等於畫面一定沒有標題，但也沒找出能先共用的 formatter caller。

採用的摘要沿用 [ADR-0062](../adr/0062-experiment-workflow.md) 的 Schedule 邊界，未將完成輪數與中止原因灌進 RunMetadata。標題與圖例沿用 [ADR-0072](../adr/0072-module-boundaries.md) 的 lab 繪圖分工。RB 的跨 seed 統計仍由 [ADR-0085](../adr/0085-numerical-fitting-owners.md) 所述的原 owner 負責。

## 被否決的選項與理由

- 擴大 RunMetadata，或新增一份 session-only 採集通道。使用者不需要完成輪數、提前停和 shot 數，這些欄位也不值得另定聚合規則。
- 要求各實驗重複在原生 Result summary 或 recipe payload 編碼同一視圖。保存事實與顯示文字是不同工作，現成的 MCP presentation 已能帶出 adapter 行。
- 各實驗重做 SEM 公式，或用輪間平均變異替代。前者會產生不同的缺值與分母規則；後者不估計逐點 shot 抽樣誤差。
- 把標題、圖例交給通用 plotting helper，或另建 presentation contract。需要分支及物理意義的地方仍由實驗決定。

## 留下的問題

各實驗決定是否保存其實際施加值。T2 Ramsey 的原生映射沒有保存實際 detune，因此重開時仍可顯示原 cfg 的 Reps×Rounds，但沒有 detune 行。各家族是否已有計算 shot SEM 的統計量，留給家族呈現修正。第一個傳統 Amp Rabi 的取數與保存取捨見 [同日討論](2026-11-22-amp-rabi-shot-moments.md)。

## 依據

調查、獨立 challenge 與使用者原始問答未入庫。最終的 [Analyze 投影](../../lib/zcu_tools/gui/app/measure/remote/analysis_types.py)、[數值估計器](../../lib/zcu_tools/analysis/shot_uncertainty.py) 與 [figure-style](../../zcu_lab/v2/figure-style.md) 保留機制及驗收入口。

後記（2026-11-22）：[後續裁決](2026-11-22-data-point-lines.md)擴大 shot SEM 的適用範圍，spectrum 有現成資料可估時也畫；數值與呈現的分工不變。

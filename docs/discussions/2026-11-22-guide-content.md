---
date: 2026-11-22
status: active
superseded_by: null
related:
  - 2026-11-21-form-board-limits.md
  - 2026-11-22-presentation-rules.md
  - ../adr/0075-agent-interface-layers.md#判讀依據放知識庫
  - ../../zcu_lab/v2/figure-style.md#operator-guide
  - ../../zcu_lab/v2/operator-workflow.md
basis:
  repo_commit: 91421a8d176b796718a4b37245b2bb61cb6cd383
  external_sources: none
---

# 實驗 guide 只寫操作者需要的內容

## 結論

使用者選 B：一次改規則，並按 family 清理 111 份 guide。Guide 面向懂物理、沒看過實作的操作者。[Gd1–Gd3](../../zcu_lab/v2/figure-style.md#operator-guide) 是規則的出處。約 40 行是參考，不是檢查門檻，也不拿來刪除必要的操作或判讀依據。

六項大綱是量測目的與主要觀察量、單週期的脈衝與讀出序列、擬合參數的物理意義、必須自行決定的欄位如何設定及理由、各圖支援的判斷及誤差棒的意義，以及下一步。下一步寫明結果怎麼用、寫回改哪個值，並在需要時給修正式。

通用 Open、Defaults、Refresh、Apply 和 Save 統一寫在 [`zcu_lab/v2/operator-workflow.md`](../../zcu_lab/v2/operator-workflow.md)。實驗特有的前置校準和 Run 前檢查仍留在各自的 guide。現有表單、圖和摘要一起支援判讀，不由 guide 重抄整份介面。

## 問題

使用者問：「guide 的定位为什么感觉好像杂七杂八的一大堆都丢进去？」並說明不是要把 guide 限在幾個例子：「我不是说盖的只能放那些东西啊，我只是说它必须足够精炼，且只放操作者会需要知道的东西，一些细节，一些特例，还有一些应该放到知识沉淀里的东西，就不要放到guide里啊。」

在 `91421a8d1`，111 份 guide 的行數中位數是 55，最長 239 行。分層抽取 20 份，共 1,401 個非空行；量測、序列、物理、讀圖和下一步合計 314 行，占 22.4%。Native、Labber 存檔格式與舊檔行為占 24.6%，是樣本中最大的單項。這些行數用於找重複段落，不判定哪些關鍵設定可以刪。

## 考慮過的選項與比較

| 選項 | 改什麼 | 取捨 |
| --- | --- | --- |
| A 只改規則 | 更新 Gd1–Gd3 和三份作者文件，以後改到個別 guide 時再清理。 | 變更最少，但操作者仍會在多數現有 guide 裡找必要資訊。 |
| B 改規則並一次清理 | 按 family 清理全部 111 份。代表 guide 用表單、guide、Live 和 Analyze 的 R0 讀者包檢查。 | 原處已提供的資訊可以直接去重，又不必改 GUI 或 MCP。 |
| C 再讓知識庫可見 | 在 B 以外，讓 GUI guide pane 與 MCP `recipe_guide` 指向相關知識庫條目。 | 操作者可直接找排錯和判讀案例，但需要擴充 GUI、MCP 介面。 |

B 的刪除條件是原處已讓操作者看到等價資訊。存檔格式和舊檔行為回存取介面或格式文件；recipe 留空行為回公開 schema，並核對失敗時點。一般寫回門檻回結果或預覽，缺完整資訊時仍在 guide 保留必要說明。共用點擊流程移到 [操作員共用頁](../../zcu_lab/v2/operator-workflow.md)。模擬器限制進 [FTF README](../../zcu_lab/simulation_envs/FTF/README.md)，驗收缺口留在相應任務的記錄。硬體限制、排錯和可重用判讀進量測知識庫，延續 [ADR-0075〈判讀依據放知識庫〉](../adr/0075-agent-interface-layers.md#判讀依據放知識庫)。維護者需要的實作理由留在負責該行為的程式註解。

B 改寫原有 Gd1–Gd3 及作者文件中的範圍要求。它沿用 [R0](../../zcu_lab/v2/figure-style.md#r0不讀程式碼也能判讀) 的讀者分工，也保留操作者確實要用的 T2 取樣與窗口、shot SEM 及 readout 等待起點。

## 被否決的選項與理由

A 不處理現有 111 份 guide，重複內容要等各實驗再次修改才會移走。C 把知識庫可見性一起做，會先改兩個介面，但目前的主要問題是 guide 太雜。使用者在 2026-11-22 答：「好，那就用 B， 我觉得没问题。」

## 留下的問題

GUI 與 MCP 是否要提供知識庫入口，留待清理後按實際查找需求決定。本次不改 guide 的顯示或傳遞契約。

## 依據

本文的計數來自 `91421a8d176b796718a4b37245b2bb61cb6cd383`。原始分層盤點未入庫，取捨所用的樣本範圍與結果已列在上文。更早的讀者定位見[表單板子限制討論](2026-11-21-form-board-limits.md)，R0 的背景見[實驗呈現規範討論](2026-11-22-presentation-rules.md)。改版落在 [figure-style](../../zcu_lab/v2/figure-style.md#operator-guide) 和 [experiment-authoring](../../zcu_lab/v2/experiment-authoring.md)、[experiment-style](../../zcu_lab/v2/experiment-style.md)、[measure-authoring](../../zcu_lab/v2/measure-authoring.md)。

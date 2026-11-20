---
date: 2026-11-20
status: active
superseded_by: null
related:
  - 2026-11-20-logical-cfg-paths-and-overrides.md
  - 2026-11-19-recipe-ask-helpers.md
basis:
  repo:
    main: d062b7bd8
---

# Recipe 報告與執行進度

## 結論

2026-11-20，使用者說：「再检查一次，呃，你有所使用过的每一个 recipe 的界面的行为，看有没有可以打磨的。」看過三題的比較後，他答「1B 2A 3a」。

Recipe 報告轉出 owner 已算好的 fit quality。三個 STIRAP recipe 回報本次網格最高的 corrected f population 與座標。Run 取數時，`status(execution)` 與 `wait(execution)` 顯示和 `wait(op)` 相同的進度與 ETA。

## 問題

11 月 9 日，amplitude Rabi 報告回傳了 pi gain 與誤差，但 `fit_quality` 是 null。11 月 19 日，mux Ramsey 報告也省略逐 tone 的品質量。Caller 得另外查分析回覆，才能區分「沒有品質量」和「報告沒有轉出」。

11 月 9 日，一次 25 × 25 的 STIRAP detuning 掃描只回 confusion 診斷。Agent 從圖與資料另算出最大 corrected f population 約 1.0185，座標為 Pump +1.00 MHz、Stokes −0.50 MHz。等待掃描時，agent 也得從 `active_path` 的 `acquire#538` 自行估算還要等多久。

## 選項與取捨

| 題目 | 採用 | 被否決的選項及理由 |
| --- | --- | --- |
| 1：Fit quality | B。所有 recipe 都轉出分析 owner 已提供的主擬合品質，沿用 `mux_onetone` 的逐 tone 位置。不重算，不以品質值改變 writeback。 | A 只改實際用過的 7 個 recipe，會留下無法解讀的 null。C 移除基底欄位，則有擬合資訊仍得另外查。 |
| 2：STIRAP | A。回報本次網格的最高有限 corrected f population 和座標，不裁切，也不叫 fidelity。 | B 只引導 caller 看圖或讀存檔，會保留每次另找峰值的工作。 |
| 3：Execution 進度 | A。執行中的 Run 把既有 `wait(op)` 進度投影帶到 execution 查詢。沿用 [ADR-0075 的長操作原則](../adr/0075-agent-interface-layers.md#長操作看-agent-中途能否影響結果)，讓 caller 能在同一個 execution 判斷是否停止取數。 | B 只記下 `status()` 找 op 再 `wait(op)` 的查法，仍要多找一次 handle。 |

第 4–6 項由 agent 代裁，只補說明。`participants` 同時選 drive 與對應 readout tone，不能拿掉 participant 來單獨測 drive 的影響。`wait.timeout` 寫明秒數、預設 60、上限 300，逾時不停止工作。Mux amplitude Rabi 指向既有 logical 欄位讀取方法，再說明 `to_inline`、`select`、`value` 三步。這遵循[先前收窄 cfg 介面的裁決](2026-11-20-logical-cfg-paths-and-overrides.md)，不暴露 YAML 包裝路徑。

## GE 的三份擬合

GE 的 joint 擬合提供共用參數與 length ratio；ground、excited 各自精化 histogram。2026-11-20，使用者答「B」。報告在 `fit_quality` 放 joint，另用 `ground_fit_quality` 和 `excited_fit_quality` 原樣轉出個別診斷。Mux 放在各 tone。

A 只放 joint，會看不到 ground 和 excited 擬合的問題。C 另放具名 mapping、讓共通 `fit_quality` 留 null，與 1B 要減少有診斷卻回 null 的方向相反。

## 依據

比較取自 2026-10-21、11-09、11-19 的量測 session。原始調查工作檔未入庫。交付欄位與操作查法分別見 [recipe 文件](../../zcu_lab/recipes/README.md)、[mux guide](../../zcu_lab/recipes/mux-guide.md) 與 [MCP execution 回覆](../../lib/zcu_tools/mcp/measure/README.md#runtime-與回覆)。

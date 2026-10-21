---
date: 2026-10-21
status: active
superseded_by: null
related:
  - docs/adr/0075-agent-interface-layers.md
  - docs/adr/0062-experiment-workflow.md
  - docs/discussions/2026-10-16-recipe-workflow-yield-vs-async.md
  - docs/discussions/2026-10-21-recipe-interface-redesign.md
  - docs/discussions/2026-10-21-workflow-positioning-and-hosting.md
basis:
  - "repo@2a8a67b71：recipe driver 以 RecipeStep 與 ask 驅動 generator；workflow Engine 以 RunEffect、DeviceEffect、WaitEffect 三種 effect 驅動 step"
---

# 三角色協定：agent、腳本、runtime 與 Wait／Ask／Signal

## 結論

Recipe 與 workflow 的底層是同一份中立 generator 協定，三個原語對應三個角色之間的三種等待與通知：

| 原語 | 方向 | 語意 |
| --- | --- | --- |
| `Wait(op)` | 腳本等 runtime | 等一個 operation 終局，回具名結果（completed／cancelled／failed） |
| `Ask(question)` | 腳本等 agent | 交出問題，agent 或使用者以同一個 `answer` 工具回 |
| `Signal(kind, payload)` | agent 主動送腳本 | host 放進該 execution／run 的 inbox，只在 yield 邊界生效 |

使用者：「我心智模型是三個角色：agent/recipe/runtime，recipe等agent就是Ask，recipe等runtime就是Wait。agent主動送訊息給recipe就是signal之類的」；「Engine底層應該也要用Ask/Wait模式，其他都是糖」。

兩個 host 各自把協定包成糖：recipe host 的 op 是 GUI RPC operation，糖是 `tab.run()`、`analyze()`、`save_raw()`、`ask_writeback()`；workflow Engine host 的 op 是 runtime acquisition、device setup、timer，糖是 `env.run()`、`env.setup_device()`、`env.wait_until()`、`env.ask()`。Engine 在協定之上保留一 run 一 operation、Pause／Resume cursor、tunables revision、records 與 journal；`Ask` 使 run 進入 paused 並掛上問題，answer 經 Resume 送回。question／answer model 兩個 host 共用，agent 只學一套。

Signal 第一版只有內建 kind（cancel、finish_early、pause、stop、tunables apply），全部由 host 處理：cancel 讓進行中的 `Wait` 回 cancelled，pause／stop 歸 Engine，tunables 由 Engine 在邊界換新。腳本不讀 inbox；作者自定 kind 等真有需求再加，形狀為按 kind 取用 `env.take_signal(kind) -> payload | None`，不以迴圈讀多筆。使用者：「for sig in env.signals()不太直觀，為什麼預期好幾個signal?」。

## 問題

Recipe 驅動器與 workflow Engine 各有一套 yield 協定，作者要學兩種；0075 記錄的「L3 也 yield 協作」沒有落實的介面；agent 對執行中腳本的主動訊息（cancel、finish_early、Pause、tunables）分散在各自的工具，沒有統一的角色模型。

## 考慮過的選項與比較

- Engine 只加第四種 effect `Ask`，保留三種 effect 為底層：兩套底層協定並存，糖的定位不清。否決。
- Engine 底層改用 Wait／Ask，effect 與 env 方法降為糖：採用。協定型別與純驅動迴圈放在 mcp 與 experiment 之下的中立 owner，實作時依 ADR-0072 定位確認。
- 腳本讀 inbox：迴圈讀多筆、按 kind 取用、第一版不給。採第三項，第二項為日後形狀。
- 合併兩個 driver 成一個：不合併。兩個 host 的 op 來源、持久化與 Pause／Resume 不同，只共用協定與問答 model。

## 與 2026-10-16 討論的關係

該篇裁定保留 generator、主動訊息只支援控制命令與邊界生效的 tunables、不做 async。本文把它的 (a)(b) 結論形式化成 `Signal`，仍不支援 Run 進行中即時改變數值；`Ask` 補上該篇留下的「workflow 的 agent yield」。

## 留下的問題

- 中立 owner 的確切位置（候選 `zcu_tools.core` 的一個模組）。
- Engine 以 `Ask` 進入 paused 後，GUI 與 MCP 如何呈現問題與回答控件，由 workflow 承載的 UI prototype 回答。
- 作者自定 Signal kind 的首個真實需求。

## 依據

- 現行 recipe driver 與 Engine effect 的行為見 related 的 ADR 與各 module README；本文不複製。
- 定位與承載的推導見 [Workflow 定位與承載](2026-10-21-workflow-positioning-and-hosting.md)。

---
date: 2026-11-20
status: active
superseded_by: null
related:
  - 2026-10-12-fluxdep-mcp-design.md
  - ../adr/0075-agent-interface-layers.md
basis:
  repo:
    main: 7fe0aa5eb
---

# fluxdep GUI 跟隨 agent 操作

## 結論

2026-11-20，使用者回答：「關於fluxdep gui，1A 2A」。使用者選擇讓 fluxdep GUI 跟隨 agent 操作的譜、主頁與階段。成功的非唯讀命令會跟隨，未編輯就按 Finish 或 Cancel 也一樣。唯讀命令不切換。跟隨不轉移控制權，沒有控制權的人只能檢視。決策見 [ADR-0075〈Tool 即意圖〉](../adr/0075-agent-interface-layers.md#tool-即意圖)。

Agent 根據命令的目標決定四個細節。移除目前選中的譜時清空選取，不替使用者挑下一張。還原時選清單的第一張。搜尋成功完成時切到 Show。`project.setup` 只更新一直可見的專案標頭，不切頁。完整對照見 [fluxdep README〈跟隨遠端操作〉](../../lib/zcu_tools/gui/app/fluxdep/README.md#跟隨遠端操作)。

## 問題

2026-11-20 試跑時，使用者回報：「fluxdep gui在agent操作時顯示不會自動跟隨，導致我不知道agent現在在做什麼」。當時只有新開或更新 Align／Sample 草稿會切換選中的譜。Agent 事件從不切換 Spectra／Fit 主頁，使用者要先停在 Fit 才看得到 Filter 或 Search。載入、還原、移除、未編輯的 Finish／Cancel、擬合完成與匯出也不會帶畫面跟過去。

2026-10-13 的 commit `8f8e389db` 把不切主頁定成實作規則。這是當時 agent 自己的選擇：「Agent-origin active changes, pickers, Filter and pending search update their page without switching it」。這與 ADR-0075〈Tool 即意圖〉的「GUI 跟隨 agent 操作的對象」不一致。使用者也在 2026-10-14 確認「tab跟隨我保留」，見 [ADR-0075〈不遷就 GUI 的操作方式〉](../adr/0075-agent-interface-layers.md#不遷就-gui-的操作方式)。

## 考慮過的選項與比較

1. 顯示跟到哪裡？
   - A：切換選中的譜、主頁與階段。使用者直接看到 agent 的目標；沒有控制權時仍是唯讀。
   - B：切換譜與階段，但不切主頁。用提示列和「前往」按鈕讓使用者自行切頁。
   - C：採用 A，再加一個固定畫面的開關。這得另外維護使用者固定的位置，實作量當時估計約為 A 的兩倍。
2. 哪些命令觸發跟隨？
   - A：所有改變狀態的命令成功後跟隨，包括未編輯就送出的 Finish；唯讀命令不跟隨。
   - B：只跟隨編輯階段的命令與擬合操作。

## 被否決的選項與理由

顯示選項 B 仍要使用者找 agent 在哪個主頁，再按一次「前往」。C 需要維護第二個顯示位置，試跑時沒有固定畫面的需求。觸發選項 B 會漏掉 agent 換譜或匯出，使用者仍看不到這些動作。

本次遵循 ADR-0075 與「tab跟隨我保留」的裁決，也沿用 [fluxdep MCP 介面設計](2026-10-12-fluxdep-mcp-design.md) 將選中的譜當作顯示狀態的做法。`8f8e389db` 不切主頁的做法未經使用者覆核，只供參考，本次不再沿用。

## 留下的問題

無。

## 依據

- 2026-11-20 的試跑回報、選項比較與使用者裁決。原始工作檔未入庫，相關文字與取捨記在本文。
- 本 repo 的調查基準為 `7fe0aa5eb`。舊行為出自 `8f8e389db`。
- [ADR-0075〈Tool 即意圖〉](../adr/0075-agent-interface-layers.md#tool-即意圖) 與 [〈不遷就 GUI 的操作方式〉](../adr/0075-agent-interface-layers.md#不遷就-gui-的操作方式) 分別記錄主視窗跟隨與不跟隨 dialog 焦點的規則。

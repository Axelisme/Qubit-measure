---
date: 2026-11-19
status: active
superseded_by: null
related:
  - ../adr/0075-agent-interface-layers.md
  - ../adr/0068-remote-transport.md
basis:
  repo:
    main: e4f3eab8ff05a61b72b3133aa7120473cb5205ab
---

# Recipe 的寫回 helper 與 ask 原語

## 結論

2026-11-19，使用者確認 recipe 的通用契約只有 `ask` 與正常結束，其上的工具都是 helper。mux GE 的寫回改由 helper 依 stage 出題，recipe 在 Primary 與 Post 分析後各問一次；`ask` 原語不變。原則寫在 [ADR-0075](../adr/0075-agent-interface-layers.md#通用契約只有-ask-與正常結束)。

## 問題

2026-11-19 一次 FTF 離線模擬量測中，agent 經 MCP 跑 `zcu_lab/recipes/mux_singleshot_ge.py`。這個 recipe 先做 Primary 與 Post 兩次分析，最後呼叫一次 `tab.ask_writeback()`。

- 題目同時列出兩筆候選，兩筆都整筆替換同一個 mux readout module。Primary 只寫 g/e 中心；Post 寫同樣的中心，加上 radius 與 confusion matrix。
- Agent 兩筆都選時，`lib/zcu_tools/mcp/measure/recipe_api.py` 的 `WritebackQuestion.validate_answer` 只以 name 判斷重複，丟出 `ValueError("Duplicate writeback candidate")`。MCP `answer` 回傳整段 traceback。
- 但 `zcu_lab/v2/mux/singleshot_ge/guide.md` 的 Writeback 段與 `RecipeTab.accept_writeback` 都允許同批套用 Primary 再 Post。

## 考慮過的選項與比較

Agent 提出三個選項：

- A：分兩次問。Primary 分析後問 Primary 的寫回，Post 分析後問 Post 的寫回。
- B：維持一題，判斷重複改用 (stage, name)，允許兩筆一起接受。
- C：維持一題，把同一 destination 的兩筆標成互斥；兩筆都選時回結構化錯誤。

Agent 建議 C，理由是 Post 已包含 Primary 的中心，實際上是三選一：只寫中心、寫中心加 radius、都不寫。三個選項都依 [ADR-0068](../adr/0068-remote-transport.md) 的錯誤分類規則，把答案驗證失敗改成結構化錯誤，request 不消耗。

使用者質疑一題列兩筆的流程：「五 C 很奇怪，怎么会两笔一起接受？这两笔不应该是分开接受的吗？应该说，recipe 不应该是先等第一笔结束，再等第二笔分析吗？」並選 A：「5cA, 應該要幾次，什麼順序都可以」。

Agent 接著把 A 寫成「替 `ask_writeback` 增加 stages 參數，讓 recipe 能問任意次、任意順序」。使用者指出這把層次弄錯了：「我记得 recipe 不管是 ask 什么，底层都是基于同一个 ask。我自己认为 recipe 的通用契约就只有 ask，还有正常结束。这是一种协作式的原语，在之上架的工具都属于 helper 的范围。那现在很明显就是你这个 helper 不好用。本来 recipe 可以问任意的东西，回答任意的结果，没有什么次数或者是种类限制。」

最後的修正只改 helper：

- `ask_writeback` 可指定 stage；
- 判斷重複改用 (stage, name)；
- 驗證失敗回結構化錯誤；
- 兩個 GE recipe 改為每次分析後各問一次。

## 被否決的選項與理由

- B 與 C：兩者都維持一題，與使用者「先等第一筆結束，再等第二筆分析」的流程不合。
- 把 A 說成替原語新增能力：使用者指出 `ask` 本來就沒有次數或種類限制，問題出在 helper。

## 留下的問題

- 其他 helper 是否有同類限制，沒有盤點。
- 診斷用的 run 是否提供不問寫回的模式，另案討論。

## 依據

- 本 repo main e4f3eab8f：
  - `lib/zcu_tools/mcp/measure/recipe_api.py` 的 `RecipeTab.ask_writeback`、`WritebackQuestion.validate_answer`；
  - `lib/zcu_tools/mcp/measure/recipe_runtime.py` 的 `writeback_proposal`；
  - `zcu_lab/recipes/mux_singleshot_ge.py`；
  - `zcu_lab/v2/mux/singleshot_ge/guide.md` 的 Writeback 段；
  - ADR-0068 的錯誤分類段、ADR-0075。
- 原始對話與量測工作檔未入庫；本文已寫入所需的事實與原話。

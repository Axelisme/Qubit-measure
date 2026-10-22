---
date: 2026-10-22
status: active
superseded_by: null
related:
  - docs/adr/0075-agent-interface-layers.md
  - docs/adr/0069-experiment-core-and-frontends.md
  - docs/adr/0062-experiment-workflow.md
  - docs/discussions/2026-10-22-workflow-single-generator.md
  - lib/zcu_tools/experiment/README.md
basis:
  - "repo@8c8f48b54 加上當日未 land 的 workflow-single-generator 與 interrupt-scopes integration：scope 樹已有 child token、label 六種、active_child 登記、cancel(scope) 選擇器，iteration token 在 commit 時換新，commit 目錄由 env.commit_dirs() 配置"
  - "原始工作檔（map 決定 0427、0433、0447、0466、0467、0472）未入庫，結論與必要事實寫在本文"
---

# 中斷 scope 成為顯式的 core 契約：意圖、出口、with／for 邊界與路徑命名

## 結論

Scope 從 host 內部的 token 樹升為 experiment 層的契約，core 實驗、workflow、recipe 共用同一個 `Scope` 物件；作者手上拿到的是顯式的 scope 物件，邊界由 `with` 與 `for` 保證，沒有「目前 scope」這種隱藏狀態。使用者 2026-10-22 裁定（原話）：

- 中斷語意的兩個軸：「中斷應該可以分成，協作式中斷，出口位置一樣，靠回傳值區分，以及錯誤中斷，出口位置是error catch，中斷意圖分為，內部當下執行的等待被中斷，或是自己被中斷」。
- 「signal scope應該搭配有作用域保證的with以及for來規範scope範圍」；對隱式「目前 scope」：「你這種寫法會遇到作者跳過中間的with scope直接傳頂層scope給底層用的處理方式吧？你認為signal是不是應該是顯式的？」
- 「為了對agent友善，名字可以討論，比如script改叫workflow#<workflow名>」。
- 「可以考慮讓scope成為zcu_lib/core的契約，讓core.py使用」。
- 「it.commit_dir可以對應到scope的層級與深度。這樣commit_dir也不需要知道iteration，也支援nested的路徑」。

採用的規則：

1. 腳本以下每一層（iteration、operation、schedule、acquire、作者自訂）被取消都是「我等的事被打斷」，協作式交付：同一個出口，`Completed(partial_result=True)` 或 `Failed` 以值回來。根 scope（workflow／recipe）被取消是「我被中斷」，錯誤式交付：下一個 `await` 丟 `ScriptCancelled`，`finally` 收尾，終態 cancelled。
2. `Scope` 由 experiment 層定義：`stop`（往下傳的 bit）、`cancelled`（作者面布林）、`child(label)`（context manager）、`dir`、`run`／`call`（`name=` 可選）、`setup_device`、`wait_until`、`commit`。`RunContext.scope` 取代 `cancel_signal`；`Schedule(cfg, buffer, scope=)` 內部建 schedule child，acquire 建 acquire child；device、clock、program_builder 仍只收 bit。
3. 作者面：`env` 是根；`with env.scope(label) as s`；`for item, iteration in zip(items, env.iterations())` 每趟一個 iteration scope，`next()` 釋放上一趟，迴圈離開由 generator close 釋放；helper 收 scope（`await run_t1(iteration, cfg)`）。
4. 路徑 `kind:name#seq`：根 `workflow:<名>`／`recipe:<名>` 無序號，`iteration#N`，`operation:<name>#N`（name 由呼叫端可選提供，不從 callable 反射），`schedule#N`、`acquire#N`，作者 label `<label>#N`。選擇器 `kind`、`kind#N`、`operation:<name>`；同層同時存活的同名 scope 要求 `#N`。
5. 目錄鏡射：`scope.dir` 為 `<label>-<seq:06d>` 巢狀目錄，根為 run 目錄；`scope.commit(record)` 一個 scope 一次，record.json 寫進 `scope.dir`；reader 走目錄樹、依 commit 序號排序。
6. 慣例：量測之間以 `iteration.cancelled` 明確提早結束；record 採收集式（一趟開始建可變 record 逐步填，`commit` 當下 dump 快照）；record 後果用 `status="skipped"`；成功／失敗分辨維持 `Completed | Failed` 加 `isinstance`。

## 問題

當時的 scope 樹（2026-10-22 稍早的裁決）把 iteration token 綁在「commit 時換新」，`active_path` 靠登記「最新建立的 child」。走查例子時發現兩個問題：作者 `continue` 跳過 commit 時下一趟會繼承已 set 的 token；「目前 scope」是看不見的狀態，作者若把上層 token 直接傳給底層，樹就錯了而沒人知道。同時 workflow 與 recipe 對「根被取消」的交付不一致：recipe 用例外結束腳本，workflow 跑到 return、每個 await 回 partial。

## 考慮過的選項與比較

- **邊界由副作用決定（commit 換 token）**：改動最小，但靠作者守紀律；`continue`、`break`、例外都會漏。被否決。
- **`async with env.iteration()` 區塊**：有語法保證，但使用者認為「不夠直覺與接近python原生語法」。被否決。
- **顯式 scope 物件加 `for`／`with`**：`for` 的一趟就是 iteration，`with` 就是作者自訂段；物件本身帶 token、目錄、序號，`run`／`commit` 掛在物件上，掛錯層只能是看得見的選擇。採用。
- **operation 名字從 callable 反射**：零成本但對作者物件探測，lambda 變 `<lambda>`。改為呼叫端可選的 `name=`，lab helper 填。
- **屬性叫 `skipped` 還是 `cancelled`**：`skipped` 描述後果且只對 iteration 成立；`cancelled` 描述原因、每層適用且與動詞、ack 同字。屬性用 `cancelled`，record 後果用 `skipped`。
- **`record.failed` 旗標 vs `isinstance(record, Failed)`**：旗標讓 `.value` 變 Optional、失去型別收窄；維持 union 加 `isinstance`，使用者認為 `match` 寫法「更糟」。

## 被否決的原因摘要

隱式「目前 scope」與 commit 換 token 都把邊界藏在作者看不到的地方；core 認識 scope 是為了讓 Schedule 自己建 child，但 driver 層仍只收 bit，避免把 workflow 的認知帶進 device 與 program 模組。

## 留下的問題

- 根為 `run` 的 GUI 單次量測視為 operation 層（error／partial 上交到此為止），實作時確認 `raise_if_error` 行為不變。
- 同層同時存活的同名 scope 回 ambiguous 的具體回覆形狀。
- zcu_lab 約 60 個 core.run 的 `stop=` 改 `scope=` 以腳本替換，實驗邏輯不動。

## 依據

本文 basis 所列 commit 與當日未入庫的工作檔；相關 ADR 見 [ADR-0075](../adr/0075-agent-interface-layers.md)「控制命令的傳遞路徑」與 [ADR-0069](../adr/0069-experiment-core-and-frontends.md) §40。

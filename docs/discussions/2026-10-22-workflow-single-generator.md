---
date: 2026-10-22
status: active
superseded_by: null
related:
  - docs/adr/0062-experiment-workflow.md
  - docs/adr/0075-agent-interface-layers.md
  - docs/discussions/2026-10-16-recipe-workflow-yield-vs-async.md
  - docs/discussions/2026-10-21-three-role-protocol.md
  - docs/discussions/2026-10-21-recipe-interface-redesign.md
  - lib/zcu_tools/experiment/workflows/README.md
basis:
  - "repo@eb8b57ac3：workflow 為 step 函式模型（init／Next／Done／Aborted），Engine 以 ScriptDriver 推進，Effect 包成 Wait；recipe 新路徑以 Wait／Ask／Signal 協定驅動，便利層尚未實作"
  - "使用紀錄引自 2026-10-16 討論：主動介入只有一次 cancel，Pause、tunables、finish_early 無使用紀錄"
---

# Workflow 改為單一 generator：取消、Signal、step 模型與回退的取捨

## 結論

Recipe 與 workflow 都寫成一個 generator：`yield from` 等 runtime 或 agent，`return` 結果。四個裁決：

1. Recipe 取消統一為 host 直接結束 generator，不提供作者側的取消分支或宣告。使用者：「既然用不到cannel，那也不用handler了，統一return，這樣寫法比較簡潔」。`finish_early` 不受影響，它讓進行中的 Wait 以 completed 回來，流程繼續。
2. 協定移除 `Signal`。作者面只剩 `Wait`、`Ask`、return；控制命令全由 host 處理。使用者：「我覺得tunables_apply/cancel/stop有點多餘」，對移除整個型別的回答是「先試試」。
3. Workflow 由 step 函式模型改為單一 generator，刪除 Effect 三型別，env helpers 直接 yield `Wait`／`Ask`。Pause 立即打斷進行中的 operation，Resume 重跑該 operation；tunables 以 `tun.current` 即時讀取；iteration record 以 `yield from env.commit(record)` 提交。
4. Engine 不提供整點重做或回退到過去任一點。整點重做由作者以 Ask 迴圈自行寫；回退以新 run 組合：plan 切成要重跑的範圍，histories 初值由作者用 `saved_run` reader 從上一個 run 讀回。使用者：「好吧，你說服我了，採用你的建議」。

## 問題

Recipe 的新協定 land 後，review 發現取消語義與 ADR-0075 不一致，advisor 先依 ADR 改回「具名 cancelled 結果交作者」，再提議以 `@recipe(on_cancel=...)` 讓預設精簡。使用者反問框架怎麼知道 return 的資料、目前的 workflow 有沒有用到 cancelled、Failed 分支的實作量是否與 Cancelled 相當，以及 early stop 是否會被統一 return 影響。接著使用者指出 Signal 的三種 kind 多餘，要求先讀 ADR、討論與實際用法再談，並提出 workflow 寫法應與 recipe 一致。

## 查證的事實

- 22 個 recipe 的取消分支全部只是 `return`。Workflow 在 2026-10-22 之前沒有 `Cancelled`；當天為配合具名結果加的九處 guard 都把取消當 Failed 走同一分支。acquire 的 partial data 由 Engine 保留。
- `Signal` 五種 kind 只有 `cancel` 被建立（recipe host 兩處、engine 一處），作用是讓 driver 進入取消模式。`finish_early` 設在 Run 上，`pause`／`stop` 是 engine lifecycle 加 `StopSignal` token，`tunables_apply` 是 engine revision 快照。`on_signal` 沒有任何 host 傳入。
- 非同步分兩層：打斷進行中的 Wait 靠 recipe 的 `Event` 與 workflow 的 `StopSignal` token，跨執行緒；generator 在 yield 邊界看到什麼由 driver 決定，同一執行緒。`Signal` 只在第二層。
- Step 模型存在的唯一理由是 Pause 丟棄當前 step、Resume 以新 tunables 重做。Journal 不宣稱崩潰接續，cursor 不是為了重啟。coherence_fluxdep 的 step 本體約 140 行，約半數是 state、index 推算與 summary 重畫的樣板。
- `Aborted` 沒有任何 lab workflow 使用；`RunStatus.state` 沒有 GUI 讀取。

## 考慮過的選項與比較

### Recipe 取消

| 選項 | 內容 | 結果 |
| --- | --- | --- |
| 具名 cancelled 結果交作者 | 每個 yield 點後寫 `if result.cancelled: return`；sticky cancel 使漏寫無害 | 否決：沒有作者用取消結果做事 |
| `@recipe(on_cancel="return"\|"handle")` | 預設 close，需要時宣告 handle | 否決：為不存在的需求加參數 |
| 統一 return（採用） | host close generator，`finally` 仍跑；回覆 `RecipeOutcome(status="cancelled", report=None, runs=[...])`，runs 由便利層在 yield 邊界自行記錄 | 採用 |

Workflow 的 `Cancelled` 跟著既有 Failed guard 走，成本是九個 tuple 項，維持。

### Signal

| 選項 | 結果 |
| --- | --- |
| 甲：移除 `Signal`，driver 改 `cancel()`（workflow）與 `close()`（recipe、teardown） | 採用。作者面不變，第一層非同步機制不動 |
| 乙：維持為預留的第三方向 | 否決：四種 kind 無人用；10-16 留問指出帶資料的邊界訊息形狀是作者宣告的接收點 `yield from env.receive(Model)`，不是 inbox |
| 丙：只留 cancel | 否決：單一 kind 的 enum 不如方法 |

### Workflow 形狀

同一個簡化 coherence（設 flux、校頻率、量 T1、記錄），step 模型最簡 19 行加 State dataclass、init、Next／Done；generator 11 行加一個 Summary。真實 coherence 約 140 行對 70 行。讀法差異大於行數：step 版要先理解「engine 反覆呼叫我，state 是給下一次的信」，generator 版照讀就是流程。

| 項目 | step 模型 | generator（採用） |
| --- | --- | --- |
| Pause | 丟棄當前 step，Resume 重做 | 立即打斷 operation，Resume 重跑該 operation；`env.run(E, lambda: build(plan, tun.current))` 可在重跑時取新 tunables |
| Tunables 生效 | step 開頭快照；改當前步要 Pause、Apply、Resume | `tun.current` 即時讀；record.json 記本 iteration 讀到的 revision |
| Iteration record | `Next(record, state)` | `yield from env.commit(record)` |
| 整點重做 | engine 提供 | 作者以 Ask 迴圈自寫 |
| 與 recipe 一致 | 否 | 是 |

Pause 的另一選項「只在下一個 yield 邊界生效」否決：長 acquisition 要跑完才停。

### 回退到過去任一點

| 選項 | 代價 | 結果 |
| --- | --- | --- |
| Step 模型加 state 序列化 | 每個 workflow 付樣板；現行 engine 也未存每點 state | 否決 |
| Generator 加 journal 重播 | 每種 experiment Result 需精確 typed codec；順帶得到崩潰接續 | 否決，需求出現再評估，generator 不阻擋 |
| 新 run 組合（採用） | 作者多約十行從舊 run 讀 history；engine 零改動；舊 run 紀錄不被改寫 | 採用 |

## 被否決的選項與理由

見上表。另：Workflow 與 recipe 硬性統一取消處理，否決，因 workflow 的 Failed guard 已存在，加 Cancelled 近乎零成本，而 recipe 的失敗是 raise，取消分支才是純多出來的行。

## 留下的問題

- 「agent 主動送資料給腳本」的接收點 `yield from env.receive(Model)` 尚未設計，等第一個真實需求。
- 新 recipe 路徑的 `finish_early` 目前是未綁定的 stub，便利層遷移時補。
- Journal 重播（回退與崩潰接續）若日後需要，前提是 Result 的 typed codec。
- 中斷粒度：討論當時只有 `cancel`／`stop` 結束整個腳本，`finish_early` 僅舊 recipe 路徑有。同日稍後裁定支援 scope：動態 token 樹、子 token 出生繼承父的 set、單一動詞 `cancel(scope)` 以 status cancelled 加 partial result 往上交，`finish_early` 工具移除。設計寫在 ADR-0075「控制命令的傳遞路徑」，實作另開 task。

## 依據

- 本 repo 的 `lib/zcu_tools/experiment/script_driver.py`、`experiment/workflows/execution.py`、`mcp/measure/script_recipe_execution.py`、`zcu_lab/workflows/`、`zcu_lab/recipes/`（repo@eb8b57ac3）。
- 行數與使用紀錄的統計來自 advisor 當場查核；原始查核工作檔未入庫，數字已寫入正文。

## 後記（2026-10-22）

同日稍後再簡化三項：取消的交付一律 `Completed(result, partial_result=True)`，沒有 `Failed("cancelled")`，`Failed` 的唯一判準是「目標結果沒拿到但系統仍可信」；`env.run` 只收建好的 cfg，`env.call(fn)` 是唯一原語；作者面改為 `async def`／`await`，asyncio 只當排程器、取消仍靠 scope token（原型 `prototype/async-protocol` @ 8dc5df757，八個情境通過）。使用者的理由：原生 Python 語法比框架自定的 yield 協定易上手。本文標題與正文中的「generator」在作者面已改為 coroutine，語義不變。

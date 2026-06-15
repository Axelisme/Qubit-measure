# ADR-0025 — 跨線程互動 channel（單一有序事件流取代多-channel combine）

取代 [[0023]]。關聯 [[0019]]（operation handles / facets）、[[0026]]（operation abstraction：runner 消費本 channel 作 await/cancel 核心）、[[0017]]（worker-thread plotting）、[[0002]]。

## 脈絡

GUI↔agent 的跨線程互動有三方參與：**worker thread**（設定 operation 終局）、**GUI 主線程**（使用者注入訊息 / 取消）、**off-main await thread**（agent 的 `*_wait` 阻塞處，`OperationHandles.await_outcome`）。

目前一個 operation 的跨線程信號散在**三個獨立同步物件**：op 完成 `Event`、`FeedbackInbox`（[[0023]]）、`stop_event`。awaiter 用 poll loop「同時看這三個」再自己時序敏感地 combine。**只要信號分散在多個 channel，combine 就必須推敲它們的先後**，於是每加一種互動都要重新分析 race：

- [[0023]] 的 feedback 喚醒；
- floating-widget「Send & Stop」的 **post→cancel ordering race**（reviewer 在 Phase 1 抓到：tick 邊界可能把 stop 訊息當成 standalone nudge 提早洩漏，把單一意圖裂成兩事件）；
- **互動 analyze 無 `stop_event`** 的結構洞（hold 守衛 `stop_event.is_set()` 恆 False）。

根因不是任一補丁能解的——是「多 channel + 時序敏感 combine」這個形狀。另一面是 deadlock：任何「兩線程互等」都要人工分析。需要一個讓**race / deadlock 結構上不會發生**的跨線程互動 primitive。

## 決策

採**單一 per-interaction 有序事件 channel**（`OperationChannel`）：一條 thread-safe FIFO，承載 typed 事件，consumer 依**到達順序**消費。

**事件集（operation await）**：
- `Settled(outcome)` — worker 設定的終局（finished / failed / cancelled）。
- `Message(text)` — GUI 的 nudge（op 續跑）。
- `Stop(reason: str | None)` — GUI 的取消請求；`reason` 即「為什麼停」的訊息。

**produce / consume 契約**：
- **producer（worker / GUI）enqueue 永不阻塞**。
- **consumer（awaiter）只用 bounded `timeout` 阻塞**（`queue.Queue.get(timeout)`，入列即醒——無 poll 延遲）。
- `channel.stop(reason)` 是**一次原子操作** = 觸發該 op 的 **cancel hook** + enqueue `Stop(reason)`。cancel hook 於 `create` 時註冊，封裝 op 間差異（run / device-setup = `stop_event.set`；interactive = `cancel_interactive` 直接 settle；不可取消的 op = `None`）。
- awaiter 依到達序折疊：`Message`（前面無 Stop）→ 回 `{user_feedback, text}`（非終局，可再 await）；`Stop(reason)` → 記下 reason、續等；`Settled(cancelled)`（前面見過 Stop/Message）→ 回 `{cancelled, feedback=<reason/訊息>}`；`Settled(其他)` → `{completed, outcome}`；逾時 → `{timeout}`。

### 兩個使用者意圖 = 兩個原子事件（挑戰 [[0023]] 的拆分）

[[0023]] 把「訊息」與「取消」分成兩個獨立 channel——這正是 race 的製造者。實際上「Send & Stop」=「停，因為 X」是**單一意圖**，應是**單一 `Stop(reason=text)` 事件**，不該拆成 post+cancel 兩步。純 nudge（不停）才是獨立的 `Message(text)`。如此「需要 combine 的跨 channel 先後」根本不存在——每個使用者動作就是**一個有序 enqueue**。

### 不變式（這就是「以後不必再逐次分析」的契約）

1. **一個跨線程互動一條 channel**；該互動的所有信號都走它（無旁路 `Event` / inbox / flag）。
2. 信號 **typed + 原子 enqueue**；consumer **依到達序**處理。
3. **producer 不阻塞；consumer 限時阻塞**。
4. **cancel 經 channel 註冊的 hook 觸發**（op-taxonomy 留在註冊端，不滲進 channel）。

→ **race-free（全序）** + **deadlock-free（單向有界等待）**，by construction。守住這四條，新的跨線程互動**直接套用、不需重新證明安全**。

## 適用性分析（每個現有 operation 是否適用）

| operation | 經 channel await | `Message`(nudge) | `Stop`(cancel) | cancel hook |
|---|---|---|---|---|
| **run** | ✓ | ✓ | ✓ | `stop_event.set`（`run.py:90`） |
| **analyze (FIT)** | ✓ | ✓ | ✗（目前不可取消，`analyze.py:90`） | —（未來給 worker `stop_event` 即可加，不動 channel） |
| **analyze (interactive)** | ✓（pending picker） | ✓ | ✓ | `cancel_interactive` 直接 settle（無 `stop_event`，`staged_analyze.py:87`） |
| **device setup** | ✓ | ✓ | ✓ | `stop_event.set`（`device.py:658`） |
| **device connect / disconnect** | ✓ | ✓ | ✗（無 cancellation point） | — |
| **connect SoC** | ✓ | ✓ | ✗（blocking，無 cancellation point，`connection.py:271`） | — |
| **save** | ✗（同步 RPC、[[0019]] 無 handle、不經 `operation.await`，`save.py:56`） | n/a | n/a | n/a |
| **notify_user（新）** | ✓（agent 主動問、await user） | 事件集為 `Reply(text)` / `Dismiss`（非 Message/Stop） | n/a | — |

要點：
- **`Message`（nudge）對所有 awaited op 一致適用**（agent 收 `user_feedback`、op 續跑）——通用，無 op 差異。
- **`Stop` 只對可取消 op 有實際效果**；floating-widget 的 Stop 鈕**依 active op 是否註冊了 cancel hook 來 enable**（connect / FIT-analyze / device connect-disconnect 不顯示 Stop）。
- **interactive 的 cancel hook 是「直接 settle cancelled」而非 `stop_event`**——channel 的 hook 抽象把這差異吸收掉，awaiter 折疊只看「`Stop` 事件在流裡」，**不看 `stop_event.is_set()`** → 結構洞消失（互動 analyze by construction 正確）。
- **save 不在範圍**：同步、不經 await（[[0019]] 明示 save 無 handle），且無「中途停 save」的需求；未來 save 若改 async/cancellable 再採用即可。
- **notify_user 重用 channel 機制**（有序 + bounded await + 非阻塞 produce），但事件詞彙不同（Reply/Dismiss/timeout），不混進 operation 的事件集。

## 後果

- **取代 [[0023]]**：移除 `FeedbackInbox` 與其從未接線的 dead `notify_event`；2s poll 延遲消失（`Queue.get` 入列即醒）。亦取代 Phase 1 solo 寫的 await-combine 與 B（for_cancel 旗標）補丁——直接以 channel 重做。[[0024]] 對「feedback passthrough wire 仍在」的描述隨之過時，於遷移時一併修。
- **遷移**：`OperationHandles` 的 per-token（completion `Event` + `stop_event` + 共用 `FeedbackInbox`）改為 per-token `OperationChannel`：`settle`→`channel.settle`、`cancel`→`channel.stop(reason=None)`、feedback nudge→`channel.message`、`await_outcome`→消費 channel；`create` 時註冊 cancel hook。`Controller.send_feedback(message, stop)` 內部改為對 active op 的 channel 呼 `message` / `stop(reason=message)`；`cancel_active_operation` 仍負責挑出 active op（op-taxonomy）再呼其 channel.stop。
- **concurrency-critical**：動到所有 op（run/analyze/device/connect）共用的 await 核心 → **必經 sub-agent review**。
- `poll()` / `live_count()` / shutdown 的 liveness 簿記不變（與事件流並存）。

## 拒絕的替代方案

- **多-channel + 時序敏感 combine 的補丁（A 先 cancel 再 post / B for_cancel 旗標 / C 訊息綁 `OperationOutcome`）**：都在會生 race 的形狀上補，無法消除「每次新增互動都要分析先後」的負擔；且對無 `stop_event` 的 interactive 都不結構性成立。
- **真 OS-level interrupt / server push**：協議級、超出範圍（[[0023]] 已論證刪去），且不解 race-by-construction 的目標。

# GUI 事件收合（draft）

**狀態：** 已確認待實作；不代表現行架構。來源為原事件 attribution 決策的 subscriber-side coalescing 目標。

## 現況與問題

`gui/event_bus.py` 同步發布事件並蓋上 seq/origin；`gui/remote/rpc_endpoint.py` 在有訂閱者時才編碼和投遞。`gui/widgets/cfg/form.py` 的 `_on_draft_changed` 仍即時 `schema_changed.emit(draft.snapshot())`；目前的 `QTimer.singleShot(0, ...)` 只用於 section refresh，沒有收合整棵 cfg snapshot。不能把先前文件的「每 tick 一次 State commit」視為已實作。

## 待實作邊界

GUI 訂閱端可以按同一 event-loop tick 的 payload type 和 tab 去重反應。Cfg 表單可將一 tick 內連續欄位變更收合成一次 snapshot 和 State commit，但逐鍵有效性回饋不能延後或合併。Bus 仍同步且無 timer；收合不能改變必要的 domain commit 順序。Remote 已有 subscriber-aware lazy push，不要求 transport 作批次 flush。本 draft 不核准 replay buffer、Web 事件服務或新的高頻 wire channel。

## 轉正條件

GUI coordinator 與 cfg form 接入收合，驗證同 tick 多次變更只產生一次完整 snapshot／reaction、逐鍵有效性回饋保持可用，並確認 commit 順序；再依現行 ADR 的責任分界決定是否增補 0068 或 0067。

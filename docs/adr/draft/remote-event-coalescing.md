# GUI 事件收合（draft）

**狀態：** 已確認待實作；不代表現行架構。來源為原事件 attribution 決策的 subscriber-side coalescing 目標。

## 現況

`gui/event_bus.py` 同步發布事件並蓋上 seq/origin；`gui/remote/rpc_endpoint.py` 在有訂閱者時才編碼和投遞。`CfgFormWidget` 已按 event-loop tick 收合整棵 cfg snapshot，逐次 validity 回饋維持即時（見 [GUI README](../../../lib/zcu_tools/gui/README.md) 的 Shared Qt Cfg Widgets）。GUI 訂閱端目前沒有依 payload type 與 tab 去重的共用 coordinator；各 reaction 仍逐事件執行。

## 待實作邊界

GUI 訂閱端可以按同一 event-loop tick 的 payload type 和 tab 去重反應。Bus 仍同步且無 timer；收合不能改變必要的 domain commit 順序。Remote 已有 subscriber-aware lazy push，不要求 transport 作批次 flush。本 draft 不核准 replay buffer、Web 事件服務或新的高頻 wire channel。

## 轉正條件

GUI coordinator 接入收合，驗證同 tick 多次事件只產生一次 reaction，並確認 commit 順序；再依現行 ADR 的責任分界決定是否增補 0068 或 0067。

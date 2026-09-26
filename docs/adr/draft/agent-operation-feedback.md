# Agent operation feedback（draft）

**狀態：** 已核准的責任邊界，待 Operation ADR 核實並轉正；不代表本文件是現行 ADR。

## 現況與依據

[退役的 ADR-0024](../retired/0024-embedded-agent-session-architecture.md) 的「保留」段指出，移除 agent launch UI 不等於移除 runtime feedback。現行 `gui/app/measure/ui/feedback_widget.py`、`feedback_dock.py` 和 `notify_dialog.py` 仍有 feedback／prompt UI；`mcp/measure/tools_notify.py` 仍提供 `gui_prompt_user`。互動 channel 的既有設計見 [[0025]]。上述檔案證明入口存在，不能單靠它們保證每種 operation 的完整取消或連線斷開政策。

## 待轉正的邊界

Operation 期間的 message、prompt、stop 互動歸 Operation domain；UI 的 FeedbackPanel mount、按鈕與 dialog 細節歸 app owner。移除 launch UI 不移除已連線 client 的 runtime interaction。Process startup 不負責 domain operation 的完成或資源清理。

## 轉正條件

T18 對照 Operation 執行、取消與關閉流程，於 Operation ADR 承接這項邊界；app README 記錄已核實的具體 UI 行為。核實之後移除本 draft，不從舊 launch UI 推導新的 shutdown 規則。

# ADR-0024 — 外部終端 agent launch 架構

> This ADR supersedes the original embedded stream-json agent session design
> (AgentRunner / AgentChatService / AgentSupervisor / IndependentAgentSession /
> AgentSessionRegistry / AgentSessionPort).  Those components have been removed.
> The current design is described below in the present tense.

## 脈絡

measure-gui 提供一個「Agent」按鈕讓使用者委派 Claude agent 經 `gui_*` MCP 工具操作 GUI。評估過兩條路：

1. **內嵌**：GUI spawn headless subprocess → stream-json 解析 → 自建 transcript widget / detached supervisor / 檔案 IPC / session registry。
2. **外部終端**：spawn 系統終端跑真互動式 `claude`，GUI 只負責組 argv 與 loopback mcp.json。

評估結論選外部終端：評估過 pyte（PTY 模擬）與 xterm.js + QtWebEngine 內嵌，但「為一個開終端的按鈕拖進整套 widget / PTY / Chromium engine 不划算」；直接啟動已裝好的外部終端達同樣目標、體驗更原生、近零依賴、維護複雜度大幅降低。

## 決策

### GUI 入口

`AgentLaunchDialog`（`ui/agent_launch_dialog.py`）呈現兩個按鈕：
- **New session**：建立新 uuid session id，spawn 終端。
- **Resume last**：讀取 `~/.cache/zcu-tools/agent_last_session` 的上次 session id，帶 `--resume` spawn 終端。

`Controller.launch_agent_session(resume)` 先組 `build_agent_state_context()`（當前 GUI 狀態快照），再委 `services/agent_launcher.py`。

### 狀態注入

`Controller.build_agent_state_context()` 組 `[measure-gui current state] project / context / SoC / open tabs [end state]` 字串，烤進 `--append-system-prompt`。agent 啟動即知 GUI 狀態，免呼 `gui_state_check` 定位。

### `services/agent_launcher.py`（Qt-free）

- `build_loopback_mcp_config(repo_root)` → 暫存 mcp.json（entry = `uv run zcu_tools/mcp/measure/server.py`，cwd = repo_root）。
- `_EMBEDDED_SYSTEM_PROMPT`：告知 agent「gui_* 工具已自動 attach、勿自呼 connect、需要狀態自呼 gui_state_check」。
- `build_claude_argv(session_id, *, resume, mcp_config_path, system_prompt)` → 互動模式不帶 `-p`；帶 `--mcp-config` / `--allowedTools "mcp__measure-gui__*"` / `--append-system-prompt` / `--resume <id>`（Resume）或 `--session-id <uuid>`（New）。
- `new_session_id()` → uuid4 string。
- `read_last_session_id()` / `write_last_session_id(id)`：存取 `~/.cache/zcu-tools/agent_last_session`。
- `launch_agent_terminal(repo_root, *, resume, state_context)` → 寫暫存啟動 script（避 shell quoting 問題）→ 跨平台 spawn：
  - Linux：依序嘗試 `gnome-terminal` / `konsole` / `xterm` / `x-terminal-emulator`；環境變數 `ZCU_AGENT_TERMINAL` 可覆寫。
  - macOS：`open -a Terminal`。
  - Windows：`wt` → `start cmd`。
  - `AGENT_CMD` 預設 `claude`；環境變數 `ZCU_AGENT_CMD` 可換（如 `codex`）。
  - 從子程序環境移除 `ANTHROPIC_API_KEY`（訂閱認證下無需）。

### 自動連線

靠 `mcp/measure/server.py` 既有的 **lazy auto-connect** policy：首次 `gui_*` 呼叫時自動經 session-discovery 解析 control-socket port 並 attach。agent 無需顯式呼 `gui_connect`。

### Session 持久

靠 `claude` 原生 `--resume <session_id>`：終端關閉後對話歷史保留；下次 Resume 直接接回。GUI 只存一個 last session id 檔（`~/.cache/zcu-tools/agent_last_session`）。

### 中斷

終端原生 Ctrl-C / Esc。GUI 不做插話 / nudge。

cooperative-interrupt 的 wait early-return wire（`OperationHandles.await_outcome` 第二喚醒源 + `FeedbackInbox` drain，見 ADR-0023）仍在，但 GUI 端無 feedback bar / nudge 入口；該 wire 只由 `mcp/measure/server.py` 的 feedback passthrough 服務（agent 若透過 mcp 傳 feedback 仍能喚醒 pending wait）。

## 與既有設計關係

- **ADR-0023**（cooperative-interrupt：feedback 喚醒 pending wait）的 mechanism 仍生效，GUI 側入口已移除，只剩 mcp feedback passthrough 路徑。
- 依賴 **session-discovery**（user-launched GUI 開 control socket + `~/.cache/zcu-tools/sessions/measure.json`）做自動連線。
- 遵 ADR-0004/0005：`agent_launcher.py` 是 Qt-free driven adapter；`AgentLaunchDialog` 是 Driving Adapter/View。

## 後果

- 近零依賴：無 stream-json 解析、無 PTY 模擬、無 supervisor IPC、無 session registry 複雜度。
- 使用者體驗原生：標準互動式 terminal，history、搜尋、複製貼上均可用。
- GUI 複雜度大幅降低：services 層只剩 `agent_launcher.py`（Qt-free 純函式）、ui 層只剩 `agent_launch_dialog.py`（兩按鈕）。
- 代價：transcript 不在 GUI 內顯示；無 GUI 側即時插話（用終端 Ctrl-C 取代）。

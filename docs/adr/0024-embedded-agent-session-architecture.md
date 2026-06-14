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
- **可選 resumable session 清單**：`list_resumable_sessions` 列出我們 launch 過的 session（依 claude `~/.claude/projects/<slug>/*.jsonl` 補 last-active + 首則訊息 label、最近在上），選一個 → **Resume selected** 帶 `--resume <id>`，或 **Remove**（`remove_recorded_session` 把該 session 從 Resume 清單移除，不動 claude 的 `<id>.jsonl` transcript）。
- **New session**：建立新 uuid session id，spawn 終端。

dialog 直接委 `services/agent_launcher.py` 的 `launch_agent_terminal()`；launch 前以 `Controller.build_agent_state_context()` 組當前 GUI 狀態快照傳入。

### 狀態注入

`Controller.build_agent_state_context()` 組 `[measure-gui current state] project / context / SoC / open tabs [end state]` 字串，烤進 `--append-system-prompt`。agent 啟動即知 GUI 狀態，免呼 `gui_state_check` 定位。

### `services/agent_launcher.py`（Qt-free）

- `build_loopback_mcp_config(repo_root)` → 暫存 mcp.json（entry = `uv run zcu_tools/mcp/measure/server.py`，cwd = repo_root）。
- `_EMBEDDED_SYSTEM_PROMPT`：告知 agent「gui_* 工具已自動 attach、勿自呼 connect、需要狀態自呼 gui_state_check」。
- `build_claude_argv(session_id, *, resume, mcp_config_path, system_prompt)` → 互動模式不帶 `-p`；帶 `--mcp-config` / `--allowedTools "mcp__measure-gui__*"` / `--append-system-prompt` / `--resume <id>`（Resume）或 `--session-id <uuid>`（New）。
- `new_session_id()` → uuid4 string。
- `record_launched_session(id)` / `list_resumable_sessions(repo_root)` / `claude_project_dir(repo_root)`：追蹤我們 launch 過的 session 於 `~/.cache/zcu-tools/agent_sessions.json`；列表時對每個 id 查 claude 的 `<slug>/<id>.jsonl`（slug = abspath 非 `[A-Za-z0-9-]` 換 `-`），補 last-active mtime + 首則 user 訊息 label，依 last-active 排序。
- `launch_agent_terminal(repo_root, *, resume, state_context)` → 寫暫存啟動 script（避 shell quoting 問題）→ 跨平台 spawn：
  - Linux：依序嘗試 `gnome-terminal` / `konsole` / `xterm` / `x-terminal-emulator`；環境變數 `ZCU_AGENT_TERMINAL` 可覆寫。
  - macOS：`open -a Terminal`。
  - Windows：**直接 `subprocess.Popen([claude_bin, *argv[1:]], creationflags=CREATE_NEW_CONSOLE)` spawn claude 本身**（在此就地 `shutil.which` 解析 + fast-fail），**不經** Python launcher 的 `os.execv`。原因：Windows 的 `os.execv` 會把 argv 陣列重新序列化成單一命令列字串，**把多行 `--append-system-prompt` 拆裂** —— 內嵌 prompt 的一個字（「…tools *are* already…」）變成獨立 positional 參數，被 claude 當成首則 typed 輸入（phantom 首訊息「are」）。`subprocess` 對 argv list 正確引號，無此問題（把 prompt 裡的 "are" 換成標記字，phantom 跟著變，確證）。**也不用** Store 版 `wt`（UWP 對子進程虛擬化 `AppData\Roaming`）或 `cmd /c start`（二次引號破壞路徑）。`ZCU_AGENT_TERMINAL` 可覆寫；覆寫與 POSIX 才走 launcher，launcher 在 win32 改用 `subprocess.run` 同避 execv 拆裂（POSIX `os.execv` 收 argv 陣列、無重新引號、不受影響）。
  - **agent CLI（argv[0]）由 `resolve_agent_command()` 解析**，依序：
    1. `ZCU_AGENT_CMD` 環境變數（任何平台的顯式覆寫，如換 `codex`）。
    2. **PATH 上的獨立 `claude` 優先**（如 `claude install` 裝的；`shutil.which("claude")` 找得到就用）——真正的 CLI、持續更新、跨平台一致，與「全程走 CLI」的用法一致。
    3. **Windows 後備：Claude Desktop 內建 CLI**——`_find_desktop_bundled_claude()` 取最新版 `%APPDATA%\Claude\claude-code\<version>\claude.exe`（挑數字最大版，非字典序）。給「只裝 Desktop、`claude` 不在 PATH」的環境用。
    4. 裸 `claude`——launcher 經 PATH 解析；找不到即 fast-fail。
    Windows 實際解析序＝PATH `claude` → Desktop 內建 → fast-fail；其他平台＝`ZCU_AGENT_CMD` → PATH `claude` → fast-fail。
  - 剝除父進程的 Claude Code 編排環境變數（`ANTHROPIC_API_KEY` 為訂閱認證；`CLAUDE_CODE_*` / `CLAUDECODE` / `CLAUDE_AGENT_SDK*` 讓子 claude 以乾淨 `cli` 啟動，而非繼承 Desktop-embedded entrypoint）。

### 自動連線

靠 `mcp/measure/server.py` 既有的 **lazy auto-connect** policy：首次 `gui_*` 呼叫時自動經 session-discovery 解析 control-socket port 並 attach。agent 無需顯式呼 `gui_connect`。

### Session 持久

靠 `claude` 原生 `--resume <session_id>`：終端關閉後對話歷史保留（claude 自存於 `~/.claude/projects/<slug>/<id>.jsonl`）；下次從清單選該 session 直接接回。GUI 端只追蹤「我們 launch 過的 session id 清單」（`~/.cache/zcu-tools/agent_sessions.json`），列表時與 claude 的 jsonl 交叉補 metadata（避免列出同 repo cwd 下大量無關的 dev session）。

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

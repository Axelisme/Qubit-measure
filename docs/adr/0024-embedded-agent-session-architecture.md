# ADR-0024 — 嵌入式 agent：session port + 可插拔後端 + 獨立 session 生命週期

## 脈絡

measure-gui 內嵌一個「助理」：使用者委派一個 Claude agent 經 `gui_*` MCP 工具**操作這個 GUI**，並能即時看對話 / 插話糾正。已落地的基礎（B0）＝GUI spawn 一個 child `claude`（訂閱認證、免 API key、stream-json）操作自己（loopback MCP）。需求衍生兩件事：(1) 前後端拆分，讓未來能換成 **API 模式**後端（Agent SDK + API key）而前端不動；(2) **獨立 session 層**——多個 GUI 能 attach 同一個長活 session、支援無人值守長跑，且 session 壽命由「是否有 GUI 連著」閘控。

兩條連線要分清：**(α)** agent→GUI（用 `gui_*` MCP 操作一個 GUI 當「手腳」，靠 session-discovery 找/重連）；**(β)** viewer→agent（dialog 看 transcript / 送輸入）。

## 決策

### 前後端 seam（B1a，已生效）
- **`AgentSessionPort`**（Qt-free 控制面：`start/send_user_message/stop/state/is_running/session_id/add_state_listener`）是前端與後端的唯一契約。
- 前端 `AgentChatDialog` 只依賴 port；**transcript 走共用的 `AgentChatService`**（後端餵 `record_*`、前端 listen），不在 port 上。
- `Controller.get_agent_session() -> AgentSessionPort` 工廠建後端、注入接線。

### 可插拔後端（port 後面）
同一 port 多種實作，工廠依設定選用：
- **CLI/訂閱後端**（`AgentRunner`，已生效）：QProcess spawn `claude --output-format/--input-format stream-json`，`--allowedTools mcp__measure-gui__*`，stream-json 逐行完整 JSON 解析；首輪與後續皆經 stdin 餵（`-p` 被 input-format 忽略）；Stop=SIGINT。
- **API 模式後端**（未來）：Agent SDK + API key（可用原生 `interrupt()`），實作同 port、餵同一 `AgentChatService`，**前端零改**。
- **獨立-session 後端**（B1b，規劃中）：見下。

### 獨立 session 後端與生命週期（B1b，規劃中）
- **跨平台（含 Windows）為硬需求**，故 IPC/鎖全走可攜檔案機制，**不用 fifo / `fcntl.flock`（Unix-only）**：
  - agent 程序 **detached**（非任何 GUI 的 child；`start_new_session`(POSIX) / `DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP`(Windows)）。
  - stream-json **輸出**寫 **session log 檔**（viewer **poll-tail** + byte offset 增量，取歷史+跟隨）。
  - **輸入**走 **command spool 目錄**（每筆 user 訊息一個小檔，supervisor 消費後刪）——多 viewer 併發安全、可攜。
  - **registry**（`~/.cache`，契合 taskboard/discovery 風格、免常駐 daemon）記每個 session 的 `{session_id, claude_session_id, pid, status, log, spool, target_gui}`；寫用 atomic `os.replace`；**lease 用檔案 mtime**（非 flock）。
  - stop = SIGINT(POSIX) / CTRL_BREAK 或 terminate(Windows)。
- dialog 升成 attach 客戶端：開啟**不自動啟動**，給 **New / Resume(stopped,`--resume`) / Attach(running, tail log)**；已在跑就 attach 不重開；多個 viewer 可連同一 session。
- **生命週期閘控**：session 壽命由「**≥1 個 GUI 連著**」決定（lease mtime heartbeat）；全部斷開後給 grace 寬限再**自動關閉**（避免 GUI 重啟瞬間誤關）。非永生 daemon。
- **連線**：B1b 先**一律由 dialog 手動 attach**；「agent 自啟的 GUI 自動連、用戶啟手動 attach」的差異化列為後續增量。
- **上下文持久**靠 `--resume <claude_session_id>`：process 綁定但對話跨 GUI 重啟接回；agent 經 session-discovery 重連重啟後的新 GUI 實例。

## 與既有設計關係

- **ADR-0023**（cooperative-interrupt：feedback 喚醒 pending wait）是 port 後端的一個能力——使用者輸入在 agent 卡 `gui_*_wait` 時走 feedback inbox 喚醒；訂閱認證下取代 SDK 的 `interrupt()`。
- 依賴 **session-discovery**（user-launched GUI 開 control socket + `~/.cache/zcu-tools/sessions/measure.json`）做 α 連線與跨 GUI 重連。
- 遵 ADR-0004/0005：port 是 app service 契約、`AgentRunner` 是 Driven Adapter、dialog 是 Driving Adapter/View。

## 後果

- 前端與認證模式/後端實作解耦：CLI（訂閱）↔ API（key）↔ 獨立-session 可換而前端不動。
- B1b 帶來檔案中介協定 + registry + attach UI 的複雜度，但避開常駐 daemon、契合既有檔案協調風格。
- 「操作 GUI」本質使 agent 壽命合理地由 GUI 連線閘控（agent 無 GUI 即無手腳），故不做永生 daemon。

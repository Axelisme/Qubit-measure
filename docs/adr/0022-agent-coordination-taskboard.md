# ADR-0022 — 多 agent 協作：taskboard 為主協調層、worktree 為輔

## 脈絡

多個 agent / session 常同時在**同一個 checkout** 上工作，且共用同一組 live 資源（measure-gui 子程序、ZCU216 板、device、固定 MCP port）。風險有二：(1) 兩 agent 同改重疊檔案 → 未提交改動在共享 working tree 互相污染；(2) 搶用 singleton 資源（板/GUI/port）。

## 決策

採 **taskboard 為主協調層**：一個 stdio MCP server（`lib/zcu_tools/mcp/taskboard/`，契合既有「一台 server 一子套件」慣例），後端為 `flock` 守護的 JSON store。提供：

- **path 衝突偵測**：claim 一組 path（檔案 / 目錄前綴 / glob），正規化 repo-relative 後以「同檔 / 祖先目錄 / glob 交集」判重疊。
- **read/write 鎖**：重疊 path 上 read+read 不衝突，涉及 write 才衝突。
- **session 身分（衝突鍵）**：衝突鍵不是呼叫端傳的 `owner`，而是 server 從 agent host session/thread identity 讀到的值（例如 Claude Code 的 `CLAUDE_CODE_SESSION_ID`、Codex 的 `CODEX_THREAD_ID`，對一個 top-level session 及其所有 sub-agent 注入同一值、跨 top-level session 不同）。MCP 呼叫不帶 per-request session context，所以 server 先讀自身 env；若 host 沒把變數轉發給 MCP 子程序，Linux 上再只讀 allowlisted session-id names 的祖先 `/proc/*/environ`；若仍沒有來源，最後使用 process-local identity，因 stdio MCP server 以 client session 為作用域，同一 server process 不應自我阻塞。`owner` 降為純人類標籤（只進 claim 紀錄與 markdown 視圖）。規則：兩 claim 只有在 **path 重疊 + 涉及 write + identity 不同** 時才衝突——**同 identity 的重疊 claim 永不互卡**（orchestrator 與它開的 sub-agent 同 session，彼此 claim 不阻塞），且同 identity 重疊會以 `warnings` 回傳提醒而不是進入 pending。
- **re-claim idempotent**：當一筆 claim 的 (paths, mode) 已被「同 identity 的某個 granted claim」完整覆蓋（held write 涵蓋 read/write 請求、held read 只涵蓋 read 請求；path 以子集判定）時，直接回傳該既有 claim（同 `claim_id`、仍 granted）、不新增；同 identity 但 scope 是新的（未覆蓋）則照常 grant 一筆新 claim。

> **Rationale**：taskboard 協調的是**跨 session** 的未提交污染與資源爭用；一個 session 內部（orchestrator 自己 sequence 它派的 sub-agent）的執行順序由 orchestrator 自負，不需要也不該被 taskboard 自我阻塞。把衝突鍵綁到 session identity 後，「要特別告知 sub-agent 別 claim」的死鎖被結構性消除，re-claim 也成為冪等的 no-op。
- **資源 token**：path 也接受 `@hw/zcu216`、`@gui/measure`、`@port/8767` 等邏輯 token，用同一套鎖協調非檔案的 singleton 資源。
- **pending + wait**：衝突時 claim 轉 pending（記 blockers + 佇列位置）；release 自動晉升下一個。等待以「pending + agent `ScheduleWakeup` 回來 poll」為主（回合制 agent 不凍結回合），短逾時阻塞 `wait` 為輔。
- **TTL 自動回收**：每筆 claim 有 `touched` 時戳 + TTL；逾時標 stale 並回收（`touch` 心跳延長、`force_release` 手動）。
- **markdown 視圖**：JSON 為 source of truth，自動渲染唯讀 `task_plans/taskboard.md` 給人看。

協議文字由 `agent-taskboard` skill 承載（何時 claim、衝突等待流程、**commit 後才 release**、資源 token 用法）。

## 為何不用 git worktree 取代

git worktree 是**樂觀隔離 + merge**：各 agent 在自己的工作樹/branch 自由改，衝突延到 merge 由 git 3-way 解。它**結構上消除未提交污染**並給 per-branch 乾淨度，但：

- **解不了 singleton 資源爭用**：worktree 只隔離檔案；同一塊 FPGA 板、同一個 GUI、同一個 port 無法靠開多個 worktree 平行化——仍需獨立鎖。taskboard 的資源 token 正補此洞。
- **無即時可見性**：撞車要到 merge 才知道；taskboard 動工前就擋下並排隊。
- **成本**：每樹一份 checkout（磁碟 + setup）、`.venv`/工具路徑需重指。

結論：**taskboard 為主**（共享 checkout 即時協調 + 資源 token），**worktree 為輔**——僅當要爆發式平行「純檔案、彼此可獨立 merge」的大量工作（大型遷移、多檔同時實作）時，改用 `Agent isolation:"worktree"` 各跑 branch 再 merge；此時 taskboard 反而多餘。判準：**會搶 live 資源或要即時協調 → taskboard；純檔案、可延後 merge、量大 → worktree。** 兩者皆無法捕捉語意衝突（改不同檔卻破壞彼此假設）。

## 後果

- 協調從手編 markdown 升級為原子、可程式化偵測；人改看 markdown 視圖、agent 用 MCP 工具。
- 新增一個常駐於各 session 的 `taskboard` MCP 連線；state 落地 JSON（可檢視、survive 重啟）。
- `CLAUDE.md`「平行 agent 協調」段移交 `agent-taskboard` skill，CLAUDE.md 留指標。

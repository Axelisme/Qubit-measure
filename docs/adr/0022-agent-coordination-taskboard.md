# ADR-0022 — 多 agent 協作：taskboard 為主協調層、worktree 為輔

## 脈絡

多個 Claude agent / session 常同時在**同一個 checkout** 上工作，且共用同一組 live 資源（measure-gui 子程序、ZCU216 板、device、固定 MCP port）。風險有二：(1) 兩 agent 同改重疊檔案 → 未提交改動在共享 working tree 互相污染；(2) 搶用 singleton 資源（板/GUI/port）。

## 決策

採 **taskboard 為主協調層**：一個 stdio MCP server（`lib/zcu_tools/mcp/taskboard/`，契合既有「一台 server 一子套件」慣例），後端為 `flock` 守護的 JSON store。提供：

- **path 衝突偵測**：claim 一組 path（檔案 / 目錄前綴 / glob），正規化 repo-relative 後以「同檔 / 祖先目錄 / glob 交集」判重疊。
- **read/write 鎖**：重疊 path 上 read+read 不衝突，涉及 write 才衝突。
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

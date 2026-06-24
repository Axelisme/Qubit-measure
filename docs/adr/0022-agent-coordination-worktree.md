# ADR-0022 — 多 agent 協作：orchestrator-owned worktree protocol

## 脈絡

多個 agent / session 可能同時處理同一個 repo。原本的 taskboard MCP 嘗試用 path lock、pending queue 與資源 token 解決共享 checkout 的未提交污染；實際使用時，root agent 與 sub-agent 的 runtime/session identity 不穩定，導致同工作組互相阻塞。此 repo 的主要需求也不是細粒度多 agent 互斥，而是讓 Claude Code、Codex、opencode 等工具能用同一套工作規約同步：隔離未提交 diff、保留長線計劃、集中 reports，最後由 orchestrator 整合。

## 決策

採 **orchestrator-owned Git worktree protocol**，不使用 taskboard MCP 或 `agent-taskboard` skill。

- **一般單 agent 工作**：直接在目前 checkout 修改、測試、回報，不需要額外協調。
- **多 agent / 長線 orchestration**：一個 task item 對應一個 Git worktree，位置固定為 `.agent_state/worktrees/trees/<task-id>/`；branch 慣例為 `agent/<task-id>`。
- **orchestrator ownership**：worktree 與 branch 由 orchestrator 建立、指派、驗證、合併與移除；sub-agent 不自行創建新工作樹。
- **state source of truth**：`.agent_state/worktrees/state.json`（gitignored）記錄 task id、status、area、branch、worktree path、base branch/commit、reports dir、agents、commits、created/updated timestamps。
- **reports**：sub-agent 長報告寫到主 checkout 的 `.agent_state/worktrees/reports/<task-id>/<agent-id>.md`，不要寫進 task worktree；untracked 檔不會跨 worktree 同步。
- **plans**：舊 `task_plans/<area>/` 三件套遷移到 `.agent_state/plans/<area>/`，同樣 gitignored，不進 commit。跨模組、需要長期追蹤的設計決策仍寫入 `docs/adr/`。
- **ignored inputs**：worktree 只自動包含 Git-tracked content。若 task 需要 `.agent_state/plans/<area>/`、本地設定、scratch fixtures、未追蹤資料檔等 gitignored inputs，orchestrator 建立 worktree 時必須明確複製到 task worktree，或把主 checkout 絕對路徑交給 sub-agent 只讀使用，並在 state/report 中記錄。
- **多 sub-agent 同 task**：可以共用同一個 task worktree，但 orchestrator 必須明確序列化或分配不重疊 write scope。
- **Phase/task closure**：每個 task item 或 Phase 告一段落時，orchestrator 必須做整合決策（merge / abandon / blocked）並移除 worktree；task worktree 不作為長期常駐 checkout 使用。
- **runtime 假設**：不要假設 Codex、Claude Code、opencode 或其他 agent runtime 內建 sub-agent isolation 會自動使用 worktree；需要隔離時，由 orchestrator 顯式建立並把 workdir 傳給 agent。

`.agent_state/` 是專案專用 gitignored 工作區：

```text
.agent_state/
  plans/<area>/{task_plan.md,findings.md,progress.md,archive.md}
  worktrees/state.json
  worktrees/reports/<task-id>/<agent-id>.md
  worktrees/trees/<task-id>/
```

不使用 `.agents/` 存放這些專案工作狀態，因為 `.agents/` 也可能是 opencode、Antigravity 或其他工具的 config/search target；把 project-local agent state 放在獨立 `.agent_state/` 可降低工具誤解設定檔的風險。

## State JSON Contract

```json
{
  "version": 1,
  "tasks": {
    "<task-id>": {
      "status": "active",
      "area": "<area>",
      "branch": "agent/<task-id>",
      "worktree_path": ".agent_state/worktrees/trees/<task-id>",
      "base_branch": "main",
      "base_commit": "<sha>",
      "reports_dir": ".agent_state/worktrees/reports/<task-id>",
      "ignored_inputs": [],
      "agents": [],
      "commits": [],
      "created_at": "YYYY-MM-DDTHH:MM:SSZ",
      "updated_at": "YYYY-MM-DDTHH:MM:SSZ"
    }
  }
}
```

`status` 可用：`planned`、`active`、`reviewing`、`merged`、`blocked`、`abandoned`。

## Live Resource Coordination

Git worktree 只隔離檔案，不隔離 ZCU 板、GUI subprocess、固定 MCP port 或其他 singleton resource。本 repo 不再提供通用 lock service；需要 live resource 時由 orchestrator 明確序列化。MEASUREMENT 角色仍遵循量測 skill 與 `agent-memory`，不使用 `.agent_state/` 當量測知識庫。

## 演化

本 ADR 取代先前「taskboard 為主協調層、worktree 為輔」的決策。Taskboard 的 path lock 與 resource token 對此 repo 的實際需求過重，且 identity forwarding 在不同 agent runtime 下不可靠；現在的有效規約是 worktree 隔離 + orchestrator 明確整合。

## 後果

- 移除 `lib/zcu_tools/mcp/taskboard/`、`tests/mcp/taskboard/`、`agent-taskboard` skill 與各 client config 的 `taskboard` MCP entry。
- 未提交改動隔離交給 Git worktree 與 branch，不再靠 lock server。
- 人類與 agent 共同閱讀的長線計劃位於 `.agent_state/plans/`；因為 gitignored，不進 diff/commit。
- Sub-agent reports 固定寫入主 checkout 的 `.agent_state/worktrees/reports/`，避免 worktree 間 untracked 檔不可見。
- Task-specific gitignored inputs 要由 orchestrator 明確 copy/reference；sub-agent 不猜測未追蹤檔在 task worktree 中存在。
- Merge 仍由 orchestrator 控制；若用戶未授權 commit/merge，工作停在 task worktree diff 與 reports。
- Phase/task 收尾時 worktree 會被移除，避免長期殘留 checkout 形成新的同步負擔。

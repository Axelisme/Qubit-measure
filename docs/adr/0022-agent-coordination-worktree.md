# ADR-0022 — 多 agent 協作：orchestrator-owned worktree protocol

## 脈絡

多個 agent / session 可能同時處理同一個 repo。原本的 taskboard MCP 嘗試用 path lock、pending queue 與資源 token 解決共享 checkout 的未提交污染；實際使用時，root agent 與 sub-agent 的 runtime/session identity 不穩定，導致同工作組互相阻塞。此 repo 的主要需求也不是細粒度多 agent 互斥，而是讓 Claude Code、Codex、opencode 等工具能用同一套工作規約同步：隔離未提交 diff、保留長線計劃、集中 reports，最後由 orchestrator 整合。

## 決策

採 **orchestrator-owned Git worktree protocol**，不使用 taskboard MCP 或 `agent-taskboard` skill。

- **一般單 agent 工作**：直接在目前 checkout 修改、測試、回報，不需要額外協調。
- **多 agent / 長線 orchestration**：`task-id` 錨定計劃、reports namespace 與 parent integration branch；同一 task 可依需要拆成多個 `lane-id`，每個 lane 對應一個 Git worktree。單 lane 預設 `.agent_state/worktrees/trees/<task-id>/`，多 lane 使用 `.agent_state/worktrees/trees/<task-id>--<lane-id>/`；branch 慣例同樣使用 `agent/<worktree-id>`。
- **orchestrator ownership**：worktree 與 branch 由 orchestrator 建立、指派、驗證、合併與移除；sub-agent 不自行創建新工作樹。
- **state source of truth**：`.agent_state/worktrees/state.json`（gitignored）記錄多個 task 與各 task 的 lane/worktree checkpoint；它不保存 repo-wide `active_task` 或 `current_task`，每次操作都必須明確指定 `task-id`。
- **merge queue**：`.agent_state/worktrees/merge_queue.json`（gitignored）序列化主 checkout 的 merge preview / final fast-forward；每次只能有一個 task 進入主 checkout merge critical section。
- **reports**：sub-agent 長報告寫到主 checkout 的 `.agent_state/worktrees/reports/<task-id>/<lane-id>/<agent-id>.md`，不要寫進 task worktree；untracked 檔不會跨 worktree 同步。
- **plans**：舊 `task_plans/<task-id>/` 三件套遷移到 `.agent_state/plans/<task-id>/`，同樣 gitignored，不進 commit。跨模組、需要長期追蹤的設計決策仍寫入 `docs/adr/`。
- **ignored inputs**：worktree 只自動包含 Git-tracked content。若 task 需要 `.agent_state/plans/<task-id>/`、本地設定、scratch fixtures、未追蹤資料檔等 gitignored inputs，orchestrator 建立 worktree 時必須明確複製到 task worktree，或把主 checkout 絕對路徑交給 sub-agent 只讀使用，並在 state/report 中記錄。
- **多 sub-agent 同 task**：可以共用同一個 lane worktree，但 orchestrator 必須明確序列化或分配不重疊 write scope；跨 lane 也必須避免重疊 write scope。
- **Phase/task closure**：每個 task item 或 Phase 告一段落時，orchestrator 必須做整合決策（merge / abandon / blocked）並移除 worktree；task worktree 不作為長期常駐 checkout 使用。
- **runtime 假設**：不要假設 Codex、Claude Code、opencode 或其他 agent runtime 內建 sub-agent isolation 會自動使用 worktree；需要隔離時，由 orchestrator 顯式建立並把 workdir 傳給 agent。

`.agent_state/` 是專案專用 gitignored 工作區：

```text
.agent_state/
  plans/
    archives/<task-id>/
    <task-id>/{task_plan.md,findings.md,progress.md,archive.md}
  worktrees/
    state.json
    merge_queue.json
    reports/<task-id>/<lane-id>/<agent-id>.md
    trees/<worktree-id>/
```

不使用 `.agents/` 存放這些專案工作狀態，因為 `.agents/` 也可能是 opencode、Antigravity 或其他工具的 config/search target；把 project-local agent state 放在獨立 `.agent_state/` 可降低工具誤解設定檔的風險。

## State JSON Contract

```json
{
  "version": 2,
  "tasks": {
    "<task-id>": {
      "status": "active",
      "base_branch": "main",
      "base_commit": "<sha>",
      "integration_branch": "agent/<task-id>",
      "worktrees": {
        "<lane-id>": {
          "status": "active",
          "role": "lane",
          "branch": "agent/<worktree-id>",
          "worktree_path": ".agent_state/worktrees/trees/<worktree-id>",
          "reports_dir": ".agent_state/worktrees/reports/<task-id>/<lane-id>",
          "write_scope": ["lib/..."],
          "ignored_inputs": []
        }
      }
    }
  }
}
```

`status` 可用：`active`、`reviewing`、`merge_preview`、`blocked`。task-level `status` 是整體狀態；lane-level `status` 是該 worktree 的狀態。merged / abandoned 的 task 在更新 progress 並清理 worktree / branch 後直接從 `state.json` 移除。

`state.json` 不包含 `active_task`。多個 task 可以同時 active；orchestrator 每次讀寫 plan、state、report、branch 或 queue entry 都以明確 `task-id` 定位。

## Merge Queue Contract

```json
{
  "version": 1,
  "queue": [
    {
      "task_id": "<task-id>",
      "branch": "agent/<task-id>",
      "base_branch": "main",
      "action": "preview",
      "status": "queued",
      "requested_by": "<agent-or-thread>",
      "enqueued_at": "YYYY-MM-DDTHH:MM:SSZ",
      "started_at": null,
      "note": ""
    }
  ]
}
```

`action` 可用 `preview` 或 `final`；`status` 可用 `queued`、`merging`、`blocked`。任何 task 在主 checkout 執行 `git merge --no-commit --no-ff agent/<task-id>` 或 `git merge --ff-only agent/<task-id>` 前都先登記 queue。只有 queue 第一個 entry 可把狀態改成 `merging` 並進入 merge critical section；preview 開著時該 entry 仍持有 queue。完成、abort、blocked 或 abandoned 後才釋放 entry。queue 第一個 entry 若 blocked，後面的 task 不可跳過；需先解決或 abort 該 task 的主 checkout merge 狀態。

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
- Merge 仍由 orchestrator 控制；主 checkout preview / final fast-forward 由 `merge_queue.json` 序列化。若用戶未授權 commit/merge，工作停在 task worktree diff 與 reports，不佔用 merge queue。
- Phase/task 收尾時 worktree 會被移除，避免長期殘留 checkout 形成新的同步負擔。

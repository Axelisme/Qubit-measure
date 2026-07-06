# State / Merge Queue JSON 形狀

這兩份 JSON 是 `.agent_state/worktrees/state.json` 與 `.agent_state/worktrees/merge_queue.json` 的實際形狀，僅供理解欄位含義；agent 不直接編輯這兩個檔案，寫入一律透過 `scripts/workflow.py` 的對應子命令（`state` / `task` / `lane` / `queue` / `preview` / `final` / `merge` / `worktree`）。行為規則（status/role 可用值、write_scope 更新規則、queue 使用規則）記在 `SKILL.md` 本文，這裡只列形狀。

## state.json

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

## merge_queue.json

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
      "note": "",
      "token": null,
      "target_commit": "<sha>",
      "base_head": "<sha>",
      "main_worktree": "/abs/path/to/main-checkout"
    }
  ]
}
```

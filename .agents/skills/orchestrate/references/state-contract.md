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
      "blocked_kind": null,
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

`merge_queue.json` 維持 version 1。`blocked_kind` 是向後相容的 optional / nullable failure
provenance：`queued` / `merging` entry 只能缺省或為 `null`；`blocked` entry 可使用下列 closed
values：

- `manual`
- `integration_refresh_failed`
- `preview_abort_failed`
- `merge_target_preflight_failed`
- `preview_target_stale`
- `preview_merge_failed`
- `preview_postcondition_failed`
- `final_fast_forward_failed`
- `final_postcondition_failed`

未知字串或非 blocked entry 的 non-null value 會 Fast Fail。舊版 blocked entry 缺少此欄位或值為
`null` 時仍可由 `state validate` / `queue status` 讀取，但只代表 provenance unknown；
`merge retry-refresh`與`merge retry-final`明確拒絕這類 entry，不從自由文字 `note` 推測或
自動 migration。前者只接受`integration_refresh_failed`；後者只接受`action=final`的
`final_fast_forward_failed`，兩個 recovery transition 都不接受其它 provenance。

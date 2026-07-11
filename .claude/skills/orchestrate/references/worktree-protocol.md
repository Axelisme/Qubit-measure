# Worktree Protocol

## Identity

- `task-id`：plan、reports namespace、parent integration branch identity；每次 durable 操作明帶此值。
- `lane-id`：同 task 可獨立整合的工作線；單 lane 慣例為 `main`。
- `worktree-id`：單 lane = task-id；多 lane = `<task-id>--<lane-id>`。
- 單 lane branch `agent/<task-id>` 同時是 integration branch；多 lane branch
  `agent/<task-id>--<lane-id>` 依序 rebase/FF 到 parent `agent/<task-id>`。

同檔案、同 API contract、同 fixture 或共同設計的行為面必須同 lane 序列化。write scope 是協調 contract，
不是 path lock；scope update 前用 `--expect-current` 防 stale writer，orchestrator 仍判斷 overlap。

## 建立 lane

用 `workflow.py worktree create-lane`，明記 base branch/commit、branch、path、reports dir、write scope。
sub-agent prompt 必須指定 workdir，不能假設 runtime 自動切 worktree。

Git-tracked content 自動出現；gitignored plan、local config、scratch fixture 等逐項記為：

- `copy`：明確複製到 lane；
- `reference`：提供主 checkout 絕對路徑，只讀；
- `omit`：確認不需要並記原因。

report 寫在主 checkout `.agent_state/worktrees/reports/...`，不要寫進 lane worktree。

## Lane 整合與 cleanup

確認 agent 已停止寫入後，在 lane 跑 validation；rebase 到目前 parent，再以 `git merge --ff-only` 推進
integration branch。衝突若代表規格/架構分歧，停下問使用者。完成後移除 worktree、prune、以
`git branch -d` 刪已整合 lane；多 lane 可在實際 Git cleanup 後用 `lane remove` 清 state。整個 task final
由 workflow 移除 state，plan 整包 archive。abandoned task 用 `task close` 後 archive，不保留長期 checkout。

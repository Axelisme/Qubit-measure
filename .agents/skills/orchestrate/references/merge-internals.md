# Queue-Managed Merge 內部行為

這是 `workflow.py merge run <task-id> --action preview|final --requested-by <agent-id> --wait` 的內部邏輯參考。agent 不需要重新實作這段邏輯，只需要知道它會做什麼、在哪些情況下會停下來要求重新驗證；日常操作照 `SKILL.md`「Merge Queue Contract」的使用規則呼叫 `merge run` 即可。

```text
reserve queue entry for task-id/action
pin current integration target_commit on first enqueue
while entry is not queue head:
  reuse the queued target_commit; do not overwrite it while waiting
  sleep and poll until timeout
if queue head is blocked:
  stop on the blocked task; do not skip it
if action is final and same task has an open preview:
  abort that preview through workflow.py-owned abort logic
ensure main checkout is on base_branch and tracked-clean
read current base_branch HEAD
if integration branch does not contain current base HEAD:
  rebase integration branch in its lane/integration worktree
  block this queue entry with blocked_kind=integration_refresh_failed on conflict
  or missing integration worktree
refresh target_commit from integration branch
if integration branch's pre-refresh head differs from pinned target_commit:
  keep queue head queued
  ask agent to rerun validation, then rerun this action
if action is final and script-controlled refresh changed target_commit:
  keep queue head queued
  ask agent to rerun validation, then rerun final
if action is preview:
  merge --no-commit --no-ff target_commit in main checkout
  keep queue entry held as merging
if action is final:
  merge --ff-only target_commit in main checkout
  verify HEAD == target_commit
  release queue and remove task from state.json
```

`merge run` 只管理 Git/state critical section；review independence gate、測試驗證、module README 更新與 task plan 歸檔仍由 orchestrator 負責。若前一個 task 推進了 `base_branch`，`merge run` 會在自己成為 queue head 後 refresh integration branch；若 integration branch 在等待或 preview 開啟期間被其它 agent 推進，script 會保留 queue head 並要求重新驗證。final action 在 target commit 因 refresh 改變時也先停下讓 agent 重新驗證，避免「等待後 main 已分叉但還直接 final fast-forward」的錯誤。

## Refresh conflict 恢復

若 integration refresh 因 rebase conflict blocked，agent在 integration worktree 解完衝突、完成
rebase 後，使用窄幅 transition：

```text
workflow.py merge retry-refresh <task-id> \
  --action preview|final \
  --requested-by <same-agent-id> \
  --expect-target <resolved-7-to-40-hex-sha>
```

命令只接受 queue head 的 `blocked_kind=integration_refresh_failed`，並在 workflow locks 內重驗
task/entry identity、main與integration worktree branch/clean/merge/rebase狀態、base ancestry、
expected target與第二次ref snapshot。成功時只把同一entry更新成 `queued`、task更新成
`reviewing`；它不執行 preview或final。Agent必須先針對回傳的新 target完成validation，再呼叫
原本的 `merge run`。其它 blocked kind與legacy missing/null provenance均Fast Fail；不得解析
`note`或使用general-purpose unblock。`--expect-target`只接受7–40位hex object id / abbrev；branch、
tag或其它live symbolic ref均拒絕，再由Git確認該object存在且為commit。

## Final fast-forward operation failure 恢復

若 final fast-forward 的 Git operation 失敗、queue head明確記錄
`blocked_kind=final_fast_forward_failed`，且main與integration refs都未改變，使用：

```text
workflow.py merge retry-final <task-id> \\
  --requested-by <same-agent-id> \\
  --expect-target <recorded-7-to-40-hex-sha>
```

命令只接受`action=final`的blocked queue head，並在workflow locks內重驗task/entry/root/branch
identity、recorded `base_head == HEAD`、recorded/expected/current integration target一致、main
clean且無merge state、untracked path不會被target覆寫、fast-forward可行性與第二次ref snapshot。
它不做integration refresh或preview；成功時直接以同一target完成fast-forward，驗證postcondition後
移除queue head與task。retry operation或postcondition再次失敗時，entry/task維持blocked且
`blocked_kind`仍為`final_fast_forward_failed`。其它blocked kind與legacy missing/null provenance均
Fast Fail，不得由`note`推測failure site。

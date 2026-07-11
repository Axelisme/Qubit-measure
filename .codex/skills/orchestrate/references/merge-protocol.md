# Merge Protocol

## Queue contract

主 checkout preview/final 只用 `workflow.py merge run ... --wait`。queue head 屬於別 task 時等待/協調；
blocked head 不可跳過。禁止直接編輯 JSON、直接 merge/abort 繞 workflow，或從自由文字 note 推測 failure。

preview 是 main 上未提交驗收 merge，queue 維持 held；不要在 preview 改碼。意見回 lane 修改前用
`preview abort`。final 只做 fast-forward；如果 refresh 或 integration ref 漂移產生新 target，queue 保留並
要求針對新 target 重驗後重跑。

## MCCT / final

MCCT 先把 task scope diff 整理成清楚 commit（無 diff 不建空 commit），完成獨立 review 與 validation，
再直接 queue-managed final。它只省人工 preview，不省 queue、refresh、revalidation、FF、clean/untracked
protection、cleanup、plan archive。缺 identity、unrelated dirty files、review/validation failure 或非 FF 即停。

## Diagnostics and recovery

- exit 20：`queue status`；等待 head，blocked 依 provenance 處理。
- exit 30：確認無同時 writer 後重試；持續發生檢查 lock owner/殘留。
- exit 40：修正 identity/transition/input，不手改 JSON。
- exit 50：檢查 Git；preview 半途失敗用 `preview abort`，不用手動 merge abort。
- refresh conflict：在 integration worktree 解 rebase、確認 main/integration branch/clean/no merge or rebase
  state/base ancestry，再用相同 requester/action 與 hex `--expect-target` 執行 `merge retry-refresh`。只接受
  `integration_refresh_failed`；命令只 requeue，回傳新 target 後重驗再 `merge run`。
- final operation failure：只有 `action=final`、`blocked_kind=final_fast_forward_failed`、base/target 未漂移，
  才以相同 requester 和 recorded hex target 用 `merge retry-final`；它不 refresh/preview，成功直接 FF。
- legacy missing/null 或其它 provenance 一律拒絕窄幅 retry。
- session 中斷且 preview `merging`：先 `queue status` + `git status`；自己的 task 用 workflow abort/續走，
  別人的 task 不碰。

低階 `preview start` / `final fast-forward` 只供 debug/legacy 明確拆步，日常入口永遠是 `merge run`。

## Cleanup

final 成功後移除所有 lane/integration worktree、prune、以 `git branch -d` 刪已整合 branch，將
`.agent_state/plans/<task-id>/` 整包移到 archives。未授權 preview 時停在 integration diff/branch 並回報，
不占 queue。`git merge --squash` 非預設，因為它破壞 ancestry 與安全 branch cleanup。

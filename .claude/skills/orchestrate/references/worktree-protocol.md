# Worktree Protocol

## Materialization gate

只有 tracked diff 需要隔離、多 writer、跨回合 implementation，或主 checkout dirty 會衝突時建立 task/lane
worktree。純規劃、唯讀研究與單回合 review 不進 state。單 writer `light` task 可直接在目前 checkout。

## Lane graph

`task-id` 是 durable plan/integration identity；`lane-id` 是可獨立驗收與整合的 writer。lane 由 dependency 與
write-scope graph 決定，不按 Phase 或 agent role 拆。同檔案、同 API/schema/fixture 或共同設計行為放同 lane
序列化。foundation contract 先完成，才從共同 base 平行下游 lanes；預設最多兩個 implementer。

## 建立與 inputs

用 `workflow.py worktree create-lane`，明記 base、branch、path、report dir、write scope。Gitignored inputs
逐項記 `copy`、`reference` 或 `omit`。Agent prompt 必須指定 workdir；report 寫主 checkout 的 reports dir。

常用命令皆以 repo `.venv/bin/python` 執行並帶 `--root <main-checkout>`：

```text
workflow.py state init
workflow.py worktree create-lane <task-id> <lane-id> --base-branch main --write-scope <path>
workflow.py report path <task-id> <lane-id> <agent-id> --mkdir
workflow.py lane scope update <task-id> <lane-id> --expect-current <path> --add <path> --reason <text>
workflow.py merge run <task-id> --action preview|final --requested-by <agent-id> --wait
workflow.py queue status [task-id]
```

## Integration and cleanup

Agent 停止寫入後，各 lane 跑 targeted validation；依 dependency 順序 rebase/FF 到 integration branch。
語義衝突停下處理，不以文字解衝突掩蓋 contract drift。Final validation 在 integration target 執行一次。
完成後移除 materialized worktrees、prune、刪已整合 branch並 archive durable plan；不要保留長期 checkout。

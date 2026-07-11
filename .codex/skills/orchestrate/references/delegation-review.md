# Delegation and Review

## Thin-slice verification

public API、module boundary、data model、workflow protocol、核心行為、薄測試或模糊 report 都由
orchestrator 親讀關鍵 README/ADR/source/diff/test output。需要廣泛搜尋或長分析再委派。

## Review Independence Gate

每個含 implementation、test、文件規範或 workflow script diff 的 lane/task item 都必須有不同 identity
的 reviewer。implementer 不能簽核自己；多 implementer lane 的 reviewer 必須與所有 implementer 不同；
orchestrator 親自產生 diff 時也視為 implementer。self-review 只能補充，不能取代獨立 review。

小型工作可 self-plan：scope 局部、必要 context 可在當回合讀完、風險主要是規則精準度。plan 應引用
source、列可執行步驟/validation/停止分叉。若 review 後需大量修改，回到 implementer lane。

## Runtime routing

- Codex planner/reviewer：`5.6-terra`，fallback `5.5-high`。
- Codex implementer：`5.6-tarra`，fallback `5.5-med`。
- Claude planner/reviewer：`opus` + high reasoning。
- Claude implementer：`sonnet` + high reasoning。

prompt 同時提供 task-id、lane-id、workdir、write scope、report 絕對路徑、ignored input policy 與 model。
report 保存於主 checkout；完成 agent 不長期占 slot。

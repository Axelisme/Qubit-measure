---
name: orchestrate
description: Act as the repo-wide orchestrator / tech lead — classify risk, resolve architecture, coordinate independent implementation lanes, integrate delegated work, and verify evidence proportionally.
skill_version: 18
---

# Orchestrate

你是 repo 的總統籌。先依風險選擇最小充分流程，再規劃、委派、整合與驗證；不要為形式建立 agent、worktree、
report 或重跑無關測試。session 與 plan 用中文，程式碼、變數名、技術名詞用英文。

## 不可妥協的 gates

- 不猜需求或擅自調整架構；重大分歧、規格不明或超出授權時停下詢問使用者。
- 設計先於實作；public contract 或跨模組方向未凍結時，不平行 implementation。
- 多 writer 必須隔離；同檔案、同 public API/contract、同 schema 或共享 fixture 放同 lane 序列化。
- `critical` diff 必須由不同 agent identity 獨立 review；implementer 不可自簽。
- 高風險 planner/reviewer 結論由 orchestrator 親讀關鍵 source/diff/test 做 thin-slice verification。
- 主 checkout preview/final 只走 merge queue；不可跳過 blocked head、手改 state/queue JSON 或繞過 workflow。
- commit / merge 只在使用者要求或授權後執行。
- Live singleton（ZCU 板、GUI、固定 port）人工序列化。

## 先分類，再 materialize workflow

入口判斷：是否產生 tracked diff、是否改 public/跨模組 contract、是否多 writer 或跨回合、是否有未決設計。

| Profile | 適用 | Plan / worktree | Review | Validation |
|---|---|---|---|---|
| `light` | 局部、機械、contract-preserving、單 writer | inline checklist；安全時直接目前 checkout | orchestrator 自審；命中 trigger 才獨立 review | targeted tests + affected type/lint |
| `standard` | 一般多檔功能或 bug fix，邊界已知 | 簡短 source-grounded plan；需要隔離才建 worktree | 一次 lane 或 integration review | targeted first；integration target 一次 broader gate |
| `critical` | public API/schema/wire/persistence、跨模組 contract、migration、concurrency、workflow/merge、硬體風險 | durable plan + worktree | 不同 identity specialist review + thin slice | risk-based full gate；refresh 依 changed surface 重驗 |

純規劃、唯讀研究與單回合 review 不建立 workflow state/worktree。Profile 因新證據向上升級；降級要記理由。
完整判定與 review triggers 見 [delegation-review.md](references/delegation-review.md)。

## Capability gates，不是固定角色流水線

收尾只要求下列結果成立：

- `design_resolved`：沒有未決重大分叉；
- `scope_isolated`：並行 writer 與 contract 不重疊；
- `evidence_sufficient`：diff、tests、文件足以支持結論；
- `integration_safe`：target clean、refresh/FF/overwrite protection 成立。

`light`/`standard` 可由同一 agent 滿足多個 capability；不要固定派 planner → implementer → reviewer。
`critical` 才強制 planner（若設計未凍結）與不同 identity reviewer。

## 多 sub-agent implementation

平行化獨立工作，序列化共享決策。先畫 dependency/write-scope graph；foundation contract 完成並驗收後，
才從同一 base 開下游 lanes。預設最多兩個 implementer，保留一個 slot 給 reviewer/investigator。只有檔案、
contract、fixture、設計決策與 targeted acceptance 皆獨立才可平行；scope overlap、contract drift 或跨 lane
finding 出現時立即收斂回單 lane。細節見 [worktree-protocol.md](references/worktree-protocol.md)。

## Execution capabilities 與 loop authority

Risk profile 與 execution mode分開選擇：`direct`、`delegated`、`parallel-burst`。`parallel-burst` 是無狀態的
平行執行元件，不擁有 persistence、verify/fix loop、commit或 completion authority；它可與 `standard` 或
`critical` 組合，只在至少兩個節點真正獨立且預期 critical-path收益高於啟動/整合成本時啟用。Non-trivial
parallel work先寫 waves、dependency matrix與每節點 acceptance，再整批啟動同 wave workers。完整契約見
[parallel-burst.md](references/parallel-burst.md)。

同一 task同時只能有一個 continuation `loop_authority`。新的 loop遇到既有 authority必須明確採
`refuse`、`adopt_existing` 或 `artifact_only`；background monitor、validation或 reviewer只提供 evidence，
不能自行宣告 task complete或另起競爭 loop。

## Agent profiles

- `contract-planner`：只在 contract/dependency/write scope 未收斂時使用；輸出 frozen contract、dependency
  graph、lane boundaries、acceptance、stop conditions、unresolved decisions。
- `lane-implementer`：擁有指定 lane 的完整交付；可處理 scope 內局部細節，不可改 frozen contract；只跑
  targeted validation。
- `lane-reviewer`：依 trigger 檢查單 lane correctness/scope/tests；不重做 planner、不做偏好式重寫。
- `integration-reviewer`：只審 lane 交界、contract parity、初始化/生命週期與整體行為。
- `Explore`、bug investigator、`mcp-skill-tester` 按需使用，不進固定 pipeline。

Prompt 只傳 `objective`、`workdir`、`write_scope`、`frozen_contract`、`acceptance_criteria`、
`targeted_validation`、`stop_conditions`、`report_path`。能從 state 推導者不重複。Report 固定為
`Outcome / Changed / Evidence / Open risks / Scope changes requested`，預設不超過30行；raw logs寫 artifact，
主 context只收 failure、evidence與next action。完成或 blocked event後立即釋放 slot；無 event才等待，單次
wait timeout不視為 stalled。

模型以 capability class 選擇：planner/critical reviewer 用 high-reasoning，standard implementer 用 balanced，
mechanical implementer 用 fast/balanced；runtime-specific model name 放 agent profile，不寫成 workflow hard gate。

## Context 與 durable state

先讀現存 task plan、相關模組 README；跨模組決策查 `docs/adr/README.md`。`task_plan.md` 是唯一預設
durable narrative；`findings.md`、`progress.md` 只在跨回合、critical 或資訊量確有需要時建立。

需要 tracked diff 隔離、多 writer、跨回合 implementation，或主 checkout dirty 會衝突時，才用
`scripts/workflow.py` 建 task/lane worktree。所有 durable workflow 操作明帶 `task-id`；狀態不直接編輯。
命令與 ignored input policy 見 [worktree-protocol.md](references/worktree-protocol.md)。

## Validation 與 integration

implementer 跑 targeted checks；reviewer只重跑 finding/聲明所需 checks；broader/full suite 在 task integration
target 最多一次。以 target SHA、command、result、affected surface 當 validation receipt；相同 SHA 不重跑。
target 變更或 base refresh 只有在 touched dependency surface 改變時才擴大重驗。完整規則見
[validation.md](references/validation.md)。

Verification failure依 `implementation|contract|environment` 分類。`light` 最多1次、`standard` 最多2次、
`critical` 最多3次 fix pass；超過上限進 terminal blocked/failed。Contract failure回到 design gate，
environment failure保留 evidence並停止重試。Finding不可因更換reviewer而關閉；target-changing fix使舊review
receipt失效，保留finding/counter並依原trigger重審新SHA，不得無限循環洗掉finding。

preview/final 仍走 queue-managed merge；MCCT 可省人工 preview，但不省 profile 要求的 review/validation、
refresh/FF/overwrite protection、cleanup 或 plan archive。細節見 [merge-protocol.md](references/merge-protocol.md)。

## 工作迴圈

1. 釐清目標，分類 `light|standard|critical`，列升級 triggers。
2. 凍結必要 contract；只在需要時建立 task plan、worktree與 lanes。
3. 選 execution mode；`parallel-burst` 先建 waves/matrix，再依 dependency graph委派，最多兩個 implementer。
4. 收短 report，抽查高風險 evidence；命中 trigger 才做對應 lane/integration review。
5. 各 lane targeted validation；整合 target做一次 risk-based final validation與必要 behavioral QA；failure走 bounded fix loop。
6. 有授權才 queue preview/final；完成後清理 materialized workflow 與 archive durable plan。
7. 有證據但不影響驗收的超 scope discovery 用 `candidate-backlog` skill 登記。

## 邊界與不用時機

單一明確小修、純問答、純規劃或唯讀 review 不需要本 skill 的 durable workflow。強型別、Fast Fail、責任
明確、最小驚訝；除非要求，不保留 legacy compatibility。不可用 `light` profile 包裝實際的 contract 變更。

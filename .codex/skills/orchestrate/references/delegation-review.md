# Delegation and Review

## Profile classification

`light` 必須同時是局部、contract-preserving、單 writer、可由 targeted checks 完整驗收。`critical` 包含
public API/schema/wire/persistence、跨模組 contract、migration、concurrency/lifecycle/security、workflow/merge
state、資料遺失或硬體風險。其餘為 `standard`。不確定時向上分類。

## Independent review triggers

以下任一成立即由不同 identity review：`critical` surface、跨模組 diff、測試薄弱、修改量超過可在單回合
完整理解的範圍、implementer 明示不確定、validation 出現非預期行為。局部 tests/docs/mechanical refactor
若 evidence 完整可由 orchestrator 自審，結案記錄免除理由。

## Thin-slice verification

public contract、module boundary、data model、workflow protocol、核心行為、薄測試或模糊 report 由
orchestrator 親讀關鍵 README/ADR/source/diff/test output。需要廣泛搜尋或長分析才委派。

## Agent contract

Prompt 只包含 objective、workdir、write scope、frozen contract、acceptance、targeted validation、stop
conditions、report path。Agent 發現需跨 scope、改 contract、碰共享 fixture、acceptance 與 source 衝突或
架構假設不成立時停止回報。Report 使用 `Outcome / Changed / Evidence / Open risks / Scope changes requested`。
預設最多30行；verbose logs寫artifact。完成、blocked或needs_decision以event回報；沒有event才wait，timeout
不是failure evidence。

模型以能力選擇：contract planner/critical reviewer 用 high-reasoning；standard implementer 用 balanced；
mechanical implementer 用 fast/balanced。runtime-specific name 由 agent profile 集中設定。

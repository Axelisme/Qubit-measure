# Candidate Backlog Schema

每項為 `.agent_state/backlog/<status>/<id>.md`。檔案開頭是 machine-readable JSON metadata block，
後接 Markdown；CLI 負責產生與驗證。

## Taxonomy

`kind` 是 closed enum：`defect`、`missing-capability`、`design-debt`、`technical-debt`、
`test-gap`、`documentation-gap`、`workflow-friction`、`observability-gap`、
`performance-opportunity`、`product-idea`。

`status` 是 `inbox`、`planned`、`resolved`、`closed`。`area` 可重複指定，但每項至少一個；值是簡短
repo area 名稱。`priority_hint` 可為 `low`、`medium`、`high`，只代表 reporter 線索，不是承諾。

## Required capture fields

- `title`：具體問題或 desired capability，不寫 solution slogan。
- `observation`：實際看到的現象。
- `evidence`：可重現位置、行為、錯誤或報告來源。
- `impact`：不處理的成本或風險。
- `desired_outcome`：描述成功結果，不強迫局部發現者設計完整解法。
- `source_task`：發現來源 task-id；無 task 時使用明確 `standalone`。

`suggested_direction`、`constraints` 可選。CLI 以 normalized title（Unicode casefold、空白折疊）做
repo-wide duplicate Fast Fail；相同 title 應補 evidence 或以 `duplicate` 指向 canonical item。

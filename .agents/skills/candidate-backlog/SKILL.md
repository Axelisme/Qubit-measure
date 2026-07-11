---
name: candidate-backlog
description: Capture evidence-backed discoveries that are valuable but outside the current task into a repo-local candidate backlog, without expanding scope or avoiding current-task obligations.
skill_version: 1
---

# Candidate Backlog

把「不影響當前 task 驗收、但有具體證據且值得未來處理」的發現寫入
`.agent_state/backlog/`。這是 local candidate inbox，不是已承諾 roadmap、issue tracker 或設計決策。

## Hard gates

- 當前 task 的 correctness、regression、review finding 或 acceptance gap 必須留在當前 task 處理。
- 需求不明、架構分叉或需要新授權時，停下詢問使用者；不可用 backlog 取代決策。
- 登記不授權 agent 擴張 scope，也不代表項目已排程。
- 只記錄有 `Observation`、`Evidence`、`Impact`、`Desired outcome` 的發現；不記個人偏好或無證據猜測。
- 不寫 credentials、硬體秘密、本機敏感設定、原始量測資料或大量 log。

## Capture workflow

1. 判斷發現不影響當前 task 驗收；若影響，回到 task finding。
2. 用 `list` 搜尋相同 title / area；`add` 也會 Fast Fail 並回報既有 ID。
3. 用 `scripts/backlog.py add` 建立一項一檔的 observation。
4. 在 task report / final 回報新建或補充的 backlog ID。
5. 不自行把項目升為 task；決定執行時才用 `plan` 綁定正式 task-id。

欄位與 taxonomy 見 [schema.md](references/schema.md)；狀態轉移與收尾條件見
[lifecycle.md](references/lifecycle.md)；人工撰寫時使用
[item-template.md](assets/item-template.md)。人工建立時必須把模板複製成與 metadata `id` 相同的
`<id>.md`，並同步替換 metadata 與 Markdown body；CLI 讀取時會驗證完整 metadata。

## CLI

文件中的 `<repo-python>` 代表 repo Python（本 repo 為 `.venv/bin/python`），`<skill-dir>` 是本
`SKILL.md` 所在目錄。所有命令明確指定主 checkout：

```text
<repo-python> <skill-dir>/scripts/backlog.py --root <main-checkout> add --kind <kind> --area <area> --source-task <task-id> --title <title> --observation <text> --evidence <text> --impact <text> --desired-outcome <text>
<repo-python> <skill-dir>/scripts/backlog.py --root <main-checkout> list [--status inbox|planned|resolved|closed] [--kind <kind>] [--area <area>] [--json]
<repo-python> <skill-dir>/scripts/backlog.py --root <main-checkout> plan <id> --task-id <task-id>
<repo-python> <skill-dir>/scripts/backlog.py --root <main-checkout> close <id> --resolution implemented --task-id <task-id> --commit <sha> --validation <text>
<repo-python> <skill-dir>/scripts/backlog.py --root <main-checkout> close <id> --resolution duplicate --duplicate-of <canonical-id>
```

CLI 使用 UTC timestamp、UTF-8、lock 與 atomic replace；不要繞過 transition 或直接覆寫 metadata。

## 收尾 routing

Development agent 在 task 收尾前檢查 findings、review report 與 workaround。只有跨 scope 且可跨 task
重現的候選事項進 inbox；當前 task 的必要修正不得被「移到 backlog」後略過。真正生效的設計仍寫入
tracked `docs/adr/` 或模組 `README.md`。

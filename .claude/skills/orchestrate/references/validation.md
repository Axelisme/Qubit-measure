# Validation

驗證與風險成比例，全部在目標 workdir 執行。先 targeted checks；不要讓 implementer、reviewer與 refresh 對
同一 target 重複跑完整套件。

## Validation receipt

每次 broader/full validation 記錄 target SHA、command、result、affected surface。相同 SHA 的成功 receipt
直接重用。只有 target code 改變，或 base refresh 影響 touched dependency surface，才擴大重驗。

## Profiles

- `light`：targeted tests、affected type/lint、`git diff --check`。
- `standard`：lane targeted checks；integration target 一次必要 broader checks。
- `critical`：integration target 依 repo 規約跑必要 full gate；orchestrator thin-slice 驗證關鍵 evidence。
- docs/skill/config-only：跑 contract/sync/parser tests與 diff-check，不跑無關產品測試。

Repo 完整 gate 的可用順序為：`uv run pyright`、`uv run pytest -n auto`、
`uv run ruff check --select I --fix`、`uv run ruff format`、`git diff --check`。它不是每個 lane 的固定清單。
格式化改變 diff 後只重跑 affected checks。新 worktree 缺 runtime dependencies 時加
`--extra development --group dev`。

最後依實際變更更新相關 module README/ADR；沒有新高層知識或未修改該模組時不做形式性 README churn。

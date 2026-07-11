# Validation

所有驗證在目標 workdir 執行。新 worktree 若需 venv，使用
`uv sync --extra development --group dev`；裸 `uv run pytest` 可能缺 qick/qtpy/scipy/h5py 等 runtime。

依 repo 規約順序：

```text
uv run pyright
uv run pytest -n auto
uv run ruff check --select I --fix
uv run ruff format
git diff --check
```

尚未 sync 時，各 `uv run` 加 `--extra development --group dev`。先跑 targeted checks 快速回饋，再跑與
風險成比例的完整檢查。格式化後重跑受影響 targeted tests。最後更新對應 lib/tests 模組 README：高層、
現在式、刷新 Last updated，不寫 commit hash；跨模組決策更新 ADR 與索引。

integration target 因 queue refresh 改變後，先針對新 target 重新完成必要 validation，再重跑 merge action。

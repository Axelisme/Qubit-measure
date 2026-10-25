# progress_bar 模組重點筆記

**Last updated:** 2026-10-25

**讀者：** framework-dev。Progress bar 的 factory、更新與關閉契約。

`progress_bar` 提供量測程式使用的最小 progress-bar seam。一般程式只 import
`make_pbar`，實際 bar 由目前 ContextVar factory 決定；沒有 factory 時使用 tqdm backend。

## API contract

- `BaseProgressBar.update(delta=1)` 是 tqdm-style 增量更新。
- `BaseProgressBar.set_progress(n)` 是 absolute progress 更新；round hook、device ramp
  這類已經持有目前絕對進度的 caller 使用它，避免 `n - pbar.n` read-modify-write。
- `reset()` 將進度歸零；`refresh()` 強制發佈目前狀態；`close()` 釋放或隱藏 bar。
- `total`、`n`、`desc` 是所有 backend 都要提供的 read surface，停用顯示時仍反映目前設定。

## Backend / factory

- `backend/tqdm.py` 是 notebook / CLI fallback，由 adapter 保存計數，避免停用顯示的
  tqdm 跳過計數更新。`set_progress(n)` 以 `update(n - current_n)` 轉成增量更新。
- GUI session 會用 `use_pbar_factory(...)` 安裝 Qt-aware worker bar；該 bar 仍實作同一個
  `BaseProgressBar` contract，發送 raw progress event，由 GUI owner 更新 Qt widget。
- fluxdep GUI 的 search worker 使用同一個 session progress factory，透過 Qt signal
  把 progress marshal 回主線程。

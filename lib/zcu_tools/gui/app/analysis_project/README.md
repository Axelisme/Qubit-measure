# Analysis project support

**Last updated:** 2026-10-21

**讀者：** framework-dev。兩個分析 GUI 共用的 project model 與路徑分工。

`gui.app.analysis_project` 是 fluxdep 與 dispersive 共用的過渡 owner。
它保留兩個分析 app 的 chip／qubit 資料流，不是量測 session 的 entry model。

- `project.py` 擁有 Qt-free `ProjectInfo`、預設路徑與 project wire payload。
- `result_scope.py` 掃描 `result/**/params.json`，保留既有 scope discovery 與 identity 行為。
- `project_dialog.py` 擁有 chip／qubit、result scope、typed path 與 Browse UI。
- `launcher.py` 擁有分析 app 的 `--chip`、`--qub` 與路徑選項。
- `readonly_handlers.py` 提供兩個分析 app 相同的 `project.info` 投影。

共用 GUI、session、remote、measure 與 workflows 不依賴本 package。
方向由 `.importlinter` 的 `shared-no-analysis-project` contract 保護，見 ADR-0072。
中立的 `nearest_existing` 位於 `gui.widgets.file_dialog`，callers 直接匯入它。

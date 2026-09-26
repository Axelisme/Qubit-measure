# `zcu_tools.plotting` — 繪圖能力家族

**Last updated:** 2026-09-27 — liveplot 搬移

本目錄收納具獨立定位或真實跨入口需求的共用繪圖能力。`liveplot/` 提供即時更新的 plotter、segment 與 frontend-neutral backend 契約，供 experiment runtime 與 GUI caller 使用；詳見 [liveplot/README.md](liveplot/README.md)。

只有需要共用的繪圖能力才納入此家族。Notebook 專用診斷圖、實驗特化圖與 app 專用 renderer 保留在各自 owner；此處不建立通用 PlotManager、統一 Figure schema 或新 plugin framework。根 package 的 `__init__.py` 不載入子 package，避免只匯入家族名稱就載入個別 frontend 或 backend。

Qt canvas、figure container 與 matplotlib GUI backend 的接入由 [gui/plotting](../gui/plotting/README.md) 負責；measure app 的 `driven/qt_liveplot_backend.py` 負責把該接入實作為 liveplot backend，app 的 `services/scopes.py` 在 worker 執行期間註冊它。依賴方向為 GUI → plotting，不由 `liveplot` 偵測 GUI。

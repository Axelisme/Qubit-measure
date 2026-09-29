# `zcu_tools.plotting` — 繪圖能力家族

**Last updated:** 2026-09-29 — 具名 Figure 所有權

本目錄收納具獨立定位或真實跨入口需求的共用繪圖能力。`liveplot/` 提供即時更新的 plotter、segment 與 frontend-neutral backend 契約，供 experiment runtime 與 GUI caller 使用；詳見 [liveplot/README.md](liveplot/README.md)。

[fluxdep/](fluxdep/README.md) 提供 Notebook 與 GUI 共用的 database search 診斷圖 builder。

`figures.FigureCollection` 提供 operation-owned 的具名原生 Figure 集合。同名同物件
登記等冪，名稱衝突、同圖別名及跨存活 owner 接管一律拒絕。`seal()` 固定集合，
不凍結 artist、不 close canvas，也不代表操作成功。集合保留期間持有圖及所有權；
登記表不延長集合生命週期。`subplots(name, ...)` 直接建立原生 Figure／Axes 與 Agg
canvas，保留 Matplotlib shape／layout，不登記 pyplot manager 或切換全域 backend。
一般圖可先繪製與保存，adapter 於操作完成後才接上呈現。此接口不處理 frontend
呈現、保存政策或 Matplotlib thread safety。
後續明確 plots factory 與 GUI 多圖採用此共用接口，目前既有 caller 尚未遷移。
只有需要共用的繪圖能力才納入此家族。Notebook 專用圖、實驗特化圖與 app 專用 renderer 保留在各自 owner；此處不建立通用 PlotManager、統一 Figure schema 或新 plugin framework。根 package 的 `__init__.py` 不載入子 package，避免只匯入家族名稱就載入個別 frontend 或 backend。

Qt canvas、figure container 與 matplotlib GUI backend 的接入由 [gui/plotting](../gui/plotting/README.md) 負責；measure app 的 `driven/qt_liveplot_backend.py` 負責把該接入實作為 liveplot backend，app 的 `services/scopes.py` 在 worker 執行期間註冊它。依賴方向為 GUI → plotting，不由 `liveplot` 偵測 GUI。

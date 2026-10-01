# `zcu_tools.plotting` — 繪圖能力家族

**Last updated:** 2026-10-02 — 單熱圖與掃描線圖共用 explicit owner

本目錄收納具獨立定位或真實跨入口需求的共用繪圖能力。`liveplot/` 提供即時更新的 plotter、segment 與 frontend-neutral backend 契約，供 experiment runtime 與 GUI caller 使用；詳見 [liveplot/README.md](liveplot/README.md)。

[fluxdep/](fluxdep/README.md) 提供 Notebook 與 GUI 共用的 database search 診斷圖 builder。

`figures.FigureCollection` 提供 operation-owned 的具名原生 Figure 集合。同名同物件
登記等冪，名稱衝突、同圖別名及跨存活 owner 接管一律拒絕。`seal()` 固定集合，
不凍結 artist、不 close canvas，也不代表操作成功。集合保留期間持有圖及所有權；
登記表不延長集合生命週期。`subplots(name, ...)` 直接建立原生 Figure／Axes 與 Agg
canvas，保留 Matplotlib shape／layout，不登記 pyplot manager 或切換全域 backend。
一般圖可先繪製與保存，adapter 於操作完成後才接上呈現。此接口不處理 frontend
呈現、保存政策或 Matplotlib thread safety。
`plots.Plots` 在此集合上提供明確 `PlotHost`、typed 1D liveplot、單熱圖
`liveplot_2d` 與含最近掃描線的 `liveplot_2d_with_line`。兩種 2D handle 共用
`(len(xs), len(ys))` 實數資料契約；uniform 與 nonuniform 座標共用具名 Figure。
一般圖在 `finish()` 時呈現，liveplot 立即呈現；update 先驗證與複製資料，
再同步送到 host owner 修改 artists。不呈現 host 仍建圖、更新 artists 並支援
原生 savefig。匯入明確 plots 不初始化 pyplot 或 Notebook display；非呈現操作
只需基本依賴，不決定 GUI backend。Caller 停止 producer 後才 finish；完成後
拒絕新 typed update。`finish()` 回傳另一個 `NamedFigures`，只有具名 Mapping
介面，沒有建圖或 host commands。它保留圖與登記 owner，不因 caller 丟棄
Plots handle 而失去所有權。Presentation owner 另持有 Plots，以 release
釋放呈現，不銷毀 NamedFigures 中的原 Figure。失敗時可
finish(present=False) 保留普通診斷圖而不呈現。GUI host 在 gui/plotting，
shared plotting 不依賴 Qt。舊 2D／scatter 與其他 experiment/application callers
仍待遷移；本 factory 不替他們改接呼叫。
只有需要共用的繪圖能力才納入此家族。Notebook 專用圖、實驗特化圖與 app 專用 renderer 保留在各自 owner；此處不建立通用 PlotManager、統一 Figure schema 或新 plugin framework。根 package 的 `__init__.py` 不載入子 package，避免只匯入家族名稱就載入個別 frontend 或 backend。

Qt canvas、figure container 與 matplotlib GUI backend 的接入由 [gui/plotting](../gui/plotting/README.md) 負責；measure app 的 `driven/qt_liveplot_backend.py` 負責把該接入實作為 liveplot backend，app 的 `services/scopes.py` 在 worker 執行期間註冊它。依賴方向為 GUI → plotting，不由 `liveplot` 偵測 GUI。

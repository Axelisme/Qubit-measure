# `zcu_tools.plotting` — 繪圖能力家族

**Last updated:** 2026-10-02 — Shared live axes 與 owner recording

本目錄收納具獨立定位或真實跨入口需求的共用繪圖能力。`liveplot/` 提供即時更新的 plotter、segment 與 frontend-neutral backend 契約，供 standalone caller 與明確 axes 的共用繪圖使用；詳見 [liveplot/README.md](liveplot/README.md)。

[fluxdep/](fluxdep/README.md) 提供 Notebook 與 GUI 共用的 database search 診斷圖 builder。

`figures.FigureCollection` 提供 operation-owned 的具名原生 Figure 集合。同名同物件
登記等冪，名稱衝突、同圖別名及跨存活 owner 接管一律拒絕。`seal()` 固定集合，
不凍結 artist、不 close canvas，也不代表操作成功。集合保留期間持有圖及所有權；
登記表不延長集合生命週期。`subplots(name, ...)` 直接建立原生 Figure／Axes 與 Agg
canvas，保留 Matplotlib shape／layout，不登記 pyplot manager 或切換全域 backend。
一般圖可先繪製與保存，adapter 於操作完成後才接上呈現。此接口不處理 frontend
呈現、保存政策或 Matplotlib thread safety。
`plots.Plots` 在此集合上提供明確 `PlotHost`、typed 1D liveplot、單熱圖
`liveplot_2d` 與含最近掃描線的 `liveplot_2d_with_line`。`liveplot_scatter` 接同長、非空一維實數 xs／ys／colors，保留 scalar color coordinate 與自動色階；更新前複製三組資料，artist mutation 仍由 host owner 執行。兩種 2D handle 共用
`(len(xs), len(ys))` 實數資料契約；uniform 與 nonuniform 座標共用具名 Figure。單熱圖可指定 `clim=(min, max)`，固定色階跨 update 保留，未指定時沿資料自動縮放。
`liveplot_1d` 的 `configure_axes` callback 在 host owner 初始化 artists 後、呈現前執行一次，供實驗設定原生 ticks／style。Callback 不得保留 active axes 供 worker 後續修改；設定失敗直接傳遞例外，不呈現未完成圖。
1D／2D factories 可接同一具名 Figure 的既有 axes，支援 workflow 合併圖。外部、移除或已被其他 handle 使用的 axes 會被拒絕；同圖只呈現一次。Heatmap 的 `mark_point` 與 scan-line 的 `mark_line` 由 host owner 更新 reference marker，不把 active artists 交回 worker。`refresh(name)` 在批次 typed update 後刷新整圖。
`record_animation(name, path)` 以 FFmpeg 保存 live figure；recorder 的 setup、grab_frame、finish 均在 host owner 執行。Caller 在 producer 停止後結束 recorder，Plots.finish 也收尾尚未結束的 recorder。錄影不取得 Figure 所有權，不依賴 ambient backend，writer 錯誤不代表操作成功。
一般圖在 `finish()` 時呈現，liveplot 立即呈現；update 先驗證與複製資料，
再同步送到 host owner 修改 artists。不呈現 host 仍建圖、更新 artists 並支援
原生 savefig。匯入明確 plots 不初始化 pyplot 或 Notebook display；非呈現操作
只需基本依賴，不決定 GUI backend。Caller 停止 producer 後才 finish；完成後
拒絕新 typed update。`finish()` 回傳另一個 `NamedFigures`，只有具名 Mapping
介面，沒有建圖或 host commands。它保留圖與登記 owner，不因 caller 丟棄
Plots handle 而失去所有權。Presentation owner 另持有 Plots，以 release
釋放呈現，不銷毀 NamedFigures 中的原 Figure。失敗時可
finish(present=False) 保留普通診斷圖而不呈現。GUI host 在 gui/plotting，
shared plotting 不依賴 Qt。Experiment 與 application callers 顯式傳遞 Plots。
只有需要共用的繪圖能力才納入此家族。Notebook 專用圖、實驗特化圖與 app 專用 renderer 保留在各自 owner；此處不建立通用 PlotManager、統一 Figure schema 或新 plugin framework。根 package 的 `__init__.py` 不載入子 package，避免只匯入家族名稱就載入個別 frontend 或 backend。

Qt canvas 與 figure container 的接入由 [gui/plotting](../gui/plotting/README.md) 負責。GUI 注入 explicit QtPlotHost，透過 owner scheduler 更新 artists 與呈現。Standalone liveplot backend 仍服務其 public plotters，與 operation-owned Plots 分開。依賴方向為 GUI → plotting，不由 liveplot 偵測 GUI。

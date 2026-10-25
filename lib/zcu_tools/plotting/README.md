# `zcu_tools.plotting` — 圖管理與 owner 更新

**Last updated:** 2026-10-25

**讀者：** framework-dev。具名原生圖、通用更新 handle 與診斷圖的分工。

`figures.FigureCollection` 擁有具名原生 Figure。同名同物件登記等冪，名稱衝突、
同圖別名及跨存活 owner 接管一律拒絕。`seal()` 固定集合，不凍結 artist，
不 close canvas，也不代表操作成功。登記表不延長集合生命週期。
`subplots(name, ...)` 直接建立原生 Figure／Axes 與 Agg canvas，保留 Matplotlib
shape／layout，不登記 pyplot manager 或切換全域 backend。

## 操作與呈現

`plots.Plots` 管理具名圖與明確的 `PlotHost`。Host 同步在自己的 owner 執行 callback，
並負責 present、refresh 與 release。Shared plotting 不 import Qt 或 lab。
GUI 注入 [QtPlotHost](../gui/plotting/README.md)，Notebook 注入自己的呈現 host。
`NonPresentingHost` 直接執行 callback，不開視窗或安排繪製，仍支援 artists 與 savefig。

`Plots.create_live(name, initialize, ...)` 在 owner 建圖或接上既有 axes，呼叫作者的
initializer，成功後立即呈現。Initializer 收到 host、figure、claimed axes 與 active
checker，可回傳自己的 handle。既有 axes 必須屬於具名圖且尚未被其他 live initializer
使用。名稱衝突、外部、移除、重複或數量錯誤的 axes 都會被拒絕。
初始化失敗不呈現新圖，不回滾 callback 已做的修改。

`draw(name, callback)` 在 owner 自由修改具名 Figure，成功後首次呈現並 refresh。
後續 draw 共用同一 presentation。Callback 失敗直接傳遞，不呈現、不 refresh，也不回滾。
Caller 不得保留 active artists 供 worker 直接修改。Callback 可以清除 axes 或重建 layout，
既有作者 handle 如何處理被移除的 artists，由該 callback 的契約決定。
`save(name, path, *, figsize=(6.4, 4.8), dpi=100)` 在 owner 以固定輸出尺寸保存，
不沿用 GUI canvas 大小，預設 PNG 為 640×480。多面板由 caller 明列較大的 figsize。
保存時沿用既有 layout engine，沒有 engine 時套 tight layout。保存成功或失敗都還原
顯示尺寸、layout 與 axes 位置。不 refresh，finish 前後皆可用。

## 通用即時更新

從 `zcu_tools.plotting` import `LiveUpdate[T]`。在 create_live initializer 中，
以其 host、figure、active checker 與接受 T 的繪圖 callback 建立 handle。
`handle.update(data)` 先 deepcopy 資料，再同步在 host owner 執行 callback，成功後 refresh。
Payload 可包含巢狀 NumPy arrays 或容器，但必須支援 deepcopy。
自訂 deepcopy 必須保留資料隔離。Copy、callback 與 host 例外直接傳回 caller。
Callback 失敗不 refresh，已修改的 artists 不回滾。

`update(data, refresh=False)` 允許批次更新，之後用 `handle.refresh()` 或
`plots.refresh(name)` 刷新整圖。Handle 在 dispatch 前及 owner 執行前都檢查 lifetime，
finish／release 後拒絕更新與 refresh。Lib 不驗證 payload 的圖格式或實驗語意。

## 完成與保留

Caller 先停止 producer，再呼叫 `finish()`。Finish 固定 membership、最後 refresh live
圖，並呈現尚未 draw 的一般圖。失敗時可用 `finish(present=False)` 保留診斷圖而不呈現。
回傳的 `NamedFigures` 只有具名 Mapping 介面，保留原生圖及登記 owner。
Presentation owner 另持有 Plots，透過 `release()` 釋放呈現，不銷毀保留的圖。
Release 逐圖嘗試，彙整失敗；失敗的項目可以重試。
`fork()` 建立同 host 的獨立通用 Plots，不共享名稱、handle 或 finish／release。

## 格式與診斷圖的 owner

四個 live 工具及其線圖、熱圖、非均勻熱圖、scatter 格式由
[lab 支援套件](../../../zcu_lab/v2/README.md#即時圖工具) 擁有。
Singleshot 分類圖與 joint-fit 診斷圖分別在 lab 的 `singleshot/plot.py` 與
`singleshot/plot_rabi.py`。實驗作者可以直接修改這些工具。
獨立 liveplot plotter／backend、reference marker 與動畫錄影已退休。

[fluxdep](fluxdep/README.md) 保留 Notebook 與 GUI 共用的搜尋、頻譜與選點診斷圖。
它自己擁有私有非均勻背景 renderer，不依賴 lab 工具。

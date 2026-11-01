# `zcu_tools.plotting` — 圖管理與 owner 更新

**Last updated:** 2026-10-31

**讀者：** framework-dev。具名原生圖、owner 更新與診斷圖的分工。

`figures.FigureCollection` 擁有具名原生 Figure。同名同物件登記等冪，名稱衝突、
同圖別名及跨存活 owner 接管一律拒絕。`seal()` 固定集合，不凍結 artist，
不 close canvas，也不代表操作成功。登記表不延長集合生命週期。
`subplots(2, 2, num="fit")` 接受 Matplotlib 的列數、欄數、sharing 與 layout 參數，
回傳原生 Figure／Axes，保留 scalar／array shape。`num` 是集合內唯一的字串名字，
重複名字 raise ValueError。省略時依序取下一個未使用的 `figure_N`。
建圖使用 Agg canvas，不登記 pyplot manager 或切換全域 backend。

## 預設外觀

`zcu_tools.plotting.DEFAULT_STYLE` 指向隨套件安裝的 `experiment.mplstyle`。
GUI runtime、`NotebookPlotHost` 建構與離線 probe 在建立實驗圖前，呼叫
`matplotlib.style.use(DEFAULT_STYLE)`，設定後續新圖的字級、格線與 tight layout。
樣式不設定 `savefig.*` 或 backend，不登記 pyplot manager，也不修改既有 artists。
直接建立 `FigureCollection` 或 `Plots` 不套用樣式。

作者用原生 Matplotlib 參數覆寫單一 artist，例如
`ax.set_xlabel("Frequency (MHz)", fontsize=16)`。
要覆寫一組新圖，在入口初始化後使用
`with plt.style.context({"axes.grid": False}): ...`。
測試或其他需要隔離全域 rcParams 的 caller，將入口與繪圖包在
`matplotlib.rc_context()` 中。

## 操作與呈現

`plots.Plots` 管理具名圖與明確的 `PlotHost`。Host 同步在自己的 owner 執行 callback，
並負責 present、refresh 與 release。Shared plotting 不 import Qt 或 lab。
GUI 注入 [QtPlotHost](../gui/plotting/README.md)，Notebook 注入自己的呈現 host。
`NonPresentingHost` 直接執行 callback，不開視窗或安排繪製，仍支援 artists 與 savefig。

`Plots.subplots` 不呈現圖。作者呼叫 `fig.show()` 時，框架的 FigureManagerBase 子類
透過 host 同步呈現，同一張圖重複 show 不重建 presentation。
Qt canvas 交換及 Notebook ipympl manager 替換後，show 仍走同一個 operation。
NonPresentingHost 的 show 在 operation 尚未 finish 時不做事。
這條路徑不使用 `plt.show()`。未 show 的圖仍在 `finish(present=True)` 時呈現。

`save(name, path, *, figsize=(6.4, 4.8), dpi=100)` 在 owner 以固定輸出尺寸保存，
不沿用 GUI canvas 大小，預設 PNG 為 640×480。多面板由 caller 明列較大的 figsize。
保存時沿用既有 layout engine，沒有 engine 時套 tight layout。保存成功或失敗都還原
顯示尺寸、layout 與 axes 位置。不 refresh，finish 前後皆可用。

## 原生 matplotlib 更新

用 `@plots.on_plot_thread` 裝飾接受單一 payload 的函式。呼叫時先檢查 lifetime，再 deepcopy
payload，在 host owner 執行本體。本體回傳 changed artists 的 iterable，Figure 也可回傳。
框架驗證所有 artists 屬於此 Plots，再對已呈現的根圖去重 refresh。None 回傳 raise TypeError，
外來圖 raise ValueError，空 iterable 不 refresh。Copy、本體與 host 例外傳回 caller，
本體失敗不 refresh，不回滾已做的修改。Finish／release 後呼叫 raise RuntimeError。

即時更新與一次性改動都用這個入口。圖顯示後的改動一律經此入口。
Show 前可先呼叫，修改只保留在 artists，不呈現、不 refresh、不載入 backend。
後續 `fig.show()` 或 `finish(present=True)` 顯示最新內容。回傳混合的圖時只刷新已呈現者，
但未呈現的外來圖仍拒絕。Decorator 不登記或佔用 axes，本體只應改回傳 artists 所屬的圖。

Lab 的 `live_line(ax)`、`live_heatmap(ax)` 與 `live_scatter(ax)` 是可選的本地 artist 工具。
它們的 `set(...)` 回傳 changed artists，可直接在裝飾後的本體中回傳。
它們不切執行緒、不 refresh、不 present，也不檢查 axes 佔用。

## 完成與保留

Caller 先停止 producer，再呼叫 `finish()`。Finish 固定 membership、最後 refresh live
圖，並呈現尚未 show 的一般圖。失敗時可用 `finish(present=False)` 保留診斷圖而不呈現。
回傳的 `NamedFigures` 只有具名 Mapping 介面，保留原生圖及登記 owner。
Presentation owner 另持有 Plots，透過 `release()` 釋放呈現，不銷毀保留的圖。
Release 逐圖嘗試，彙整失敗；失敗的項目可以重試。
Finish 後的 `fig.show()` 一律 raise RuntimeError，release 前後相同。
Release 還原帶框架 manager 的 Agg canvas，並解除 manager 對舊 Plots 的參照。
另一 host 可以經自己的 `present(figure)` 呈現保留的圖，但不恢復已結束 operation 的 show。
`fork()` 建立同 host 的獨立通用 Plots，不共享名稱、更新或 finish／release。

## 格式與診斷圖的 owner

三個 live 工具及其線圖、熱圖、非均勻熱圖、scatter 格式由
[lab 支援套件](../../../zcu_lab/v2/README.md#即時圖工具) 擁有。
Singleshot 分類圖與 joint-fit 診斷圖分別在 lab 的 `singleshot/plot.py` 與
`singleshot/plot_rabi.py`。實驗作者可以直接修改這些工具。
獨立 liveplot plotter／backend、reference marker 與動畫錄影已退休。

[fluxdep](fluxdep/README.md) 保留 Notebook 與 GUI 共用的搜尋、頻譜與選點診斷圖。
它自己擁有私有非均勻背景 renderer，不依賴 lab 工具。

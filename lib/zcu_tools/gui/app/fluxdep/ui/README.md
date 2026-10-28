# Fluxdep UI

**Last updated:** 2026-10-28

**讀者：** framework-dev、agent。查找 fluxdep 的 Qt 呈現 owner，以及 widget 與分析命令的接縫。

## 視窗與分析頁

`MainWindow` 接收 Controller，呈現光譜清單、互動 editor 與 Fit 頁。
載入與匯出 dialog 收集輸入，將命令交給 Controller，不另建領域 State。
`AnalyzePanelWidget` 在第一次進入 Fit 頁時建立，之後切頁只隱藏，保留分析階段狀態。
`FilterStage` 與 stage controls 呈現已提交的選點與遮罩，數值計算交給 app owner。

`interactive/` 擁有定線、OneTone、TwoTone 與預覽的 Qt widgets 和 canvas。
Widget 連接 `FluxDepInteractiveOwner` 的 live context，不以畫面是否可見決定草稿是否存在。
Active spectrum 只決定顯示，切譜不退休其他 spectrum 的草稿。
Owner 與完整 pipeline 見 [fluxdep README](../README.md#architecture-and-owners)。

## 命令、結果與拆除

UI 與 remote 使用同一個 Controller 與 interactive owner。
背景計算透過 owner-loop delivery 交回結果，不從 worker 直接寫 State。
TwoTone preview 只更新呈現，完成提交才更新已提交的選點狀態。

搜尋與擬合數值由 `analysis.fluxdep` 擁有。
搜尋診斷圖由 `plotting.fluxdep` 建立原生 Figure，再由共用 `gui.plotting` 呈現。
Entry／component 選擇使用 [analysis_project](../../analysis_project/README.md)，不接 Measure 的硬體或 tab owners。

UI 回呼與訂閱經 LifetimeScope 登記，替換 editor 前先關閉它的 scope。
Window 關閉時 dispose interactive contexts 並 quiesce runner，單純 detach view 不終止 Session。
跨 app 的 owner 分界見 [ADR-0067](../../../../../../docs/adr/0067-gui-application.md)。

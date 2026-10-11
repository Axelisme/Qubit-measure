# GUI plotting

**Last updated:** 2026-10-10，live presentation 收合與 final flush

此目錄提供 explicit Matplotlib／Qt 接入，不決定各 app 的 figure 接受或保存政策。
根 package 延遲載入 Qt 與 Matplotlib 匯出。Runtime 在 QApplication 建立後初始化
host、shutdown guard 與 mathtext lock／prewarm，不切換 process-wide backend。

## Figures 與 presentation

Caller 透過 Plots 持有具名原生 Figure，並注入 QtPlotHost。Host 使用 owner scheduler
執行 live artist 更新與 presentation。普通圖首次成功 draw 後即呈現，後續 draw 共用同一 canvas；未 draw 的圖仍完成後才呈現。Callback 失敗不呈現或 refresh，也不 rollback，診斷結果仍保留。
QtPlotHost 只收合 presentation，不丟棄資料更新。每次 live callback 都同步在 owner
更新 artists；密集 refresh 可以合成較少的 canvas draws。Producer 停止後的 final
refresh 立即 flush 最新圖，不等待下一次呈現。RPC PNG 讀同一份原生 Figure，
不從已呈現的畫面反推資料。
Qt canvas 與 Figure 的生命週期分開。Release 取消待呈現工作並釋放 presentation，保留原圖及 Agg canvas，
caller 仍可修改 artists 或 savefig。

FigureContainer 包裝 QStackedWidget 與 placeholder。host.py 維護 weak-key
Figure-to-container registry，並透過 GUI-thread QObject 處理 explicit attach 與
canvas removal。容器可以同時保留多張圖，重新 attach 會選取對應 canvas。
容器 clear 清掉動態 canvas 與 registry entry，不清空原生 Figure。

`figure_export.render_figure_png` 提供固定 geometry 的原生 Figure PNG。Measure agent screenshots 使用這個 shared rendering owner，save／preview 政策仍屬於 app。Caller 持有 exclusive rendering slot；成功與失敗都恢復原尺寸，不轉移 canvas ownership，也不註冊 pyplot Figure。

## Thread 與 rendering 邊界

Host 必須先在 GUI thread 初始化。Runtime 與 FigureContainer 建構都確保此條件。
Worker 不直接操作 Qt widget；liveplot update 透過 host owner 執行。
Shared mathtext lock／prewarm 保護字型解析，不代表任意 Matplotlib 操作都 thread-safe。

各 app 的互動 widget 也可以在主執行緒自持 FigureCanvasQTAgg。
Qt、Agg 與 Notebook ipympl 能力保留，不依賴專案自訂 pyplot routing backend。
數據、取消及 operation 的收尾仍由各 app／session owner 負責。

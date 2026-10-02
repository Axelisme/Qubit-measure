# GUI plotting

**Last updated:** 2026-10-02

此目錄提供 explicit Matplotlib／Qt 接入，不決定各 app 的 figure 接受或保存政策。
根 package 延遲載入 Qt 與 Matplotlib 匯出。Runtime 在 QApplication 建立後初始化
host、shutdown guard 與 mathtext lock／prewarm，不切換 process-wide backend。

## Figures 與 presentation

Caller 透過 Plots 持有具名原生 Figure，並注入 QtPlotHost。Host 使用 owner scheduler
執行 live artist 更新與 presentation。普通圖完成後才呈現，診斷失敗不必丟棄計算結果。
Qt canvas 與 Figure 的生命週期分開。Release 釋放 presentation，保留原圖及 Agg canvas，
caller 仍可修改 artists 或 savefig。

FigureContainer 包裝 QStackedWidget 與 placeholder。host.py 維護 weak-key
Figure-to-container registry，並透過 GUI-thread QObject 處理 explicit attach 與
canvas removal。容器可以同時保留多張圖，重新 attach 會選取對應 canvas。
容器 clear 清掉動態 canvas 與 registry entry，不清空原生 Figure。

## Thread 與 rendering 邊界

Host 必須先在 GUI thread 初始化。Runtime 與 FigureContainer 建構都確保此條件。
Worker 不直接操作 Qt widget；liveplot update 透過 host owner 執行。
Shared mathtext lock／prewarm 保護字型解析，不代表任意 Matplotlib 操作都 thread-safe。

各 app 的互動 widget 也可以在主執行緒自持 FigureCanvasQTAgg。
Qt、Agg 與 Notebook ipympl 能力保留，不依賴專案自訂 pyplot routing backend。
數據、取消及 operation 的收尾仍由各 app／session owner 負責。

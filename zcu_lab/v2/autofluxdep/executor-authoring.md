# Autofluxdep executor authoring

**Last updated:** 2026-10-05，executor 與 leaf 同族搬遷

`FluxDepExecutor` 在 `core.py` 繼承 framework 的 `MultiMeasurementExecutor`，以 flux values、FluxDepCfg 與 FluxDepEnv 執行一組 MeasurementTask。`run()` 接收 device cfg、predictor、RunContext 與 ModuleLibrary。每個 flux point 更新 tracker、設定 flux，再執行 measurement batch。`save()` 分別保存每個已註冊 measurement 的結果。各 leaf 的 `core.py` 定義 executor task、cfg 與結果處理。

`_support/env.py` 擁有 FluxDepEnv 與 FluxDepInfoTracker。Tracker 的 current／first／last、required input 與 fallback 契約見 [experiment authoring](../experiment-authoring.md#executor-模式autofluxdep--overnight)。

Liveplot 使用 context.plots 中具名 `measurement` 合併 Figure。Host owner 用 typed handles 更新曲線、熱圖與 reference marker，也操作 MP4 recorder。Executor 不 close 圖。Caller 停止 producer 後 finish／release Plots，保留的 NamedFigures 可繼續原生保存。

這套接口供 direct caller 使用 [framework runtime](../../../lib/zcu_tools/experiment/v2/runtime/README.md) 編排 flux 掃描。[GUI Node authoring](node-authoring.md) 使用另一套 Builder／Node 接口，由 app Orchestrator 管理 placement、Patch 與 run artifact。兩者不共用實驗 class；此搬遷不宣告任一接口退役。

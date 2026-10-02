# `experiment/v2/autofluxdep/` — executor workflow

**Last updated:** 2026-10-02 — explicit workflow plots

`FluxDepExecutor` 在 `executor.py` 繼承 `MultiMeasurementExecutor`，以 flux values、
`FluxDepCfg` 與 `FluxDepEnv` 執行一組 `MeasurementTask`。`run()` 接收 device cfg、
predictor、RunContext 與 ModuleLibrary；每個 flux point 更新 tracker、設定 flux，
再執行 measurement batch。`save()` 將每個已註冊 measurement 的結果分別保存。
各實驗檔案定義這套 executor 使用的 task、cfg 與結果處理。

Liveplot 使用 context.plots 中的具名 `measurement` 合併 Figure，typed handles
在 host owner 更新曲線、熱圖與 reference marker。MP4 recorder 也在 host owner
操作。Executor 不 close 圖，caller 負責停止 producer 後 finish／release Plots；
保留的 NamedFigures 可繼續原生保存。

這個接口適用於直接使用 experiment/v2 runtime 組合和執行 flux 掃描的 caller。
[`experiment/v2_gui/autofluxdep/`](../../v2_gui/autofluxdep/README.md) 則提供
Autofluxdep GUI 的 Builder／Node 接入，透過 app Orchestrator 管理 placement、Patch
與 run artifact。兩者的適用 caller 和長期差異尚待逐項核實；本 README 不將任一接口
標為退役，也不宣稱它們共用同一實驗 class。

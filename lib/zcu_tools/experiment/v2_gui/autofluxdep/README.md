# `experiment/v2_gui/autofluxdep/` — Autofluxdep GUI experiment integration

**Last updated:** 2026-10-02 — explicit GUI plot factories

本分支提供 autofluxdep GUI workflow 的 concrete measurement experiments。
每個 `<name>.py` 擁有自己的 Builder／Node、cfg/schema、acquire／fit／Patch policy 與
Result／Plotter factory，檔尾輸出 `EXPERIMENT`。這些 Builder 實作
[app 的 Builder／Node／RunEnv 契約](../../../gui/app/autofluxdep/README.md#workflow-與執行契約)；
workflow 排程、run lifecycle、cfg snapshot 與 artifact 仍屬 app。
這不是 `experiment/v2/autofluxdep/` 的 executor／MeasurementTask 介面。
Plotter factory 接收 app 的 run-owned Plots 與 figure name，以 typed factories 建立原生
subplot。主線程更新 Result 的 scalar／curve／heatmap 投影，再刷新 canvas；Plotter 不取得
Qt widget 或保存權威，也不依賴 ambient liveplot backend。

`catalog.py` 明確收集 qubit_freq、lenrabi、ro_optimize、t1、t2ramsey、t2echo 與 mist 的
`EXPERIMENT`，供 GUI 建立 placement。catalog 順序只決定新增選單，不重排使用者保存的
workflow。`__init__.py` 不匯入 catalog 或 concrete experiments；使用者應從
`zcu_tools.experiment.v2_gui.autofluxdep.catalog` 明確匯入 `builders()`／`create_placement()`。
`_support/` 只保留多個 experiment 共用的量測 mechanics，詳見
[_support README](_support/README.md)。

新增實驗時，在此分支新增與 `Builder.name` 同名的 `<name>.py`，實作 app contract、
在 `catalog.py` 明確登記 `EXPERIMENT`，並在
[`tests/experiment/v2_gui/autofluxdep/`](../../../../../tests/experiment/v2_gui/autofluxdep/README.md)
驗證該實驗契約。不要依賴 filesystem discovery，也不要把具體實驗政策放入 app Orchestrator。

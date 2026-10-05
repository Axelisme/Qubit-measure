# Autofluxdep Node authoring

**Last updated:** 2026-10-05，使用者 leaf 與顯式 catalog

每個 `zcu_lab/v2/autofluxdep/<name>/autofluxdep.py` 擁有自己的 Builder／Node、cfg/schema、acquire／fit／Patch policy、Result／Plotter factory，檔尾輸出 `EXPERIMENT`。Builder 實作 [app 的 Builder／Node／RunEnv 契約](../../../lib/zcu_tools/gui/app/autofluxdep/README.md#workflow-與執行契約)。Workflow 排程、run lifecycle、cfg snapshot 與 artifact 屬於 app。這套接口不同於 [executor](executor-authoring.md) 的 MeasurementTask。

Plotter factory 接收 app 的 run-owned Plots 與 figure name，以 typed factories 建立原生 subplot。主線程更新 Result 的 scalar／curve／heatmap 投影，再刷新 canvas。Plotter 不取得 Qt widget 或保存權威，也不依賴 ambient liveplot backend。

`zcu_lab.autofluxdep_catalog.build_catalog()` 明列各 leaf 的 `EXPERIMENT` declarations，供組合根注入 app。通用 ExperimentCatalog 與 placement lookup 屬於 app。Catalog 順序只決定新增選單，不重排使用者保存的 workflow。Family 的 `__init__.py` 不載入 GUI attachments。Result 以 class-level result_kind 宣告 app archive representation；array layout 與量測／fit policy 留在 leaf。

共享量測 mechanics 位於 [domain support](../_support/autofluxdep/README.md)。Support 不選 workflow order，也不 import concrete experiments。

新增實驗時，在與 Builder.name 相同的 leaf 建立 `autofluxdep.py`，實作 app contract，並在 `zcu_lab.autofluxdep_catalog` 明列 `EXPERIMENT`。在 `tests/zcu_lab/v2/autofluxdep/<name>/` 驗證可觀察契約。不要依賴 filesystem discovery，也不要把具體實驗政策放入 app Orchestrator。

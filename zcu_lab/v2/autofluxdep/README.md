# Autofluxdep workflows

**Last updated:** 2026-10-05，兩套 workflow 同族分址

這個 family 保留兩套不同的接口。`core.py` 擁有 FluxDepExecutor，leaf 的 `core.py` 提供 executor tasks。Leaf 的 `autofluxdep.py` 提供 GUI app 的 Builder／Node。

- [Executor authoring](executor-authoring.md) 說明 Schedule、MeasurementTask、tracker 與 direct caller。
- [Node authoring](node-authoring.md) 說明 GUI workflow 的 Builder、Node、Patch 與 catalog declarations。

通用 runtime 留在框架，GUI app 擁有 workflow 編排與 artifact。兩套接口不轉接，也不因搬到同一 family 而合併。

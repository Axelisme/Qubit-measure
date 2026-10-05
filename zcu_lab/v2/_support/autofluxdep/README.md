# Autofluxdep domain support

**Last updated:** 2026-10-05，實驗定義搬遷

這個 private package 保留多個 Autofluxdep concrete experiments 共用的 acquire、Result、
Plotter、dependency/module/readout/timing defaults、schema、module values、OverridePlan 與
sweep/timing mechanics。只屬單一實驗的 cfg、fit 與 Patch policy 留在相應 leaf 的 `autofluxdep.py`。

`_support` 可以使用 [app 的 Node contract](../../../../lib/zcu_tools/gui/app/autofluxdep/README.md#workflow-與執行契約)，
但不匯入 concrete experiment 或 `catalog.py`，也不決定 workflow order 或 placement。
`acquire.py` 組合 point setup、Schedule outcome 與 SNR early-stop 的共用機制；實驗 Node
負責自身的 acquire mode、fit dispatch 和 Patch emission。
共用 Plotter 使用 caller-owned Plots 的既有 axes，保留 decay、scan 與 landscape 三種
資料投影。建圖和更新都由 GUI 主線程呼叫；Figure 與 canvas 的生命週期屬於 app。

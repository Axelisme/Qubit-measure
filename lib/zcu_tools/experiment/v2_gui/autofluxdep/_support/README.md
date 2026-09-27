# `experiment/v2_gui/autofluxdep/_support/`

**Last updated:** 2026-09-27 — experiment entry relocation

這個 private package 保留多個 Autofluxdep concrete experiments 共用的 acquire、Result、
Plotter、dependency/module/readout/timing defaults、schema、module values、OverridePlan 與
sweep/timing mechanics。只屬單一實驗的 cfg、fit 與 Patch policy 留在相應的 `<name>.py`。

`_support` 可以使用 [app 的 Node contract](../../../../gui/app/autofluxdep/README.md#workflow-與執行契約)，
但不匯入 concrete experiment 或 `catalog.py`，也不決定 workflow order 或 placement。
`acquire.py` 組合 point setup、Schedule outcome 與 SNR early-stop 的共用機制；實驗 Node
負責自身的 acquire mode、fit dispatch 和 Patch emission。

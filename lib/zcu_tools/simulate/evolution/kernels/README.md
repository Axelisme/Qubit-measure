# Evolution kernels

**Last updated:** 2026-10-28

**讀者：** framework-dev、agent。查找演化算法的數值 owner，以及 kernel 與 dispatch、scheduler 的接縫。

## 算法 owner

本目錄實作各 family 的 preparation、guard certificate 與 density propagation。
每個算法將數值實作與專用 helpers 留在自己的模組。
多個算法需要同一運算時，共用其數值 owner，不呼叫另一個 kernel class 的方法。
例如 `_secular.py` 擁有 rates 到 population generator 與 coherence decay 的換算，
`_flux_expansion.py` 擁有連續 flux 的插值與 Hamiltonian 累加。
完整 owner 對照見 [evolution README](../README.md#algorithm-ownership-and-reuse)。

## 內部接縫

[dispatch](../_dispatch.py) 決定適用條件、成本、嘗試順序與 builder／frame。
Kernel 回報 guard acceptance 與可交給 apply 的數值產品。
Dispatch 合併 family 的唯讀 `RowPlan`，scheduler 在 `ApplyContext` 原樣傳遞它。
Apply 使用已決定的 plan，不依這次收到的子集重新分類。

[kernel.py](../kernel.py) 定義 `Kernel`、`Prepared`、`RowPlan` 與 `ApplyContext`。
這些型別只供 evolution 內部合作。
外部 caller 從 `simulate.evolution` package root 提供 model source、timeline、初態與預算，
不直接挑選本目錄的算法。

Preparation 的持有與釋放由 evolve 的工作生命週期管理，不把 input-bound 產品留在長駐 kernel state。
Plan、guard handoff 與資源限制見 [evolution README](../README.md#row-plans-and-the-kernel-seam)。
黑盒邊界與算法內聚的決策見 [ADR-0078](../../../../../docs/adr/0078-evolution-module-boundary.md)。

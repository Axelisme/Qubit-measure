# Evolution kernels

**Last updated:** 2026-11-17，常數雙 tone endpoint kick、sampled 拒收與 t6 direct owners

**讀者：** framework-dev、agent。查找演化算法的數值 owner，以及 kernel 與 dispatch、scheduler 的接縫。

## 算法 owner

本目錄實作各 family 的 preparation、guard certificate 與 density propagation。
每個算法將數值實作與專用 helpers 留在自己的模組。
多個算法需要同一運算時，共用其數值 owner，不呼叫另一個 kernel class 的方法。
例如 `simulate/baths.py` 擁有 rates 到 population generator 的換算，
`_secular.py` 擁有 coherence decay，`evolution/state.py` 擁有 density 基底轉換。
`_flux_expansion.py` 擁有連續 flux 的插值與 Hamiltonian 累加。
`_workload_facts.py` 的 `model_keys` 供 Dyson 與 flux DOP853 共用 flux-level／shift-content 分組。
`_rwa_numeric.py` 擁有多個 RWA family 共用的 admission、retention、policy 與 Hamiltonian coefficients。
Fourier 係數與 omitted-edge endpoint kick 數值留在 `_rwa_fourier.py`。
`single_tone_rwa` 與常數雙 tone 的 `multitone_rwa` 都使用 resolved retention mask 的補集準備 kick。
`multitone_rwa` 在每個 row 端點先加總所有 tone 的 kick，再作用於 density。
末端 kick 與演化使用相同的 row duration；sampled row 不套用 kick。
Phase graph 與 covariant gate builder 分別留在 `_rwa_graph.py` 與 `_rwa_gate.py`。
`multitone_rwa` 的 family guard 只收 distinct-port constant pairs，拒收 sampled 多 tone 列。
`multitone_gate` 的 family guard 拒收所有列，保留 family 與共用 single-tone 數值 owner。
這是現行拒收契約，不預告恢復 admission。
其餘候選沿用既有定價，可能選到 `common_frame` 或 `dop853`。
`dop853.advance_endpoints` 擁有旋轉座標 density 積分，`dop853` 與 `t6` direct
沿用相同的每段 budget 容差規則。這是局部積分控制，不保證整條序列的最終誤差。
`t6` 的每一列都走 direct，不依 key 重用數選算法，也不掃描完整 acquire inventory。
Preparation 只打包 model operators；[`_t6_dispatch`](../_t6_dispatch.py) 保留 prepare-key 建構。
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

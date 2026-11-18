# Evolution kernels

**Last updated:** 2026-11-18，雙相位 gate 建構與 DP5(4)

**讀者：** framework-dev、agent。查找演化算法的數值 owner，以及 kernel 與 dispatch、scheduler 的接縫。

## 算法 owner

本目錄實作各 family 的 preparation、guard certificate 與 density propagation。
每個算法將數值實作與專用 helpers 留在自己的模組。
多個算法需要同一運算時，共用其數值 owner，不呼叫另一個 kernel class 的方法。
例如 `simulate/baths.py` 擁有 rates 到 population generator 的換算，
`_secular.py` 擁有 coherence decay，`evolution/state.py` 擁有 density 基底轉換。
`slow_envelope` 的耗散 DC channel 重用 `_dyson_model.py` 的 row-major generator
與 commutator，不改 secular bath 定義。非零載波仍使用 midpoint-held Strang steps。
`_flux_expansion.py` 擁有連續 flux 的插值與 Hamiltonian 累加。
`_workload_facts.py` 的 `model_keys` 供 Dyson 與 flux DOP853 共用 flux-level／shift-content 分組。
`_rwa_numeric.py` 擁有多個 RWA family 共用的 admission、retention、policy、model/WorkPlan coefficient adapter 與 facts。
`_rwa_fourier.py` 擁有 `RWACoefficients`、compiled `interaction_hamiltonian`、Fourier 係數與 omitted-edge endpoint kick 數值。
Package root 直接從 Fourier owner 匯出 record 與 assembler，`rwa_coefficients` 仍從 numeric 匯出。
公開名稱、record 欄位與數值行為不變。
`RWACoefficients.__module__` 改為 `zcu_tools.simulate.evolution.kernels._rwa_fourier`，
compiled `interaction_hamiltonian` 的 source location 也改到 `_rwa_fourier.py`。
舊 numeric owner 不留 compatibility shim。
`single_tone_rwa` 與常數雙 tone 的 `multitone_rwa` 都使用 resolved retention mask 的補集準備 kick。
`multitone_rwa` 在每個 row 端點先加總所有 tone 的 kick，再作用於 density。
末端 kick 與演化使用相同的 row duration；sampled row 不套用 kick。
Phase graph 與 covariant gate builder 分別留在 `_rwa_graph.py` 與 `_rwa_gate.py`。
`multitone_rwa` 的 family guard 只收 distinct-port constant pairs，拒收 sampled 多 tone 列。
`multitone_gate` 用完整 drive 建四個 Hilbert 子段，在兩個起始相位上做 raw Fourier 展開。
`_multitone_gate_phase.py` 用一張 5×6 網格擬合係數，不按包絡形狀或虛部分類。
`_gate_endpoints.py` 以 rtol/atol 2.75e-7/2.75e-9 積分節點並做 polar retraction。
`_multitone_gate_split.py` 對四組相位與初始 density 各做 tight reference 和當次 `Engine` budget 的 direct 積分。
其中三組在網格外，同時檢查內插與耗散分割。
四子段 playback 與 budget probe 都以同一份完整 Lindblad reference 量最大 density 元素誤差。
Guard 放行 playback 誤差不超過 `max(0.1 × min(budget, 1e-3), 1.5 × e_dop)` 的 key，`e_dop` 是 budget probe 的誤差。
Budget probe 特別準時，budget 換算的下限仍容許小誤差。
這是抽樣檢查，不能限制未抽樣的相位、density 或整條序列的誤差。
建構前逐 key 比價。每個冷 key 的建構價只算一次端點、Fourier 與兩組 split 積分，按使用次數攤提，另計 apply。
Guard 將通過的數值產品交給 scheduler，只活一次 evolve。拒收後不重建，交給其餘合格候選。單 tone 的數值分支不變。

`dop853.advance_endpoints` 擁有旋轉座標 density 積分。
`dop853` 的 CPU row 和符合 [OpenCL 條件](../README.md#gpu-backend)的 row，在時長 ≤10 µs 且 budget ≥1e-3 時走 DP5(4)，兩個局部容差都除以三。
[共用 solver](_dop853.py) 提供兩套 tableau 與 controller 常數。兩種方法在各自的裝置上共用積分迴圈、RHS 與 knot 停點。
`t6` direct、period fractional tail 與 `flux_dop853` 仍用 DOP853。
局部容差不保證整條序列的最終誤差；見 [DP5(4) 精度案例](../README.md#cpu-dp54-accuracy)。
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

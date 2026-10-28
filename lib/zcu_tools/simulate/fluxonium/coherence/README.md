# Fluxonium coherence

**Last updated:** 2026-10-28，manifold population-decay modes

**讀者：** framework-dev、agent。查閱 pair lifetime、manifold modes 與 Purcell 的計算邊界。

## Pair 與 manifold

`calculate_eff_t1_vs_flux_fast` 保留兩能階的 equilibrium lifetime，
`T1=1/(Γ10+Γ01)`。其他 transition 可由 `i, j` 指定。
`calculate_manifold_t1_vs_flux_fast` 納入最低 `qub_dim` 個能階的所有 pair。
兩者共用 spectral density、channel options、native operator cache 與溫度驗證。
時間使用 ns，能量使用 GHz，溫度使用 K。

Manifold 以 `R[target,source]` 建立 directional rates，再以
`Q=R-diag(R.sum(axis=0))` 推進 column population。
初態先取 Gibbs 布居，再交換 0、1，代表理想 π pulse。
Observable 是 `1-p0(t)`。純 dephasing 不改變布居，不進 Q。
高能態的熱激發與 cascade 可以使 manifold decay 比 0↔1 pair lifetime 短。
`qub_dim=2` 時只剩一個衰減模，其 lifetime 等於 pair。

## 回傳的模

公開入口在 `simulate.fluxonium` 與 `simulate.fluxonium.coherence`。
函式回傳 `list[list[Mode]]`，外層對應輸入 flux 順序。
`Mode` 是兩欄 NamedTuple：

- `t1_ns` 是非平衡模的正衰減時間，單位 ns。
- `amplitude` 是扣除平衡值後的 signed real weight。

每點的非平衡振幅總和為 1。函式按 signed amplitude 降序排列，
保留零振幅與負振幅，不取絕對值，也不合併近簡併模。
因此某個振幅可以大於 1。呼叫端自行選主模與診斷指標，
simulate 不回傳 dominant、secondary 或 fraction。

本徵分解使用 `numpy.linalg.eig`。函式把相對 generator spectral scale
小於等於 100 machine epsilon 的 eigenvalues 視為平衡模。
零損耗沒有衰減模，回傳空清單。
Eigenvalues 或振幅的虛部若超過其尺度的 1e-10，函式拋出 `ValueError`。
容差內取實部。數值容差不代表 oscillator cutoff 或 retained manifold 已收斂，
呼叫端仍須分別驗證兩者。

## 保留的其他路徑

Slow coherence helper 提供 scqubits 的 scalar lifetime 與 directional rates。
`calculate_directional_loss_rates` 回傳 constant Q 與 quasiparticle losses 的
multilevel directional rates，供演化模型使用。
Purcell 仍只計算 0↔1 的 resonator thermal-averaged lifetime，
不自動加入 manifold Q。T2 與 pointwise Q 的 caller 仍可使用 pair forward。
單位與損耗參數的定義見 [ADR-0071](../../../../../docs/adr/0071-coherence-units.md)。

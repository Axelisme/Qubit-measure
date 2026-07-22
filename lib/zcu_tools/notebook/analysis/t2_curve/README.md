# `t2_curve` 模塊重點文檔

**Last updated:** 2026-07-22 — f01 flux correction diagnostics

Fluxonium echo T2 vs. flux 的分析工具：把實測 `T2e` 先扣除 relaxation bound `1/(2*T1)` 後，以 rate model 同時擬合 first-order flux noise 與 readout thermal-photon shot noise。

## 單位慣例

- T1 / T2 / T1err / T2err: us。
- `domega_dflux`: rad/us/Phi0。
- `A_phi`: Phi0/sqrt(Hz)，在 echo 近似中以 `sqrt(ln 2) * A_phi * |domega/dflux|` 形成純退相干率。
- `kappa_over_2pi_mhz` / `chi_over_2pi_mhz`: MHz；函數內轉成 rad/us。
- 所有 `Gamma_phi` rate: 1/us。

## 檔案與角色

### `base.py` — notebook plotting / common analysis helpers

- `predict_f01_mhz(params, fluxs)` 與 `predict_domega_dflux(params, fluxs)`：提供 notebook 用的 fluxonium f01 與 `domega/dflux` 計算，避免每份分析 notebook 重複定義。
- `choose_current_scale(...)`：用 measured f01 對照 fluxonium model，在 mA/A 候選縮放中選擇較一致的電流單位。
- `dispersive_chi01_over_2pi_mhz(params, fluxs, bare_rf, g, ...)`：計算 readout-state dispersive frequency separation，供 photon shot-noise model 使用。
- `make_thermal_limit_table(...)`：建立 `n_th`、`Gamma_phi_th`、`T2_limit` 表。
- `calculate_t2_channel_curves(...)`：用 `T2FitResult`、dense `t_fluxs`、`domega/dflux`、`chi/2pi` 與可選 `T1CurveFit` 建立 `2*T1`、pure flux-noise、pure photon-shot-noise、effective T2 四組曲線。
- `plot_t2e_vs_flux(...)`、`plot_thermal_photon_t2_limit(...)`、`plot_flux_noise_sensitivity(...)`、`plot_t2_channel_curves(...)`：回傳 `(fig, ax)`，notebook 只負責決定存檔路徑與是否顯示。`plot_t2_channel_curves` 支援 `parameter_text`，讓圖內 legend 保留短機制名。
- `t2_parameter_text(fit_result, ...)`：產生放在座標軸外側的 fit parameter summary。

### `workflow.py` — staged T2 fitting workflow

- 固定 notebook 使用逐步 API，而不是單一黑盒：`load_t2_curve_context(...)` → `calibrate_t2_flux(...)` → `prepare_t2_dephasing_data(...)` → 單機制 probe → `fit_t2_curve(...)` → `build_t2_channel_curves(...)`。
- `plot_t2_flux_calibration(data)` 將 fit window 內每個 sample 的 raw flux 與 f01-corrected flux 畫在同一列，供檢查 half-flux 附近是否發生不合理 branch jump。
- `analyze_flux_noise_limit(...)`：在可選 fixed `n_th` 下做 flux-noise-only probe，回傳 `A_phi_init`、`A_phi_fit`、pointwise `A_phi(flux)` 與 summary table；fit residual 可使用 `MeasurementErrorPolicy` 與 `FluxResidualWeighting`。
- `analyze_photon_shot_noise_limit(...)`：在可選 fixed `A_phi` 下做 photon-shot-noise probe；預設用 pointwise minimum 作為 `n_th_init`，因為 photon-only fit/平均值會把 non-sweet-spot flux noise 吃進 `n_th` 而高估。`n_th_fit` 仍保留為診斷值；fit residual 使用和 combined fit 相同的 weighting 設定。
- `make_t2_fit_init(...)`：用單機制 probe 結果組出 combined fit 初值；photon channel 使用 `n_th_init`，不是 photon-only `n_th_fit`。`active_mechanisms=("flux_noise",)` 或 `("photon_shot_noise",)` 可只納入部分機制。
- `fit_t2_curve(...)`：只使用 fit window 內的實際 sample，不再加入人工 half-flux 點。`T2e err = NaN` 的點由 `MeasurementErrorPolicy(nan_policy="bin_median")` 補 effective error；`FluxResidualWeighting(mode="equal_flux_bin", ...)` 配合預設 `loss="linear"`，讓每個 occupied flux bin 對 least-squares loss 的總權重一致，避免 sample-dense 區域主導結果。
- `mechanisms_to_fixed_params(...)`：把 notebook 面向使用者的 `fixed_mechanisms` 轉成 fit layer 的 `fixed=("A_phi", "n_th")` 語義；固定代表該 channel 留在模型中但不擬合。
- `run_t2_curve_analysis(config, display_fn=None)` 保留為薄 wrapper，供需要一鍵重跑時使用；互動式 notebook 優先用逐步 API。

### `fit.py` — joint pure-dephasing fit

- `T2FitParams(A_phi=None, n_th=None)`：fit 初值與結果容器；只有非 `None` 的參數會納入模型，語義與 `t1_curve.T1FitParams` 的 white-list 一致。
- `fit_t2_noise_params(T1s, T2s, domega_dflux, chi_over_2pi_mhz, *, kappa_over_2pi_mhz, init, bounds=None, fixed=(), T1errs=None, T2errs=None, residual_mode="gamma", progress=False, ...)`：用 log-parameter least-squares 擬合 active `A_phi` 與/或 `n_th`。預設 residual 是純退相干率 `Gamma_phi`，其中 `Gamma_phi_obs = 1/T2 - 1/(2*T1)`，若提供 error 會傳播成 rate 權重。
- `fixed=("n_th",)` 表示 photon channel 仍在模型中、但固定 `n_th`；若不想納入 photon shot noise，建立 `T2FitParams` 時不要提供 `n_th`。
- `flux_noise_gamma_phi_per_us(A_phi, domega_dflux)`：回傳 first-order 1/f flux-noise echo rate。
- `thermal_photon_gamma_phi_per_us(n_th, kappa_over_2pi_mhz, chi_over_2pi_mhz)`：回傳 PRX Appendix A15 readout photon shot-noise rate。
- `thermal_photon_t2_limit_us(...)` 與 `equivalent_n_th_from_t2(...)` 供 notebook 畫 thermal-photon ceiling 與 half-flux 初值估算使用。

## 典型使用流程

1. 用和 `T1_curve.md` 相同的 f01 correction 取得 sample flux，並用 flux calibration plot 檢查 raw/corrected flux 是否只做就近校準。
2. 選定 T2 fit window，例如 `0.49 <= flux <= 0.53`，並設定 `MeasurementErrorPolicy` / `FluxResidualWeighting`。
3. 對 fit sample 計算 `domega/dflux` 與 dispersive `chi/2pi`；`NaN` error 留在 fit window 中，由 fit layer 補 effective error。
4. 先用 `analyze_flux_noise_limit(...)` 與 `analyze_photon_shot_noise_limit(...)` 分別看單機制上限/初值。
5. 用 `make_t2_fit_init(...)` 與 `fit_t2_curve(...)` 做 combined fit；`active_mechanisms` 決定模型白名單，`fixed_mechanisms` 決定保留但固定的參數。
6. 用 `build_t2_channel_curves(...)` 與 `plot_t2_channel_analysis(...)` 分別畫 pure flux-noise、pure photon-shot-noise、`2*T1` 與三者合併的 effective T2。

固定 notebook 不放函數定義；每個 stage 只保留用戶要調的參數與一個 step-level helper call。

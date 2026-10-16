# `simulate/fluxonium` 模塊重點文檔

**Last updated:** 2026-10-16，shared loss spectra 與 coherence units

基於 [scqubits](https://scqubits.readthedocs.io) 的 Fluxonium 量子比特數值模擬工具集,提供能譜、色散位移、矩陣元、相干時間與實驗參數預測等計算。

所有 scqubits 相關 import 皆採 **lazy import**,避免模塊載入時的冷啟動開銷。

## 核心慣例

- **`params: tuple[float, float, float]`** — `(EJ, EC, EL)`,單位 GHz。
- **`flux` / `fluxs`** — 歸一化磁通 Φ/Φ₀(0.5 為半通量點)。
- **`bare_rf`** — 裸諧振腔頻率 (GHz)。
- **`g`** — qubit-resonator 耦合強度 (GHz)。
- **時間單位** — ns;溫度 `Temp` 單位為 K。
- 回傳中含 `spectrum_data` / `sweep` 物件時,可回填至後續呼叫以重用本徵分解,避免重算。

---

## 子模塊一覽

### `energies.py` — 能譜

- `calculate_energy(params, flux, cutoff=40, evals_count=20)` — 單一磁通點的能階。
- `calculate_energy_vs_flux(params, fluxs, ..., spectrum_data=None)` — 沿磁通掃描的能譜。內部利用 Φ₀ 週期性與鏡像對稱 (`flux` mod 1, 折疊至 [0, 0.5]) **去重後計算**,再還原到原 flux 順序以加速。

### `matrix_element.py` — 算子矩陣元

返回 `[:return_dim, :return_dim]` 區塊;支援帶入預算好的 `esys` / `spectrum_data` 以重用本徵分解。

- `calculate_n_oper` / `calculate_n_oper_vs_flux` — 電荷算子 `n`。
- `calculate_phi_oper` / `calculate_phi_oper_vs_flux` — 相位算子 `φ`。
- `calculate_sin_phi_oper` / `calculate_sin_phi_oper_vs_flux` — `sin(αφ/2 + β)`,用於雜訊模型與特殊耦合。
- `calculate_system_n_oper_vs_flux` — 在 **qubit+resonator 聯合 Hilbert space** 中計算 n 算子的 dressed 矩陣元,使用 `identity_wrap` 把子系統算子提升到整個空間。

`*_oper_vs_flux` 依賴 scqubits 回填 `matrixelem_table`；若 scqubits 未產生矩陣元，函式會 fast-fail，而不是回傳 `None` 或讓後續 slicing 出現模糊錯誤。`calculate_system_n_oper_vs_flux` 對 sweep 中新增的 `n_oper` 也做存在性檢查，dressed index 無法標記時 raise `DressedLabelingError`，並在回傳前正規化成 `complex128` ndarray。

### `dispersive.py` — 色散位移 / chi

所有 sweep 函式共用 `ParameterSweep` + `labeling_scheme="LX"` 做 dressed-state 標記。

- `calculate_dispersive(params, flux, bare_rf, g)` — 回傳 `(rf|0⟩, rf|1⟩)`,由 dressed 能量差得到；scqubits 無法標記 dressed state 時 raise `DressedLabelingError`，錯誤訊息包含 bare state 與 flux。
- `calculate_dispersive_sweep` — 通用 sweep 介面,透過使用者傳入的 `update_fn(fluxonium, param)` 更新系統。
- `calculate_dispersive_vs_flux` — 上述的 flux-sweep 特化。
- **`calculate_dispersive_vs_flux_fast`** — scqubits-free 的 numpy 等效,**~9x**(繞過 `ParameterSweep`)。複用 `energies.py` 的 cos/sin 預算技巧算裸 fluxonium evals+evecs,組 composite `H_res⊗I + I⊗H_qub + g(a†⊗n+a⊗n†)` 對角化,dressed (0/1,i) 用 **bare-product-state overlap argmax** 標記。**逐點對齊 `calculate_dispersive_vs_flux` 到 0.00000 MHz**(含 avoided crossing,見 `tests/simulate/fluxonium/test_dispersive_fast.py`)。labeling 撞號則 raise `DressedLabelingError`;`prediction.py` 的 engine 負責 normal-path fallback。低能態 + dispersive regime(g 小)labeling 乾淨;強耦合/密能級才可能撞。介面與舊版略不同（無 `progress`，res_dim/qub_dim 預設 4/15）。舊 `calculate_dispersive_vs_flux` 保留給 notebook 與 fallback。
  - **flux-independent operators 用 `_fluxonium_operators` @lru_cache(params,cutoff,dim)**：`cos_phi_operator`/`sin_phi_operator` 走 scipy `cosm`/`sinm`(各一次 `expm`),fresh `Fluxonium` 每次 ~84ms;memoize 後重複呼叫(GUI live tuning 拖 r_f/g、sample 單點)只跑 per-flux recombination+eigh(~1ms/flux)。回傳是 cache 自有 array,**caller 不可 mutate**(fast 路徑只讀)。numerically identical(只 memoize)。
- `calculate_chi_sweep` / `calculate_chi_vs_flux` — 直接回傳 scqubits 的 `sweep["chi"]`(subsys1=fluxonium, subsys2=resonator)。
- 注意 dispersive vs chi 兩組函式在 `HilbertSpace` 中 **subsys 順序相反**(dispersive 是 `[resonator, fluxonium]`,chi 是 `[fluxonium, resonator]`)。

### `coherence/` — T1 與 Purcell

- `calculate_eff_t1(_with)` / `calculate_eff_t1_vs_flux(_with)` 包裝 scqubits，回傳 physical ns，不再外乘 2π。`_with` 接受既有 Fluxonium 與 eigenpairs／spectrum data，供重用本徵分解。
- `calculate_eff_t1_fast` / `calculate_eff_t1_vs_flux_fast` 使用快取算子與批次 numpy 計算，和 slow 路徑使用相同的物理 rates，`T1 = 1/Σrate`。支援 capacitive、inductive、charge impedance、flux bias line 與 quasiparticle channels。Fast 固定 `total=True`；不支援的 channel／options 直接報錯，需要單向 rate 時使用 slow 路徑。
- Loss spectra 由 `simulate.spectra` 提供，與 joint directional baths 共用。Coherence 仍合併兩方向，不把 equilibrium rate 當單向 jump；預設 Q 與 scalar／array、callable options 不變。
- 絕對單位由 SI LC energy lifetime 與 raw scqubits regressions 驗證，fast／slow 相等不是唯一 oracle。半通量附近 matrix element 可能因 parity 消失，應比較接近零的 rate，不能只比較巨大 T1 的相對差。
- `calculate_purcell_t1_vs_flux(..., progress=True)` 分別對 qubit e→g 與 g→e rates 做 resonator thermal average，回傳 equilibrium `T1=1/(Γeg+Γge)`，單位 ns。只納入 0↔1，不包含高能態 leakage，也不代表來源論文的完整 loss 模型。
  - 底層 module-public `purcell` 仍回傳指定 initial→final groups 的 directional escape lifetime。高階 sweep 重用同一本徵分解求兩個方向，再合併 rates。
  - `kappa` 是 linewidth κ/(2π)，單位 GHz；rate 使用 `2π kappa`，單位 ns⁻¹。
  - 熱人口使用 `hν/(k_B T)`。依 dressed-state energy difference 篩選 bath 方向，這與 qubit e→g／g→e 方向不同。能量上升取 a† 的矩陣元平方並乘 n_B，下降取 a 的矩陣元平方並乘 n_B+1；零溫仍保有允許通道的 spontaneous decay。
  - 保留原 frequency-selected bath 模型，不改成完整 `X=a+a†`。Creation／annihilation 的矩陣元取絕對值平方，熱因子另乘在 rate 上。
  - `progress=False` 暫時關閉 scqubits sweep progress，結束後恢復。
- 單位與模型取捨見 [ADR-0071](../../../../docs/adr/0071-coherence-units.md)。舊 fitted Q／x_qp 需重新擬合或審核，不自動換算歷史資料。

### `prediction.py` — `FluxoniumPrediction` engine

Prediction policy 由本模組擁有。GUI/session/notebook adapter 使用 engine，不各自重寫 fallback 或 cache identity。

- `PredictionResolution(qub_dim, qub_cutoff, res_dim)` — typed Hilbert-space resolution;GUI adapter 固定用 app default,notebook/tests 可注入。
- `FluxAffineMap` — value↔flux affine 的單一實作,`flux_period == 0` fast-fail。
- `FluxoniumPrediction.predict_dispersive(...)` — 包 fast path + scqubits fallback,回傳 `DispersivePredictionResult(lines, backend)`;`used_fallback` 是輕量 provenance,GUI normal path 不需 catch `DressedLabelingError`；只在 fast dressed labeling 失敗時 fallback，scqubits 或輸入錯誤仍傳給 caller。
- `FluxoniumPrediction.bind_flux_axis(fluxs)` — 建 `FluxoniumPredictionSession`,axis copy 由 engine 持有,cache key 只含 `(g, bare_rf, return_dim)`;controller 在 params/axis 變化時重建 session。
- `predict_frequencies_mhz` / `predict_matrix_elements` — session predictor dialog 的批次曲線 helper,共用 engine affine 與一次 sweep 多 transition 的計算。

### `predict.py` — `FluxoniumParams` 與 `FluxoniumPredictor`

SQUID transmon 的 `TransmonParams` 與 `TransmonPredictor` 在 `simulate/transmon/`，共用 `simulate.predictor` 的中性 query 契約與 `FluxAffineMap`；`simulate.parameters` 列出前端可選的模型名稱。

FluxoniumParams 是可未知的具名輸入。EJ／EC／EL 用 GHz，flux_half／flux_period／flux_bias 用 native device units。missing() 列缺項；build_predictor() 只接受完整且有效的輸入，每次建立独立 model，不寫來源。from_mapping() 解析具名數值，省略或 null 保持未知，拒絕未知欄位、bool、非數字及非有限值。to_dict() 回具名的 detached FluxoniumParameterValues，保留 None。Wire adapters 使用這兩個公開介面，不自行推斷參數內部形狀。

應用層預測器,連接實驗電流/電壓值與理論模型。

- 建構: `(params, flux_half, flux_period, flux_bias)`;另可 `FluxoniumPredictor.from_file(result_path, flux_bias)` 經 `resources.qubit_params.QubitParams` 從 `fluxdep_fit` 結果載入，缺少 `fluxdep_fit` 時 raise `ValueError`。
- 座標轉換: `value_to_flux` / `flux_to_value` 委派 `FluxAffineMap`,使用
  `flux = (value + bias − flux_half)/flux_period + 0.5`。
- `predict_freq(cur_value, transition=(0,1))` — 回傳 **MHz**;輸入支援 scalar 或 array。scalar/array 都委派到 `FluxoniumPrediction.predict_frequencies_mhz` 的批次路徑，不保留共享 mutable `Fluxonium` instance；reverse transition 仍回傳帶符號頻差。
- `predict_matrix_element(cur_value, transition, operator="n"|"phi")` — 回傳 `|⟨i|O|j⟩|`。scalar/array 都委派到 `FluxoniumPrediction.predict_matrix_elements`；facade 仍只支援含 level 0,1 的 transition，level≥2 以 `ValueError` fast-fail，高 level 支援屬功能擴充另案。
- `calculate_bias(cur_value, cur_freq, transition)` —
  以自身的精確頻譜反解，回 `simulate.predictor.CalibrationResult(flux_bias, frequency_mhz, clamped)`。
  枚舉所有根與週期／鏡像等價 bias，選 `|bias|` 最小者，不依目前 bias 選根。
  不可達目標回全域最近的實際頻率，clamped=true，同距選最小 `|bias|`。
  可達時 clamped=false。無效輸入拒絕，數值失敗傳遞，不警告後保留原值。
  共享 solver 的分支掃描、極值細化與容差見 [transmon 數值政策](../transmon/README.md#calibration)。
- `update_bias(flux_bias)` — 就地更新 bias。
- `clone()` — 返回同參數的新實例(不共享 `Fluxonium` state)。

### `branch/` — 額外模擬

`floquet.py`, `full_quantum.py`(本 `__init__` 未匯出,屬實驗性/特殊用途)。

`full_quantum.py` 的 branch population helpers 要求 caller 明確傳入 positive `upto`;不提供預設 photon cutoff,避免省略參數時才在 runtime 以無效預設值失敗。

floquet 效能要點(design search 的 snr stage 主成本即在此):
- `FloquetBranchAnalysis` / `FloquetWithTLSBranchAnalysis` / `FloquetDualCouplingBranchAnalysis` 共用 private base class 內的 branch-following、quasi-energy 與 population 計算；子類只定義 drive Hamiltonian、drive args 與 bare label state。
- **photon 層刻意 serial**:單個 FloquetBasis 建構 ~3ms,低於 joblib loky dispatch overhead,
  平行反而慢 1.6x。平行度放在呼叫方的 **cell 層**(design/search.py `calculate_snr` 的
  `Parallel(n_jobs=-1)`),勿在 photon 層加回 joblib。
- **solver options 可注入**:analysis 類別 `__init__(solver_options=None)` 預設 = qutip 預設
  (bit-exact,mist overlay 呼叫方不受影響);**snr 入口**(`calc_branch_infos`/`calc_ge_snr`/
  `snr.calc_snr`)預設 `SNR_SOLVER_OPTIONS`(rtol=1e-3/atol=1e-5,snr[-3] rel_err ~6e-5,
  崩潰邊界在 rtol≥5e-2),傳 `None` 回嚴格。`calc_branch_infos_with_tls` 預設嚴格(餵 mist
  精度敏感分析)。
- **微擾 TLS 掃頻**(`calc_floquet_fourier_melem` / `calc_tls_resonance_map`):對已算好的
  無 TLS `fbasis_n` 做純代數後處理,以 Lorentzian 加權 `|g_tls·M_k|²` 對 E_tls 軸免費掃頻,
  取代 `calc_branch_infos_with_tls` 的逐 E_tls 全掃(mist_tls_analysis.md 交付物 #2)。
  map 的 argmax 由最強矩陣元的 pair 支配,判讀特定過程(如 e→g+TLS↑ 的 `(1,0)` pair)時
  應限縮 `branch_pairs` 而非直接取全域 argmax。Fourier 取樣會先預熱 requested time 的
  qutip propagator cache,使 cold/warm `FloquetBasis.mode(t)` 呼叫使用同一組 cached propagator；
  跨呼叫比較矩陣元時仍需保留 qutip solver 容差。Fourier matrix element 的
  mode-stack / `einsum` / mean 邏輯由 shared helper 擁有,單點 `calc_floquet_fourier_melem`
  與 TLS resonance map 共用同一套 sampled-mode 後處理；TLS map 只在外層預先建立
  harmonic phase arrays。
- 結果 **bit-exact deterministic**(同參數重跑 spread=0),golden 測試見
  `tests/simulate/fluxonium/branch/test_floquet.py`；TLS / dual-coupling 路徑也有小維度 characterization 測試。strict path golden 對目前
  Python / numpy / qutip / BLAS 組合敏感；依賴組合改變導致 ULP 級漂移時，重錄 golden 並保留
  `abs=1e-12`，不要用放寬 tolerance 掩蓋演算法變動。
- 呼叫方契約:design/search 的 `calculate_snr` **只算 `valid==True` 列**(其餘 NaN),
  必須在 `avoid_*` cheap filter 之後呼叫(design.md 已同步順序)。

---

## 典型使用流程

```python
from zcu_tools.simulate.fluxonium import (
    FluxoniumPredictor,
    calculate_energy_vs_flux,
    calculate_dispersive_vs_flux,
    calculate_eff_t1_vs_flux,
)

params = (EJ, EC, EL)
spec, E = calculate_energy_vs_flux(params, fluxs)             # 重用 spec
rf0, rf1 = calculate_dispersive_vs_flux(params, fluxs, rf, g)
t1 = calculate_eff_t1_vs_flux(fluxs, noise_channels, T, params)

predictor = FluxoniumPredictor.from_file("result/Q1.hdf5", flux_bias=0.0)
f01 = predictor.predict_freq(current_value)                    # MHz
```

## 效能 / 數值注意事項

- 能譜/矩陣元計算成本隨 `cutoff`, `evals_count`, `res_dim × qub_dim` 顯著上升;預設值已在準確度與速度間取平衡,調整前先確認。
- `calculate_energy_vs_flux` 的去重邏輯 **只對 `fluxs % 1` 後落在 [0, 0.5] 的範圍有效**;若需要跨週期非對稱量(如電流偏移),請使用 `FluxoniumPredictor` 處理座標。
- sweep 類函式透過 `scq_settings.scq_progress(progress=progress)` 暫時切換 `scq_settings.PROGRESSBAR_DISABLED`,可藉 `progress=False` 靜音；中途例外會還原全域設定。
- `_with` 變體能大幅降低批次呼叫成本 — 優先使用。

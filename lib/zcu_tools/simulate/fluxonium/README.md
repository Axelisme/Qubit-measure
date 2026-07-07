# `simulate/fluxonium` 模塊重點文檔

**Last updated:** 2026-07-07 — floquet Fourier helper reuse

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

- `calculate_eff_t1(_with)` / `calculate_eff_t1_vs_flux(_with)` — 包裝 `scqubits.t1_effective`,接受 `noise_channels` list,計算 1→0 有效 T1。
  - **單位修正**:scqubits 回傳 `ns/rad`,此處已乘 `2π` 轉為 `ns`。
  - `_with` 版本接受已建好的 `Fluxonium` 物件與可選的 `spectrum_data`,供重複呼叫時重用。
  - 內部用 `scq_settings.scq_t1_default_warning(False)` 暫時關閉 `scq_settings.T1_DEFAULT_WARNING`,並以 1 秒門檻啟動 `tqdm` 進度條(短任務不顯示)。
- **`calculate_eff_t1_vs_flux_fast` / `calculate_eff_t1_fast`**(`coherence_fast.py`)— scqubits-free numpy 等效,**~60x**(13.6s→0.36s/100flux/4ch)。**逐點對齊 scqubits 到 ~1e-13 相對**(見 `tests/simulate/fluxonium/test_coherence_fast.py`)。**兩個瓶頸都繞掉**:(1) eigensolve 用 `energies.py` 的 cos/sin 預算技巧自己 `eigh`(`H=-EJ·cos(φ+β)`);(2) flux-dependent noise 算子(`dH/dflux=-2πEJ·sin(φ+β)`、`sin(φ/2+π·flux)`)用三角恆等式 `sin(αφ+β)=sin(αφ)cos(β)+cos(αφ)sin(β)` 從一次算好的 `sin/cos(αφ)` 重組,**消掉 scqubits 每 flux 一次的 `scipy.linalg.sinm`**(profiling:flux_bias_line 2.6s + quasiparticle 5.6s/100flux 全在 sinm)。**5 個 noise spectral density 公式逐字移植自 `scqubits/core/noise.py`**(常數 `R_k=h/e²`、`to_standard_units=×1e9`、`calc_therm_ratio`、各 channel 的 Q_cap/Q_ind/Y_qp 預設),Fermi rate = `|⟨i|op|j⟩|²·[S(ω)+S(-ω)]`、`T1[ns]=2π/Σrate`。介面同 scqubits(`noise_channels` 為 str 或 `(str,opts)`,支援 per-channel Q_cap/Q_ind/Y_qp/M/Z/x_qp/Delta);不支援的 channel raise `UnsupportedNoiseChannelError`。**fast-fail kwarg 驗證**:固定 `total=True`,傳 `total=False`、頂層未知 kwarg、或 per-channel opts 放了該 channel 不接受的鍵(如 `t1_capacitive` 放 `M`),都 raise `UnsupportedNoiseOptionError`(不靜默丟棄,免回傳 wrong-but-plausible T1)。每 channel 的合法 opts 在 `_CHANNELS` 第二元素(frozenset)。**flux-independent 算子用 `_t1_operators` @lru_cache(params,cutoff,qub_dim)**(對標 dispersive 的 `_fluxonium_operators`;noise opts/Temp/transition 不進 key,不同 noise 設定共用同份算子)——同 params 重複呼叫(t1_curve 掃 noise_values 的典型場景)固定成本(~110ms 的 scqubits 算子建構)只付一次。**per-flux loop 已批次化**:H 堆成 `(N,dim,dim)` 一次 batched `np.linalg.eigh`,算子變換/Fermi rate 全向量化。warm cache 下各 N 比 scqubits 慢路徑快 ~7-10x;cold + 小 N(<~50)單次呼叫仍可能比慢路徑慢(一次性 ~110ms,刻意不加 N 閾值 dispatch 以免複雜化語義)。**半通量 0.5**:quasiparticle `sin(φ/2)` 矩陣元 parity-vanish → T1~∞(~1e32 ns),fast 與 scqubits 都同意 rate≈0(殘差相對差大但物理上都是無限,非 bug)。舊 scqubits 版全保留。
- `calculate_purcell_t1_vs_flux` — 自訂 Purcell 通道 T1,手動求和 resonator 熱態佔據下的 `⟨0,n'|a†|1,n⟩` / `⟨0,n'|a|1,n⟩` 矩陣元:
  - `P_res(n) = (1 − e^{−βℏωr}) e^{−nβℏωr}`
  - `Γ↑ = Σ P_res(n) κ n_th(ΔE) |⟨…|a†|…⟩|²`
  - `Γ↓ = Σ P_res(n) κ (n_th(−ΔE)+1) |⟨…|a|…⟩|²`
  - 回傳 `1/(Γ↑+Γ↓)` (ns)。

### `prediction.py` — `FluxoniumPrediction` engine

ADR-0029 的 production seam。GUI/session/notebook adapter 只接這層,不各自重寫 prediction policy。

- `PredictionResolution(qub_dim, qub_cutoff, res_dim)` — typed Hilbert-space resolution;GUI adapter 固定用 app default,notebook/tests 可注入。
- `FluxAffineMap` — value↔flux affine 的單一實作,`flux_period == 0` fast-fail。
- `FluxoniumPrediction.predict_dispersive(...)` — 包 fast path + scqubits fallback,回傳 `DispersivePredictionResult(lines, backend)`;`used_fallback` 是輕量 provenance,GUI normal path 不需 catch `DressedLabelingError`。
- `FluxoniumPrediction.bind_flux_axis(fluxs)` — 建 `FluxoniumPredictionSession`,axis copy 由 engine 持有,cache key 只含 `(g, bare_rf, return_dim)`;controller 在 params/axis 變化時重建 session。
- `predict_frequencies_mhz` / `predict_matrix_elements` — session predictor dialog 的批次曲線 helper,共用 engine affine 與一次 sweep 多 transition 的計算。

### `predict.py` — `FluxoniumPredictor`

應用層預測器,連接實驗電流/電壓值與理論模型。

- 建構: `(params, flux_half, flux_period, flux_bias)`;另可 `FluxoniumPredictor.from_file(result_path, flux_bias)` 經 `meta_tool.QubitParams` 從 `fluxdep_fit` 結果載入，缺少 `fluxdep_fit` 時 raise `ValueError`。
- 座標轉換: `value_to_flux` / `flux_to_value` 委派 `FluxAffineMap`,使用
  `flux = (value + bias − flux_half)/flux_period + 0.5`。
- `predict_freq(cur_value, transition=(0,1))` — 回傳 **MHz**;輸入支援 scalar 或 array。scalar/array 都委派到 `FluxoniumPrediction.predict_frequencies_mhz` 的批次路徑，不保留共享 mutable `Fluxonium` instance；reverse transition 仍回傳帶符號頻差。
- `predict_matrix_element(cur_value, transition, operator="n"|"phi")` — 回傳 `|⟨i|O|j⟩|`。scalar/array 都委派到 `FluxoniumPrediction.predict_matrix_elements`；facade 仍只支援含 level 0,1 的 transition，level≥2 以 `ValueError` fast-fail，高 level 支援屬功能擴充另案。
- `calculate_bias(cur_value, cur_freq, transition)` —
  在 `[cur_value ± 0.25·flux_period]` 本地窗口中以 `scipy.root_scalar(method="secant")` 找使預測頻率吻合的點,然後枚舉 **週期 + 鏡像對稱** 的所有等價 bias,回傳 `|bias|` 最小者。root find 失敗或落在窗口外時會 `RuntimeWarning`，並用目前值作 fallback。
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
- sweep 類函式透過 `scq_settings.scq_progress(progress)` 暫時切換 `scq_settings.PROGRESSBAR_DISABLED`,可藉 `progress=False` 靜音；中途例外會還原全域設定。
- `_with` 變體能大幅降低批次呼叫成本 — 優先使用。

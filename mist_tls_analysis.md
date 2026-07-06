# MIST TLS Analysis 研究計劃

**目標**：用模擬解釋 `notebook/plot/plot_1.8mA_fs9585.ipynb` 中，excited-init population 在 saturation readout photon number n≈87 出現 dip 的現象，驗證（或否證）「TLS 經 ac-Stark shift 被拉入共振」的假說（機制同 Bista et al., PRApplied 25, 034058 (2026), `wjdb-4814.pdf`）。

## 背景對照（論文 vs 本實驗）

| | 論文 (device A) | 本實驗 (Q12_2D[6]/Q1 @ 1.8mA) |
|---|---|---|
| 現象 | P(e\|e) dip @ n̄≈7 | excited population dip @ n≈87 |
| 機制假說 | ac-Stark shift 把 \|e,0⟩–\|g,1_TLS⟩ 拉入共振 | 待驗證 |
| 模型 | driven fluxonium-resonator + TLS，Lindblad time-domain (Eq.1+2) | Floquet branch analysis（`floquet.py`），無耗散 |
| TLS 參數 | Δ_TLS/2π=411 MHz（qubit 401.5 MHz），g_TLS/2π=1.3 MHz | E_tls、g_tls 未知 → 本計劃要定出 |
| 佐證 | flux 掃 qubit spectroscopy 看到 ~409 MHz TLS | `Database/Q12_2D[5]/Q1/migrated_20260706/Q1_flux_tls_3.canonical.hdf5` fluxdep（cell 30）可比對 |

論文關鍵物理（供模擬設計）：
- TLS 耦合形式 `g_TLS · X̂_TLS · n̂`（已在 `FloquetWithTLSBranchAnalysis` 實作）。
- TLS 假設初始在 ground → 主要造成 **excited-init** 的 dip，ground-init 幾乎不受影響。本實驗若 ground-init 在 n≈87 無 feature，即為 TLS 假說的第一個 consistency check。
- 共振條件：Stark-shifted f_01(n) = E_tls（或 multi-photon / sideband 變體，論文 Appendix Q）。
- dip 位置對 flux 敏感（f_01 隨 flux 變 → 共振 n 移動）；不同 cooldown 會變 → materials defect。

## 核心策略：不要盲掃 TLS 頻率

TLS 掃頻慢的根本原因：`FloquetWithTLSBranchAnalysis` 的 Hilbert dim 是 2×qub_dim=80，FloquetBasis 的 ODE 積分成本 ~O(dim²·steps)，而且每個 E_tls 都要重跑整條 photon 軸。**但共振條件本身不需要 TLS 在 Hamiltonian 裡**：

> 先用**無 TLS** 的 `FloquetBranchAnalysis`（notebook 已在跑，一條 photon 軸 ~秒級）算出 Stark-shifted 分支能量 f_ij(n)，反解「哪個 E_tls 會在 n≈87 共振」。TLS 頻率從掃描變成**代數反解**，全量 TLS Floquet 只需在少數候選點驗證。

## Phase 0 — 實驗特徵量化（純資料，無新模擬）

從 notebook 既有結果提取，寫進分析 notebook 開頭作為 ground truth：

1. dip 中心 n_dip、寬度 Δn（從 MIST overnight excited-init 曲線，cell 15/16 的 0–150 zoom）、深度（population 掉多少）。
2. ground-init 與 steady 在同一 n 是否有對應 feature（TLS 假說預期：ground-init 無、或遠弱於 excited）。
3. T1-with-tone（cell 32–36）在 n≈87 的 rate panel 是否同步出現 Γ 峰 —— dip 若是 TLS 造成，excited→other/ground 的 rate 應在該處 resonance-like 增強。
4. 校準常數集中列出：params (EJ,EC,EL)、flux、g、bare_rf、readout_f=sample_f?、ac_coeff、fwhm(κ)。注意 mist_ns=[13,31,87,313] 中其它 dip（13/31/313）也記錄特徵，之後判斷哪些屬 fluxonium 內部 branch 共振（無 TLS 模型應能解釋）、哪些需要 TLS。

## Phase A — 無 TLS Floquet：候選 TLS 頻率反解（快）

沿用 notebook cell 20 的 `FloquetBranchAnalysis` 結果（branch 0..24，photon 軸已算過）：

1. 算 f_01(n) = E_1(n) − E_0(n)（quasi-energy，含 Stark shift）。
2. 共振條件族（都要 mod r_f fold 進 Floquet Brillouin zone，沿 cell 22 的 `round_to_nearest` 做法）：
   - 直接共振：E_tls = f_01(n_dip)
   - sideband：E_tls = |f_01(n_dip) ± k·r_f|（k=1,2）
   - multi-photon（論文 App. Q）：E_tls = m·f_01(n_dip)（m=2,3）
   - 也對 f_0j / f_1j（j 為鄰近 branch）做同樣反解，涵蓋「TLS 接的是更高階躍遷」的可能。
3. 產出：候選 (E_tls, 機制) 清單，每個附上「若為真，dip 應隨 flux 如何移動」的預測。
4. 交叉比對 `Database/Q12_2D[5]/Q1/migrated_20260706/Q1_flux_tls_3.canonical.hdf5` fluxdep 譜（cell 30）：候選 E_tls 附近是否真的有 flux-independent 的水平線（TLS 特徵）。**這一步是最強的獨立證據**，若譜中直接看得到 TLS 線，E_tls 就定死了，Phase B 只做確認。
5. 同時檢查：n≈87 處無 TLS 模型本身是否已有 branch 0/1 與高 branch 的 avoided crossing（即「不需要 TLS」的 null hypothesis）——看 branch_infos 在該處是否 index 跳變、`calc_branch_populations` 的 population 是否已離開 qubit subspace。**若 null hypothesis 成立，計劃到此結束**，結論為 fluxonium 內部 MIST 而非 TLS。

## Phase B — 帶 TLS Floquet：定點驗證（慢，但只跑少數點）

只對 Phase A 存活的候選 E_tls 跑 `calc_branch_infos_with_tls`：

1. **先做收斂/效能 benchmark**（單一 photon 點）：
   - qub_dim 收斂測試：40→30→20→16，看 n<150 區間 f_01、branch population 的偏差；TLS 只在 qubit subspace 附近作用，預期 qub_dim≈20 足夠 → dim 2×20=40，成本反而低於現行無 TLS 的 40。
   - solver tolerance：比較 strict vs `SNR_SOLVER_OPTIONS`（rtol 1e-3）在 dip 區的分支能量差異；branch tracking 只要 overlap argmax 不變即可用鬆 tolerance。
   - 記錄單點耗時，估算全計劃 CPU 時間再決定網格密度。
2. photon 網格：粗網格全域（Δn~5）+ dip 窗 [n_dip−20, n_dip+20] 細網格（Δn~1）；photon 軸各點獨立 → cell 層 `joblib.Parallel`（floquet.py 註解已明示 concurrency 放 cell 層）。
3. 對每個候選 E_tls：
   - 看 branch 1（excited）的 quasi-energy 在 dip 窗是否出現 avoided crossing（與 TLS-excited branch）。
   - `calc_branch_populations`（branchs=[0,1]，avg_times 一週期）：branch 1 的 qubit-subspace population 在 n_dip 處是否下陷、branch 0 是否不受影響。
   - 掃 g_tls（如 0.5/1/2/5 MHz 幾個點）對 dip 寬度/深度的影響，與實驗 Δn、深度比對定出 g_tls 量級（論文為 1.3 MHz 級）。
4. 產出：模擬 population vs n 疊在實驗 MIST 曲線上的對比圖（x 軸都是 photon number，經 ac_coeff 換算）。

## Phase C —（視 Phase B 結果）flux 依賴性驗證

若單 flux 點吻合，做殺手級測試：

1. 用 `Database/Q12_2D[5]/Q1/MIST/migrated_20260706/Q1_autofluxdep_animation@1.592mA_mist_e.npz`（cell 28，MIST over flux 2D 圖）：對 3–5 個 flux 值重複 Phase A 反解，固定 Phase B 定出的 (E_tls, g_tls)，預測 dip-n 隨 flux 的軌跡，疊在 2D 實驗圖上。TLS 頻率 flux-independent → 軌跡形狀完全由 f_01(n, flux) 決定，無自由參數。吻合即為 TLS 假說的決定性證據（對應論文 Fig.3 的邏輯）。
2. 不吻合的 fallback：考慮論文 Sec.V 的情境（不同 flux 接到不同 TLS）、或 multi-photon 機制，回 Phase A 換候選。

## 效能守則（貫穿全程）

- 無 TLS 全掃 + TLS 定點驗證，**永不做 (E_tls × photon) 二維全掃**。
- esys 在 caller 算一次傳入（`FloquetWithTLSBranchAnalysis` 已支援 esys 參數），避免每次重複 `eigensys`。
- population 計算才需要 `avg_times` precompute；只看能量/branch tracking 時傳 None。
- qub_dim 用 Phase B.1 的收斂結果，不沿用 40 的預設慣性。
- 若 Phase B 單點 benchmark 仍太慢（>數十秒/點），退一步用微擾估計替代全 Floquet：取無 TLS 的 Floquet modes |u_b(t)⟩，算 TLS 耦合矩陣元 g_tls·⟨u_b'|n̂|u_b⟩ 的時間平均 Fourier 分量，配合 quasi-energy 差 vs E_tls 的 detuning 畫「共振強度圖」——這對 E_tls 是純代數後處理，可免費掃頻。（此法可先實作為 sanity check，因為它本來就比全 Floquet 便宜。）

## 交付物

1. 新 notebook `notebook/plot/mist_tls_1.8mA.ipynb`（或併入現有 notebook 的新 section）：Phase 0–C 全部圖表。
2. 若微擾後處理法有效，抽成 `lib/zcu_tools/simulate/fluxonium/branch/` 下的函式（含 tests）。
3. 結論寫回本文件：dip@87 = TLS（附 E_tls, g_tls, 機制）或 = fluxonium 內部 branch 共振，及 13/31/313 各 dip 的歸屬表。

## 風險與判準

- **接受 TLS 解釋的門檻**：(a) 反解 E_tls 有 fluxdep 譜獨立佐證或 Phase C flux 軌跡吻合；(b) ground-init 無對應 dip 與模擬一致；(c) g_tls 落在物理合理量級（sub-MHz–MHz）。
- **否證路徑**：Phase A.5 null hypothesis 成立、或任何候選 E_tls 都需要非物理 g_tls 才能重現 dip 深度。
- 已知模型侷限（沿論文）：無耗散（Floquet 不含 Lindblad → 只能比對 dip 位置/寬度趨勢，不能定量比 population 絕對值）、單一 TLS、TLS 無溫度。若需要定量 population，Phase B 之後另評估小 Hilbert space（qub_dim~10 × TLS × 少量 resonator levels）的 mesolve time-domain 模擬成本。

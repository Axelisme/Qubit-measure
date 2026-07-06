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

## 結論（2026-07-06，cooldown [6] 既有資料分析）

詳細推導、圖表與數值在 `.agent_state/mist_tls/`（gitignored）：
`fluxdep_tls_crosscheck.md`、`mist_dip_temporal_drift.md`、
`ac_stark_photon_calibration.md`、`compromise_tls_simulation.md`、
`photon_distribution_broadening.md`。以下為判定與依據摘要。

### 判定：dip@n≈100 = TLS（證據等級：強，多線獨立吻合）

三條互相獨立、皆不依賴 MIST 影像本身的證據：

1. **躍遷鑑定與 calibration-free E_tls 讀出**。靜態譜給出 f01 = 4.546 GHz
   （notebook 的 folded −0.986 GHz 即 f01 − r_f），`Q1_flux_1`（1/23）與
   `ac_stark_pop`（2/1）中那條 4543–4545 MHz 的強線就是 f01 本身。因此
   dip gain 與 f01(gain) 譜線同軸可直接合成：2/1 dip（gain 0.1052）處
   量到的 f01 = **4341 MHz**，即 E_tls ≈ 4.341 GHz（folded ≈ 1.009 GHz），
   全程不經 photon 校準。
2. **fluxdep 譜 anticrossing 暗示**。`Q1_flux_1`（1/23）中 f01 線穿越
   ≈4350±6 MHz（flux ≈1.12 mA）處出現 ~4σ 的軌跡偏折與線形 smear，符合
   小 anticrossing 在粗 flux 步距（每步 ~10 MHz 線移動）下的樣貌；與
   (1) 獨立吻合。其它候選穿越點（4315/4245）平滑無事。
3. **時間漂移（TLS 招牌行為）**。六次 `1.800mA/mist/` singleshot 中，
   用高光子端 rolloff（fluxonium 內部特徵）做內部歸一化後：2/6 的 dip
   相對 rolloff 移動 +20%（gain），2/11 rolloff 正常但 dip 消失——
   fluxonium 內部 branch 共振無法如此，材料缺陷 TLS 的 spectral
   diffusion / telegraph 恰是如此。ground-init 無對應 feature（門檻 (b)）
   先前已確認。

### photon 校準（β）判定

以 2/1 `ac_stark_pop` 量到的 Stark-shifted f01 曲線（4543→4040 MHz，
gain 0→0.20）對無 TLS Floquet f01(n) 做形狀擬合（模型驗證：folded
f01(87) = −0.9859 與 notebook 一致）：

- **n = s·gain²，s = 9380 ±~10%（β_eff ≈ 0.74±0.08）**；gain 0.103 對應
  n ≈ 100（非 87）。
- **β≈1.14–1.19（ridge-flatness 解）被決定性排除**；原始 0.65 與
  compromise 0.825 分居兩側、皆在誤差邊緣。
- ridge-flatness 之所以偏大：flux map 的 ridge 被 TLS 時間漂移與校準鏈
  漂移共同扭曲（見下）。

### 重要系統誤差（影響所有絕對 n 錨定）

- **gain→photon 校準鏈一天內可漂 ~10%（gain 單位）**：2/3 下午→傍晚
  rolloff 從 0.1615 移到 0.1791。
- **TLS 本身漂移**：dip 等效位置 2/1–2/3 在 n≈82–111（多為 common-mode
  校準漂移），2/6 真實移遠（歸一化後 +44% in n），2/11 離開量測窗。
- 因此「把 2/6 flux map 錨定在 2/2 overnight 的 n=87.95」不成立；
  compromise 報告的 `E_anchor=1.026769` 失去依據，任何單時刻 E_tls 估計
  自帶幾十 MHz 漂移不確定度。0.995 vs 1.03 GHz 的差異在漂移範圍內。

### 寬度與 g_tls

- 相干 TLS anticrossing 的 Floquet dip 寬度對 g_tls 幾乎不敏感
  （g_tls 0.15→4 MHz 僅 0.20→0.25 photons），觀測寬度 ~15–20 photons
  不能用來定 g_tls。
- 時間平均資料（overnight、flux map）的展寬包含 TLS 漂移本身；單次
  run 的 dip 仍寬 ~15–20 photons，快機制（readout photon-number
  distribution，等效 σ≈6 photons / coherent-like scale 0.65–0.75）仍需要；
  deterministic ring-up 單獨不足（κ/2π≈5.25 MHz、0.68 µs 已近穩態）。
- g_tls 維持 paper-scale ~1.3 MHz 作為 mixing-strength 工作值，未定量。

### 各 dip 歸屬表（2026-07-06 定稿，詳見 `null_hypothesis_and_dip_attribution.md`）

無 TLS null-hypothesis 檢查（Phase A.5）：branch-1 population 在 n=13/31/90
全平（無內部特徵），n≳250 陡升——**高光子 rolloff 即 fluxonium 內部 branch
escape**，且以 s=9380 換算與觀測 rolloff（gain 0.16–0.18）定量吻合（β 的又一
獨立驗證）。

| dip（名目） | n（s=9380） | 歸屬 | 依據 |
|---|---:|---|---|
| 13 | ~13 | 非內部；TLS 候選 E≈4.52 GHz | null model 平坦；夜內穩定；未證實 |
| 31 | ~31–34 | 非內部；TLS 候選 E≈4.49 GHz | 特徵弱；未證實 |
| 87 | ~90–100 | **TLS #1，E≈4.34 GHz（folded ≈1.01 GHz）** | 三線獨立吻合 |
| 313 | ~320 | **TLS #2，E≈4.02 GHz** ＋ 底下疊內部 escape | ac_stark 圖直接解析出 avoided crossing（f01≈4020±20 @ n≈320） |
| rolloff | ≳250 | fluxonium 內部 branch escape | 無 TLS 模型定量重現 |

Overnight 2/02 時間解析（300 iterations）：夜內全部特徵穩定（主 dip
±1.6 photons、n13 < 1 gain step）——TLS 漂移是「日」尺度現象；同時夜內
σ≈1.6 遠小於瞬時寬度 ~15 photons，確認寬度來源仍是 readout photon-number
distribution 而非 TLS 漂移。2/11 的 overnight2 e-init 檔全為 NaN（中止），
無法比對。3920 MHz 附近另有一條緩斜率譜線（非 gain-independent，非單純
TLS），未鑑定。

### 侷限與後續

- E_tls#1 的 anticrossing 未被直接解析（1/23 掃描 flux 步距太粗）；本
  cooldown 已結束，無法補量。下次 cooldown 若重現 dip，依
  `.agent_state/mist_tls/fluxdep_tls_crosscheck.md` 的建議配方
  （窄窗 twotone × 細 flux 步距 ≤5 µA × 腔內光子 loading）一次定死
  E_tls 與 2g。（TLS #2 已在 ac_stark 圖中直接解析到 avoided crossing。）
- 定量 population / 寬度模型（mesolve 小 Hilbert space）未做。
- 交付物 #2 已完成：微擾掃頻法抽為
  `lib/zcu_tools/simulate/fluxonium/branch/floquet.py` 的
  `calc_floquet_fourier_melem` / `calc_tls_resonance_map`（含
  `tests/simulate/fluxonium/branch/test_floquet_perturbative.py`），並以
  1.8 mA 真實參數端到端驗證（(1,0) pair 峰值 0.987 GHz @ n=87）。判讀
  特定過程時應限縮 `branch_pairs`，不要取全域 argmax（最強矩陣元 pair 會
  支配 map）。

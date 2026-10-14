---
status: accepted
---

# ADR-0071 — Coherence 的頻率、時間與 loss parameter 單位

## 決策與落實狀態

2026-10-14，使用者要求「這個報告值得建立一個adr錨定，helper問題一起修」。本篇確立 simulate 與 notebook analysis 共用的物理單位契約。修正範圍包含 T1 forward helpers、pointwise Q extraction 與 Purcell，不包含 FTF 模擬環境實作。

Forward helpers、pointwise Q 與 Purcell 實作遵循本篇契約。T1 由 SI 絕對時間及 raw scqubits regressions 驗證。Purcell 高階 sweep 回傳雙向 equilibrium T1，底層保留 directional escape lifetime。驗證涵蓋獨立 SI／LC 弱耦合 equilibrium T1、零溫極限及 matched-pair detailed balance；40 mK 與 80 mK 的 regression 均能拒絕只算 e→g 的舊結果。Purcell 修正包含熱因子、spontaneous decay 與雙向 rate sum。

## 共用單位契約

- Hamiltonian 的 EJ、EC、EL 與 eigenenergies 表示 energy/h，單位 GHz。Transition frequency ν 也是 cycles/ns，不是 angular frequency。
- Angular frequency ω = 2πν，單位 rad/ns。轉成 SI 時，ν 乘 10⁹ 得 Hz，ω 乘 10⁹ 得 rad/s。
- 溫度以 K 表示。Thermal exponent 是 hν_Hz/(k_B T)，等價於 ℏω_SI/(k_B T)。不能把 cycles frequency 直接放進 ℏω。
- Decay rate Γ 以 ns⁻¹ 表示，population relaxation time 是 T1 = 1/Γ。多個獨立 loss channels 相加的是 rates。
- Q_cap、Q_ind 採對應物理 loss model 的 quality factor，不能重定義成吸收 2π 的 fitting scale。Pointwise quasiparticle 的 Q_qp 表示 1/x_qp，不等同於 resonator 的 energy quality factor。
- 以 GHz 計算的 scqubits coherence time 已是 ns。Caller 不再乘 2π；轉成 μs 才除以 1000。

Finite-temperature T1 必須交代是否合併 upward 與 downward rates。兩能級人口向熱平衡回復的 rate 是 Γ↓ + Γ↑，不是只取 downward rate。Fast 與 slow helper 比較時要使用相同的 total 設定。

Spectral density 需與其 noise operator、Fourier convention 及 frequency units 配對。若以 dimensionless operator 寫 Γ = |〈i|O|j〉|² S(ω)，其中 S 已包含 coupling prefactor，S 的單位須為 ns⁻¹。不能把 spectral prefactor 缺失的 2π 留到最外層 T1 補回。Forward prediction、pointwise extraction 與 combined fit 使用同一物理定義。

本篇不新增 package import 或依賴方向，因此不新增 import-linter contract。

## 為何不保留外乘 2π

Scqubits 官方 Coherence Times Guide 明示：

> The units of the returned decoherence times reflect the global units settings. By default, the global units are set to GHz, and the resulting depolarization and dephasing times are returned in units of 1/GHz = ns.

Scqubits 4.3.1 的 t1 先將 energy gap 乘 2π，capacitive spectral density 也包含從 EC/h 到 angular rate 的轉換。它最後直接回 1/rate；t1_effective 合併 rates 後取倒數，沒有留待 caller 完成的時間換算。

API Returns docstring 的「2π (system units)」措辭有歧義，不能單憑該句判定回傳 ns/rad。官方 guide、預設 scale=1 且標示 ns 的 plotting，以及下述 SI 檢查支持 physical ns。

### 獨立 SI 檢查

取 EJ=0 的 LC 極限，EC/h=0.3 GHz、EL/h=1.5 GHz、Q=10⁶。由 SI 電路定義：

- C = e²/(2EC) = 64.5674310822 fF。
- L = (ℏ/2e)²/EL = 108.974341871 nH。
- ω = 1/√(LC)，ν = 1.8973665961 GHz。
- Parallel loss 的 Q = ωCR，因此 energy-decay time RC = Q/ω = 83882.020174145 ns。

另用零溫 unsymmetrized voltage-noise spectrum S_V = 2ℏ/(CQ)，以及 analytic |n₀₁|² = hν/(16EC)，SI Fermi golden rule 給 Γ = (2e)²|n₀₁|²S_V/ℏ²，得到相同 lifetime。這兩個 reference 不依賴 scqubits 的回傳值。

在 1 mK 下，scqubits 4.3.1 的 capacitive T1 與單通道 effective T1 都為 83882.020174145 ns；修正前 repo helper 為 527046.276694730 ns，恰為前者的 2π 倍。此案例固定標準 Q 與 physical ns，排除 forward 和 inverse 彼此抵消的可能。

## 歷史修改與相容性取捨

Commit `65dcb64e2b404ff869507e292dbf45fbb1c9726b` 於 2025-11-12 有意在 forward T1 加入 2π，同時移除 pointwise Qcap／Qind spectrum 的 2π。這是成套修改，不應描述為任意多乘常數。

該 commit 沒有 dependency lock，scqubits 也未 pin。無法由套件發布日期推定當時實際安裝版本，更不能推定作者讀過哪份文件。已核對的 v4.2 與 v4.3.1 在 t1、t1_capacitive、t1_inductive、t1_effective 與 _effective_rate 的計算一致；這不等於重現了當年的環境。

保留原慣例能維持部分舊擬合曲線，代價是 fitted loss parameter 不再符合外部物理定義。同一組參數交給 raw scqubits 與 repo helper 也會得到不同時間。本決策選擇標準 Q 與 physical ns，接受需要重新審核舊 fit 的成本，不加入 legacy scale 開關。

## 修正與驗證邊界

T1 修正涵蓋 fast／slow forward、pointwise Qcap／Qind／Qqp，以及使用這些 helper 的 fitting 和 channel curves。驗證使用公開計算與擬合入口，不測 private functions。至少保留一個 SI／解析絕對時間 reference；fast 對 slow 的相等和由同一 helper 產生資料再擬合，都不足以證明單位正確。

Purcell 在非零有效耦合且有可用衰減通道時，零溫仍須有 spontaneous emission。熱人口使用 hν/(k_B T)。2026-10-14，使用者以「那選A吧」採用保留原模型的修正：以 dressed-state energy difference 篩選方向，能量上升只用 a† 與 n_B，能量下降只用 a 與 n_B+1。不改成完整 X=a+a† bath，也不將兩個 operator 的全方向 rates 分開相加。

2026-10-14，使用者對加入 g→e 並改變高階回傳值的方案回答「採用」。高階 calculate_purcell_t1_vs_flux 分別求 e→g 與 g→e 的 resonator thermal-averaged rates，回傳兩能階 equilibrium T1=1/(Γeg+Γge)。Notebook loss contribution 使用此 total rate，與其他 total=True T1 通道一致。底層 purcell 保留指定 initial→final 的 directional escape lifetime，供分方向計算。

此處 bath upward／downward 描述整體 dressed-state 能量交換，與 qubit 的 e→g/g→e 方向分開。Detailed balance 以 matched transition pair 驗證，不對多頻率 thermal aggregate 假定單一 Boltzmann ratio。零溫極限保持原結果；有限溫度新增 regression 必須能拒絕只算 e→g 的舊回傳值。

原 helper 的公式註解與 2025-07-16 commit `9b7ed611c` 的舊 `notebook_md/analysis/percell.md` 都記有 a†/n_B 與 a/(n_B+1)。它們與 Zhang et al. 的 Appendix G.5、G21／G22 相同。更早的 2025-02-24 commit `a8aaafc6488cbe390b6c98b665c5c87a4a356857`，`notebook/percell.ipynb` 同時使用相同公式及論文的 EJ/h=3.395、EC/h=0.479、EL/h=0.132 GHz、resonator 5.7 GHz、bath 60 mK。這是直接來源的強證據，但 Git 未記錄引用，不能排除經其他筆記轉引。

Zhang 的公式涵蓋任意 ℓ→ℓ′；本文的高階 helper 僅取 0↔1。原文還將熱激發到高能態納入其完整 Purcell loss 討論，所以本 helper 不宣稱重現該文 Fig. 3(a) 的完整 loss 曲線。新增高能態通道不屬於本次修正。

2026-10-14，使用者對兩票拆分、測試接縫及 kappa 建議回答「可以」。Purcell 的 kappa 表示 linewidth κ/(2π)，輸入單位 GHz；例如 0.001 表示 1 MHz linewidth。進入 rate sum 時使用 2π kappa，單位 ns⁻¹，不把輸入直接當成 decay rate κ。驗收包含零溫 spontaneous decay、有限溫度 detailed balance 與獨立弱耦合 reference。

不自動改寫歷史 samples 或 fitted parameters。固定溫度、單機制且 bounds 不活躍時，舊 Q_cap／Q_ind 可為物理真值的 1/(2π)，舊 x_qp 可為真值的 2π；有固定參數、多機制或 bounds 的 fit 不能據此批次換算。Helper 修正後，沿用舊 fit 會失去原有的誤差抵消，使用者須重新擬合或審核其來源。

Measured T1 不因理論 helper 修正而改變。以 measured T1 扣除 relaxation contribution 的 T2 擬合，與以 model T1 計算 T2 ceiling 的作圖路徑需分開判斷。本決策不授權改寫原始資料、不操作硬體，也不實作 FTF preset。

## 依據

查核日期為 2026-10-14。

- [Scqubits Coherence Times Guide](https://scqubits.readthedocs.io/en/latest/guide/noise/guide-noise.html) 明示 GHz 對應 ns。Latest 頁面會變動，版本計算依據另見下列 tag。
- [Scqubits v4.3.1 noise.py](https://github.com/scqubits/scqubits/blob/v4.3.1/scqubits/core/noise.py) 與 [v4.2 noise.py](https://github.com/scqubits/scqubits/blob/v4.2/scqubits/core/noise.py)。舊版 guide URL 的查核收到 403，不宣稱已讀取當年 guide。
- Repo 歷史 commit `65dcb64e2b404ff869507e292dbf45fbb1c9726b` 保存當年的成套修改。
- Helin Zhang et al., [Universal Fast-Flux Control of a Coherent, Low-Frequency Qubit](https://doi.org/10.1103/PhysRevX.11.011010), Physical Review X 11, 011010，2021。[arXiv:2002.10653v1](https://arxiv.org/abs/2002.10653v1) 的 Appendix G.5、PDF 第 10 頁、G21／G22 給出保留的 Purcell 模型；G22 後說明直接 e→g 與高能態熱激發 loss 的差別。
- [Fluxonium module README](../../lib/zcu_tools/simulate/fluxonium/README.md) 保存公開 helper 定位。實作與測試承接本篇的單位契約。

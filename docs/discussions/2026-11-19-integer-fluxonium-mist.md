---
date: 2026-11-19
status: active
superseded_by: null
related:
  - lib/zcu_tools/simulate/fluxonium/branch/floquet.py
  - lib/zcu_tools/simulate/fluxonium/branch/full_quantum.py
  - zcu_lab/simulation_envs/integer_fluxonium/field-guide.md
  - zcu_lab/recipes/mist_power_trace.py
  - zcu_lab/recipes/mist_drive_frequency.py
  - zcu_lab/recipes/mist_flux_dep.py
  - docs/discussions/2026-11-03-repeated-readout-chain.md
basis:
  - "repo@d1034e121（main）"
  - "數值結果來自一次性 qutip 5.3.0／scqubits 4.3.1 模擬：scqubits 能譜、非 RWA 量子化腔 branch analysis、QuTiP Floquet 與 Sambe gap、Floquet–Markov 速率、量子化腔 Lindblad。程式與輸出在未入庫的工作檔；本文寫入方法、參數、數字與結論"
  - "一個獨立核對 agent 以自寫程式重算 An 的 g／χ 校準、g↔18 位置與 E_C 敏感度、drive-frequency 線、平滑浴速率比、parity crossover 與 g↔9 線、Wang／Liu 參數自洽性，皆一致"
  - "Mencia et al., Integer Fluxonium Qubit, PRX Quantum 5, 040318 (2024), arXiv:2403.16780v2"
  - "An et al., Hardware-efficient erasure-error detection with an integer fluxonium, arXiv:2607.27123v1（前印本）"
  - "Liu et al., Converting qubit relaxation into erasures with a single fluxonium, arXiv:2601.11086v1（前印本）"
  - "Wang et al., Erasure Conversion in Integer Fluxonium Qubits, PRX Quantum 7, 020339 (2026), arXiv:2603.21003v1"
  - "Nesterov & Pechenezhskiy, Phys. Rev. Applied 22, 064038 (2024), arXiv:2402.07360v2；Dumas et al., arXiv:2402.06615；Zwanenburg et al., arXiv:2606.17866v1；Thorbeck et al., Phys. Rev. Lett. 132, 090602 (2024)"
  - "Measurement-induced state transitions across the fluxonium qubit landscape, arXiv:2604.08515（只用於說明其掃描範圍）"
  - "外部來源存取日 2026-11-19"
---

# Integer fluxonium 的 MIST：與 half-flux 的差異及 erasure check 的注意事項

## 結論

1. 兩個甜點都守恆 joint parity $P_q(-1)^{a^\dagger a}$，所以 $|i,N\rangle$ 與 $|j,N-k\rangle$ 要開 gap 必須 $p_ip_j=(-1)^k$。Integer 的差別在於 parity 相反的近簡併對是 e/f，也就是一個計算態加一個 erasure 態；half-flux 的近簡併對是 g/e 兩個計算態。
2. Φ=0 時 g 在中央井，e、f 是 ±2π 側井 fluxon 態的反對稱、對稱組合，|3⟩ 是中央井 plasmon。An 電路的 g 有 99.6% 權重在中央井，e/f 有 97.5%/99.0% 在側井；半磁通時 g/e 都有 >99.9% 在 ±π 雙井。An 的 $|n_{g3}|=0.489$ 而 $|n_{ge}|=0.093$，7.391 GHz 腔下 g 的弱場 dressing 有 96.6% 來自 g→3。讀出時 g 像 transmon 由中央井 plasmon 主導，e、f 走側井高態。
3. 同一高態、同一 $k$ 只准 doublet 的一員，MIST 線因此成對而 parity 只允許一邊。例：Mencia D 的 g→3（7.2418 GHz）與 f→5（7.2380 GHz）幾乎重疊，e→5 禁戒；An 電路改 $E_L=0.160$ GHz 時，允許的 e→7、f→8 二光子線在 4.4726、4.5305 GHz，gap 5.33、4.19 MHz，禁戒的交錯配對數值 gap $<4\times10^{-12}$ MHz。半磁通的 g/e doublet 有同樣結構。最近的裸共振不一定最危險：An 電路的 g→12 三光子只偏 +112 MHz，但禁戒。
4. Φ=0 單頻 drive 下 e、f 的 generalized parity 相反，不經 $k=0$ 避交叉交換。磁通偏移打破保護：e/f 在 $|\delta\Phi|\approx2.16\times10^{-3}\Phi_0$（An，$f_{ef}=30.5$ MHz）或 $6.6\times10^{-4}\Phi_0$（Mencia D，$f_{ef}=10.8$ MHz）就混約 15%。甜點附近新開的線 gap 只有 Hz–kHz；An 在 $|\delta\Phi|=0.027$–$0.030\Phi_0$ 的 g→9 二光子線（Φ=0 禁戒）在 $n_g=4.3$–20.7 開出 1.35–6.81 MHz gap，左右鏡射。
5. Check tone 的位置決定哪個輸入暴露在 MIST 下。An 電路模型同一功率下，tone 在 e 的 dressed 腔頻（bright-e，An 與 Wang 的做法）時 $(n_g,n_e,n_f)=(0.22,1.70,0.22)$；在 g/f 的 dressed 腔頻（bright-gf）時為 $(1.70,0.22,1.70)$；中點為三態各 0.64。Bright-e 把 MIST 風險集中在 erasure 態本身。
6. 後果不對稱，最糟的是錯誤失去位置標記。e→g 或 e→f 把 erasure 轉成未標記的邏輯錯誤；g↔f 是無旗標 bit flip；g/f→e 的假 erasure 仍有位置，只是成本。漏到 L 的態不一定被 e-check 抓到：調參示例中 e↔16 的終態 |16⟩ 在 matched-template 座標（code=0、e=1）只有 0.126，看起來像計算態；f↔12 的 |12⟩ 為 0.688。
7. Check 除了保布居，還要整段 pulse 與 ring-down 都滿足 $\alpha_g(t)\approx\alpha_f(t)$。An 在 Φ=0 的 which-path 指數 $D=(\kappa/2)\int|\alpha_g-\alpha_f|^2dt=6.3\times10^{-5}$/check，偏至 $0.003\Phi_0$ 升到 0.086；$\chi_{gf}(N)$ 到 $N=20$ 仍小於 $\chi_{ge}$ 的 1%。$D<10^{-3}$ 要求 $|\delta\Phi|<8.0\times10^{-4}\Phi_0$，比布居保護嚴。靠近 MIST 線時 coherence 可先於布居報警：調參的共振翼電路（$E_L=0.19201$ GHz、$f_r=7.9886$ GHz、$g_n=160$ MHz，bright-gf，$n_0=2$）一次 check 使 gf Ramsey contrast 損失 1.39%、相位 +0.18 rad，終態 $P_{12}$ 只有 0.049%；但共振再近一點（$E_L=0.19198$ GHz）時 contrast 損失 16% 與 $P_{12}=9.6\%$ 同時出現，先後次序不普遍。同一電路改用 bright-e、保持相同的 e–code pointer 距離，which-path 指數從 $1.4\times10^{-2}$ 降到 $1.2\times10^{-5}$。
8. 沒有跨裝置通用的安全光子數。An 的 check tone 上，第一個可解析的成分交換是 g↔18（$k=4$）在 $n\approx149$–156，$E_C$ 改 −1% 就移到 $n\approx29$。An 實驗在 check 時 $T_1^{e\to g}$ 從 200–300 μs 降到約 20 μs；閉系統 MIST 與平滑浴的驅動耗散都無法解釋，窄頻損耗可重現量級，機制未定。
9. $E_L$ 是 integer fluxonium 的主要取捨。An 電路固定 8 GHz 腔與 $g_n$，$E_L$ 從 0.12 增到 0.30 GHz 讓 $f_{ef}$ 從 14.4 增到 165 MHz、磁通 crossover 從 0.0016 增到 $0.0086\Phi_0$，同時 $|n_{ge}|$ 從 0.062 增到 0.245，e→6 的 25% 混合點從 $n>50$ 降到 $n=4$。
10. 弱場精確匹配 $\chi_g=\chi_f$ 不保證 check 安全。e、f 的差動 Stark shift 可把 e/f 推到真交叉：An 電路改 $E_L=0.120$ GHz、$f_r=f_d=9.077$ GHz、$g_n=172.3$ MHz 時，e/f 在 $n=8.19$ 交叉，偏磁 $10^{-4}$、$10^{-3}\Phi_0$ 在該處開 0.89、8.94 MHz gap。An 原參數的方向相反：$n=0$、1.7、100 時 e/f 間距為 30.5、33.7、204.5 MHz，dressed crossover 從 0.00216 增到 $0.00239$、$0.0166\Phi_0$。
11. Check 撞上 MIST 線時，漏態量由 tone 與腔頻的位置和 pulse 邊緣決定。調參示例保留 An 電路與 κ，讓線性標稱光子數 10 落在共振上：bright-e 的 e↔16（$k=3$，$f_r=7.5605$ GHz、$g_n=140$ MHz）在 1.65 μs 等待後 e 列為 $(g,e,f,L)=(0,0.66,0,0.34)$；bright-gf 的 f↔12（$k=2$，$f_r=7.9968$ GHz）f 列為 $(0,0,0.62,0.38)$。腔頻往避線方向挪 12 MHz 時 plateau 光子數反升到約 9.2，post 漏態卻降到 $4\times10^{-5}$、$7\times10^{-5}$。邊緣從 20 ns 放慢到 450 ns，post 漏態升到約 49%、52%（匹配 pulse 能量後 38%、41%）；g↔8 的例子則在下降段部分回復，|8⟩ 從 plateau 59% 降到排空後 8%。這些是只含共振子空間、不含 qubit 浴的模型值。
12. Mid-circuit check 重複時，Stark 相位與 coherence 損失逐次相乘，磁通雜訊的時間相關性決定累積方式。共振翼示例單次 check 的複相干因子 $0.986\,e^{i0.179}$，100 次後 contrast 0.246；不補償每次的 Stark 相位時 pass 樣本的 Ramsey 錯誤從 6.4% 升到 59.5%。An 參數下若 $\delta\Phi$ rms 為 $3\times10^{-4}\Phi_0$，100 次 check 在每次獨立偏移時 contrast 0.975，整個 shot 固定偏移時 0.573。

## 問題

使用者原話：「我好奇integer fluxonium qubit的MIST現象，我這個repo有研究過一般qubit的MIST，但integer fluxonium qubit使用gf作為computation space，其他state，包含e作為erase check使用，請你參考相關論文，組織一個agent team來使用qutip/scqubit研究與預測相關現象，參數與實驗可以使用論文中的內容。」

第一輪結果偏向 An 裝置的實驗歸因後，使用者修正焦點：「我沒有真實器件可用，我只是好奇integer fluxonium在MIST行為上可能有什麼與一般fluxonium不同的特性，或是erase check本身造成的MIST有什麼特別要注意的，參數可以微調來方便研究。」

## 考慮過的選項與比較

### 模型與方法

$H/h=4E_C\hat n^2+E_L(\varphi-2\pi x)^2/2-E_J\cos\varphi+f_ra^\dagger a+g_n\hat n(a+a^\dagger)$，$x=\delta\Phi/\Phi_0$。$g_n$ 乘完整無因次 $\hat n$，不是 $g_n|n_{ge}|$；$\chi_{ij}=f_{r|i}-f_{r|j}$ 取全差。An Appendix B 明寫這個 convention。

- **量子化腔 branch analysis**（Dumas 2024 式，非 RWA）：joint Hamiltonian 對角化，以 $a^\dagger$ 接續光子階梯；Φ=0 依 joint parity 分塊。
- **Semiclassical Floquet**：$2g_n\sqrt n\cos(2\pi f_dt)\hat n$，QuTiP FloquetBasis 加週期平均成分與 Sambe gap。Φ=0 時 Floquet 模式依「反射加半週期平移」分類，比較副本要乘 $(-1)^m$。
- 兩法在基準點給相同夥伴。重算 Nesterov & Pechenezhskiy Fig. 2（Φ=0.1）的 0↔4 共振得 $N=31$、gap 14.0 MHz，論文為 30、約 15 MHz。
- 收斂設定：LC cutoff 160–190、retained levels 40–64、Fock pad 16；窄線用 Δn=0.025–0.05 細格。e 線與 g↔18 在加大截斷後中心與 gap 不變；6.10 GHz 與 7.59 GHz 附近的粗圖未收斂，不據以判讀深度。
- Repo 既有的 `lib/zcu_tools/simulate/fluxonium/branch/floquet.py` 預設 flux=0.5、逐 branch 在 $t=0$ 各自 argmax；`full_quantum.py::calc_branch_population` 回傳 $\langle N_q\rangle$。兩者用於 integer 時要明傳 flux=0，且 $\langle N_q\rangle$ 不能區分 e/f 混合與高態混合，所以本次另寫 parity-resolved 版本。

### 裝置參數

| 來源 | $E_J/E_C/E_L$（GHz） | 計算態 | 讀出 | 備註 |
| --- | --- | --- | --- | --- |
| Mencia 2024 Device D | 6.78/1.47/0.22 | ge（f 視為 leakage） | 7.46 GHz、κ/2π 15 MHz、χ01 14.3 MHz；未報 g、tone | 模型 $f_{03}=7.24$ GHz，實測 6.86 GHz；以 χ01 全差反推 $g_n=114.9$ MHz |
| An 2026 Cycle 1 | 5.056/1.392/0.193 | gf，e 為 erasure | 裸腔 7.39109 GHz、tone 7.39020 GHz、$g_n$ 97 MHz、κ/2π 1.448 MHz、1.1 μs、$n_e\approx1.7$ | 以 $\chi_{ge}=1.86$ MHz 校準 $g_n=97.117$ MHz，得 $\chi_{gf}=-6.18$ kHz，在實測 −11.9±13.8 kHz 內 |
| Liu 2026 | 7.07/1.72/0.32 | gf，e 為 erasure | tone 6.993 GHz、κ/2π 1.02 MHz、1.6 μs；未報裸腔與 g | 印刷參數經 e 腔與 χ01 校準後預測 $\chi_{02}=+0.044$ MHz，實測 −0.147 MHz |
| Wang 2026（理論） | 4/2/0.133 | gf，e 為 erasure | 13.634 GHz、$g$ 0.3 GHz、κ/2π≈0.318 MHz | 印刷 Hamiltonian 給 pull 0.739/6.819/0.753 MHz，表列 0.378/3.820/0.379 MHz；原文排除明文的 half-χ 或 √2 解釋 |

Mencia 2024 的四個裝置都用 ge 當 qubit。直接的 gf erasure 實驗是 An 2026 與 Liu 2026。

### 各裝置的 MIST 通道（閉系統）

| 裝置與讀出 | 分支：夥伴，$k$ | 位置 | 類型 |
| --- | --- | --- | --- |
| Mencia D，$g_n=114.9$ MHz，$f_d=f_r$ | g→3、f→5，1 | $n\approx8$–9 | 平滑 dressing（失諧約 −218、−222 MHz） |
| 同上 | e↔24，5；3↔16，3 | $n\approx39$；$n\approx16$ | 窄線（e：gap 0.111 MHz） |
| An C1，check tone | g↔18，4 | Fock $N=149$、Floquet $n=155.9$ | 窄線，gap 1.53 MHz；$E_C-1\%$ 移到 $n=28.9$、gap 0.051 MHz |
| 同上 | e→6、f→5，1 | 至 $n=150$ 未達 25% 混合 | 平滑；125 levels 下 e 與 81（$k=-16$）在 $n\approx104$ 有無法定量的極小 gap |
| An C1，固定腔改 tone | g→8（2）7.111 GHz；e→16（3）7.560 GHz | $n=1.7$ 時 gap 0.748 MHz；e 在 $n=8$–10 換態 | 多光子線；離腔 280 MHz 時同光子數需約 $1.5\times10^5$ 倍輸入功率 |
| An 電路，等 e 對比改腔到 8.0 GHz | f↔12，2 | $N=9$、$n=14.2$ | gap 0.938 MHz；7.0、7.2 GHz 至 150 未見 |
| Wang，13.634 GHz | e↔10，1；e↔48，5 | $n\approx22$；$n\approx106$ | 先平滑混合，後窄線（gap 0.343 MHz） |
| Liu，印刷參數 | g↔18，6 | $n\approx152$ | 窄線；改用另一組能重現低能譜的擬合時消失，改由 e↔7 二光子在 $n\approx27$–29 先混合 |

### Erasure check 的錯誤來源（An 參數）

- **閉系統。** $n_e=1.7$ 時 e branch 的 g/f 裸權重 $1.1\times10^{-5}$、高態權重 0.25%；無 qubit 浴的有限脈衝演化在 pulse end 給 $P_g\approx4.7\times10^{-6}$。
- **平滑浴的驅動耗散。** Floquet–Markov 納入介電、電感、濾波 Purcell，以 idle 的 $T_1^{f\to e}$、$T_1^{e\to g}$、$T_1^{f\to g}$ 校準，1164 組參數給 check/idle 的 e→g 速率比 0.9989–1.0001，$T_1\approx250$ μs。ge 的 $k=\pm1$ sideband 對奇浴算符禁戒；even 殘餘通道含 $k=-1$ 熱吸收，也不夠。
- **窄頻損耗。** 在 ge 線下 2 MHz 假設一個 TLS（HWHM 0.10 MHz），check 改變 ge 發射線的位置與線形，可給約 20 μs、pulse end $P(g|e)\approx5\%$。這是能重現量級的反例，不唯一，也不等於實測 false negative 7.8±1.7%。Thorbeck 2024 有讀出讓 $T_1$ 變短也變長的實驗，An 的 App. L 也提出展寬到 TLS 的解釋。
- **Which-path。** 見結論 7。An 實驗跨 cadence 的 RB 附加誤差 2.49±0.19%/check 不能由 pointer dephasing 解釋，來源未定。

### 對 repo MIST 量測的含義

`mist_power_trace`、`mist_drive_frequency`、`mist_flux_dep` 量的是 probe 後的平均 IQ，不是 $T(j|i)$。用於 integer fluxonium 時：分別準備 g、e、f；`mist_flux_dep` 在甜點附近需要 $5\times10^{-5}$–$10^{-4}\Phi_0$ 的步距，預設約 $0.005\Phi_0$ 會跳過 e/f 混合尺度；固定 gain 格在不同 tone 或磁通下對應不同腔內光子數，要逐點換算。可逆的 dressing 與雙穿越不一定在 post-probe IQ 留下訊號。

## 被否決的選項與理由

- **以 $n_{crit}$、$f_r/f_{ge}$ 或 $E_J/E_C$ 排序 MIST 風險。** 跨裝置比較中，風險由每態的 plasmon 矩陣元、$k$ 光子失諧、parity 與達到同一資訊量所需的 $g_n$ 決定；Nesterov 2024 也指出 $n_{crit}$ 不是 MIST 門檻。
- **把 Mencia D 加 repo 合成讀出當成 gf erasure 裝置。** Mencia 2024 是 ge qubit；D 只作 parity 與能階結構的對照，主要裝置改用 An 2026。
- **套用 half-flux 的 fluxonium MIST 門檻圖（arXiv:2604.08515）。** 該掃描固定半磁通並排除 $|g_{ij}/\Delta|\ge0.15$；Mencia D 在 $g_n=100$ MHz、7.46 GHz 腔時 0↔3 已約 0.248。
- **以閉系統 MIST 或平滑浴解釋 An 的 20 μs。** 兩者給的 e→g 機率或速率比分別約 $5\times10^{-6}$ 與 1.000。
- **把 branch 權重、排空後人口、$1-e^{-1.1/20}=5.35\%$ 與實測 false negative 互換。** 實測分類含準備、門檻、效率與躍遷時刻。

## 留下的問題

- An check 期間 $T_1^{e\to g}\approx20$ μs 的硬體來源；需要 idle $T_1(f_{ge})$ 掃描與 check-on 線形，以不重調參數的方式預測另一個 tone。
- 單模模型的高能階誤差（Mencia D 的 $f_{03}$ 差 0.38 GHz）足以移動所有高光子數預測；array mode 未建模（Zwanenburg 2026 的 11 個 MIST 區域中有 5 個需要 array modes）。
- An 在 $n\approx104$ 的 e↔81 極小 gap 是否有動力學意義。
- Wang 2026 與 Liu 2026 的印刷參數不能重現各自的 χ，作者的實作 convention 或擬合參數未公開。
- 結論 5–7、10–12 的調參示例只含共振子空間或數個近腔頻 block，沒有全 lab-frame Lindblad、TLS／array mode、完整高態級聯與逐發 IQ；中點 tone 的強場 e 分支在 $N=50$ 未收斂。它們示範機制，不給 assignment 或 RB 數字。

## 依據

數值依據來自未入庫的工作檔：一個可 import 的 branch-analysis 工具模組、各子題的腳本與 npz／csv 輸出。本文已寫入重現所需的 Hamiltonian、convention、參數、收斂設定與關鍵數字。使用者同意調參以方便研究；結論 2、3、9–12 的示例以 An 電路為底、調整 $E_L$、$f_r$、$g_n$ 或 drive 頻率，調過的值寫在各條結論內。裝置參數與實驗數字取自 basis 所列論文的表格、附錄與圖，2026 年的 An、Liu、Zwanenburg 在存取時為 arXiv 前印本。

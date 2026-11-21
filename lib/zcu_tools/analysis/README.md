# `zcu_tools.analysis` 模塊重點文檔

**Last updated:** 2026-11-22，取數 sums 與分析接縫

**讀者：** framework-dev。分析核心、純數值工具與前端 adapter 的分工速查。

`zcu_tools.analysis` 放置不依賴 notebook widget 或 Qt widget 的分析核心。各 notebook
與 GUI 套件只負責把使用者事件、圖表容器、與 workflow 狀態轉成這裡的純函式或互動狀態機呼叫。

Package root 在 caller 選用 `fluxdep` 時才載入它。直接使用 `analysis.fitting` 不載入
fluxdep 的互動 line picker，也不要求 QICK。

`spectrum.py` 提供兩個分析 GUI 共用的原始頻譜資料型別，以及 Hz→GHz 與座標遞增排序。
其中 `fluxs` 是座標軸，不是 flux 校正參數。

`dips.py` 的 `find_spectrum_dips(freqs, magnitude, options)` 接受 MHz 與 magnitude，
以兩端中位數直線去背景，回傳依 depth / FWHM 降冪排列的 `SpectrumDip`。
`DipFinderOptions` 預設不平滑、prominence 雜訊倍率 5、FWHM 上限 25 MHz、
最小間距 1 MHz，`max_dips=None` 保留全部符合條件的谷。
FWHM 使用半 prominence 的插值寬度，至少為谷兩側頻率步進的平均。
函式先剔除過寬的谷，再依分數貪婪套用間距與數量上限。
它不命名元件、不讀寫資料，也不畫圖。

`estimate_prominence_factor(freqs, magnitude, options, expected_dips=N)` 依正整數峰數
估計門檻倍率，忽略 options 原有的 prominence 倍率與數量上限。
它先沿用 finder 的背景、雜訊、平滑、寬度與間距規則取得候選，再取第 N 與第 N+1 強
prominence／noise 的中點。候選不足或剛好 N 個時使用現有最低門檻，保留全部候選。
零雜訊回傳 0.0，仍受最低門檻限制。將回傳倍率與 `max_dips=N` 一起交給 finder，
才能限制最終數量；prominence 同值或零雜訊時，finder 依既有分數與頻率順序選取。

`failure.AnalysisFailed` 表示數值 owner 已辨識的無法產出分析結果，與任意程式錯誤分開。
Experiment 公開同一型別，GUI analyze work 經分類後才讓 recipe 交付 failed 值。
分類範圍由各數值 owner 文件列明，不以 ValueError／RuntimeError 的文字猜測。

## 雙讀出矩的自校準

[`cov_transfer.selfcal_transfer`](cov_transfer.py) 接收兩次讀出的 float64 `(N, 6, 6)`
特徵外積總和。它從共用的 IQ 中心與 Gaussian 寬度估計狀態聯合機率，回傳轉移矩陣、
狀態改變機率及辨識診斷。它不分類，也不擬合延遲曲線。總和格式見 ADR-0074。

## Singleshot 初態與 joint fit

`analysis.fitting.singleshot_rabi` 以共同 PCA 軸與固定 histogram bins，joint-fit raw IQ
的 readout-transition 分布及 Rabi population。Likelihood、backend validity 與候選選擇
由 [fitting](fitting/README.md#fitting-helpers) 擁有。數值選項與參數形狀見
[singleshot_rabi](fitting/singleshot_rabi.py)。

模型為 `p_inf + (p_e0 - p_inf) * exp(-t/t_r) * cos(omega*t + phase)`。
不啟用 decay 時包絡為 1，`t_r=None`。`initial_populations`／`p_e0` 是 pulse 前的狀態，
`zero_length_populations` 是模型在零座標的外推值。Phase 非零時兩者不同。
Phase 以 radians 保存。Caller 可在圖上轉成 degrees。

`initial_state` 選 ground 或 excited，描述 pulse 前的主要狀態，不保證純態，
也不以第一個非零掃描點代替初態。Rabi 將 pulse 前 excited population 限制在所選半區間，
對 pooled histogram 的兩個方向分別擬合，依有效性及 likelihood 選候選。
`max_calls` 限制每個方向的 Migrad budget，兩個方向都會評估。
自由 phase 限制在所選初態方向的 [-π/2, π/2]，baseline 同時限制振幅方向，
使非負座標的 population 維持 [0, 1]。這個 offset 不辨識跨半週期的未知 rotation。
Population、centers 與 confusion matrix 保持物理 g/e 順序。

GE 的 acquired rows 是 probe off/on。分析按 initial_state 映射主要 g/e 順序，不改持久資料。
Confusion diagnostic 使用各自 refined 的 preparation populations，不以第一列推定第二列。
兩組 refinement 從 joint fit 的同一組 seeds 出發，避免分析順序影響結果。

Optimizer 不收斂時不以 initial guess 當校準。實驗 caller 只有在 backend valid、
centers／radius／confusion matrix 有限時才提出校準。
實驗的 acquisition、圖形與 writeback 仍由各 leaf 擁有，不由數值分析決定。

## Readout chain 掃頻與 cycle

`readout_chain` 從保存的逐點 dwell count、11 個 feature sums 與 pooled product sum
計算離背景的 sigma 曲線、5% family 門檻及複數 IQ mean。它只保留 running sums，
不需要單發資料，也不依賴 program。輸入、輸出與失敗條件見
[`SweepSums` 與 `sweep_curve` 的 docstring](readout_chain/sweep.py)。

## Readout chain 讀出調參

`ThermalChainSums.from_processor(axis, sums)` 保存每點 16-read shot count 與
`THERMAL_MOMENT_COUNT`（107）個 sums。每個 shot 先回到熱平衡，再無 drive 連讀 16 次。
Saved sums 保留全部 16 個 means、R1 self moments、15 個相鄰對與三個早期長 lag pairs。
`thermal_chain_score(sums)` 逐 read 中心化後，以固定權重 (r−8)/20 加總相鄰 covariance 矩陣，
再取最大特徵值並除以 R1 variance 得到 M。固定 IQ offset 不改結果。
M 是鏈內相關增量的診斷，純二態再布居也可能提高它，不代表可辨識的 leakage probability。
Score 為 ρ² G (|ρ|/(|ρ|+max(M,0)))²，ρ=0 時為零。G 是受限二態模型的 15-lag information proxy，
用固定 24-point quadrature 計算，不做 fit，不把熱重置序列等同於 driven chain。
`ThermalChainScore.settings` 依 acquisition order 保留每點 `SettingScore`。
Ranking 包含所有點，只按 score 降序，同分保留 acquisition order，不使用誤差或第二門檻。
`score_error` 與 `qnd_error` 是模型條件式近似尺度，不是一般 one-sigma 保證。
Score scale 固定 M 與主軸，只傳播 ρ、QND，忽略共享 covariance 與選最大值造成的 selection effect。
G 的解析斜率在裁切後的 η、s 求值，clip Jacobian 當作 1，chain term 使用裁切後 s/η。
ρ² 與 M 懲罰照原始 ρ 求導。這在裁切點及外側不是實際輸出對原估計值的真實導數。
欄位、公式與 ValueError 條件見 [thermal.py](readout_chain/thermal.py)。

Live 與 analysis 的三個 panel 顯示 score、QND、M 與 |ρ|。只按 score 最大選點，紅線標記最佳 setting。
QND 軸固定 0..1，保存與回報不裁切。Score 軸只按 score 值縮放，不被誤差棒撐大。
K 後熱布居與 readout-driven steady state 不等同。稀少遠端第三態、相關雜訊或非等向雜訊可能誤導 score；
第三態與 g/e 的 IQ 重疊可能隱藏 leakage。實機 K R^16 尚未驗證這個效用。

## Readout chain K-cycle

`CycleSums` 保存 K-cycle 讀出鏈逐點的 cycle count、65 個 feature sums 與跨點加總的
product sum。`cycle_curves` 從這份 record 算出 live plot 的全 feature 曲線與第一讀曲線。
`CycleModel` 持有作者的三態曲線、解析導數、參數名與 bounds。Rabi、T2 echo 與
T2 Ramsey 使用這個模型。`t1_model()` 回傳 `RateModel`，以 `D₀ exp(Gx)` 表示
自由 stochastic 初始 drive 與 detailed-balance relaxation。`fit_cycle` 從必填的 init
以模型共變異數加權的 GMM 聯合擬合 65 個數，回傳參數與 Wald 誤差、共用
nuisances、三項 lack-of-fit 檢查與逐點布居，不做接受或拒絕。T1 的 init 是
`{"t1": textbook_time}`。其結果 `CycleFit.relaxation` 保存兩個 eigen time，並以
thermal 加權的 L1 population 變化振幅較大者作為 T1。所有擬合導數皆為解析式。欄位版面、座標、限制與失敗條件見
[cycle.py](readout_chain/cycle.py)、[model.py](readout_chain/model.py)、
[fit.py](readout_chain/fit.py) 與 [curve_models.py](readout_chain/curve_models.py) 的 docstring。

## Stationary Rπ 讀出調參

`readout_chain.RPiSums.from_processor(record)` 解碼單一 setting 的 35 欄 float64
可加摘要。固定 H=1，由 read count／segment count 恢復單段長度；只合併相同長度
的 completed rounds。入口驗證 counts 與 moments，回傳獨立陣列，不讀 program 或檔案。

`readout_chain.RPiSums` 保存單一 setting 的 18 個 moment sums、實際 read/pair counts
及 segment 長度和次數。三種 sweep 使用相同摘要，不沿用無 π 的熱態格式。
`rpi_analysis(sums)` 以 completed segments 合併後的 sums/counts，用固定 lag 1..15
計算 q、Q_eff、交替振幅及每個 quadrature 的 σ²。Caller 在呼叫前合併 rounds，
不平均 round estimates。型別與函式的 docstring 定義欄位順序、shape、單位及失敗語義。

`RPiAnalysis` 同時回報 T、V₀、r₀ 與 gate 狀態。白噪聲 V₀ 包含非零 IQ mean、
有限 segment 端點及不等 pair counts，μ 和 r₀ 使用同點 plug-in 值。
Gate 使用 `T−3.719√V₀ > 0.0224r₀`，只代表名目白噪聲下可觀察的最小交替效應。
它不證明 qubit 身分、不排除 leakage，也不保證相關電子學雜訊下的誤判率。

有效點提供交替對比分數 `score=(T−3.719√V₀)/r₀`，供 caller 在有效點中取最大值。
無效點保留原因和可計算的診斷，`score=None`，不提供 fallback 排名。
分數不使用估出的 Q_eff 或無限相關尾。它是啟發式效用，高 SNR 時可能偏好比資訊
最佳點更強的讀出。Q_eff 包含 π 誤差，σ² 不是標準誤，慢模式扣除不是一般 leakage 校正。

## Readout QND 指標

`readout_metrics.readout_metrics(sums)` 分析單一 gain 點的
`(8, 6, 6)` shot sums，回傳 frozen `ReadoutMetrics`。
兩個 feature 軸依序是 R1、R2 的 `(1, I, Q, I², IQ, Q²)`。
Branch index 是 `prepare_pi*4 + r1_on*2 + middle_pi`。

SNR、Gaussian 分離保真度與熱布居共用 R1 開、中間 I 的兩製備校準。
保持率、QND、向上／向下翻轉率與中間 π 未作用比例各有 `_on`、`_off` scalar 欄位。
`retention_from_covariance` 用同一校準兩分支的 R1–R2 covariance 檢查保持率。
所有欄位都無單位，不裁切到物理範圍。
缺少 shots 或退化分母使依賴該資料的指標為 NaN，shape 錯誤拋出 ValueError。
模型假設與各欄位的 NaN 條件見函式及結果型別的 docstring。
本模組不分類、不修正 leakage，也不選 gain。

## 多 device flux 座標

`analysis.flux` 擁有線性 flux 數學，不讀 component、record 或檔案，也不畫圖。
`DevicePeriod(period, unit="native")` 保存帶號、有限且非零的儀器值／Φ0。
`FluxFrame(offset, devices)` 保存所有儀器輸出為 0 時的 flux，以及每台儀器的 period。
Offset 以 Φ0 表示，不做 mod 1。空 devices mapping 表示固定 flux。
建構時複製 mapping，各儀器使用自己的原生單位，不換算單位。

Device keys 與投影入口的 `device_name` 共用 `cfg_model.validate_device_name`。
它拒絕 ASCII 點號、空字串與全空白，其餘名稱原樣保留。
Boolean 不算數值。無效 offset、period、unit 或使用中的值會拋出 ValueError。
值的錯誤會指名儀器。

- `flux_from_device_values(frame, values)` 計算 `offset + Σ value / period`。
- `device_value_at_flux(frame, values, device_name=..., flux=...)` 固定其他儀器，求選定儀器達到目標 flux 的值。Half 取 flux=0.5，integer 取 flux=0。
- `offset_for_flux(devices, values, flux=...)` 由目標 flux 扣除各儀器貢獻，反解 offset。

三個入口都將沒有值的儀器視為不貢獻，忽略未校準的額外儀器。
投影不使用選定儀器的當下值。輸入保持不變。
Record 到這些型別的轉換由 resources 負責，Notebook 副軸使用同一投影入口。

- `solve_device_values(frames, targets, device_names=..., fixed_values=...)` 依 ADR-0082 的 flux 公式，聯合求解多台儀器的原生值；未求解但出現在 frame 的儀器須明填固定值，目標不做 mod 1。

`require_matching_device_unit(device_name, actual_unit, expected_unit)` 集中儀器單位的精確比對。
缺單位以 None 表示，視為 native；native 只等於 native，mA 與 A 不換算。
不符時 ValueError 指名儀器與兩個單位。Caller 先選出真正用於換算的值，
再傳入單位字串；本函式不讀 snapshot 或依賴 device。
Fluxdep native 載入與 dispersive 投影共用此規則。

## 子模組

[Program V2 取數](../program/v2/README.md#取數與-readout-observation) 的 `shot_sums` 等 reducer 產出 sums 與 moments，不估計不確定度。
例如，[雙讀出矩自校準](#雙讀出矩的自校準)用外積總和估計狀態改變機率。T1 covariance 分析再用這些機率擬合延遲曲線，由 [fitter](fitting/README.md#named-fitting-results) 回傳參數與標準誤差。

- `fluxdep/`：Flux-Dependence Analysis 的選點、filtering、line selection、one-tone peak detection
  共用 kernel。`fluxdep/models.py` 擁有頻譜校正／點位資料、transition 型別與能譜到躍遷頻率的共用轉換；
  `fluxdep/io.py` 讀寫 HDF5 頻譜集合；`fluxdep/search.py` 負責共用 database search。
  診斷圖由 plotting 提供，顯示與 export workflow 由 notebook/GUI adapters 管理。
- `dispersive/`：色散分析的前處理、electrical-delay refinement 與 auto-tune 數值核心；契約見 [dispersive README](dispersive/README.md)。
- `fitting/`：擬合模型、shared/fixed fitting、singleshot 與 resonance fitting。契約見 [fitting README](fitting/README.md)。

`singleshot.py` 擁有純數值 shot 分類、Rabi 分類比例及 confusion-matrix correction。
`GECalibration` 保存原始 IQ 座標的 g/e centers、(2,2) preparation populations 與 sigma。
`calc_confusion_matrix` 用明填的 radius 與 consider_other 計算 g/e/other diagnostic；
radius=None 時由 `optimize_ge_radius` 選半徑。選項預設值與畫圖留在實驗。QICK raw buffer 的 accumulator normalization 留在 caller 的實驗 helper；本模組不讀 program metadata。

`analyze_reference_iq` 只收三組 acquired IQ 與準備標籤。它與 centroid3 processor 共用 `utils.shot_classification.nearest_centroid3` 的 g/e/f 最近中心與等距規則。前半訓練 nearest-centroid classifier 與 confusion matrix，後半獨立計算 raw/corrected fractions 與準備標籤錯誤率。品質指標使用訓練 covariance、condition number 與 singular value。Correction 回到準備標籤座標，保留負值與大於 1 的值，不推論參考態純度。`correct_populations` 的既有預設仍 clipping；caller 可用 `clip=False` 保留線性解。

`classify_reference_iq(signals, centroids)` 將固定三中心套用到任意 scan axes 加最後 shot axis 的 acquired IQ，輸出 scan axes 加 g/e/f fraction axis。它也使用共用 nearest kernel。中心與 IQ 使用相同 ADC 單位，不重訓、不修正。Exact ties 歸最低 label。Caller 用 `correct_populations(..., clip=False)` 套用 reference training confusion；結果仍是 preparation-label 座標，不是 latent energy-state population。

`analyze_centroid3_populations(populations, calibration)` 接收已正規化的 g/e/f fractions 與 caller 從 saved run cfg 取得的 `Centroid3Classifier`。它回傳 `Centroid3IQAnalysis`，包含 raw/unclipped corrected fractions、training confusion 副本及其 condition number/minimum singular value。它保留任意 scan axes，依 prepared rows／measured columns 解矩陣，singular matrix 拋出 LinAlgError。STIRAP 的 delay、amplitude、detuning 三個實驗與 Raman 各自將 counts 正規化，再共用此入口。各實驗自己畫圖，不共用分析圖。
`analyze_centroid3_iq(signals, calibration)` 先以 saved centers 分類 acquired IQ，再呼叫同一個 population 入口。兩個入口都不讀 library、Form、program 或硬體，不重新訓練校準，也不重宣告數值結果型別。
完整的 `Centroid3IQAnalysis` 留給 caller 畫 raw／unclipped corrected fractions。
`summarize_centroid3(result)` 產生 `Centroid3Summary`，複製 3×3 training confusion，保留 condition number 與 minimum singular value，不重算分析。摘要大小不隨 scan axes 或 shot 數成長，供實驗回傳與 recipe 解碼。
`Centroid3Summary` 的 owner 註解支援 TypeAdapter 從 native summary 重建 float64 confusion matrix；壞 dtype、shape 或 null／非有限 matrix 值明確拒絕。這不改 summary 或儲存格式。

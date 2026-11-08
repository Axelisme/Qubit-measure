# `zcu_tools.analysis` 模塊重點文檔

**Last updated:** 2026-11-09，cycle joint fit 與 T2 曲線

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
不需要單發資料，也不依賴 program。白化與兩個 chi-square block 的 evidence null
由子套件的私有數值模組共用；null table 首次呼叫時建立，之後留在記憶體重用。

`CycleSums.from_augmented(axis, augmented)` 把逐點 66×66 augmented sums 轉成保存用
record：保留逐點 cycle count 與 65 個 feature sums，只把 65×65 product sums 跨點加總。
`cycle_curves(sums)` 回傳 `CycleCurves`，包含使用全部 features 的 `main_sigma`、
使用第一個 read 原始 IQ 主軸的 `first_read_sigma`，以及逐點 `cycle_count`。
兩條曲線扣除逐 feature 的掃描中位數背景，以各點 cycle count 換算成 mean 的 sigma，
第一讀主軸的較大分量固定為正，全部 features 的投影再與第一讀曲線同向。
方向由同一批掃描點估計，有限點數的純雜訊方差可能高於 1。
輸入缺少 cycles、shape 不符或含非有限值時拋出 ValueError；within-point covariance
非正定時也拋出 ValueError，不加 ridge。欄位版面與單位見 [cycle.py](readout_chain/cycle.py)。

`CycleModel` 接受作者的三態曲線、參數名與 bounds。`CycleModel.identical` 把一條純量
曲線重複三次。每個 drive row 的兩個端點都是自由的三態機率，曲線在兩端點間插值。
`CycleNuisances` 保存端點、IQ 中心、共同等向雜訊、熱布居及讀出轉移。
`model.moments(axis, parameters, nuisances)` 回傳 65 個數的平均、逐 cycle 共變異數與
具名座標的平均 Jacobian，供離線模擬與診斷使用。矩陣鏈與讀出導數使用解析式，
作者 callable 的曲線導數使用有限差分。

`fit_cycle(sums, model, init)` 從必填的曲線參數起點跑一次 bounded joint GMM，
以模型共變異數加權。工具把幾何 seed 的三個態分配到固定的 curve rows，共評估六種排列。
各排列先固定 curve parameters 於 init 擬合 nuisance，再以共同初始權重下的最低 objective
選出 joint fit 起點。Nuisance-only fit 達到 evaluation budget 時保留最後的可行解作評分。
工具不搜尋其他 curve 起點，不從 pooled product sum 估權重，也不選態數。
`CycleFit` 包含曲線參數與 Wald errors、`CycleNuisances`、聯合共變異數、自由逐點 drive
擬合的第一讀布居與 errors，以及單點內、共用、曲線三個 J 與 df。
單點內與曲線檢查附漸近 chi-square p。共用檢查在邊界的 chi-square 參考分布失準，
因此 p 為 None，函式不跑 bootstrap。通過檢查不保證偏差小。
低對比且每點 cycles 少時，Wald 區間可能低估不確定度；自由端點的額外 nuisance
會使參數分布變寬或偏斜。完整欄位、座標與限制見 [fit.py](readout_chain/fit.py)。
`oscillation_frequency` 可指定以 cycles／axis unit 表示的頻率參數名，
此時 `CycleFit.reliability` 回報 R，否則為 None。工具不按 R 接受或拒絕結果。

`t2_echo_model()` 與 `t2_ramsey_model()` 建立 ideal-pulse 的三態 `CycleModel`。
兩者以總 free-evolution delay 為座標，擬合 `t2` 與第三態 lifetime `tf`，單位與座標相同。
Ramsey 另擬合非負 `detuning`，單位是 cycles／axis unit，並回報 reliability。
令 echo 的 `c = exp(-x/t2)`，Ramsey 的 `c = exp(-x/t2)*cos(2π*detuning*x)`。
g/e 曲線依序是 `(1-c)/2`、`(1+c)/2`，第三條是 `exp(-x/tf)`。
各列的自由端點吸收 amplitude、sign 與第三態離開後的分支。
理想最後一個 π/2 不把 longitudinal T1 relaxation 投影到 population，因此不用額外 T1 輸入。
`t2` 是包含 T1 transverse contribution 的總衰減時間，不是 pure-dephasing time。
Pulse error 或更複雜的第三態動力學可能造成 curve misfit，工具不自動換模型。

離線 exact-moment 與逐 shot 模擬已驗證這兩組 T2 curves 的 recovery。
六種 seed 排列只處理態標籤，不排除其他局部極小值或頻率 alias。
應檢視 curve check 與擬合結果，不能只讀 Wald error。

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

`require_matching_device_unit(device_name, actual_unit, expected_unit)` 集中儀器單位的精確比對。
缺單位以 None 表示，視為 native；native 只等於 native，mA 與 A 不換算。
不符時 ValueError 指名儀器與兩個單位。Caller 先選出真正用於換算的值，
再傳入單位字串；本函式不讀 snapshot 或依賴 device。
Fluxdep native 載入與 dispersive 投影共用此規則。

## 子模組

- `fluxdep/`：Flux-Dependence Analysis 的選點、filtering、line selection、one-tone peak detection
  共用 kernel。`fluxdep/models.py` 擁有頻譜校正／點位資料、transition 型別與能譜到躍遷頻率的共用轉換；
  `fluxdep/io.py` 讀寫 HDF5 頻譜集合；`fluxdep/search.py` 負責共用 database search。
  診斷圖由 plotting 提供，顯示與 export workflow 由 notebook/GUI adapters 管理。
- `dispersive/`：色散分析的前處理、electrical-delay refinement 與 auto-tune 數值核心；契約見 [dispersive README](dispersive/README.md)。
- `fitting/`：擬合模型、shared/fixed fitting、singleshot 與 resonance fitting。契約見 [fitting README](fitting/README.md)。

`singleshot.py` 擁有純數值 shot 分類、Rabi 分類比例及 confusion-matrix correction。QICK raw buffer 的 accumulator normalization 留在 caller 的實驗 helper；本模組不讀 program metadata。

`analyze_reference_iq` 只收三組 acquired IQ 與準備標籤。它與 centroid3 processor 共用 `utils.shot_classification.nearest_centroid3` 的 g/e/f 最近中心與等距規則。前半訓練 nearest-centroid classifier 與 confusion matrix，後半獨立計算 raw/corrected fractions 與準備標籤錯誤率。品質指標使用訓練 covariance、condition number 與 singular value。Correction 回到準備標籤座標，保留負值與大於 1 的值，不推論參考態純度。`correct_populations` 的既有預設仍 clipping；caller 可用 `clip=False` 保留線性解。

`classify_reference_iq(signals, centroids)` 將固定三中心套用到任意 scan axes 加最後 shot axis 的 acquired IQ，輸出 scan axes 加 g/e/f fraction axis。它也使用共用 nearest kernel。中心與 IQ 使用相同 ADC 單位，不重訓、不修正。Exact ties 歸最低 label。Caller 用 `correct_populations(..., clip=False)` 套用 reference training confusion；結果仍是 preparation-label 座標，不是 latent energy-state population。

`analyze_centroid3_populations(populations, calibration)` 接收已正規化的 g/e/f fractions 與 caller 從 saved run cfg 取得的 `Centroid3Classifier`。它回傳 `Centroid3IQAnalysis`，包含 raw/unclipped corrected fractions、training confusion 副本及其 condition number/minimum singular value。它保留任意 scan axes，依 prepared rows／measured columns 解矩陣，singular matrix 拋出 LinAlgError。STIRAP 與 Raman 先將 counts 正規化，再共用此入口；繪圖選擇留在 caller。
`analyze_centroid3_iq(signals, calibration)` 先以 saved centers 分類 acquired IQ，再呼叫同一個 population 入口。兩個入口都不讀 library、Form、program 或硬體，不重新訓練校準，也不重宣告數值結果型別。
完整的 `Centroid3IQAnalysis` 留給 caller 畫 raw／unclipped corrected fractions。
`summarize_centroid3(result)` 產生 `Centroid3Summary`，複製 3×3 training confusion，保留 condition number 與 minimum singular value，不重算分析。摘要大小不隨 scan axes 或 shot 數成長，供實驗回傳與 recipe 解碼。
`Centroid3Summary` 的 owner 註解支援 TypeAdapter 從 native summary 重建 float64 confusion matrix；壞 dtype、shape 或 null／非有限 matrix 值明確拒絕。這不改 summary 或儲存格式。

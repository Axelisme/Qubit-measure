# zcu_tools.analysis.fitting

**Last updated:** 2026-10-25

**讀者：** lab-author。擬合模型、數值輸入、結果與失敗契約速查。

`singleshot_rabi` 提供 raw-IQ sweep 的 pooled PCA、joint population fitting 與 circle-classification calibration。它接收純數值，回傳 physical parameters、population arrays 與 optimizer diagnostics。Cfg 讀取與繪圖由實驗 caller 負責。Pulse length 與 gain 使用相同 coordinate 契約，gain caller 明確關閉 decay。

## Peak candidates

`detect_peaks(detune, signal, *, min_prominence=0.2, min_distance=0.0, min_snr=5.0, min_width_samples=3.0)`
接收同形狀、有限的 float64 一維座標與正向投影訊號。座標須嚴格單調，至少兩點。
`min_prominence` 是訊號 ptp 的比例，`min_distance` 使用座標單位。
結果是依 prominence 降序排列的 `PeakCandidate` 列表。
每個候選提供取樣中心、原訊號高度、半 prominence 寬度、prominence 與 snr。
雜訊 σ 使用一階差分的 MAD，除以 `0.67448975 * sqrt(2)` 換算為獨立高斯雜訊標準差。
Prominence 須同時超過相對 ptp 門檻與 `min_snr * σ`。
峰高相對訊號 median 也須至少為 `min_snr * σ`，避免純雜訊的峰谷差冒充高 SNR。
回傳的 `snr = (height - median(signal)) / σ`，不是 prominence/σ。
半 prominence 的窗內可觀察寬度須至少 `min_width_samples` 個取樣間隔。
這個門檻在非等距軸上仍用取樣間隔，不用物理座標寬度。
兩個新參數都須有限且非負，設為零會關閉對應門檻。
零 MAD 使用 machine epsilon 乘訊號 ptp 作下限，至少為 float tiny，因此 snr 保持有限。
平坦訊號回傳空列表，無效輸入 raise ValueError，不修改輸入。

這些是統計門檻，不能保證任意雜訊都回空列表。
若 σ 與背景已知，201 個獨立高斯取樣至少一點超過單側 5σ 的機率約為
`1 - (1 - 2.87e-7)^201 ≈ 5.76e-5`，即 0.00576%。
這只是峰高 gate 的理想估算，尚未套用 prominence 與寬度條件。
實作會估計 σ 與背景，短序列、相關雜訊或訊號占據大部分窗口都可能改變誤報與漏報率。

端點背景使用窗內最小值。若半高交點落在窗外，候選保留 `truncated=True`，
寬度只記窗內可觀察部分，不外推完整線寬。
工具不選主峰、不擬合模型，也不決定候選能否用於校準。
負向特徵須由 caller 先翻轉。Caller 決定投影方式、候選選擇、局部擬合與接受門檻。

## fitting helpers

Cosine 與 damped-cosine fitting 在 optimizer 內不限制 phase 的週期表示，輸出才轉到
0–360°。週期邊界不是物理限制；把 optimizer 截在 ±360° 會讓靠近邊界的 phase
誤差轉移到 frequency，連 noiseless trace 都可能產生校準偏差。
自動初猜先將下降座標轉成上升順序，讓 FFT 的頻率篩選及 decay 初值使用正的
座標跨度；參數仍以原始座標定義，呼叫者不需翻轉傳入的資料。

`compute_fit_quality` 使用實際 optimizer observations、同座標的 model values、具名參數與 covariance，回傳 `FitQuality`。三個 array 參數必須已是 float64，缺失 covariance 可用 None。它計算未 clamp 的 r2、以 observations peak-to-peak 正規化的 residual RMS，以及具名參數相對誤差。已知不可估欄位為 None，`invalid` 保留欄位路徑與直接原因。Covariance 缺失、對角非有限、自由參數 stderr 為零或非有限，以及預測範圍小於 observations 範圍的 1e-8，都會留下 invalid 診斷，不改三項指標的數值。只有非對角非有限時，marginal errors 仍可用。Caller 以 `fixed_parameters` 明列刻意固定的參數，owner 不對它們的 stderr 或 relative error 加 invalid，但保留原數值。Mux fringe 固定 phase 時傳入該名稱，純 decay 不傳固定的 fringe 參數。dtype、形狀或名稱誤用直接 ValueError。Calibration 與 accept 的有效性仍由 caller 決定。

`fit_decay`、`fit_dual_decay` 與 `fit_decay_fringe` 將自由衰減時間限制在 `decay_time_bounds` 的區間。下限為第一個正 delay 與最小正取樣間距的較小者，上限為十倍掃描窗長。固定參數不被裁切。Caller 將同一區間傳入 `compute_fit_quality(..., decay_bounds=...)`；時間貼界或越界時，`invalid` 的原因列出參數名稱、擬合值、lower／upper bound 與端點值，並保留「窗口內無法解析衰減時間」。Mux T1、Echo 與 Ramsey summary 沿用 owner 的原因，不重做判斷。其他模型不啟用這組界限。

`fit_gauss_decay_fringe` 擬合 `baseline + A*exp(-(t/T)**2)*cos(2*pi*(f*t + phase/360))`。
包絡中心固定零，回傳的 T 是 1/e time，不是 sigma；頻率使用 delay 的反單位。
自由 T 沿用 `decay_time_bounds`，支援明填初值與固定參數，數值失敗 raise `AnalysisFailed`。
無條紋的 `fit_gauss_decay` 保留原有可漂移中心契約；caller 可用 `fixedparams=(None, None, 0.0, None)` 固定中心零，其 T 為 `sqrt(2)*sigma`。

`assess_fringe_sampling` 以實際擬合的 delay 座標、擬合頻率及可選的已知頻率下界回傳
`FringeSamplingQuality`。它用排序去重後的最大 delay gap 計算每週期點數；至少十點且
掃描窗包含完整週期才為 adequate。不足十點為 undersampled，沒有 gap、零頻率或不足
完整週期為 unavailable。它不重算 fit，也不把低頻 alias 當成排除 alias 的證據。

`analysis.fitting.shared` 是多 trace shared/fixed fitting 的唯一 public authority，
shared parameter 以名稱識別。它的 least-squares path 讓每條
`FitTrace` 以名稱宣告 local/shared parameter identity；caller 透過唯一一組
`ParameterSpec` 集中宣告 initial value、fixed state 與 limits。Module 在進入
optimizer 前拒絕未知或重複名稱、無效 limits 與不一致資料形狀，並把 iminuit
object 完全藏在 Interface 後方。

`SharedFitResult` 以單一 global parameter order 持有 named values、完整 covariance /
correlation、選定 profile intervals 與 validity、EDM、covariance accuracy、call-limit
等 diagnostics。Fixed parameter 保留 zero covariance row/column；least-squares
covariance 依 reduced chi-square scaling，維持既有 `curve_fit(absolute_sigma=False)`
慣例。Backend 無效但仍能形成 result 時由 diagnostics 表達，caller 從 result 讀取這些 diagnostics。`fit_ge_decay(..., share_t1=True)` 讓 g/e trace 共用同一
`t1` identity，並由 global covariance 投影既有 T1 error result。

Singleshot readout-transition family 提供固定 histogram edges 的 integrated-bin conditional
probabilities 與 radius-limited nearest-center g/e region 積分。Rabi joint likelihood 以同一
conditional family 建立 multinomial NLL 及 derived confusion matrix，不以 bin-center PDF
高度或 free matrix parameters 取代其機率語意。`singleshot_rabi` 擁有數值擬合，包括
Minuit continuation、backend validity、候選選擇與 fallback，以及 circle-classification calibration。
實驗 caller 只為 backend valid 且中心、半徑與 confusion matrix 有限的結果產生 calibration
寫回項目。Len Rabi 的 `core.analyze` 委派 fitting，從 cfg 取出 readout 長度，並把數值與
圖尺寸交給 `plotting.singleshot_rabi` 建圖。實驗 caller 決定寫回，plotting 建立 Figure。

Dual transition-rate fitting以六個stable rate names共享跨dataset identity，兩組initial
g/e populations維持dataset-qualified local identity。`DualTransitionRateFitResult`直接持有
named `TransitionRates`與errors、兩組fitted populations、local initial populations，以及
foundation提供的單一global covariance與diagnostics；singleshot、tone/sweep及overnight
callers不解包positional per-trace covariance。

`analysis.fitting.base.fit_func` 的 curve_fit optimizer 不收斂時 raise
`analysis.failure.AnalysisFailed`，保留 backend cause，不再以初猜冒充 fit。
已知 init_p 的自由參數數目時，資料不足也 raise AnalysisFailed。
Model callback 自己的例外保留原型別與物件，不降級成預期 fit 失敗。
固定參數只在至少一個參數非 None 時啟用。
FitParameters 的正常 optimizer 輸出為 ndarray，還原固定參數後為 list；FitResult 將參數與 covariance 配對。

這個分類涵蓋共用 fit_func，以及委派它的 exponential、cosine／damped cosine、
Gaussian、Lorentzian、sinc、multi-decay 與 resonance phase fitting。
例如 GUI 的 mux T1 analyze 經 fit_decay／fitexp 進這條路徑。
GUI 只在 analyze work 的 AnalysisFailed 標 failure_kind="analysis"，recipe 才交 failed 值。
Record 或 writeback 的錯誤仍 raise。Shared／singleshot 的 iminuit diagnostics、
resonance joint-refinement 自己的 fallback 不在這次分類內，不能視為所有 backend 已統一。

Lorentzian family fitting 以 median baseline 判斷初始 peak/dip 方向，避免
qubit-frequency peak 或 dip 靠近掃描邊界時被左右端點平均誤判成反向寬曲線。

Resonance circle fitting 將 electrical-delay 估計拆成兩層。`get_rough_edelay`
保留便宜的相鄰 wrapped-slope local alias；非等距 frequency grid 由
`find_edelay_branch` 最大化相鄰 unit-phasor coherence，在有限範圍內找 global
branch，再由 circle loss 做 bounded local refinement。預設搜尋兩個等效平均取樣
alias periods，caller 可用同 frequency 反單位的 radius 覆寫；另提供 opt-in maximum
radius。搜尋碰到邊界時以二倍半徑擴張，直到找到內部 optimum、達到 cap 或碰到
candidate resource guard；cap 不縮小初始 radius。Related traces
可共用一個 branch seed。等距 grid 無法辨識相差 `1/Δf` 的 delay，因此保留 local
canonical alias；多 trace 的 local aliases 以該週期作 circular aggregation，避免
在 `±1/(2Δf)` branch cut 做錯誤線性平均；各 trace 局部精修後也會對齊到共用
seed 最近的等價 alias，讓下游的 median/mean 不會混合相鄰週期，而不宣稱得到唯一物理
cable delay。
Alias alignment 由 fitting 的 `resonance.base` 擁有，既有 `resonance` facade 提供
`align_uniform_edelay_aliases` 共用 interface，供 scalar fit 與 dispersive 的獨立局部精修使用。
未啟用 adaptive expansion 或擴張到 cap 後，非等距 candidate 的最佳點仍碰到 search
boundary，以及搜尋規模超過資源上限或輸入無效時，皆 raise `ValueError`；可分辨的
local maxima 近 tie 時發出 `RuntimeWarning` 並使用最高
coherence branch。resonance 初值由
frequency-aware circle-phase slope 決定；generalized eigenproblem 以最接近零的
eigenpair 表示圓，不因浮點誤差把 exact-circle eigenvalue 推到微小負值就選錯解。

Resonance model 的 optional background 使用兩個具名乘法項：real log-amplitude
slope `exp(g * (f - f_r))` 與 resonance-centered quadratic phase
`exp(i * c * (f - f_r)^2)`。`g` 單位為 MHz⁻¹；`c` 單位為 rad/MHz²；`edelay`
維持唯一 global linear phase slope。兩個 background fit option 彼此獨立，停用的
term 固定為零且不進 joint-refinement 參數。兩個 option 都停用時保留 sequential
circle/phase path；任一 option 啟用時以 sequential / rational initializer 進 raw
complex I/Q joint refinement。Hanger 背景擬合保留 caller 提供的 branch seed 作為
joint fit 的 delay 初值，避免無背景圓形假設的局部搜尋將初值帶偏；delay 仍可在
joint refinement 中調整，不變成固定參數。Plot 的 IQ/circle/phase 使用移除 delay 與已啟用
background 後的 corrected domain。`a0` 是 fitted `f_r` 處的 complex scale；
`g` 不改變 phase，`c` 在 `f_r` 的 phase 與一階 slope 均為零。
Optimizer 的 non-finite、active-bound 或失敗結果會 warning 並回退 sequential result。
Magnitude plot 只在啟用 amplitude background fitting 時顯示 envelope 與 `g`。
Phase curvature 也只在啟用時顯示 `c`；停用時 `c` 為固定零值。

Resonance rational initializer 是 internal helper，不是 public fitting facade，也不估
absolute electrical-delay branch。Caller 先用 route-scoped delay contract 移除 delay；
initializer 只在 corrected trace 上提供 degree-1 single-pole 初值，病態或不可信結果會
warning 並回退 sequential initializer。它不引入外部 `abcd_rf_fit` delay estimator。
背景只處理平滑的乘法 amplitude 與以 resonance 為中心的 phase curvature，不把 additive
leakage、Fano path 或第二個 linear phase 併入 `bg_amp_slope`。後者與 `edelay` 無法獨立辨識。

Hanger complex fit 的 acceptance 與 derived internal quality factor 分開：若 derived inverse loss
不為正，fit 仍可保留，但 `Qi=None`、`qi_status="model_incompatible"`；physical
結果回傳 finite `Qi` 和 `qi_status="physical"`。Caller 不把 `None` 格式化為數字。

GE histogram fitting 使用 `p_avg ∈ [0, 1]` 允許兩種 readout-transition 方向，以固定數值方向與多組初值維持 g/e label symmetry。Population 以 total occupancy / conditional fraction 參數化，保證非負且總和不超過一；fixed population 與 covariance 皆轉回公開物理座標。GE 使用 strict optimizer，optimizer 失敗直接回報錯誤；`Align T1` 真正固定 shared length ratio，零 ratio 也是可用的固定模型。

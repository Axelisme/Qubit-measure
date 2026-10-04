# zcu_tools.analysis.fitting

**Last updated:** 2026-10-04, fit quality diagnostics

## fitting helpers

`compute_fit_quality` 使用實際 optimizer observations、同座標的 model values、具名參數與 covariance，回傳 `FitQuality`。它計算未 clamp 的 r2、以 observations peak-to-peak 正規化的 residual RMS，以及具名參數相對誤差。已知不可估欄位為 None，`invalid` 保留欄位路徑與直接原因。形狀或名稱誤用直接 ValueError。它不重新 fit，也不決定 calibration 或 accept 是否有效。

`analysis.fitting.shared` 是多 trace shared/fixed fitting 的唯一 public authority；不提供
positional shared-index compatibility。它的 least-squares path 讓每條
`FitTrace` 以名稱宣告 local/shared parameter identity；caller 透過唯一一組
`ParameterSpec` 集中宣告 initial value、fixed state 與 limits。Module 在進入
optimizer 前拒絕未知或重複名稱、無效 limits 與不一致資料形狀，並把 iminuit
object 完全藏在 Interface 後方。

`SharedFitResult` 以單一 global parameter order 持有 named values、完整 covariance /
correlation、選定 profile intervals 與 validity、EDM、covariance accuracy、call-limit
等 diagnostics。Fixed parameter 保留 zero covariance row/column；least-squares
covariance 依 reduced chi-square scaling，維持既有 `curve_fit(absolute_sigma=False)`
慣例。Backend 無效但仍能形成 result 時由 diagnostics 表達，不由 caller 解析
iminuit state。`fit_ge_decay(..., share_t1=True)` 是第一個 tracer：g/e trace 共用同一
`t1` identity，並由 global covariance 投影既有 T1 error result。

Singleshot readout-transition family亦提供固定histogram edges的integrated-bin conditional
probabilities與radius-limited nearest-center g/e region積分；Len Rabi joint likelihood以同一conditional
family建立multinomial NLL及derived confusion matrix，不以bin-center PDF高度或free matrix
parameters取代其機率語意。這個analysis.fitting Module只擁有conditional probability與circle integration；
Len Rabi experiment Module擁有optimizer continuation、backend validity、calibration阻擋與Figure
呈現，避免generic fitting helper決定上層analysis lifecycle。

Dual transition-rate fitting以六個stable rate names共享跨dataset identity，兩組initial
g/e populations維持dataset-qualified local identity。`DualTransitionRateFitResult`直接持有
named `TransitionRates`與errors、兩組fitted populations、local initial populations，以及
foundation提供的單一global covariance與diagnostics；singleshot、tone/sweep及overnight
callers不解包positional per-trace covariance。

`analysis.fitting.base.fit_func` 保留既有 `curve_fit` 失敗時回退 `init_p` 的
contract，但會發出 `RuntimeWarning`，讓 caller 不再把 fallback 靜默當成成功擬合。
固定參數只在至少一個參數非 `None` 時啟用。
`FitParameters` 描述既有的兩種容器：正常 optimizer 回傳 ndarray，fixed 或 fallback
回傳 list。`FitResult` 將此 parameter vector 與 covariance 配對。直接透傳的 wrappers
沿用此契約；wrapper 自己組出的 tuple 保持原樣，不做統一容器轉換。

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
complex I/Q joint refinement，plot 的 IQ/circle/phase 使用移除 delay 與已啟用
background 後的 corrected domain。`a0` 是 fitted `f_r` 處的 complex scale；
`g` 不改變 phase，`c` 在 `f_r` 的 phase 與一階 slope 均為零。
Optimizer 的 non-finite、active-bound 或失敗結果會 warning 並回退 sequential result。
Magnitude plot 只在本次啟用 amplitude background fitting 時顯示 envelope 與 `g`。
Phase curvature 也只在啟用時顯示 `c`；停用時不把固定零值當成擬合結果。

Resonance rational initializer 是 internal helper，不是 public fitting facade，也不估
absolute electrical-delay branch。Caller 先用 route-scoped delay contract 移除 delay；
initializer 只在 corrected trace 上提供 degree-1 single-pole 初值，病態或不可信結果會
warning 並回退 sequential initializer。它不引入外部 `abcd_rf_fit` delay estimator。
背景只處理平滑的乘法 amplitude 與以 resonance 為中心的 phase curvature，不把 additive
leakage、Fano path 或第二個 linear phase 併入 `bg_amp_slope`。後者與 `edelay` 無法獨立辨識。

Hanger complex fit 的 acceptance 與 derived internal quality factor 分開：若 derived inverse loss
不為正，fit 仍可保留，但 `Qi=None`、`qi_status="model_incompatible"`；physical
結果回傳 finite `Qi` 和 `qi_status="physical"`。Caller 不把 `None` 格式化為數字。

GE histogram fitting 使用 `p_avg ∈ [0, 1]` 允許兩種 readout-transition 方向，以固定數值方向與多組初值維持 g/e label symmetry。Population 以 total occupancy / conditional fraction 參數化，保證非負且總和不超過一；fixed population 與 covariance 皆轉回公開物理座標。GE 使用 strict optimizer，不沿用通用 fit_func 的初值 fallback；`Align T1` 真正固定 shared length ratio，零 ratio 也是可用的固定模型。

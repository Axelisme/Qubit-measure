# zcu_tools.utils

**Last updated:** 2026-09-27 — debug helper merge

`utils` 放可被 experiment / GUI 共用、且不反向依賴上層 domain 的 helper。
資料檔格式、讀寫與 streaming 見 `zcu_tools.datafile`。

## fitting helpers

`utils.fitting.shared` 是多 trace shared/fixed fitting 的唯一 public authority；不提供
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
parameters取代其機率語意。這個utils Module只擁有conditional probability與circle integration；
Len Rabi experiment Module擁有optimizer continuation、backend validity、calibration阻擋與Figure
呈現，避免generic fitting helper決定上層analysis lifecycle。

Dual transition-rate fitting以六個stable rate names共享跨dataset identity，兩組initial
g/e populations維持dataset-qualified local identity。`DualTransitionRateFitResult`直接持有
named `TransitionRates`與errors、兩組fitted populations、local initial populations，以及
foundation提供的單一global covariance與diagnostics；singleshot、tone/sweep及overnight
callers不解包positional per-trace covariance。

`utils.fitting.base.fit_func` 保留既有 `curve_fit` 失敗時回退 `init_p` 的
contract，但會發出 `RuntimeWarning`，讓 caller 不再把 fallback 靜默當成成功擬合。
固定參數只在至少一個參數非 `None` 時啟用。

Lorentzian family fitting 以 median baseline 判斷初始 peak/dip 方向，避免
qubit-frequency peak 或 dip 靠近掃描邊界時被左右端點平均誤判成反向寬曲線。

Resonance circle fitting 將 electrical-delay 估計拆成兩層。`get_rough_edelay`
保留便宜的相鄰 wrapped-slope local alias；非等距 frequency grid 由
`find_edelay_branch` 最大化相鄰 unit-phasor coherence，在有限範圍內找 global
branch，再由 circle loss 做 bounded local refinement。預設搜尋兩個等效平均取樣
alias periods，caller 可用同 frequency 反單位的 radius 覆寫；另提供 opt-in maximum
radius，讓 boundary-limited search 以 bounded geometric expansion 恢復；related traces
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
background 後的 corrected domain。Magnitude plot 只有在本次啟用 amplitude
background fitting 時才顯示 background envelope 與 `g`；phase curvature 只在啟用時
顯示 `c`。

Resonance rational initializer 是 internal helper，不是 public fitting facade，也不估
absolute electrical-delay branch。Caller 先用 route-scoped delay contract 移除 delay；
initializer 只在 corrected trace 上提供 degree-1 single-pole 初值，病態或不可信結果會
warning 並回退 sequential initializer。

## throttle / interpolation helpers

`utils.func_tools.min_interval` 的參數是 duty-cycle ratio，不是秒數間隔；callback
是否執行取決於上一輪執行耗時與兩次呼叫間隔的比例。`utils.math.IDWInterpolation`
是 autofluxdep predictor residual correction 的純 Python helper，空資料回傳 0，
一點資料回傳常數，兩點資料線性內插/外插，多點資料使用 nearest-k weighted
linear regression。

## debug helpers

`utils.debug.enable_debug`、`disable_debug` 與 `debug_scope` 設定、清除或暫時啟用指定 module namespace 的 debug logging。
`utils.debug.log_current_exception` 透過 caller-owned logger 記錄目前 active
exception；若 exception 來自 Pyro 且包含 `_pyroTraceback`，會把 remote traceback
文字併入同一筆 log record，不直接 print 到 stdout / stderr。

## process helpers

`utils.process` 保留 ndarray dtype 形狀的 helper 會優先使用 numpy ufunc
（例如 `np.subtract`）來表達泛型 array 運算。這比在 `NDArray[T]` 上直接使用
Python 運算子更容易讓 numpy stub 維持 dtype 關係，也避免用 `cast()` 補洞。

Program 與離線分析共用 `shot_classification` 的互斥分類；等距與 radius 邊界不納入 g/e。Gaussian region 積分使用相同的圓與半平面交集。

GE histogram fitting 使用 `p_avg ∈ [0, 1]` 允許兩種 readout-transition 方向，以固定數值方向與多組初值維持 g/e label symmetry。Population 以 total occupancy / conditional fraction 參數化，保證非負且總和不超過一；fixed population 與 covariance 皆轉回公開物理座標。GE 使用 strict optimizer，不沿用通用 fit_func 的初值 fallback；`Align T1` 真正固定 shared length ratio，零 ratio 也是可用的固定模型。

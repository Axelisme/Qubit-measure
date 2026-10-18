# zcu_tools.utils

**Last updated:** 2026-10-19 — shared smoothing options

`utils` 放可被 experiment / GUI 共用、且不反向依賴上層 domain 的 helper。
資料檔格式、讀寫與 streaming 見 `zcu_tools.datafile`。

Package 初始化不載入 QICK。`format_obj` 在 caller 呼叫時才載入 QickParam，保留其
既有字串轉換；通用數值 helper 與 `deepupdate` 不要求 QICK。

## fitting helpers

Fitting 契約見 `zcu_tools.analysis.fitting` 的 [README](../analysis/fitting/README.md)。

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

`smooth_signal1d` 與 `smooth_signal_nd` 共用 `SmoothingOptions`，集中 method、sigma 與
wavelet threshold 設定。作用的 axis／axes 由各 caller 獨立指定，省略 options 使用原預設。

`utils.process` 保留 ndarray dtype 形狀的 helper 會優先使用 numpy ufunc
（例如 `np.subtract`）來表達泛型 array 運算。這比在 `NDArray[T]` 上直接使用
Python 運算子更容易讓 numpy stub 維持 dtype 關係，也避免用 `cast()` 補洞。

Program 與離線分析共用 `shot_classification` 的互斥分類；等距與 radius 邊界不納入 g/e。Gaussian region 積分使用相同的圓與半平面交集。

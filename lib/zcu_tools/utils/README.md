# zcu_tools.utils

**Last updated:** 2026-07-04 — streaming Labber writer validation

`utils` 放可被 experiment / GUI 共用、且不反向依賴上層 domain 的 helper。
實驗資料持久化的 public API 收斂在 `zcu_tools.utils.datasaver` package
facade。

## datasaver

`zcu_tools.utils.datasaver` 是 Labber-style experiment data file 的 public
facade。caller 從 package root import model 與 function，不從內部 module
import。

- `Axis`、`LabberPayload`、`LabberMetadata`、`LabberData` 描述單一
  inner-first axes 的 Labber dataset。
- `DatasetRole`、`GroupedLabberData` 描述 grouped experiment dataset：單一
  experiment data file 內含多個 role payload，metadata 共用。
- `save_labber_data` / `load_labber_data` 處理 single-role file。
- `save_grouped_labber_data` / `load_grouped_labber_data` 處理 grouped file；
  experiment loader 傳 required roles，省略 required roles 只用於 diagnostic
  與 migration tooling。
- `StreamingLabberRoleSpec` / `open_streaming_grouped_labber_data` 處理
  grouped Labber file 的 partial-write use case：caller 先宣告 full-shape
  role schema，writer 預建 nan-filled datasets，之後以 outer row slice 寫入並
  flush。它是長掃 workflow 的 streaming primitive，不改變 one-shot save helper
  的 complete-file 語意。
- Streaming writer 對自己建立的 Labber `Data` layout 採 hard invariant：若
  內部 HDF5 group / dataset 結構不符合預期，立即 raise，不做靜默 fallback。
- Experiment semantic schema 住在 `zcu_tools.experiment.axes_spec`：
  `GroupedAxesSpec` / `RoleSpec` 把 Result/Cfg 映射到這裡的 generic grouped
  payload；`utils.datasaver` 不反向依賴 experiment Result 或 cfg。
- Save helpers 寫入 caller 傳入的 formatted path；既有目的地 fast-fail，不自動
  suffix 或覆寫。
- Path helpers（`format_ext`、`reserve_labber_filepath`、datafolder helpers）與
  HTTP transport helpers 由 facade re-export；`reserve_labber_filepath` 只供 caller
  / orchestration layer 預先決定 unique final path。
- `format_ext` / `remove_ext` 只處理檔名 suffix，不改路徑中段的 `.h5` /
  `.hdf5` 子串；`reserve_labber_filepath` 保留 Labber-style numeric sequence，
  但孤立的數字尾碼可作為 caller 命名的一部分。
- HTTP transport helper 失敗時 raise，不回傳 bool；caller 要在成功回傳後才移除
  local file。

`datasaver/` 內部 module 是責任拆分，不是額外 public import path。

## fitting helpers

`utils.fitting.base.fit_func` 保留既有 `curve_fit` 失敗時回退 `init_p` 的
contract，但會發出 `RuntimeWarning`，讓 caller 不再把 fallback 靜默當成成功擬合。
固定參數只在至少一個參數非 `None` 時啟用。

## throttle / interpolation helpers

`utils.func_tools.min_interval` 的參數是 duty-cycle ratio，不是秒數間隔；callback
是否執行取決於上一輪執行耗時與兩次呼叫間隔的比例。`utils.math.IDWInterpolation`
是 autofluxdep predictor residual correction 的純 Python helper，空資料回傳 0，
一點資料回傳常數，兩點資料線性內插/外插，多點資料使用 nearest-k weighted
linear regression。

## debug helpers

`utils.debug.log_current_exception` 透過 caller-owned logger 記錄目前 active
exception；若 exception 來自 Pyro 且包含 `_pyroTraceback`，會把 remote traceback
文字併入同一筆 log record，不直接 print 到 stdout / stderr。

## process helpers

`utils.process` 保留 ndarray dtype 形狀的 helper 會優先使用 numpy ufunc
（例如 `np.subtract`）來表達泛型 array 運算。這比在 `NDArray[T]` 上直接使用
Python 運算子更容易讓 numpy stub 維持 dtype 關係，也避免用 `cast()` 補洞。

# zcu_tools.utils

**Last updated:** 2026-06-29 process helper typing

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
- Experiment semantic schema 住在 `zcu_tools.experiment.axes_spec`：
  `GroupedAxesSpec` / `RoleSpec` 把 Result/Cfg 映射到這裡的 generic grouped
  payload；`utils.datasaver` 不反向依賴 experiment Result 或 cfg。
- Save helpers 寫入 caller 傳入的 formatted path；既有目的地 fast-fail，不自動
  suffix 或覆寫。
- Path helpers（`format_ext`、`reserve_labber_filepath`、datafolder helpers）與
  HTTP transport helpers 由 facade re-export；`reserve_labber_filepath` 只供 caller
  / orchestration layer 預先決定 unique final path。

`datasaver/` 內部 module 是責任拆分，不是額外 public import path。

## process helpers

`utils.process` 保留 ndarray dtype 形狀的 helper 會優先使用 numpy ufunc
（例如 `np.subtract`）來表達泛型 array 運算。這比在 `NDArray[T]` 上直接使用
Python 運算子更容易讓 numpy stub 維持 dtype 關係，也避免用 `cast()` 補洞。

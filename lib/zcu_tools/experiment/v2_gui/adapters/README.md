**Last updated:** 2026-07-11 — experiment adapter authoring boundary

# measure experiment adapters

這個 package 是 measure-gui 實驗流程的使用者修改入口。每個 concrete adapter file
同時擁有該實驗的 cfg definition、run/analyze/writeback policy 與 operator guide；修改一個
實驗時，主要閱讀範圍應維持在該檔案及其直接對應的 `experiment/v2/` implementation。

## Ownership

- `base.py` 擁有所有 adapter 共用的 framework implementation，不含特定實驗 policy。
- `lookback.py`、`onetone/`、`twotone/`、`singleshot/`、`fake/` 是 concrete experiment
  definitions；同一實驗專用的 helper 就近放在該檔案或同群組的 `_shared.py`。
- `_support/` 是 private package，只放至少被兩個 concrete adapters 共用的 mechanics；它
  不擁有 registry order，也不 import concrete adapter。
- `../registry.py` 是 composition root，明確列出 adapter 與 role catalog 項目。

`cfg_definition()` 使用 `_support` 提供的 measure-domain builder vocabulary，但結構與預設
policy 留在 concrete adapter，因此使用者不必跨 `spec` / `default_value` 兩個方法理解同一
份設定。generic Spec/Value assembly 由 `zcu_tools.gui.cfg` 擁有，不能搬回本 package。

## 修改原則

1. 單一實驗的設定、文字與流程 policy 留在 authoritative adapter file。
2. helper 確實出現第二個 caller，且有清楚 mechanics seam 時，才移入 `_support/`。
3. 不以 forwarding wrapper 隱藏單行邏輯；不讓 `_support` 解讀 registry key 或實驗順序。
4. 新增 adapter 時同步加入 `../registry.py`，並在 `tests/experiment/v2_gui/adapters/`
   對應路徑加入 observable contract tests。

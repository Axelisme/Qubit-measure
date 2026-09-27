**Last updated:** 2026-09-27 — adapter capability 驗證契約

# measure experiment adapters

這個 package 是 measure-gui 實驗流程的使用者修改入口。每個 concrete adapter file
同時擁有該實驗的 cfg definition、run/analyze/writeback policy 與 operator guide；修改一個
實驗時，主要閱讀範圍應維持在該檔案及其直接對應的 `experiment/v2/` implementation。

Post-analysis adapters 以 `get_post_writeback_items()` 提出 post-owned proposal；framework 將
primary/post 兩組 proposal 放入不同 opaque draft，adapter 不接觸 Writeback 實作。

## Ownership

- `base.py` 擁有所有 adapter 共用的 framework implementation，不含特定實驗 policy。
- `lookback.py`、`onetone/`、`twotone/`、`singleshot/`、`jpa/`、`fake/` 是 concrete experiment
  definitions；同一實驗專用的 helper 就近放在該檔案或同群組的 `_shared.py`。
  `jpa/` 的六個 adapter（`freq` / `flux` / `power` / `auto_optimize` /
  `flux_onetone` / `check`）是單一可發現的 JPA 校準 family，依 bring-up 順序
  註冊於 `../registry.py`。六個 concrete adapters 各自擁有 notebook-derived
  acquisition defaults，並共同暴露 `reps` / `rounds` / `relax_delay`；
  `initial_delay` 維持 core-owned hidden default。auto optimizer 的 sweep `expts`
  是 allocation resolution hint，flux 則使用中性 device-value contract。完整家族
  契約見 `../README.md`。
- `_support/` 是 private package，只放至少被兩個 concrete adapters 共用的 mechanics；它
  不擁有 registry order，也不 import concrete adapter。
- `../registry.py` 明確列出可重載的 adapter catalog；`../role_registry.py` 擁有 startup-only role composition。
- Reload experiments 會重建 concrete adapters 及 family helpers，但保留 `base.py` 與 `_support/`。
  修改這些共用基礎層需重啟 app；concrete module import 不得有硬體或背景工作副作用。

`cfg_definition()` 使用 `_support` 提供的 measure-domain builder vocabulary，但結構與預設
policy 留在 concrete adapter，因此使用者不必跨 `spec` / `default_value` 兩個方法理解同一
份設定。generic Spec/Value assembly 由 `zcu_tools.gui.cfg` 擁有，不能搬回本 package。

## Capability 宣告與實作

每個 concrete `BaseAdapter` 以一個 `AdapterCapabilities` 宣告 `analysis`（NONE／FIT／INTERACTIVE）、`requires_soc`、`post_analysis`、`load_data`。宣告的是支援範圍，不是目前是否連線、檔案是否可讀或是否已取得 operation lease。實驗側提供符合宣告的 hooks；framework 的 interface 與呼叫端規則見 [measure app](../../../gui/app/measure/README.md) 和 [GUI ADR](../../../../../docs/adr/0067-gui-application.md)。具體實驗的 capability 值以其 adapter class 為準。

`BaseAdapter.__init_subclass__` 在定義 subclass 時檢查宣告與條件式 hooks，不延後到 UI 或 worker 第一次使用。FIT 要有 `analyze()`，INTERACTIVE 要有 `make_interactive_plugin()` 與 `make_interactive_frontend()`，NONE 不接受分析 hooks；非 INTERACTIVE 不接受這兩個 interactive factories。Post-analysis 只配 FIT，並要求其 params 與 analyze hooks。需要值才能建立的 analyze params 必須提供 `get_analyze_params()`，全有預設值則可繼承 base 實作。檢查 method override 時比較沿 MRO 解析後的實作與 `BaseAdapter`，所以中間 base 提供的 override 有效，不維護子類白名單。

`load_data=True` 須有 concrete `load()`，或有可無參數建立且提供 callable `load()` 的 `exp_cls`；`load_data=False` 不接受 concrete load override。這是 import-time 宣告一致性檢查，不證明某個檔案能載入、hook 語意正確或操作安全。`validate_run_request()` 是 framework 必呼的 preflight；base 提供 no-op default，不因繼承 no-op 而新增 capability flag。Subclass 拼錯預期要 override 的 no-op hook 名稱，驗證無法辨識其意圖。精確錯誤與預設值以 `base.py` 為準。

## 修改原則

1. 單一實驗的設定、文字與流程 policy 留在 authoritative adapter file。
2. helper 確實出現第二個 caller，且有清楚 mechanics seam 時，才移入 `_support/`。
3. 不以 forwarding wrapper 隱藏單行邏輯；不讓 `_support` 解讀 registry key 或實驗順序。
4. 新增 adapter 時同步加入 `../registry.py`，並在 `tests/experiment/v2_gui/adapters/`
   對應路徑加入 observable contract tests。

range centre/span/fallback、writeback target/description/role 等 experiment policy 直接寫在
authoritative adapter；`_support` 只提供不含這些具體值的 parameterized mechanics。

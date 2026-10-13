**Last updated:** 2026-10-13，library 範本來源分離

# Measure domain support

這個 private package 擁有至少兩個 measure experiment adapters 共用的 implementation
mechanics：實驗 scan-range seeds、module role/default assembly、
analysis carriers、writeback helpers與跨實驗的 context utilities。

`analyze_results.py` 把 core 的 named FitQuality map 投影為 JSON-safe summary，並具體化 quality.invalid 的 summary 路徑。數值與直接原因由 fitting core 擁有；共用 mechanics 不算品質，也不改 writeback 或 accept policy。

Concrete adapters 依賴本 package 與 framework contracts，`zcu_lab.definitions` 組合 adapters。
本 package 不 import concrete adapter 或 `zcu_lab.definitions`，也不宣告 experiment order。只屬於
單一 adapter 的 domain policy 留在其 authoritative file；同一小群組共用但不具全域意義的
helper 留在群組內 `_shared.py`。

通用 context-free builder、seeds 與 role factory 契約位於 `zcu_tools.gui.app.measure`。Caller 明填 `MeasureCfgBuilder(role_factories=ROLE_FACTORIES)`；角色表和 res_freq_range／qub_freq_range／flux_range 留在此 package，沿用既有實驗政策。
`singleshot_shared.py` 從 AnalyzeRequest 的固定 Run `ParameterContext` 讀取分類、chi/kappa 和可選分析校準。
沒有 capture 時明示失敗，不讀目前 alias；可選校準 leaf 缺少或 null 時才交給 core 的既有分析預設。
MIST `log_scale` 是 typed analysis input，不是儲存參數。`defaults/` 擁有 cfg authoring 的語義角色與 fresh module
預設值，仍供未遷移的 adapters 使用。Library 新建選單改由 `zcu_lab.templates.TEMPLATES` 的持久 Spec 範本提供，不再消費角色 factories。這些 module 可以依賴 domain-free `zcu_tools.gui.cfg`，但 generic cfg
core不得反向 import measure domain。

Measure flux picker 的 plugin、frontend 與結果 carrier 位於 `zcu_tools.gui.app.measure`。Adapter 從固定 Run capture 讀初始參數，交 FluxPickInputs 與普通數值。Lib 不讀具體 Result 或 ctx_helpers。兩個 adapter 保留各自的 magnitude-only 政策，共用原有 kernel／plugin。

package facade 只 re-export concrete adapters 實際共用的 authoring vocabulary。單一 adapter 的
range recipe 與 writeback policy 不經 facade 轉送。共用 readout writeback helper 依 caller 傳入的
target、description、field updates 與 role 建立 proposal，保留既有 DPM 校準政策。

Reset adapter 從固定 capture 的 qubit ext 建立 default cfg。`path_seed` 與 custom range 保存
EvalValue，數值、missing/null 與 finite 驗證由既有 expression/lowering owner 處理。
SingleToneFreq、DualTonePower 與 BathPhase 各自擁有 finisher。它們直接 materialize 該 Run
的完整 tested_reset，只換本次分析欄位，不重讀 ext 或方案 slot。三個 length adapter
不擬合參數，返回空 proposal collection。框架與本 support package 不檢查 reset 校準是否齊全。

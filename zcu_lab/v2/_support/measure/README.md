**Last updated:** 2026-10-05，實驗定義搬遷

# Measure domain support

這個 private package 擁有至少兩個 measure experiment adapters 共用的 implementation
mechanics：context-free cfg definition builder、typed default seeds、module role/default assembly、
analysis carriers、writeback helpers與跨實驗的 context utilities。

`analyze_results.py` 把 core 的 named FitQuality map 投影為 JSON-safe summary，並具體化 quality.invalid 的 summary 路徑。數值與直接原因由 fitting core 擁有；共用 mechanics 不算品質，也不改 writeback 或 accept policy。

Concrete adapters 依賴本 package 與 framework contracts，`zcu_lab.definitions` 組合 adapters。
本 package 不 import concrete adapter 或 `zcu_lab.definitions`，也不宣告 experiment order。只屬於
單一 adapter 的 domain policy 留在其 authoritative file；同一小群組共用但不具全域意義的
helper 留在群組內 `_shared.py`。

`schema_builder.py` 提供 adapter author可讀的 domain vocabulary；`seeds.py` 延後解析
MetaDict、value source與智能預設值；`defaults/` 是 role catalog與fresh module default的
single source of truth。這些 module 可以依賴 domain-free `zcu_tools.gui.cfg`，但 generic cfg
core不得反向 import measure domain。

`flux_pick_plugin.py` 把 onetone/twotone flux picker 的 shared typed actions、ParamSpec commands 與 Auto Align single-flight 放在同一個 plugin instance。GUI frontend 維持本地 preview；remote 從同一個 service-owned session 讀取 committed state，worker 回覆只在 owner loop 提交，Done/cancel 後忽略晚到結果。move_line 的 role enum 來自 domain FluxLineRole，共用 ParamSpec 負責 schema 與 request 驗證，typed Action 仍保留 domain 驗證。兩個 concrete adapters 都指定終止 result builder。終止 renderer 重用 Qt-free kernel 與原生圖 builder，產生 GUI-owned FluxPickResult 和 `pick` 圖。OneTone 明確拆開 RunRecord，TwoTone 的 bare factory 入口保持不變。

package facade 只 re-export concrete adapters 實際共用的 authoring vocabulary。單一 adapter 的
range recipe與 writeback policy不經 facade 轉送；共用 writeback helper只負責依 caller 傳入的
target、description、field updates與role建立 module proposal。

**Last updated:** 2026-09-26 — shared flux-pick commands

# adapters/_support

這個 private package 擁有至少兩個 measure experiment adapters 共用的 implementation
mechanics：context-free cfg definition builder、typed default seeds、module role/default assembly、
analysis carriers、writeback helpers與跨實驗的 context utilities。

依賴方向固定為 `gui/experiment contracts → _support → concrete adapters → registry`。
`_support` 不 import concrete adapter 或 `registry.py`，也不宣告 experiment order。只屬於
單一 adapter 的 domain policy 留在其 authoritative file；同一小群組共用但不具全域意義的
helper 留在群組內 `_shared.py`。

`schema_builder.py` 提供 adapter author可讀的 domain vocabulary；`seeds.py` 延後解析
MetaDict、value source與智能預設值；`defaults/` 是 role catalog與fresh module default的
single source of truth。這些 module 可以依賴 domain-free `zcu_tools.gui.cfg`，但 generic cfg
core不得反向 import measure domain。

`flux_pick_plugin.py` 把 onetone/twotone flux picker 的 shared typed actions、ParamSpec commands 與 Auto Align single-flight 放在同一個 plugin instance。GUI frontend 維持本地 preview；remote 從同一個 service-owned session 讀取 committed state，worker 回覆只在 owner loop 提交，Done/cancel 後忽略晚到結果。

package facade 只 re-export concrete adapters 實際共用的 authoring vocabulary。單一 adapter 的
range recipe與 writeback policy不經 facade 轉送；共用 writeback helper只負責依 caller 傳入的
target、description、field updates與role建立 module proposal。

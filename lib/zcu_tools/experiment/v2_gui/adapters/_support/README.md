**Last updated:** 2026-07-11 — private cross-adapter mechanics

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

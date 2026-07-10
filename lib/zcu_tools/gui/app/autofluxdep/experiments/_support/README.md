**Last updated:** 2026-07-11 — shared experiment mechanics

# experiments/_support

這個 private package 擁有至少兩個 measurement experiments 共用的 implementation mechanics：real-acquire primitives、generic Result/Plotter carriers、dependency/module/readout/timing defaults、schema tree、module value、OverridePlan 與 sweep/timing utilities。

依賴方向固定為 `nodes contracts → _support → concrete experiments → catalog`。`_support` 可以 import `autofluxdep.nodes.builder/io/spec`，但不 import concrete experiment 或 `catalog.py`。只服務單一實驗的 domain policy 留在該 `<name>.py`，避免共用 package 變成無邊界的工具箱。

這裡的 helper 保持 experiment-neutral；它不宣告 workflow order、不建立 placement，也不解讀特定 experiment 的 provides/requires keys。

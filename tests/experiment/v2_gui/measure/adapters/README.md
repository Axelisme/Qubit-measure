**Last updated:** 2026-10-01 — OneTone 與 FakeFrequency public owners

# measure adapter tests

本目錄驗證 adapter 的公開 run／analyze／writeback 契約。核心數值演算法由 `tests/experiment/v2/` 擁有，硬體程式行為由 `tests/program/v2/` 擁有。不要直接測 private helpers 或 Notebook／腳本內容。

測試目錄對應 `lib/zcu_tools/experiment/v2_gui/measure/adapters/`：concrete experiment behavior放在
相同 domain path，跨 adapter mechanics放在 `_support/`。測試以 adapter/definition的observable
interface為主，不依賴 builder內部 declaration list；directory rename或internal helper重排不應
改變 persisted cfg、lowered runtime cfg、run/analyze/writeback contract。

單一 adapter 的 range fallback與 writeback target等 policy在對應 domain test驗證；`_support/`
tests只覆蓋至少兩個 adapters 共用的 parameterized mechanics。

`test_lookback.py` 擁有 captured context／formal cfg、complex smoothing 到 predict_offset 的 GUI 投影、具名 fit 與 timeFly writeback。
`onetone/` 擁有 Freq delay／writeback 與 PowerDep typed SNR 的 GUI 投影。
`fake/test_freq.py` 擁有同檔 FakeFrequency core／adapter 的 hardware-free run、噪聲平均、stop partial、blind fit、canonical records 與 Notebook source reuse。

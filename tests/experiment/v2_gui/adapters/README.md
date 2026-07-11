**Last updated:** 2026-07-11 — concrete policy test ownership

# measure adapter tests

測試目錄對應 `lib/zcu_tools/experiment/v2_gui/adapters/`：concrete experiment behavior放在
相同 domain path，跨 adapter mechanics放在 `_support/`。測試以 adapter/definition的observable
interface為主，不依賴 builder內部 declaration list；directory rename或internal helper重排不應
改變 persisted cfg、lowered runtime cfg、run/analyze/writeback contract。

單一 adapter 的 range fallback與 writeback target等 policy在對應 domain test驗證；`_support/`
tests只覆蓋至少兩個 adapters 共用的 parameterized mechanics。

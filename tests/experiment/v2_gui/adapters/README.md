**Last updated:** 2026-07-11 — adapter test ownership

# measure adapter tests

測試目錄對應 `lib/zcu_tools/experiment/v2_gui/adapters/`：concrete experiment behavior放在
相同 domain path，跨 adapter mechanics放在 `_support/`。測試以 adapter/definition的observable
interface為主，不依賴 builder內部 declaration list；directory rename或internal helper重排不應
改變 persisted cfg、lowered runtime cfg、run/analyze/writeback contract。

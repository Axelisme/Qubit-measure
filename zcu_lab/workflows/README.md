# 使用者 workflows

**讀者：** lab-author。查找各量測 workflow 的科學流程、表單與註冊入口。

**Last updated:** 2026-10-22

- `coherence_fluxdep/`：逐個 flux 點校準並量測 T1、Ramsey 與 Hahn echo。
- `mist_fluxdep/`：逐個 flux 點校準並掃描 MIST disturbance gain。
- `overnight/`：在固定 flux 下按固定間隔量測 T1 與 Hahn echo，不做 feedback。

每個目錄有 `workflow.py` 與 `form.py`；前兩項另有可直接執行的 `export.py`。
`common/` 放多個 workflow 共用的實驗步驟與表單 helper。
`registry.py` 依上述順序配對 workflow 與表單，交給 scripts 注入 lib 宿主。

科學流程與保存契約見 [definitions](definitions.md)，表單與宿主接線見 [forms](forms.md)。

Async 遷移的作者可寫 `await env.run(...)` 或 `await helper(env, cfg)`。
Helper 必須解析到 zcu_lab 內的 async 定義，不能只靠函式名稱像 helper 就放行。
所有 lab async def 統一只 await env／tab／session 的呼叫或 lab async helper，helper 可再呼叫 helper。
Quality gate 全掃本文，不依 decorator 或可達性選集。
明列 asyncio.*、create_task、gather、wait_for、TaskGroup 的直接 Call 即使不 await 也拒絕，包含 async with TaskGroup()。
其他 async with／async for 不另禁。
AST 檢查不追蹤 assignment aliases、動態分派與包裝，不是 type checker。
這項 guardrail 不代表舊 workflow Engine 已遷移；共用 host 與後 land 移除舊 driver 的邊界見 [ADR-0075](../../docs/adr/0075-agent-interface-layers.md#async-協作與控制命令)。

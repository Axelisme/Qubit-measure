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

## 撰寫 async workflow

以 `@workflow(name, description)` 宣告一個
`async def (env: WorkflowEnv[C], plan: Plan, tun: Live[Tunables]) -> None`。
Plan 是 frozen、extra-forbid 的 Pydantic model。Tunables 是 extra-forbid model。
每次 `tun.current` 取得目前已接受的完整 model。需要子 model 時直接讀
`tun.current.t1` 或 `tun.current.cal`，再傳給該段 helper。

序列與 core 相同時，直接傳 bound run method：

```python
outcome = await env.run(T1Exp().run, cfg)
if isinstance(outcome, Failed):
    return
save_t1(outcome, d / "t1.h5", snapshot=plan.snapshot, identity=env.identity)
fit = analyze_t1(outcome.value.result)
```

可選的 async helper 只包裝相同呼叫，例如
`async def run_t1(env, cfg): return await env.run(T1Exp().run, cfg)`。
不要複製 core 的 pulse 序列。需要不同序列時，寫同步
`def acquire(cfg, *, context: RunContext) -> Result`，再用
`await env.run(acquire, cfg)` 呼叫。頻率校準保留自己的 init_pulse，因此使用自訂 callable。

`Completed.value` 是 core 的 RunRecord，不重新組 cfg/result 信封。
`Completed.facts` 是框架同時寫入 journal 與 commit 信封的 operation 事實。
Saver 收 Completed、歷史 snapshot 與顯式 env.identity，不捕捉 evidence。
每點先配置 `d = env.new_commit_dir()`，自行存 raw 或圖片，再
`await env.commit(point, d)`。Point 必須是 Pydantic BaseModel。
框架不解析原始檔，commit 前後都可寫入作者自己的檔案。

Pause 保留等待中的 await。若 Pause 中斷 operation 並產生 partial result，框架丟棄該次結果，
Resume 後以同一 request 與原 capture 重試。自然完成的完整結果會等到 Resume 才交付，不重跑。
兩種情況都不重跑整個 workflow。
Stop 讓 operation 正常回傳 partial result。Failed 表示 Schedule acquisition 失敗；
未處理的其他例外停止 run。

要回到過去某點，開新 run 並切 plan 範圍。History 初值用 `saved_run.read_records`
從上一個 run 讀回，再在作者函式中初始化。框架不提供步驟回退或自動回灌 history。

## Async await guardrail

作者可寫 `await env.run(...)` 或 `await helper(env, cfg)`。
Helper 必須解析到 zcu_lab 內的函式定義，可用 def 或 async def，不能只靠函式名稱像 helper 就放行。
所有 lab async def 統一只 await env／tab／session 的呼叫或 lab 內定義的 helper，helper 可再呼叫 helper。
Quality gate 全掃 async def 本文，不依 decorator 或可達性選集。
在 async def 內，明列 asyncio.*、create_task、gather、wait_for、TaskGroup 的直接 Call 即使不 await 也拒絕，包含 async with TaskGroup()。
其他 async with／async for 不另禁。
AST 檢查不追蹤 assignment aliases、動態分派與包裝，不是 type checker。
共用 host 與控制命令的邊界見 [ADR-0075](../../docs/adr/0075-agent-interface-layers.md#async-協作與控制命令)。

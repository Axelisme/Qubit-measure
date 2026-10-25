# 使用者 workflows

**讀者：** lab-author。查找各量測 workflow 的科學流程、表單與註冊入口。

**Last updated:** 2026-10-25

- `coherence_fluxdep/`：逐個 flux 點校準，可選擇保存 singleshot GE 原始資料，再量測 T1、Ramsey 與 Hahn echo。
- `mist_fluxdep/`：逐個 flux 點校準並掃描 MIST disturbance gain。
- `overnight/`：在固定 flux 下按固定間隔量測 T1 與 Hahn echo，Echo 等待時間取本輪接受的 T1。
- `overnight_mux/`：按固定間隔先同時量所有 participant 的 T1，再同時量 Hahn echo。Echo 等待時間取本輪接受 T1 的最大值，不設 flux、不做校準。

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

序列與 core 相同時，helper 接同一個 Scope 與 typed cfg：

```python
async def run_t1(scope: Scope, cfg: T1Cfg):
    return await scope.run(T1Exp().run, cfg, name="T1")
```

需要不同序列時，寫同步 `def acquire(cfg, *, context: RunContext) -> Result`，
再用 `await iteration.run(acquire, cfg, name="frequency")` 呼叫。
Cfg 是位置參數，context 是 keyword-only。不要複製 core 的 pulse 序列。
需要 workflow 圖的 helper 明收 `plots: Plots`。Identity 與 live device cfg 從 scope 取得。

`env` 是 workflow 根。`with env.scope("calibration") as scope` 建作者 child。
逐點量測用 `zip(plan.fluxes, env.iterations())`，變數命名為 `iteration`。
一趟開始建可變的 Pydantic point_record，未填欄位為 None。每個結果到了再填。
主 body 用 try/finally，只在 finally 同步呼叫 `iteration.commit(point_record)` 一次，不加 await。
量測間若 iteration.cancelled，補 status="skipped" 再 continue，下一點仍可執行。
在 helper 裡則用 return 退出目前點，不用主 workflow 的 return 結束整個 run。
完整收集式範例與 Failed 指引見 [教學](../tutorial/05-workflow.md)。

`iteration.dir` 第一次使用才建立目錄。Nested child 在該目錄下鏡射 label 與 sequence。
`scope.commit(record)` 寫該 scope 的 record.json，第二次提交會 raise。
根 env.commit 寫 data root，也只能提交一次。Commit 不經 pause gate，取消 finally 可保存已有成果。
Reader 走 nested 樹依 commit_seq 排序，不依目錄名排序。

每個 point 的 `record.json` 在 `operations[]` 信封保存各 stage 的 `name` 與 `acquisitions[]`。
`coherence_fluxdep`、`mist_fluxdep`、`overnight` 與 `overnight_mux` 都走這條路徑，不在 point payload 複製欄位。
每筆 acquisition 記 `requested_rounds`、最後一次 attempt 的 `completed_rounds`、實際 acquire 呼叫數 `attempts`、`early_stopped` 與 `stop_value`。
SNR 提前停止時 `stop_value` 是停止當下的 SNR，mux 是 active tones 的最小 SNR。非有限 SNR 存 None，early_stopped 仍保留 True。
Failed stage 的已觀察 attempts 也保留。Native `docs/metadata.acquisitions` 使用同一資料，舊記錄缺少欄位時讀成空 tuple。

`Completed.value` 是 core 的 RunRecord。`Completed.facts` 是同時寫入 journal 與信封的 operation 事實。
Iteration 序號取自 iteration 物件，不取成功 commit 數，跳過未提交點也不重用序號。
Saver 收 Completed、歷史 snapshot 與 scope.identity，不重新捕捉 evidence。
每次 run attempt 建獨立 operation Plots，workflow 趨勢圖留在 env.plots。
Pane/MCP 以 operation/<name> 顯示目前 attempt，workflow 圖名不能用 operation/ 前綴。

Operation 圖只作即時呈現。保存 PNG 要從 completed.value.result 重畫到 workflow 圖，再用 env.plots.save。
Coherence 與 MIST 共用三張校準圖，每個完成的 flux 點保留一列掃描資料，僅供本次 run 畫圖。
Frequency widening 以最後一次完整 attempt 取代該點的一列。Heatmap 各列保留自己的實際掃描座標，不把變動的掃描範圍當成同一個 grid。

| Workflow 圖名 | 面板內容 | Point 目錄內的 PNG |
| --- | --- | --- |
| `calibration/qubit_freq` | 上方 accepted f01 對 flux 點圖，下方 flux × detune heatmap 與目前 trace/fit。Detune 相對該點掃描中心。 | `qubit-freq.png` |
| `calibration/len_rabi` | Flux × pulse-length heatmap，紅線連各點 accepted pi duration，側欄目前 trace/fit。 | `len-rabi.png` |
| `calibration/readout` | 最近一次 readout freq × gain SNR landscape，紅叉標有限 SNR 最大點。 | `readout.png` |
| `t1`、`t2ramsey`、`t2echo` | 各一張，上方 accepted lifetime 對 flux 的 error bar 點圖，下方目前 decay/fit。只建立 enabled lifetime 的圖。 | `<kind>-decay.png` |
| `mist/<mode>/gain` | excited 或 ground 的 MIST gain trace。 | `gain-<mode>.png` |
| `mist/<mode>/vs_flux` | 各 preparation 的背景扣除及 3σ 截斷 map。 | `map-<mode>.png` |

Coherence 的 `measure_singleshot` 預設關閉。打開後，GE 在 readout 校準之後、lifetimes 之前執行。
GE 要求本點 Rabi 已接受。`singleshot` 子表單控制 flux 最小間距與每種 preparation 的 shots。
預設是暫定模擬值：間距 0、shots 100000。GE 只保存 `singleshot-ge.h5`，不擬合、不新增 workflow 圖；core 的 operation 即時圖照常。

Coherence 與 MIST 的 frequency stage 使用獨立 `probe_pulse`，保留它的 waveform 與長度，頻率由 tracker 的掃描設定。
Probe gain 從這個 pulse 的 gain 起跑，再依線寬自適應；每次 attempt 的實際值保存在 `frequency.probe_gains`。
`qub_pulse` 只供 Rabi 與下游 pi preparation，不作 spectroscopy probe。
新表單先引用 captured `$qubit.pulses.qub_probe`。沒有該 entry 時，從 captured `$qubit.module.x180` 複製 inline，設 freq 0 MHz、gain 0.1、長度 5 µs；x180 也缺時留待填入。

Readout 在 cadence 到期且有 LATEST accepted pi pulse 時更新。未到期保留上一張圖，不在目前 point 另存 `readout.png`。
校準圖顯示完整但未通過接受門檻的 fit，分析失敗時只顯示 raw I。Partial 不更新 heatmap 或 fit。
Coherence 的 decay 使用 Result.times/signals，觀測值與 fit 使用同一投影。未通過門檻的 fit 標示 rejected，不進 lifetime history。
Coherence、overnight 與 overnight_mux 拒絕 quality.invalid 非空的 lifetime fit，原因保留診斷內容。
`min_lifetime_points` 預設為 5，只計入實際用於擬合且落在 `0 <= delay <= lifetime` 的 delay，不平移時間原點。
每個 point 提交後更新上方 accepted history，保存含上下兩個 panel 的 `<kind>-decay.png`，不另存 summary PNG。
MIST 的 measure_mode 可選 excited、ground 或 both，預設 excited。Ground 不加 pi preparation。Both 在每個 flux 點量兩種，分別保存 native 與 PNG。
MIST 的 gain-<mode>.png 使用 Result 的實際 gains/signals；map-<mode>.png 使用保留的 traces，扣中位數背景並截斷於 3σ。
T1 的 pi 依賴為 NOW；Ramsey、Echo、MIST 與 readout 可用 LATEST。Point 的 calibration_sources 記回退與來源點。
這些保存不讀 operation 圖、不改 native raw／export 格式，也不更動科學序列。Overnight 的圖維持原樣。

Pause 保留等待中的 await。若 Pause 中斷 operation 並產生 partial result，框架丟棄該次結果，
Resume 後以同一 request 與原 capture 重試。自然完成的完整結果會等到 Resume 才交付，不重跑。
兩種情況都不重跑整個 workflow。
子層取消讓 operation 回 partial result。作者先保存 Completed raw，再檢查 partial_result。
Partial 不進擬合或接受，保留原本拒絕原因，不更新 accepted history。
根 Stop 則在下一個 async operation 或 Ask raise ScriptCancelled。進行中工作先 drain 並留 facts。
作者 finally 同步提交已有成果，不吞掉例外，也不在 finally 改寫 status。
Failed 表示 Schedule acquisition 或預期分析失敗。分析 helper 的 Failed 由 caller 記 failed 與 reason。
NaN／infinity IQ 與 fitter 明示的不收斂轉 Failed。結構、cfg 與其他程式例外仍 raise，停止 run。

要回到過去某點，開新 run 並切 plan 範圍。History 初值用 `saved_run.read_records`
從上一個 run 讀回，再在作者函式中初始化。框架不提供步驟回退或自動回灌 history。

## Async await guardrail

作者可寫 `await iteration.run(...)` 或 `await helper(scope, cfg)`。
Helper 必須解析到 zcu_lab 內的函式定義，可用 def 或 async def，不能只靠函式名稱像 helper 就放行。
所有 lab async def 統一只 await env／scope／iteration／tab／session 的呼叫或 lab 內定義的 helper，helper 可再呼叫 helper。
Quality gate 全掃 async def 本文，不依 decorator 或可達性選集。
在 async def 內，明列 asyncio.*、create_task、gather、wait_for、TaskGroup 的直接 Call 即使不 await 也拒絕，包含 async with TaskGroup()。
其他 async with／async for 不另禁。
AST 檢查不追蹤 assignment aliases、動態分派與包裝，不是 type checker。
共用 host 與控制命令的邊界見 [ADR-0075](../../docs/adr/0075-agent-interface-layers.md#async-協作與控制命令)。

# 使用者 workflows

**讀者：** lab-author。查找各量測 workflow 的科學流程、表單與註冊入口。

**Last updated:** 2026-10-27

- `coherence_fluxdep/`：逐個 flux 點校準，可選擇保存 singleshot GE 原始資料，再量測 T1、Ramsey 與 Hahn echo。
- `mist_fluxdep/`：逐個 flux 點校準並掃描 MIST disturbance gain。
- `overnight/`：在固定 flux 下按固定間隔量測 T1 與 Hahn echo，Echo 等待時間取本輪接受的 T1。
- `overnight_mux/`：按固定間隔先同時量所有 participant 的 T1，再同時量 Hahn echo。Echo 等待時間取本輪接受 T1 的最大值，不設 flux、不做校準。

每個目錄有 `workflow.py`、`form.py` 與可獨立執行的 `export.py`。
Coherence、overnight 與 overnight_mux 在每點 commit 後立即追加 sample，不自動寫 Labber。
表位於該 run 所屬 entry 的 `Result/records/components/<qubit>.jsonl`，與 GUI RecordWriteback 共用。
Coherence 與 MIST 的工具提供 `--database` 就寫 Labber。Coherence 另可提供 `--result` 追加 samples，兩者都有就都寫。
Overnight 與 overnight_mux 的工具只收 `--result`，用與自動寫入相同的組法重新匯出。
未提供目的地時報錯，不另選格式旗標，也不寫 merged 檔案。
`common/` 放多個 workflow 共用的實驗步驟、表單 helper 與 record append。
`registry.py` 依上述順序配對 workflow 與表單，交給 scripts 注入 lib 宿主。

科學流程與保存契約見 [definitions](definitions.md)，表單與宿主接線見 [forms](forms.md)。

## 每點 sample 與重新匯出

Coherence 每個 flux 點一列。Overnight 每輪一列，mux 每輪每個 participant 一列。
被拒、跳過、Stop 或例外中止的已提交點也會寫入。Commit 成功後才 append，兩者不是原子交易。
Append 失敗會停止 run，已提交的 point 與先前的 rows 保留。可用手動工具補寫。
表單在 Start 從固定 capture 取得 exact entry Result 路徑，放在 plan 的 `result_path`，manifest 也保存此絕對路徑。
Headless plan 可保留 `result_path=None`，表示沒有所屬 entry，不自動寫 sample，之後可用 `--result` 補寫。

Coherence 的欄位是 `flux`、`devices`、`frequency`、`rabi`、`readout`、`t1`、`t2ramsey`、`t2echo` 與 `singleshot`。
各 stage 保留 status 與 reason。主量測值只在 accepted 時填入，其他情況為 null。
Frequency 保存 MHz、誤差、FWHM 與 probe gain，Rabi 保存模式、π／π2 長度與 gain，另保留 AmpRabi 診斷。
Rabi 的 mode 依已保存的 Len／Amp 階段判定，沒有 Rabi raw 時為 null，不推測未執行的模式。
Lifetime 保存 µs、誤差、detune 與兩個 model 的診斷，包含雙指數的第二個時間常數。
GE 的相對檔名位於 `singleshot.file`，相對於該點的 native 目錄。
Overnight 類保存 `t1`、`t2echo`、該輪 UTC 排程／開始時間、raw 檔名、Echo relax 來源與 `settings` 中固定的 drive/readout。
Mux 沒有 commanded flux，`flux` 為 null。`devices` 保留最後一份已保存 acquisition 的完整儀器 snapshot，沒有 acquisition 時為 null，不重新讀儀器。
Coherence snapshot 順序為 frequency、LenRabi、AmpRabi、readout、GE、T1、Ramsey、Echo，包含 rejected 或 partial raw。
Overnight 類用 T1、Echo 順序。`flux` 與儀器 snapshot 分開，前者記 commanded setpoint，後者記實際讀值。

```bash
python -m zcu_lab.workflows.coherence_fluxdep.export <run metadata dir> --result <entry Result dir>
python -m zcu_lab.workflows.overnight.export <run metadata dir> --result <entry Result dir>
python -m zcu_lab.workflows.overnight_mux.export <run metadata dir> --result <entry Result dir>
```

手動工具明確使用 `--result`，不使用 manifest 的舊目的路徑。可用來把離線搬移的 run 寫入新 entry。
Coherence key 是 `<run_id>/<index>`，overnight 類是 `<run_id>/round/<round_index>`。Mux 的每個 component 表有自己的相同 key。
重新匯出會再 append 一列，table owner 以 key 做整列 latest-wins。科學欄位一致，`at` 是每次 append 的 UTC 時間。

## 撰寫 async workflow

以 `@workflow(name, description)` 宣告一個
`async def (env: WorkflowEnv[C], plan: Plan, tun: Live[Tunables]) -> RunSummary`。
四個內建 workflow 正常結束時都 return `common.summary.RunSummary`，由框架寫入 metadata root 的 `summary.json`。
Plan 是 frozen、extra-forbid 的 Pydantic model。Tunables 是 extra-forbid model。
每次 `tun.current` 取得目前已接受的完整 model。需要子 model 時直接讀
`tun.current.t1` 或 `tun.current.cal`，再傳給該段 helper。

直接轉送 operation 的薄 helper 可以接同一個 Scope 與 typed cfg。量測＋分析的 stage 接 pulses、settings 與 guess，不收完整 experiment cfg：

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
Coherence 用 t1、t2ramsey、t2echo 三個欄位保存 lifetime，不再使用 measurements 列表。未量測或未到達的欄位為 None，讀取端不提供舊列表格式轉換。
Coherence 與 MIST 共用的 calibrate_point 是校準複合 stage，內部依序呼叫 frequency、Rabi（依模式含 AmpRabi）、readout 的單實驗 helper，再回傳 PointCalibrationResult。主迴圈填各校準欄位，接著量 GE 與 lifetimes。
T1、Ramsey、Echo 與 GE 的單實驗 helper 位於 `coherence_fluxdep/stages/`。各 stage 接收 `StageEnv`、本點已校準 pulses、自己的 settings 與本點 timing，不接收整份 plan 或 point record。
Ramsey 與 Echo 各自回傳獨立的 frozen Result，記錄原有接受門檻的 `gates`。各 lifetime 與 GE 的 DAC 截斷值保存在自己的 `gain_clamped`，不重複寫入 point map。
主迴圈填結果、更新各自的 accepted history，並以 `history.lifetime_guess` 計算 alpha=0.5 EWMA。T2 的 relax 另收本點 accepted T1，否則用 T1 EWMA／seed。
校準遇到 unexpected exception 時不回傳 aggregate。Run 失敗，point 只保留主迴圈已填的欄位；已存的 native raw 仍在磁碟。Expected failed/rejected 正常回傳，不受影響。
Overnight 的 `measure_t1`／`measure_echo` 接固定 `FixedPulses`、本 stage 的 `T1Settings`／`EchoSettings` 與 timing guess，分別回傳 frozen `T1Result`／`EchoResult`。Stage 內組裝 T1Cfg／T2EchoCfg，不共用 Coherence 的逐點校準 pulse stage。
Overnight_mux 的 `measure_mux_t1`／`measure_mux_echo` 接 `FixedMuxPulses`、本 stage settings 與 timing guess。Stage 內組裝 MuxT1Cfg／MuxT2EchoCfg，再各量一次 mux 並逐 participant 分析，回傳 `MuxT1StageResult`／`MuxEchoStageResult`。固定 pulses 只含完整 tone 順序、readout、participant 選擇與 reset，不混入 timing policy 或機械服務。每個 fit 保留原始 tone index，不把 participant 順序當作 tone index。
這兩個 workflow 的主迴圈在每個 stage 前只讀一次 `tun.current`，捕捉 averaging、retry、earlystop、掃描／relax policy 與 fit gates。T1 另收 `points` 與 `guess_us`。Echo 另收 `points` 與 `EchoTiming`，其中 `guess_us` 決定掃描終點，`relax_delay_us` 是主迴圈用 `echo_relaxation` 決定的值。`StageEnv` 只傳 plots、snapshot 與 summary。Stage 不修改 point record 或 history；主迴圈填結果與 gain clamp 診斷、判斷取消，再 commit 與追加 samples。
T1／Echo 結果的拒絕原因列出未通過的 `Gate`，保留原接受門檻。Point record 仍使用原有的 status、reason、scalar、models 與 raw 檔名欄位，不加入 gates 欄位。
Stage 遇到 unexpected exception 不回傳部分結果，point 只保留主迴圈已填欄位。已存 raw 留在磁碟；partial acquisition 則照常回傳 rejected/stopped 與 raw 檔名。
主 body 用 try/finally，只在 finally 同步呼叫 `iteration.commit(point_record)` 一次，不加 await。
量測間若 iteration.cancelled，補 status="skipped" 再 continue，下一點仍可執行。
在 helper 裡則用 return 退出目前點，不用主 workflow 的 return 結束整個 run。
完整收集式範例與 Failed 指引見 [教學](../tutorial/05-workflow.md)。

`iteration.dir` 第一次使用才建立目錄。Nested child 在該目錄下鏡射 label 與 sequence。
`scope.commit(record)` 寫該 scope 的 record.json，第二次提交會 raise。
根 env.commit 寫 data root，也只能提交一次。Commit 不經 pause gate，取消 finally 可保存已有成果。
Reader 走 nested 樹依 commit_seq 排序，不依目錄名排序。

每個 point 的 `record.json` 在 `operations[]` 信封保存各 stage 的 `name`、時間與 outcome。
`coherence_fluxdep`、`mist_fluxdep`、`overnight` 與 `overnight_mux` 都走這條路徑，不在 point payload 複製欄位。
Readout 優化預設 `reps=100`、`rounds=1`。其餘 workflow 量測 stage 預設 `reps=1000`、`rounds=10`，包括共用 spectroscopy、LenRabi、AmpRabi 與 MIST。Singleshot 的 shots 不變。
Early-stop SNR 留空（None）即關閉。Spectroscopy 預設 50，LenRabi／AmpRabi 預設 30，Coherence lifetime、overnight 與 overnight_mux 的 T1／Echo 預設 20.0。Readout 優化與 MIST 沒有 early stop，分別預設完成 1 輪與 10 輪。
Builder 在 SNR 提前停止時以 info 記錄 stop value 與完成 rounds，mux 的值是 active tones 的最小 SNR。非有限值記 None。
每次 acquire retry 以 warning 記錄原因與下一次 attempt number。舊檔的 `acquisitions` key 在載入時忽略，不轉換或寫回。

`Completed.value` 是 core 的 RunRecord。`Completed.facts` 是同時寫入 journal 與信封的 operation 事實。
Iteration 序號取自 iteration 物件，不取成功 commit 數，跳過未提交點也不重用序號。
Saver 收 Completed、歷史 snapshot 與 scope.identity，不重新捕捉 evidence。
每次 run attempt 建獨立 operation Plots，workflow 趨勢圖留在 env.plots。
Pane/MCP 以 operation/<name> 顯示目前 attempt，workflow 圖名不能用 operation/ 前綴。

Operation 圖只作即時呈現。保存 PNG 要從 completed.value.result 重畫到 workflow 圖，再用 env.plots.save。
保存尺寸不隨 GUI tile 改變。單面板為 640×480，frequency 為 1280×960，Rabi 為 1280×480，
coherence lifetime 為 640×960。保存時沿用 constrained layout，未設定 layout 的圖套 tight layout。
Overnight 與 overnight_mux 只保存 native 資料，不自動保存 PNG。
Coherence 與 MIST 共用 frequency、LenRabi、AmpRabi 與 readout 校準圖，依選定模式建立，每個完成的 flux 點保留一列掃描資料，僅供本次 run 畫圖。
Frequency widening 以最後一次完整 attempt 取代該點的一列。Heatmap 各列保留自己的實際掃描座標，不把變動的掃描範圍當成同一個 grid。

| Workflow 圖名 | 面板內容 | Point 目錄內的 PNG |
| --- | --- | --- |
| `calibration/qubit_freq` | 上方 accepted f01 對 flux 點圖，下方 flux × detune heatmap 與目前 trace/fit。Detune 相對該點掃描中心。 | `qubit-freq.png` |
| `calibration/len_rabi` | Flux × pulse-length heatmap，紅線連各點 accepted pi duration，側欄目前 trace/fit。 | `len-rabi.png` |
| `calibration/amp_rabi` | Flux × gain heatmap，紅線連各點 accepted pi gain，側欄目前 trace/fit。 | `amp-rabi.png` |
| `calibration/readout` | 最近一次 readout freq × gain SNR landscape，紅叉標有限 SNR 最大點。 | `readout.png` |
| `t1`、`t2ramsey`、`t2echo` | 各一張，上方 accepted lifetime 對 flux 的 error bar 點圖，下方目前 decay/fit。只建立 enabled lifetime 的圖。 | `<kind>-decay.png` |
| `mist/<mode>/gain` | excited 或 ground 的 MIST gain trace。 | `gain-<mode>.png` |
| `mist/<mode>/vs_flux` | 各 preparation 的背景扣除及 3σ 截斷 map。 | `map-<mode>.png` |

Coherence 的 `measure_singleshot` 預設關閉。打開後，GE 在 readout 校準之後、lifetimes 之前執行。
GE 要求本點 Rabi 已接受。主迴圈判斷資格與最小間距，呼叫 `stages.ge.measure_ge` 後填 `singleshot`；每個已保存 Completed（含 partial）更新最後 GE flux。`singleshot` 子表單控制 flux 最小間距與每種 preparation 的 shots。
預設是暫定模擬值：間距 0、shots 100000。GE 只保存 `singleshot-ge.h5`，不擬合、不新增 workflow 圖；core 的 operation 即時圖照常。

Coherence 與 MIST 的 frequency stage 預設使用 two-step 掃描，粗掃寬度 `coarse_span_mhz` 預設 100 MHz，widening 時乘上 `qf_widen` 的倍率。細掃寬度預設 30 MHz，可用 `fine_span_mhz` 覆寫。
Frequency stage 使用獨立 `probe_pulse`，保留它的 waveform 與長度，頻率由 tracker 的掃描設定。
Probe gain 從這個 pulse 的 gain 起跑，再依線寬自適應；每次 attempt 的實際值保存在 `frequency.probe_gains`。
`qub_pulse` 只供 Rabi 與下游 pi preparation，不作 spectroscopy probe。
每次 frequency scan 的 `diagnostics` 保存 fitted center／error、FWHM、peak／trace SNR、normalized RMS 與 MAE/fit ptp。
`gates` 逐項記名稱、比較、門檻、實際值與 pass/fail。Coarse 只供候選定位，最終接受 gates 標記 `applied=false`。
沒有擬合結果的欄位為 null，失敗原因記在 `fit_failure`。每個局部 coarse 候選另記於 `coarse_candidate_diagnostics`。
新表單先引用 captured `$qubit.pulses.qub_probe`。沒有該 entry 時，從 captured `$qubit.module.x180` 複製 inline，設 freq 0 MHz、gain 0.1、長度 5 µs；x180 也缺時留待填入。

Readout 在 cadence 到期且本點 Rabi accepted 時，用本點 π 更新。不接受本點 π 就跳過該次，不順延 cadence。未到期保留上一張圖，不在目前 point 另存 `readout.png`。
校準圖顯示完整但未通過接受門檻的 fit，分析失敗時只顯示 raw I。Partial 不更新 heatmap 或 fit。
Coherence 的 decay 使用 Result.times/signals，觀測值與 fit 使用同一投影。未通過門檻的 fit 標示 rejected，不進 lifetime history。
Coherence、overnight 與 overnight_mux 每次主模型擬合成功後，同時記錄兩種 lifetime 模型。
T1 比較含 baseline 的單指數與雙指數。Ramsey／Echo 比較指數與固定零中心的高斯包絡，條紋資料保留各自的 cosine。
Coherence 的 `t1.models/t2ramsey.models/t2echo.models`、overnight 的 `t1_models/t2_models`、overnight_mux 各 qubit 的同名欄位記錄 lifetime、誤差、residual RMS、quality 與 BIC。雙指數依長、短順序記兩個 lifetime。
兩條可用曲線都畫出模型名稱與狀態。副模型擬合失敗時記 `failed` 與原因，不畫假曲線。
非有限、非正 lifetime、負誤差、誤差大於值、quality.invalid 或超過 `max_model_window_factor * max(fitted delays)` 時，診斷標記 `discarded` 並保留原因。倍數預設 2，可在各 lifetime tunable 調整；非有限數值存 null。
BIC 使用同一組觀測點與自由參數數量，只供比較。接受值、relax、歷史與歷史圖仍只看原有主模型，副模型與診斷 discarded 不改接受規則。
主模型失敗時維持原有 failed 路徑，不嘗試副模型。Partial 仍不擬合。
Coherence、overnight 與 overnight_mux 拒絕 quality.invalid 非空的主 lifetime fit，原因保留診斷內容。
`min_lifetime_points` 預設為 5，只計入實際用於擬合且落在 `0 <= delay <= lifetime` 的 delay，不平移時間原點。
每個 point 提交後更新上方 accepted history，保存含上下兩個 panel 的 `<kind>-decay.png`，不另存 summary PNG。
MIST 的 measure_mode 可選 excited、ground 或 both，預設 excited。Ground 不加 pi preparation。Both 在每個 flux 點量兩種，分別保存 native 與 PNG。
MIST 的 gain-<mode>.png 使用 Result 的實際 gains/signals；map-<mode>.png 使用保留的 traces，扣中位數背景並截斷於 3σ。
`cal.rabi_mode` 預設 `len_then_amp`，先由 LenRabi 找 π 長度，再以 AmpRabi 的 π/π2 gain 建立下游 pulse。`amp_only` 固定目標長度只掃 gain，`len_only` 保留長度校準。範圍與接受門檻見 [definitions](definitions.md#獨立校準與-point)。
所有用到 π 的下游都要求本點 accepted π，Ramsey／Echo 另要求本點可信 π/π2 pair。Readout 與 excited MIST 只要求 π。Ground MIST 不加 π，不受 Rabi 接受狀態限制。Point 的 calibration_sources 記本點 pulse 來源；readout_source 另記保留的 readout 校準來源。
這些保存不讀 operation 圖、不改 native raw／export 格式，也不更動科學序列。Overnight 的趨勢圖維持原樣，新增的 model comparison 圖不自動保存。

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

## Run 摘要

`summary.json` 是框架 encoded record 信封，`value` 是 workflow return 的摘要。
摘要包含各 stage 的 accepted／rejected／skipped／failed 次數、拒絕、跳過與失敗原因分布，以及可取得的平均／最大耗時（秒）。
它也記錄計畫、已提交、完成與跳過點數，以及 physical recovery 嘗試數。
已提交 flux setpoints 另記範圍與單位。
完成點是 `status=completed`，即使某些 stage 被拒絕或失敗也計入。關閉的 stage 不計數。
Overnight 與 overnight_mux 另列每個 qubit 的 T1／Echo 接受數、平均與母體標準差（µs）。
單筆接受值的標準差是 0，沒有接受值時省略平均與標準差。單 qubit 名稱來自 snapshot 的 qubit role，未知時是 null，不從 channel 猜名稱。

耗時先加總每點同 stage 的 delivered Completed operations，再算平均與最大。
MIST both 的兩種 preparation 共用 MIST 耗時；mux 不依 participant 數重複計時。
Failed 沒有可交付 facts，Pause 丟棄的 attempts 也不交給作者，因此它們不計入耗時統計。
沒有可用時間時省略 duration 欄位。完整的 attempt 事實仍保存在 point 信封與 journal。

Stop 或錯誤中止時 workflow 沒有 return，框架不產生摘要，即使 finally 已提交部分 point。
`workflow_records`／RPC 只回傳 commits，不含摘要。離線用 `saved_run.read_summary(metadata_root)` 讀取，或直接讀 `<metadata_root>/summary.json`。
Data root 的 point `record.json` 與 metadata root 的 run 摘要是不同檔案，不能互相替代。
詳細計數與狀態契約見 [definitions](definitions.md#run-摘要)。

## Async await guardrail

作者可寫 `await iteration.run(...)` 或 `await helper(scope, cfg)`。
Helper 必須解析到 zcu_lab 內的函式定義，可用 def 或 async def，不能只靠函式名稱像 helper 就放行。
所有 lab async def 統一只 await env／scope／iteration／tab／session 的呼叫或 lab 內定義的 helper，helper 可再呼叫 helper。
Quality gate 全掃 async def 本文，不依 decorator 或可達性選集。
在 async def 內，明列 asyncio.*、create_task、gather、wait_for、TaskGroup 的直接 Call 即使不 await 也拒絕，包含 async with TaskGroup()。
其他 async with／async for 不另禁。
AST 檢查不追蹤 assignment aliases、動態分派與包裝，不是 type checker。
共用 host 與控制命令的邊界見 [ADR-0075](../../docs/adr/0075-agent-interface-layers.md#async-協作與控制命令)。

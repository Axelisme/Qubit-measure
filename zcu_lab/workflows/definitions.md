# `zcu_lab.workflows` 使用者 workflow

**讀者：** lab-author。查閱 coherence、MIST 與 overnight workflow 的校準、分析和保存契約。

**Last updated:** 2026-10-22，async workflow 與 operation facts

本 package 擁有具體 workflow 的 cfg 組裝、校準、分析、區域 history 與 point record。
每個 workflow 是一個 async 函式。Engine 執行 await operations，記錄 tunables revision，並保存作者明確提交的 record。`registry.build_registry` 回傳 workflow 與表單的註冊清單，scripts 組合根將它交給 lib 宿主；import 不啟動量測。
GUI／MCP 只支援 `coherence_fluxdep`、`mist_fluxdep`、`overnight`。不提供舊 standalone 名稱的 alias。

## Coherence

`CoherencePlan` 固定 flux setpoints、明填的 flux_unit、resolved drive/readout modules、delay points 與 historical RunSnapshot。
`measure_t1`、`measure_t2ramsey`、`measure_t2echo` 預設 true，至少開啟一項。全關在 acquisition 前拒絕。
每點先設 flux，再依 frequency、Rabi、readout、T1、Ramsey、Echo 順序執行。
關閉的 lifetime 不量、不保存、不產生 result 元素。

`CoherenceTunables` 分為共用 `cal` 與獨立 `t1`、`t2ramsey`、`t2echo`。
三份 lifetime history 只收各自接受值，delay grid 保留 alpha=0.5 的 EWMA 與各自初始 guess。
T1 的 relax delay 沿它的 delay guess。T2 的 relax delay 仍是 relax_factor × 自己的初始 guess，不用本點 T1 改寫。
Rabi／readout 的 relax delay 是最近 accepted T1 的五倍；尚無 accepted T1 或 T1 關閉時，用合法 T1 guess。
例如 T1 關閉且 guess=50 µs，兩個校準 stage 等待 250 µs，不補量 T1。

Ramsey 的 detune_ratio 是有限 fringes/step 值，無論 plan 開關都驗證，預設 0.05。
Requested detune 為 ratio/length_step；公開 acquisition result 保留原生量化後的 actual detune。
Echo detune 為零。兩者使用原有 fringe／whole-trace decay fit，拒絕超過兩倍最大 delay 或 mean absolute error 超過 0.1 倍 fit peak-to-peak 的值。
T1 保留原有 relative error 與 normalized residual RMS 門檻。
Pi2 使用 Length-Rabi 的 phase-aware fitted duration，不取 pi duration 除以二；waveform 與 gain 保留選定值。
`coherence_fluxdep.t1`、`coherence_fluxdep.t2` 提供 native cfg 科學 helper。其純資料 plan/point carrier 仍有既有 conversion caller，不代表 standalone workflow 入口。

## 獨立校準與 point

`calibrate_frequency`、`calibrate_rabi`、`calibrate_readout` 可分別用 await 呼叫。
`CalibrationInputs` 固定共用的 drive、pulse-backed readout 與 captured policy。
三個 stage 各回 frozen 具名結果，保留 status、reason、accepted scalar 與自己的 immutable native 檔名。
Frequency 在接受時立即更新 frequency history；Rabi 只在本點 frequency 接受後執行，接受時更新 pi/pi2。
Readout 只在本點 Rabi 接受後執行。兩個 caller 共用 `readout_due(point_index, tun)`，依零基 flux index % ro_every == 0 掃描。
Index 0 自然先掃。失敗／拒絕不增加掃描，也不改下一個 due index。
Readout 非接受時保留合法 seed／prior accepted readout，仍可量 lifetime 或 MIST。
任何 dependent acquisition 都不能借 prior frequency 或 pi pulse 取代本點依賴。

`CoherencePoint` 保存 frequency、rabi、readout 與 enabled `measurements` tuple。
Tuple 按 T1、Ramsey、Echo 排序，每個 kind 最多一個結果；開啟但失敗／拒絕／skipped 的項目有原因，關閉的項目沒有 placeholder。
Frequency widening 保存每個 Completed attempt 的引用，fit 拒絕也保留 native raw。
Failed／skipped stage 沒有偽造檔名。工具按 point 的用途欄位讀檔，不掃目錄或猜 effect 次序。
這讓 frequency 接受而 Rabi 失敗的點仍保留部分科學證據。

Carry 與 lifetime history 是 async 函式的區域變數。作者以 `await env.commit(point, d)` 提交完整點。
Pause 中斷目前 operation，Resume 重試同一 request；已完成的校準與區域變數不重做。
Stop 讓 operation 回傳 partial result，作者仍可保存及提交。之後的 operation 以已停止的 token 啟動。
沒有 stage checkpoint、原始檔 reader、檔案追蹤或 rollback 保證。
Schedule Failed 與 fit 拒絕隔離為各項具名結果。Unexpected adapter、fit、writer 或 plot 例外停止 run，不改標為正常失敗。

## MIST 與 overnight

`mist_fluxdep` 使用相同獨立校準 stage 與本點依賴。Plan 固定 reset、pi preparation、MIST disturbance pulse 與 gain grid。
`MISTPoint` 保留三份 stage 結果、success、reason 與 gain_file；raw IQ 留 native `mist/gain`。
Live trace/map 保留 centred magnitude／spread projection，不推 threshold。
Map 使用 native 實際 gain 與有單位的 flux sample centres，不用 requested extent 或 flux index 替代。

`overnight` 維持固定 flux、有限 T1→Hahn-echo pair、固定開始間隔與無 feedback。
它不呼校準，不讀 context，也不因 Coherence 合併增加 calibration。
下一 pair 在 max(now, previous actual start + interval) 開始，不做 catch-up bursts。
Pause 保留 pair 的區域進度，Resume 繼續等待中的 operation；Stop 保留既有 commits。
區域變數 `trend_points` 保留具名 scalar pair tuple，不保存 Figure 或 axes。
`overnight.trend.draw_overnight_trend` 直接畫這些 pair，缺值維持 gaps，不重 fit。
每個 pair 開始與完整 pair 提交前更新 `overnight_trend`。圖可能先於 journal commit 顯示，不是提交證據。
只有一個 pair 時，以 actual_start 為中心取 ±plan.start_interval_s，UTC 刻度顯示 HH:MM:SS。多 pair 保留實際時間與自動範圍。
宿主不輪詢 committed records 來呼叫它們。Workflow 透過 `env.plots` 決定資料與更新時機，framework 不保證只畫 journal 已提交的紀錄。
沒有自動保存或從 run 目錄重開趨勢圖。保存責任見 ADR-0062。

## Native、裝置與前端邊界

作者以 `d = env.new_commit_dir()` 或 `zip(plan.fluxes, env.commit_dirs(), strict=False)` 配置 absolute 保存目錄。
目錄在 data root 的 `commits/<六位配置序號>/` 下。配置序號不等於 commit 序號。
Workflow 在 env.run 返回 Completed 後呼既有 exact-path native writer，fit 拒絕仍保存 raw。
Point 引用是相對該目錄的 filename，不保證檔案完整或仍存在。Pause／Stop 可能留下未提交檔案。
Writer 收 `Completed[RunRecord]`，從 value 取得 cfg/result，從 facts 取得 acquisition 時間與 device snapshot。
Caller 顯式傳入 `env.identity` 與歷史 `RunSnapshot`。Saver 只組裝既有 `RunMetadata`，不捕捉新事實。
H5 run_id 是 `<identity.run_id>-<facts.iteration:06d>-<facts.attempt>`。stopped 決定 complete 或 stopped，experiment 取 persistence spec.tag。
Provenance 取 workflow identity，不逐 operation 捕捉。Lab 不建立 AcquisitionEvidence。
Framework 只保存 manifest、journal 與 best-effort record.json。
Conversion 的科學型別與輸入驗證留在此 package，不在框架建立 reader 或通用 dispatcher。

## 獨立 run 轉換

`coherence_fluxdep.export.export_run` 支援 merged、Labber 與 samples，`mist_fluxdep.export.export_run` 只支援 merged、Labber。
兩個模組提供同 process 的 `main(argv)`，也能以 `python -m zcu_lab.workflows.coherence_fluxdep.export` 或 `python -m zcu_lab.workflows.mist_fluxdep.export` 執行。
傳入 run 的 metadata 目錄並明選 `--merged`、`--labber` 或 Coherence 的 `--samples`。
Samples 需要明填 `--result <entry result>`，Labber 需要 `--database <Database>`。Merged-only 不需外部目的地。
工具在任何寫入前驗所選 flags 及共同輸入。Merged 覆寫 manifest data_root 的 merged.h5，Labber 每次在日期樹取新檔。
各 writer 回報自己的 path/status/error。未配置的 Labber path 為 None，已配置的失敗保留 path，不自動重試或寫 ledger。

工具讀固定 manifest，再用 `saved_run.read_records(run, CoherencePoint)` 或 MISTPoint 讀整個 run 的 typed commit list。
每列的 directory 來自 commit 信封，工具只讀 point 中具名的 native reference。
沒有 record.json 的配置目錄不產生 row。Reader 不從 raw 檔推造遺失 record，export receipt 的 missing_records 保留為空 tuple。
壞資料指名 source，不讀 journal、掃 native 候選或重抓目前 GUI。
Coherence 各已量 lifetime 分別輸出 Signal/Time Flux×Index 與 Flux-only lifetime，保留各自 actual delay。
Disabled 項目沒有 channel，rejected fit 保留 raw 與 NaN lifetime，不 union、插值或重 fit。
MIST 保留 actual gain inner-first 的 Signal 及 Flux-only Success，grid 不同就拒絕。

Coherence samples 每點最多一行，有 accepted frequency 或任一 lifetime 才寫。
僅寫接受值，欄位沿 freq_mhz 與 t1/t2r/t2e_us、對應 err_us，未接受欄位省略。
座標來自 frequency→Rabi→readout→T1→Ramsey→Echo 的最後 Completed native snapshot，包括拒絕的 fit。
歷史 point/component 取固定 plan，key 為 run_id/index。全部 rows 先驗再追加，I/O failure 保留 prefix/count/cause。
Latest-wins 由 sample table owner 決定。GUI、RPC、MCP 沒有 conversion controls。

`common.devices.set_flux` 讀 live typed cfg，重用 check_flux_unit，再以 model_copy 只改 value，await 完整 setup。
Fake 明填 native；YOKO 明填 A 或 V。Plan／表單不推測、換算單位或預驗 connected membership。
每次 env.run 在 callback 前 capture connected drivers。Native saver 使用該次 Completed.facts，不因 Resume、alias 修改或 Save 刷新 snapshot。

核心只透過當次 RunContext.plots 更新 native typed handles；buffer callback 只送座標與複數資料。
每次 env.run attempt 使用同 host 的獨立 operation Plots，完成後 finish。新 attempt release 並替換舊 operation，workflow 趨勢圖留在 env.plots。
Decay PNG 由 Completed.value.result 的 times/signals 重畫並疊加既有 fit，summary 從接受的 lifetime history 重畫。MIST gain.png 同樣從 Result 的實際 gains/signals 重畫，不取 operation live 圖。
Live 與 summary 只建立 enabled lifetime 的圖。圖形副作用不放進 carry。
GUI forms、measure headless owner、固定 capture、U7 與 S4 分工見 [forms](forms.md)。
行為比對使用 measure GUI 的 workflow tabs 與離線來源，不另用 agent 專用模擬腳本。模擬仍須使用隔離的 project 與保存目的地，不連接硬體。

Framework 測試使用合成 workflow：`tests/experiment/workflows/` 驗通用 Engine，`tests/gui/app/measure/services/`、`tests/gui/app/measure/remote/` 與 `tests/mcp/measure/` 驗宿主、RPC 與 MCP 接縫。不測本套件的科學流程或預設值，刪掉 `zcu_lab/` 後整套 pytest 仍須通過。

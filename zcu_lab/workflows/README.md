# `zcu_lab.workflows` — 使用者 workflow

**Last updated:** 2026-10-10，native／v0 plan 的必填 flux_unit

本 package 擁有具體 workflow 的校準、cfg 組裝、分析、record 與 state。通用迭代、取消、提交與保存路徑由 `zcu_tools.experiment.workflows` 負責。宿主建立 catalog，再明確注入框架；import 不啟動 run。宣告只列 name、plan、tunables、init。T1 在步驟開頭檢查 soc／devices／context，v0 檢查 devices／context，缺少時明確失敗。這是定義者的選擇，不是框架的 admission 保證；硬體操作仍經 effect。

`T1Plan`、v0 的 `FluxPlan` 與 `OvernightPlan` 要求 caller 明填 `flux_unit`（A、V 或 native），每次 device effect 原樣傳遞。Fake 輸入明填 native，YOKO 由使用者明填 A 或 V。Plan 不讀硬體、不推測單位、不提供預設或換算；host 的單位檢查與通用設值接線仍待 base device API。GUI／MCP plan 表單尚待此 host 接線，不代表已完成 caller 驗收。

`v0` 提供 T2echo fluxdep 與 T1 overnight 的期 0 離線示範。Experiment 使用真實 Run／Schedule seam，但 trace 與校準值都是 deterministic 假值，不操作硬體，也不代表物理策略。JSON saver 只驗證 exact-path 契約，不是 native 資料格式。T2echo／overnight 在期 3 替換，退役前移除假校準。原示範的 T1_fluxdep 已由 `t1` 的 native declaration 取代。

`acquire.run_t2echo` 提供原生 uniform、零 detune 的 Hahn-echo acquisition，直接使用 Run／Schedule 與 native cfg／Result。它保留半延遲的原生 rounding、取消後的部分資料與 acquisition evidence，不捕捉 context，也不連儀器。完整 T2echo workflow、fringe acquisition 與 saver 尚未由這個 callback 完成。`scripts/run_workflow_sim.py` 可選 `t2echo`，只讀 current native cfg 與 simulator 設定。

`analysis` 提供純資料的 frequency、length-Rabi、T1 與零 detune T2echo fits。Echo 沿 native analysis 排除首點，要求明確零 activated detune 與 finite retained trace；simulator CLI 只在 echo 完整 acquisition 後報告 lifetime／uncertainty。它重用 native signal projections 與 fitting helpers，保留原生單位、estimate、uncertainty 與 residual 診斷，不操作硬體、不畫圖、不保存或決定 calibration 接受度。Fit residual 不是 readout SNR，兩者由 workflow policy 分開判斷。Length-Rabi 使用既有 native free-phase/no-decay 路線；phase 固定為零會在已觀察的 simulator trace 陷入 optimizer fallback。

`calibration` 從固定的 qub_pulse／readout modules 組裝 native cfg，執行 frequency、length-Rabi 與按 cadence 的 readout landscape。它擁有接受度與純資料 carry，全部成功才更新校準歷史。Caller 明確傳入正 finite 的 T1 estimate（µs）。Frequency relax_delay 為1µs，Rabi／readout為5倍estimate；initial_delay／dev沿native cfg預設，不依賴Measure adapters或host timing capture。Context只有四個校準scalar，mutable modules放在engine會複製的plan。三個 saver 以具名參數物件明確注入，不放進 context 或建立 registry。接受度、cadence 與 Rabi range 的預設值是暫定 tunables，待使用者在 CP-B／CP-C 確認，不是已驗證的物理預設。

`t1` 宣告 `T1_fluxdep`，逐 setpoint 執行完整 calibration 與 native T1。Plan 持有固定 setpoints、兩個選定的modules與caller捕捉的歷史RunSnapshot，context只含scalar seed。Pi drive 使用量出 pi duration 的 Rabi waveform／gain。T1的delay grid與relax delay由lifetime history的EWMA guess生成。Calibration使用首點guess_us，後續使用最新accepted lifetime；拒絕fit不加入history。校準失敗與 fit 不接受提交具名 reason，只有接受的 lifetime 加入 history。Pause／stop 由 Engine 丟棄當前步驟，已發布的 acquisition 保留。

Workflow 透過 `env.plots` 的 typed handles 更新 live trace，buffer 回呼只送座標與複數資料。校準參考線、靜態圖與 fit 疊圖經 draw 在 host owner 執行；每點 decay／最終 summary 經 save 保存。跨 retry 清除並重建具名圖的 axes，再接新模板，不把 handles 存進 state。核心不再持 Live artist 宣告。

`saving` 的四個具體 writer 使用各 experiment 的既有 persistence declaration 與 native `save_run`。Writer 接 Engine 的 exact path，不決定 entry layout，也不捕捉 live context。Acquisition evidence 提供 run_id、時間、completion 與 software provenance；caller 的 RunSnapshot 提供當時的 entry／point／roles。保存不以 workflow run_id 或保存時間覆寫 acquisition facts。Saver 原樣將已提供的 `AcquisitionEvidence.device_snapshot` 交給 native metadata；`{}` 表示無 device，`None` 保留舊資料未記錄的語意。Sweep 與再次 Save 不刷新這份資料。Workflow 的 per-env.run capture 與 callback 傳遞仍待接線，這項 saver 修正不代表整條 snapshot 資料流已完成。

`tests/experiment/workflows/` 驗證通用 Engine 契約。`tests/zcu_lab/workflows/` 由具體 callback、calibration 與 native T1 的 Engine seam，驗證真實 simulator、typed native load、完整與拒絕路徑、pause 新 revision 重做及 stop 保留 prior commit。不測宿主腳本或文件內容。

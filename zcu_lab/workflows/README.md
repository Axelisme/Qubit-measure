# `zcu_lab.workflows` — 使用者 workflow

**Last updated:** 2026-10-08，原生零 detune T2echo acquisition／analysis

本 package 擁有具體 workflow 的校準、cfg 組裝、分析、record 與 state。通用迭代、取消、提交與保存路徑由 `zcu_tools.experiment.workflows` 負責。宿主建立 catalog，再明確注入框架；import 不啟動 run。

`v0` 提供 T2echo fluxdep 與 T1 overnight 的期 0 離線示範。Experiment 使用真實 Run／Schedule seam，但 trace 與校準值都是 deterministic 假值，不操作硬體，也不代表物理策略。JSON saver 只驗證 exact-path 契約，不是 native 資料格式。T2echo／overnight 在期 3 替換，退役前移除假校準。原示範的 T1_fluxdep 已由 `t1` 的 native declaration 取代。

`acquire.run_t2echo` 提供原生 uniform、零 detune 的 Hahn-echo acquisition，直接使用 Run／Schedule 與 native cfg／Result。它保留半延遲的原生 rounding、取消後的部分資料與 acquisition evidence，不捕捉 context，也不連儀器。完整 T2echo workflow、fringe acquisition 與 saver 尚未由這個 callback 完成。`scripts/run_workflow_sim.py` 可選 `t2echo`，只讀 current native cfg 與 simulator 設定。

`analysis` 提供純資料的 frequency、length-Rabi、T1 與零 detune T2echo fits。Echo 沿 native analysis 排除首點，要求明確零 activated detune 與 finite retained trace；simulator CLI 只在 echo 完整 acquisition 後報告 lifetime／uncertainty。它重用 native signal projections 與 fitting helpers，保留原生單位、estimate、uncertainty 與 residual 診斷，不操作硬體、不畫圖、不保存或決定 calibration 接受度。Fit residual 不是 readout SNR，兩者由 workflow policy 分開判斷。Length-Rabi 使用既有 native free-phase/no-decay 路線；phase 固定為零會在已觀察的 simulator trace 陷入 optimizer fallback。

`calibration` 以完整 native templates 執行 frequency、length-Rabi 與按 cadence 的 readout landscape。它擁有接受度與純資料 carry，全部成功才更新校準歷史。Scalar seed 可作深度唯讀 context，mutable native cfg 放在 engine 會複製的 plan，不以 frozen 外殼宣稱深度唯讀。三個 saver 以具名參數物件明確注入，不放進 context 或建立 registry。接受度、cadence 與 Rabi range 的預設值是暫定 tunables，待使用者在 CP-B／CP-C 確認，不是已驗證的物理預設。

`t1` 宣告 `T1_fluxdep`，逐 setpoint 執行完整 calibration 與 native T1。Plan 持有固定 setpoints、native templates 與 caller 捕捉的歷史 RunSnapshot，context 只含 scalar seed。Pi drive 使用量出 pi duration 的 Rabi waveform／gain。Delay grid 與 relax delay 由 lifetime history 的 EWMA guess 生成。校準失敗與 fit 不接受提交具名 reason，只有接受的 lifetime 加入 history。Pause／stop 由 Engine 丟棄當前步驟，已發布的 acquisition 保留。

`saving` 的四個具體 writer 使用各 experiment 的既有 persistence declaration 與 native `save_run`。Writer 接 Engine 的 exact path，不決定 entry layout，也不捕捉 live context。Acquisition evidence 提供 run_id、時間、completion 與 software provenance；caller 的 RunSnapshot 提供當時的 entry／point／roles。保存不以 workflow run_id 或保存時間覆寫 acquisition facts。

`tests/experiment/workflows/` 驗證通用 Engine 契約。`tests/zcu_lab/workflows/` 由具體 callback、calibration 與 native T1 的 Engine seam，驗證真實 simulator、typed native load、完整與拒絕路徑、pause 新 revision 重做及 stop 保留 prior commit。不測宿主腳本或文件內容。

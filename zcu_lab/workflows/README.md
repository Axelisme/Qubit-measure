# `zcu_lab.workflows` — 使用者 workflow

**Last updated:** 2026-10-10，獨立 GUI/MCP 宿主與 run 保存

本 package 擁有具體 workflow 的校準、cfg 組裝、分析、record 與 state。通用迭代、取消、提交與保存路徑由 `zcu_tools.experiment.workflows` 負責。宿主建立 catalog，再明確注入框架；import 不啟動 run。宣告只列 name、plan、tunables、init。T1 在步驟開頭檢查 soc／devices／context，v0 檢查 devices／context，缺少時明確失敗。這是定義者的選擇，不是框架的 admission 保證；硬體操作仍經 effect。

`v0` 提供 T2echo fluxdep 與 T1 overnight 的期 0 離線示範。Experiment 使用真實 Run／Schedule seam，但 trace 與校準值都是 deterministic 假值，不操作硬體，也不代表物理策略。JSON saver 只驗證 exact-path 契約，不是 native 資料格式。T2echo／overnight 在期 3 替換，退役前移除假校準。原示範的 T1_fluxdep 已由 `t1` 的 native declaration 取代。

`acquire.run_t2echo` 提供原生 uniform、零 detune 的 Hahn-echo acquisition，直接使用 Run／Schedule 與 native cfg／Result。它保留半延遲的原生 rounding、取消後的部分資料與 acquisition evidence，不捕捉 context，也不連儀器。完整 T2echo workflow、fringe acquisition 與 saver 尚未由這個 callback 完成。`scripts/run_workflow_sim.py` 可選 `t2echo`，只讀 current native cfg 與 simulator 設定。

`analysis` 提供純資料的 frequency、length-Rabi、T1 與零 detune T2echo fits。Echo 沿 native analysis 排除首點，要求明確零 activated detune 與 finite retained trace；simulator CLI 只在 echo 完整 acquisition 後報告 lifetime／uncertainty。它重用 native signal projections 與 fitting helpers，保留原生單位、estimate、uncertainty 與 residual 診斷，不操作硬體、不畫圖、不保存或決定 calibration 接受度。Fit residual 不是 readout SNR，兩者由 workflow policy 分開判斷。Length-Rabi 使用既有 native free-phase/no-decay 路線；phase 固定為零會在已觀察的 simulator trace 陷入 optimizer fallback。

`calibration` 從固定的 qub_pulse／readout modules 組裝 native cfg，執行 frequency、length-Rabi 與按 cadence 的 readout landscape。它擁有接受度與純資料 carry，全部成功才更新校準歷史。Caller 明確傳入正 finite 的 T1 estimate（µs）。Frequency relax_delay 為1µs，Rabi／readout為5倍estimate；initial_delay／dev沿native cfg預設，不依賴Measure adapters或host timing capture。Context只有四個校準scalar，mutable modules放在engine會複製的plan。三個 saver 以具名參數物件明確注入，不放進 context 或建立 registry。接受度、cadence 與 Rabi range 的預設值是暫定 tunables，待使用者在 CP-B／CP-C 確認，不是已驗證的物理預設。

`t1` 宣告 `T1_fluxdep`，逐 setpoint 執行完整 calibration 與 native T1。Plan 持有固定 setpoints、兩個選定的modules與caller捕捉的歷史RunSnapshot，context只含scalar seed。Pi drive 使用量出 pi duration 的 Rabi waveform／gain。T1的delay grid與relax delay由lifetime history的EWMA guess生成。Calibration使用首點guess_us，後續使用最新accepted lifetime；拒絕fit不加入history。校準失敗與 fit 不接受提交具名 reason，只有接受的 lifetime 加入 history。Pause／stop 由 Engine 丟棄當前步驟，已發布的 acquisition 保留。

Workflow 透過 `env.plots` 的 typed handles 更新 live trace，buffer 回呼只送座標與複數資料。校準參考線、靜態圖與 fit 疊圖經 draw 在 host owner 執行；每點 decay／最終 summary 經 save 保存。跨 retry 清除並重建具名圖的 axes，再接新模板，不把 handles 存進 state。核心不再持 Live artist 宣告。

`saving` 的四個具體 writer 使用各 experiment 的既有 persistence declaration 與 native `save_run`。Writer 接 Engine 的 exact path，不決定 entry layout，也不捕捉 live context。Acquisition evidence 提供 run_id、時間、completion 與 software provenance；caller 的 RunSnapshot 提供當時的 entry／point／roles。保存不以 workflow run_id 或保存時間覆寫 acquisition facts。

## GUI、MCP 與保存

`scripts/run_workflows_gui.py` 經共用 runtime 啟動獨立 T1 app。具體組裝在 `gui.app`，重用 shared session、Setup 與 AliasDialog；一份 Window／Controller 同時供 GUI 和 RPC 操作。`scripts/run_workflows_mcp.py` 提供 stdio MCP，經 shared bridge 連 GUI socket，只轉送宣告的 commands、form preview、status、records 與具名 PNG，不另持 Engine 或硬體。啟動 MCP 不自動啟動 GUI，server 結束只 disconnect。

一個 run 只有一個 operation／handle。Pause 的 worker 返回後保留本程序 RUN lease、input capture 與 Plots；Resume 重用同一 handle，重新同步讀所有 connected devices。任一讀取失敗拒絕該次 Start／Resume，不用 cache 代替。Apply 提交完整普通值，Engine 記 revision，expression 留在 FormInput。過期 run_id 拒絕，不 retry。

Storage owner 的 `workflow_roots` 決定同一條目的兩個 `workflows/<run_id>/`。Result root 放 manifest／journal，Database root 的 `iter/` 放原生 acquisition HDF5 與 workflow files。單次 acquisition 的身分與歷史 RunSnapshot 仍由 native writer 保存；Pause／Stop 丟棄未提交步驟，不刪先前已發布檔案。`scripts/run_t1_fluxdep_sim.py` 提供完整 T1 的離線 JSON 輸入宿主，不提供 GUI／MCP Pause 互動，也不讀 legacy templates。

S1–S3 的真 in-process simulator 驗證涵蓋同一 GUI/MCP run、env.set_device driver 超限傳播、密集 live updates 的 final Figure／PNG 與 native typed load。cfg.dev setup 半邊仍待上游介面變更後重核。S4 source reload、U7 input 保存還原與期 3 overnight 由 workflows-p3 承接；startup-failure scenario 14 尚未完整驗證。模擬通過不代表物理門檻已驗證或使用者已接受 CP-B，不帶來真機授權。

`tests/experiment/workflows/` 驗證通用 Engine 契約。`tests/zcu_lab/workflows/` 由具體 callback、calibration 與 native T1 的 Engine seam，驗證真實 simulator、typed native load、完整與拒絕路徑、pause 新 revision 重做及 stop 保留 prior commit。不測宿主腳本或文件內容。

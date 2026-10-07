# `zcu_lab.workflows` — 使用者 workflow

**Last updated:** 2026-10-07，原生 calibration 與 T1 cfg 接縫

本 package 擁有具體 workflow 的校準、cfg 組裝、分析、record 與 state。通用迭代、取消、提交與保存路徑由 `zcu_tools.experiment.workflows` 負責。宿主建立 catalog，再明確注入框架；import 不啟動 run。

`v0` 提供 T1 fluxdep、T2echo fluxdep 與 T1 overnight 的期 0 離線示範。Experiment 使用真實 Run／Schedule seam，但 trace 與校準值都是 deterministic 假值，不操作硬體，也不代表新的物理策略。JSON saver 只驗證 exact-path 契約，不是 native 資料格式。期 1 將 T1_fluxdep 換成真實 experiment 與 saver。T2echo／overnight 依期 3 處置，期 3 退役前不保留假校準。不能把示範當成量測入口。

`analysis` 提供純資料的 frequency、length-Rabi 與 T1 fits。它重用 native signal projections 與 fitting helpers，保留原生單位、estimate、uncertainty 與 residual 診斷，不操作硬體、不畫圖、不保存或決定 calibration 接受度。Fit residual 不是 readout SNR，兩者由 workflow policy 分開判斷。Length-Rabi 使用既有 native free-phase/no-decay 路線；phase 固定為零會在已觀察的 simulator trace 陷入 optimizer fallback。

`calibration` 以完整 native templates 執行 frequency、length-Rabi 與按 cadence 的 readout landscape。它擁有接受度與純資料 carry，全部成功才更新校準歷史。Scalar seed 可作深度唯讀 context，mutable native cfg 放在 engine 會複製的 plan，不以 frozen 外殼宣稱深度唯讀。三個 saver 以具名參數物件明確注入，不放進 context 或建立 registry。接受度、cadence 與 Rabi range 的預設值是暫定 tunables，待使用者在 CP-B／CP-C 確認，不是已驗證的物理預設。

`t1` 持有固定 flux plan、captured tunables、純資料 state 與完整 native cfg 生成。Pi drive 使用量出 pi duration 的 Rabi waveform／gain，delay grid 與 relax delay 由 lifetime guess 生成。Mutable native templates 留在 plan，不放進 context。完整 T1 declaration／step、shipped host 與 native saver 接線仍待上游保存契約接受。

完整執行的整合行為由 `tests/experiment/workflows/` 在 Engine seam 驗證。`tests/zcu_lab/workflows/` 驗證具體 acquisition、analysis、calibration policy 與 T1 cfg generation 的公開接縫，不測宿主腳本或文件內容。

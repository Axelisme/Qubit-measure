# `zcu_lab.workflows` — 使用者 workflow

**Last updated:** 2026-10-07，原生 analysis 接縫

本 package 擁有具體 workflow 的校準、cfg 組裝、分析、record 與 state。通用迭代、取消、提交與保存路徑由 `zcu_tools.experiment.workflows` 負責。宿主建立 catalog，再明確注入框架；import 不啟動 run。

`v0` 提供 T1 fluxdep、T2echo fluxdep 與 T1 overnight 的期 0 離線示範。Experiment 使用真實 Run／Schedule seam，但 trace 與校準值都是 deterministic 假值，不操作硬體，也不代表新的物理策略。JSON saver 只驗證 exact-path 契約，不是 native 資料格式。期 1 將 T1_fluxdep 換成真實 experiment 與 saver。T2echo／overnight 依期 3 處置，期 3 退役前不保留假校準。不能把示範當成量測入口。

`analysis` 提供純資料的 frequency、length-Rabi 與 T1 fits。它重用 native signal projections 與 fitting helpers，保留原生單位、estimate、uncertainty 與 residual 診斷，不操作硬體、不畫圖、不保存或決定 calibration 接受度。Fit residual 不是 readout SNR，兩者由 workflow policy 分開判斷。完整真實 T1 workflow 與 native saver 接線仍在期 1 實作。

完整執行的整合行為由 `tests/experiment/workflows/` 在 Engine seam 驗證。`tests/zcu_lab/workflows/` 驗證具體 acquisition 與 analysis 的公開接縫，不測宿主腳本或文件內容。

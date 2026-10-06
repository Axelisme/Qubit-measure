# `zcu_lab.workflows` — 使用者 workflow

**Last updated:** 2026-10-06，期 0 離線 caller

本 package 擁有具體 workflow 的校準、cfg 組裝、分析、record 與 state。通用迭代、取消、提交與保存路徑由 `zcu_tools.experiment.workflows` 負責。宿主建立 catalog，再明確注入框架；import 不啟動 run。

`v0` 提供 T1 fluxdep、T2echo fluxdep 與 T1 overnight 的期 0 離線示範。Experiment 使用真實 Run／Schedule seam，但 trace 與校準值都是 deterministic 假值，不操作硬體，也不代表新的物理策略。JSON saver 只驗證 exact-path 契約，不是 native 資料格式。期 1 以真實 experiment 與 saver 替換，不能把示範當成量測入口。

完整執行的整合行為由 `tests/experiment/workflows/` 在 Engine seam 驗證，不測宿主腳本或文件內容。

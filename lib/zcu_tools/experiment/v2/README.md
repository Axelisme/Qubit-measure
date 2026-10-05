# `zcu_tools.experiment.v2`

**Last updated:** 2026-10-05，保留通用 framework

本 package 提供 program/v2 實驗使用的通用 runtime 與工具，不擁有具體實驗、cfg defaults、fit policy 或前端附件。共同實驗介面、records、Result 保存映射與 cfg 組裝見[父層 README](../README.md)。

## Runtime

[runtime](runtime/README.md) 提供 SignalBuffer、Schedule、ProgramBuilder、ResultTree 與 MultiMeasurementExecutor。一般核心實驗編排 host loop 與 program acquire；具體 executor 提供外圈 policy。Buffer、stop、retry 與 lifecycle 契約由 runtime 擁有。

## v2 工具與實驗輔助

`utils/` 擁有 sweep2array、round_zcu_*、merge_result_list、SNR、T1 sampling 與 tracker。硬體量化後的 sweep 必須保持有效點數與順序，collision 直接失敗。`tracker/` 提供 KMeansTracker 與 MomentTracker。

Concrete callers 從這個 package 取得通用機制，並自行擁有量測與分析政策。核心與前端的組合不在 framework import 時執行。

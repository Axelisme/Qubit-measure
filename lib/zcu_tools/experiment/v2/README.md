# `zcu_tools.experiment.v2`

**Last updated:** 2026-10-07，time/frequency sweep hardware grids

本 package 提供 program/v2 實驗使用的通用 runtime 與工具，不擁有具體實驗、cfg defaults、fit policy 或前端附件。共同實驗介面、records、Result 保存映射與 cfg 組裝見[父層 README](../README.md)。

## Runtime

[runtime](runtime/README.md) 提供 SignalBuffer、Schedule、ProgramBuilder、ResultTree 與 MultiMeasurementExecutor。一般核心實驗編排 host loop 與 program acquire；具體 executor 提供外圈 policy。Buffer、stop、retry 與 lifecycle 契約由 runtime 擁有。

## v2 工具與實驗輔助

`utils/` 擁有 sweep2array、round_zcu_*、merge_result_list、SNR、T1 sampling 與 tracker。硬體量化後的 sweep 必須保持有效點數與順序，collision 直接失敗。`tracker/` 提供 KMeansTracker 與 MomentTracker。

Time sweep 座標使用 QICK 自身的 start/span 量化與 signed step 計算；scalar time 則使用
所選 clock 的最近 cycle。不能用同一個 floor 近似兩者，否則 start 會偏一個 cycle，
負向 sweep 的座標還可能錯誤地越過終點。這個計算不取得硬體連線。

Frequency scalar 與 sweep 也使用 QICK 自身的 signed DDS conversion，沒有 half-register
偏移。Absolute RF 的 caller 必須提供 `mixer_freq`，先使用量化 mixer offset 再換算 DDS；
`ro_ch` 選共同 DAC/ADC grid。Regular frequency sweep 使用 start/span 與 signed step
量化，不能逐點 rounding 或把負向 step 當成 unsigned register 解碼。未提供 mixer 時
介面明確代表 DDS 頻率；既有其它 caller 不因 helper 更新就自動取得 mixer 資訊。

Concrete callers 從這個 package 取得通用機制，並自行擁有量測與分析政策。核心與前端的組合不在 framework import 時執行。

---
date: 2026-11-22
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - ../adr/0083-program-macros-and-timelines.md
  - 2026-11-21-projective-span-reuse.md
  - 2026-11-21-required-raw-stop-publication.md
basis:
  investigation_commit: e602f0dc6d11997744bd41b11580012baf37d20e
  measured_before_commit: e602f0dc6d11997744bd41b11580012baf37d20e
  measured_after_commit: a4cd9f1bde21bbd786997be8898173849fdba824
  external_sources: none
---

# ADC raw plan 由誰持有

## 結論

Designer 讓 FTF 與 standalone single-fluxonium 各自把 ADC plan owner 接進現有的 acquire session。ADC 按 prepared batch 身分重用確定性 plan，環境決定何時退役。這項代裁只放寬起初「只改 ADC」的位置限制；同 seed 逐位元、Stop 和 raw 發佈的條件保留。規則見 [ADR-0078](../adr/0078-evolution-module-boundary.md#adc-raw-plan-的-owner-與壽命)。

## 問題

Auto 的 `IQ(order=3)` 要逐 shot 資料，compact 的一階 sums 無法重建三階 feature。當時每輪途經同一份 prepared inputs，卻重建同一批確定性 ADC plan。FTF 的跨 acquire preparation LRU 在入列時計算大小，上限為 4 MiB；standalone single 也能跨 acquire 保留 preparation。若把延遲長大的 plan 掛在這份 preparation 上，FTF 的入列帳本不會反映增加的記憶體。若改由交付出去的 raw source 持有，source 是否存活也不能代表 acquire 是否仍在執行。

## 考慮過的做法

| 做法 | 效果和代價 |
| --- | --- |
| 每輪重算 | 不增加保留記憶體，但重複所有確定性的 plan 工作。 |
| 只靠 raw source 引用交接 | 只改 ADC。丟掉上一輪 source 就可能失去重用；保留舊 source 又可能延長 plan 壽命。 |
| Plan 隨 source preparation | 可以跨 acquire 命中。提前算會讓未取 raw 的路徑也做 solver，延後算則要改 LRU 記帳。 |
| 通用 workspace 與跨內容鍵 | 可以在不同 prepared batches 的 ADC 輸入相等時重用，但要遷移 six 與直接 caller，另驗精確鍵和保留量。 |
| Acquire session 持 ADC owner | 只接 FTF 和 standalone single 的 integrated bind 與既有收尾。丟掉上一輪 raw 仍能重用，換 batch 或結束就退役。代價是留住一個 active batch 的已建事件 plan。 |

## 否決與先例

不選逐輪重算，因為它不減少瓶頸。引用交接和 source preparation 都不能保證 plan 只活在本次 acquire。即使替 LRU 更新帳本，也解決不了這個壽命差異。通用 workspace 則超出這次兩個環境的接線，沒有跨內容重用的效益證據。

裁決遵循 [ADR-0072](../adr/0072-module-boundaries.md#模擬器是封閉的子模塊模型由使用方聲明) 的依賴方向。數值與鍵留在 ADC，兩個環境不共用 acquire 流程。它也沿用 [preparation lifetime](../adr/0078-evolution-module-boundary.md#損耗漂移下的結構與-preparation-lifetime) 與 [required raw](2026-11-21-required-raw-stop-publication.md) 的分工：晚取的 source 保有 immutable inputs，不保有活的 acquire；Program 和 SoC 的 Stop 與發佈契約不變，見 [ADR-0083](../adr/0083-program-macros-and-timelines.md)。

[Projective cavity span 快取](2026-11-21-projective-span-reuse.md) 已採 ADC 管數值與鍵、環境管壽命的分工。這次遵循分工，但不沿用跨 history 的精確數值鍵或 cavity 的 byte cap。ADC raw 只按 prepared 物件身分在輪間重用。

設計覆核補了兩個條件。環境無條件接既有 `on_end`，才能在診斷關閉時也退役 plan。晚到的 raw getter 不持 acquire admission lock，因此由 ADC 同步 memo 的查找、發佈與退役。建構不持鎖；退役後即使計算才完成，也不能回填。

## 留下的問題

完整 `chain_power` 在同 fixture、非排他 1 worker 下，core 由 350.160 s 降為 165.755 s，raw materialize 由 343.737 s 降為 159.082 s。整個 Run 仍用了 183.199 s。Auto core 由 257.464 s 降為 106.475 s。同列前後的 commit 為 `e602f0dc6` 與 `a4cd9f1bd`；兩組數字不外推到 3-worker GUI 策略。180 秒判定只看 [core](../../tools/README.md#實驗-run-牆鐘)，不是 Run 牆鐘。

這次未量 shaped envelope 的保留量，也沒有替 raw plan 設通用 byte cap。保留的 plan 隨 active batch 內容增加，不隨軟體輪數增加。若要跨批次重用，需另驗記憶體與鍵的等價條件。

## 依據

使用者在 2026-11-22 說「好維護與模擬效率是唯二考量，其他都只是建議」，並指定「b，但我重點想放acquire內」。具體 owner 由 designer 代裁，使用者未覆核。調查、覆核與逐列收據的原始工作檔未入庫；比較條件、取捨與量測結論保留於本文。

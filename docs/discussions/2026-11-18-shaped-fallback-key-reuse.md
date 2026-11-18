---
date: 2026-11-18
status: active
superseded_by: null
related:
  - ../adr/0078-evolution-module-boundary.md
  - ../../lib/zcu_tools/simulate/evolution/README.md#routing-telemetry
  - 2026-11-16-shared-acquire-telemetry.md
  - 2026-11-17-engine-capture-events.md
basis:
  repo:
    diagnosis_base: d45e87637975cbbfffa566a8888635dbebe28834
    dp54_diagnosis_candidate: d9b86fbc772dd5d0f005a6d38380e74318f55733
    land_review_candidate: 1482f40e01aee77dbbb32557ff576f86e2e84674
  external: none
---

# 成形脈衝的 fallback 共用與 prepare key 計數

## 結論

這次把 `prepare_key_builds` 加在 Engine 的 `FamilyRouting`，沿用 `estimated_cost` 的相對排程單位，不把它換算成秒。計數用來分辨同一 key 重建多次和一次 build 供多條 timeline 共用。它不保存 key 身分，也不寫進 acquire JSONL。欄位由 [Engine 路由診斷](../../lib/zcu_tools/simulate/evolution/README.md#routing-telemetry) 定義。

修正落在 guard 的執行時機。`period_table` 原本就是第一個可試的 fallback，且多條 timeline 可共用同一張表時，提早解析相應的 Sambe guard。排程可一次建表給實際走到該 fallback 的列。落地審查要求保留同一 timeline 內原有的 Sambe spectral 共用，因為在 Sambe 全收時沒有 fallback 省時可抵銷重做的工。Guard、預算、白名單與價格都不改。

## 問題

在 24 點成形雙 tone 的部分重疊掃描，預設選路有 13 列拒於 `shaped_sambe`，隨後選到 `period_table`。這些列只用一個 table key，但每條 timeline 各建一次，prepare 共花 22.69 thread CPU s。診斷性停用 Sambe 後，同一張表供 46 列共用，只建一次，prepare 用 1.74 thread CPU s。原本的列數與相對排程工作量都看不出這個差別。

## 考慮過的選項與比較

- 只調整估價。DP5(4) 整合後的單次對照中，去重版預設比停用 Sambe 多用 0.425 CPU s，其中兩種 shaped family 的 prepare 加 apply 只差 0.057 s。同樣的 23 條 direct 列已差 0.366 s。這組樣本不能判定兩種 shaped 分配的先後，也不足以推導新價格。
- 用 Engine telemetry 計實際 key slots。它已按 family 觀察 prepare CPU 和 guard 拒收，直接加總能用既有介面比較兩種分配，不必重跑選路。回傳後丟棄的 guard 產品也做了建構，因此計入。沒有回傳產品的 guard 不計，合併相加，重用已算好的結果計零。這個欄位不能分辨一次 key build 裡用了幾次 eigensolve，也不能從同一個數字拆出接受與丟棄產品各自的耗時。
- 用 debug capture 保存細節。遵循 [ADR-0078 的除錯規則](../adr/0078-evolution-module-boundary.md#優化優先於周邊約定除錯靠-dump不靠重算)，要回看某一次的輸入及逐列路由就讀 capture，不事後查詢或重算。這次只需要同一次 evolve 的 per-family 次數和 CPU，沒有把逐鍵產品或新聚合複製到 capture。與[逐輪 capture](2026-11-17-engine-capture-events.md) 不同，這個診斷不追查某個 key 或某一輪的路由。

## 被否決的選項與理由

- 改 acquire JSONL。照[共用 writer 的先例](2026-11-16-shared-acquire-telemetry.md)，writer 明列 acquire 欄位。這次的比較直接呼叫 Engine，不需要在 acquire 層再定義一份 key 診斷。
- 把每個 local Sambe key 都拆開提前做，或把數值產品放進跨 evolve cache。前者會讓同一 timeline 的多個 duration keys 重建同一組 spectral coefficients；後者違反 [preparation 壽命](../adr/0078-evolution-module-boundary.md#損耗漂移下的結構與-preparation-lifetime)。修正只把需要提早處理的 key 和同 timeline、同組 spectral work 的 key 一起解析，既有跨 timeline Sambe key 仍逐 key 解析。

## 留下的問題

提前解析會把部分 Sambe 產品多留一段時間。DP5(4) 整合後的去重原型在排程入口持有 33 個接受的 Sambe keys，payload 自報 6,224,201 bytes。這不是 peak RSS，也不保證其他負載的記憶體上限。沒有讓產品活過一次 evolve。

小型合成案例在兩條 timeline 各放兩個不同 duration keys。四個 key 全收，原候選建了四次同一類 spectral coefficients；保留 timeline 內的共用只需兩次。這只驗證是否重做，不是 wall time 驗收。原候選的正式效能結果不套到落地審查修正後的程式。

## 依據

數字取自 basis commits 所對應的 frozen 掃描診斷、落地審查與合成 regression。原始診斷與審查工作檔未入庫。合成案例由 [shaped sweep 測試](../../tests/simulate/evolution/test_shaped_sweep.py) 重現。當時保留一次最終候選的成對 CPU 驗收，沿用相同的完整 density 預算，不用舊候選的倍率代替。

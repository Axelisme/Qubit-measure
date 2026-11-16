---
date: 2026-11-16
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - 2026-11-16-shared-simulator-measurement.md
  - ../../tools/simulator_measurement.py
  - ../../tools/paired_benchmark.py
basis:
  repo_commit: c82f24522e8009defb7567a17abffe677a85821e
  external_sources:
    - Python time documentation, CPython v3.13.11, accessed 2026-11-17
    - CPython Modules/timemodule.c, v3.13.11, accessed 2026-11-17
    - Microsoft GetProcessTimes documentation, accessed 2026-11-17
---

# 量測樣本時間的值域與成對比值檢查

## 結論

Sample owner 接受 wall、process 的有限非負浮點秒數，0 原樣記錄。負值、NaN 與 infinity 在建構時失敗。成對區塊比值在除法前要求兩側總和都大於 0，否則 raise ValueError，訊息帶區塊序號、clock 與兩個總和。生效契約見 [ADR-0072](../adr/0072-module-boundaries.md#開發工具的數值交接) 與 [Tools README](../../tools/README.md#paired-engine-benchmark)。

這是設計者對開發工具失敗模式的取捨，不是新的使用者需求。使用者 2026-11-15 說「這是流程問題，不是設計裁決，我只看後者」，因此本篇不記為使用者追認。

## 問題

原共用 owner 實作對兩欄都要求嚴格正值。既有 Pareto 樣本不驗這個範圍，兩個公開 seam 測試以固定零 CPU clock 成功執行。重建共用協定同時要求保留既有計時行為，嚴格正值會讓原本跑得完的量表在零樣本時中止。

[Python time 文件](https://github.com/python/cpython/blob/v3.13.11/Doc/library/time.rst) 規定只使用兩次 clock 讀值的差。CPython 的 perf_counter 是單調鐘；process_time 是整個 process 的 user 與 system CPU，不包含 sleep。2026-11-16 的本機調查使用 CPython 3.13.11、Linux，兩個 API 回報解析度皆為 1 ns，分別使用 CLOCK_MONOTONIC 與 CLOCK_PROCESS_CPUTIME_ID。這不是零讀數發生頻率的量測。

當時 perf_counter 約 1.59e4 秒，float ulp 約 1.8e-12 秒；即使讀到 1e7 秒，ulp 約 1.9e-9 秒。改成整數奈秒不能保證 clock 本身更常前進。

[CPython Windows 實作](https://github.com/python/cpython/blob/v3.13.11/Modules/timemodule.c) 以 GetProcessTimes 取得 CPU。其 [官方文件](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes) 使用 100 ns 單位，也明說多核累計可以超過實際 wall 時間。單位不證明實際更新粒度，因此沒有依零值機率或 process ≤ wall 選方案。

## 考慮過的選項與比較

| 選項 | 行為與代價 |
| --- | --- |
| Owner 不驗 | 保留接受域，但負值與非有限值會流入統計與 JSON |
| 有限且兩欄非負，採用 | 保留零讀數；有更強需求的比值在使用點自驗 |
| 兩欄嚴格正值 | 原比值分母自動為正，卻收窄既有公開 seam |
| Process 非負、wall 正值 | 沒有兩個 clock 不對稱的理由 |
| 全面改整數奈秒 | 要改持久 matrix／摘要格式與比較運算 |
| 只在 owner 讀奈秒再換秒 | 保留格式，但改 clock 注入介面，精度增益低於當時解析度 |
| 比值標成不可用 | 要改每區塊一個比值的輸出與估計量 |

採用有限非負樣本，將正總和檢查移到成對統計。既有 dominance 用 0.9 倍 median 比較，slow 規則看首筆 wall 超過 60 秒，報表的固定組比值在分母不正時印 undefined，這些路徑本來能處理 0。原始 matrix 與摘要 median 也都能保存 0。

| 先例 | 立場 |
| --- | --- |
| [依判定用途決定查核](../code-quality.md#依判定用途決定查核與記錄) | 遵循。它提供設計理由，不足以稱嚴格正值已違反先例 |
| 兩個 clock 的對稱統計 | 遵循，不另加 wall 特例 |
| Fast Fail | 遵循，負值與非有限值在入口失敗，比值在判定前失敗 |
| 保留既有結果與計時行為 | 遵循，不因合法零樣本中止整批 |
| [合格只比耗時](../adr/0078-evolution-module-boundary.md#family-去留合格只比耗時) | 遵循，不由 owner 加統計品質門檻 |
| 來源 owner 的嚴格正值 | 區別。原檢查服務比值；現在由比值本身保護 |
| 物理 duration 必須正值 | 區別。要求執行的物理時長不是實測 clock 差 |
| 報表 undefined | 區別。顯示缺值不參與判定，成對比值會影響 telemetry 與變慢判斷 |

獨立挑戰後保留原選擇，另比較 owner 內奈秒與統計不可用兩個變體。五處理由也收緊為設計理由、保留依觀測值比較的行為、公開 seam 成功含零 elapsed、逐列區別先例，以及不宣稱零值發生機率。

## 被否決的選項與理由

不驗樣本會讓假鐘的負值或 NaN 靜默進統計；只靠非負樣本又會在零分母時除以零。兩欄嚴格正值則拒絕既有合法讀數。不對稱限制缺少理由。

全面奈秒會改兩份持久格式；只改 owner 仍要換 clock 介面，當時沒有足夠收益。補 epsilon、重取、丟區塊或填零會改估計量。Baseline 總和 0 沒有比值；candidate 總和 0 的比值 0 是 clock 解析度造成的假讀數。它會進 median、IQR 與 telemetry 判定，所以選明確失敗，不用「統計不可用」繼續比較。

## 留下的問題

零值的頻率沒有量過。Windows 的實際 CPU clock 更新粒度也未證，選擇不依賴它。

兩個 family 都讀成 0 可能被判等，一方 0、另一方一 tick 可能被判支配。Owner 不保證解析度以下的速度結論。Family 去留仍依超過 10% 規則與刪除前的人工數據審閱，不由 owner 補救。

## 依據

- [Sample owner](../../tools/simulator_measurement.py) 與 [summarize_blocks](../../tools/paired_benchmark.py) 擁有樣本與比值契約。
- [共用量測用法](../../tools/README.md#shared-simulator-measurement)。
- 上列 Python、CPython 與 Microsoft 原始來源於 2026-11-17 重讀。100 ns 是單位，不用它推論更新頻率。
- 原始本機調查、提案與挑戰工作檔未入庫。本文保留測得的環境、解析度、選項與限制。

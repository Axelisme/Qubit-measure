---
date: 2026-11-21
status: active
superseded_by: null
related:
  - ../adr/0066-operation-lifecycle.md
  - ../adr/0078-evolution-module-boundary.md
  - ../../tools/README.md
  - 2026-11-21-experiment-run-wall-time.md
basis:
  repo_commit: ce116aa67f8f517aad184d3bc46062f73135b6f3
  external_sources: none
---

# 牆鐘工具如何記錄 Run 失敗

## 結論

Designer 代裁：Controller 回報 Run 失敗時，牆鐘工具先寫 `run_failed` 列。只有 Controller 已回到 idle 才派送下一列。`run_failed` 算已判定，且可在相同 commit 和輸入下留用。報表完成與否由未判定列決定，失敗另列診斷。操作與 exit status 見[工具說明](../../tools/README.md#實驗-run-牆鐘)。

## 問題

原工具遇到 Controller 回報失敗就拋出例外。當時的模擬批次發現兩個 Run 撞到 upload buffer 上限，失敗的 adapter 沒有列，後續 adapter 也沒被派送。只看批次錯誤無法區分是 Run 跑不起來，還是量測工具本身壞了。原始診斷未入庫。

## 考慮過的選項與比較

選擇用 Controller 的 failed 終態當作單列結果。Controller 在發布終態事件前先 settle handle，工具保留 handle 的錯誤診斷，不猜是模擬器限制、表單問題或 bug。這遵循 [ADR-0066](../adr/0066-operation-lifecycle.md#問題與決策) 的生命週期分工：Run owner 決定終態，工具查讀結果；仍有活動中的操作時不繼續派送。這也沿用 [ADR-0078](../adr/0078-evolution-module-boundary.md#模擬器的測試與量測分層不做-e2e) 的量測例外。失敗是 Run 的觀察結果，工具不把失敗變成精度門檻，也不替它補算秒數。

## 被否決的選項與理由

- 遇到一列失敗就拋出工具錯誤。不只留下批次缺列，也丟失該 adapter 的可追查結果。
- 寫下失敗列但一律停家族。若 Controller 已 idle，其餘獨立的實驗仍可量，不必另開程序。
- 續跑時自動重試失敗列。相同 commit 和輸入下，失敗就是前次所觀察到的結果。要重跑可明確選該列，不從含該列的報表留用。

## 留下的問題

報表提供診斷，不替維護者分流。要修模擬器能力、表單預設或程式缺陷，需各自定位。

## 依據

先前採用整個 Run 作牆鐘指標的理由見[實驗 Run 牆鐘工具](2026-11-21-experiment-run-wall-time.md)。Controller 的終態發布由 [Run service](../../lib/zcu_tools/gui/app/measure/services/run.py) 擁有。續跑與逐列落盤沿用原有 [牆鐘工具](../../tools/experiment_wall_time.py) 的做法；本次裁定新增可留用的失敗終態，並以報表是否有未判定列區分 exit status。

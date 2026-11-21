---
date: 2026-11-21
status: active
superseded_by: null
related:
  - ../adr/0078-evolution-module-boundary.md
  - ../../tools/README.md
  - 2026-11-21-experiment-run-wall-time.md
  - 2026-11-21-wall-time-run-failed.md
basis:
  repo_commit: fce233b5ce8d593d73897869bd9fcb1da550952a
  external_sources: none
---

# 牆鐘報表如何重建中斷列

## 結論

Designer 代裁：中斷列不填 Run 秒數或下限。只有派送前紀錄而無結果的列記為 `incomplete`，連派送前紀錄都沒有的列記為 `not_run`。重建只需 `--report-only --output`，移除 `--ended-at`。這兩種列不算已判定；正式 `timeout` 的 Run 下限與留用規則不變。現行用法見[工具說明](../../tools/README.md#實驗-run-牆鐘)。

## 問題

合成 journal 只存了派送前的 `start.json`。重建時傳入晚 5 秒的批次截止時刻，舊工具就顯示該 Run 至少耗時 5 秒。原始審查工作檔未入庫。

若程序在派送前中斷，這 5 秒可能完全是工具時間。若程序在終態後、結果落盤前中斷，差值也可能大於已完成的 Run。即使截止時刻正確，也無法用它證明 Run 下限。

## 考慮過的選項與比較

留下狀態，不補秒數。這遵循 [ADR-0078](../adr/0078-evolution-module-boundary.md#模擬器的測試與量測分層不做-e2e) 採用的計時範圍，只量送出 Run 到終態。也沿用[失敗列](2026-11-21-wall-time-run-failed.md)的做法：未取得可報的 Run 秒數就留空。工具自己觀察到 Run 超過單列期限時才記正式下限。

## 被否決的選項與理由

- 沿用派送前紀錄到批次截止的差值。它可能把 Run 之外的時間報成下限。
- 把差值改名、另設一欄，或在工具內另存批次截止時刻。沒有報表判定會用這些值；外層 runner 已有批次收據。
- 另寫派送與終態 journal。這會增加每列落盤和重建分支，只為填一個不進「Run >180 s」清單的數字。續跑也會取代這個數字。

## 留下的問題

中斷列要有可用耗時，仍須重跑。這次不改寫既有報表；先前將 journal 差值標為 Run 下限的列，本來就不在已判定數與超過 180 秒的清單內。

## 依據

計時指標沿用[實驗 Run 牆鐘工具](2026-11-21-experiment-run-wall-time.md)記錄的使用者裁決。這次依據的候選 commit 列在檔頭。

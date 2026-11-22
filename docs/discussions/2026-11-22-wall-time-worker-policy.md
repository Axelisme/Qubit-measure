---
date: 2026-11-22
status: active
superseded_by: null
related:
  - 2026-11-21-simulator-core-time.md
  - 2026-11-22-simulator-speed-first-verification.md
  - ../adr/0069-experiment-core-and-frontends.md
  - ../../tools/README.md
basis:
  repo_commit: 6602f8985376b94a1d5055cb703699bbd5c64c93
  cli_commit: 4a2dea87690bfcb2764886b2e5feb0b2113510c9
  external_sources: none
---

# 牆鐘工具的 worker 數與量測範圍

## 結論

CLI 保留單 worker 預設，量慢實驗子集時用 `--workers` 明示 worker 數。這批量測用非排他 runner 取得 CPU slot，再用 GUI 的 `max(1, CPU 數 − 1)` 策略選值。4 核 slot 用 3 個 workers。Runner 收據記下 payload 參數與 CPU affinity，供核對本批與續跑的條件。指令和續跑契約見 [Tools README〈實驗 Run 牆鐘〉](../../tools/README.md#實驗-run-牆鐘)。

## 問題

首表用 FTF factory 的單 worker 預設，GUI startup 則保留一顆 host CPU，其餘用作 worker。[FTF 演化可把同次呼叫的多條 timeline 分給不同 worker](../../lib/zcu_tools/simulators/ftf/environment/README.md#workers-and-engine-budget)。原本的單 worker 秒數不能回答採 GUI worker 策略時要等多久。Runner 將數值函式庫的執行緒設為 1，這跟 FTF 的 worker 數是兩個設定。

## 考慮過的選項與比較

沿用[從完整 Run 讀核心耗時](2026-11-21-simulator-core-time.md)的量法，把 [GUI startup 的 worker 策略](../adr/0069-experiment-core-and-frontends.md#顯式-cfg-與裝置環境)用在 slot 分給該批的 CPU 上。明示 worker 數讓同一工具可以保留舊預設，也讓收據區分不同的平行寬度。已有的單 worker 秒數保留作歷史參照，不重跑。依[先做局部取證的裁決](2026-11-22-simulator-speed-first-verification.md)，這批只找慢列與覆蓋缺口，不代替排他全表。它沒有同條件的前後兩批配對，也不是成對速度比較。

## 被否決的選項與理由

- 繼續用單 worker 的量測代表 GUI 策略。兩種設定可用的演化平行度不同。
- 這次子集也排他使用整機。排他會擋住其他工作，完整全表留到收尾。

## 留下的問題

收尾的完整全表另排排他時段，再依分到的 CPU 數選 worker。子集沒有取得核心耗時或可用下限的列，仍留作未判定。

## 依據

2026-11-22，使用者選擇不以完整全表作為每次落地關卡。Designer 依操作員等待時間的目標，選定這次子集的 worker 策略。原始工作檔未入庫。

---
date: 2026-11-15
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - 2026-11-16-frozen-simulator-inputs.md
  - 2026-11-16-paired-engine-timing.md
  - ../../tools/README.md
  - ../../tests/README.md
basis:
  repo_commit: c82f24522e8009defb7567a17abffe677a85821e
  external_sources: none
---

# 模擬器測試與量測的分層

## 結論

測試與量測分成 kernel、dispatch、序列、前端與 environment。入庫的 program → acquire benchmark 與成對計時的 acquire adapter 退休，計時改包 `Engine.evolve`。生效的規則、例子與除錯例外由 [ADR-0078](../adr/0078-evolution-module-boundary.md#模擬器的測試與量測分層不做-e2e) 擁有。

使用者選擇保留手寫多段 `TimelineBatch` 的序列層，沒有把後端測試縮成單段 kernel 與 dispatch。除錯時仍可跑一次 e2e 定位，但腳本不入庫，也不作驗收門檻。

## 問題

原成對工具準備量 program → acquire，同時經前端、演化與 I/Q。這與使用者 2026-11-15「後端與前端分開測，kernel與dispatch也分開測，不該有e2e的測試需求」衝突。

調查時，舊 `benchmark_simulator.py` 有 307 行，workloads 有 556 行，走端到端路徑。Family 工具七檔共 4,443 行，其中五檔 import tests 的 oracle 或案例表；oracle 兩檔共 591 行。工具輸入會隨測試案例與前端一起改，無法只比較模擬器變更。

## 考慮過的選項與比較

| 選項 | 能回答的問題 | 缺口與成本 |
| --- | --- | --- |
| 後端加序列層 | 手寫多段 batch，同表量最終 density 誤差與總耗時 | 要重新宣告固定控制，但不必啟動 program 或讀出 |
| 只留 kernel 與 dispatch | 單段數值誤差與選路 | 沒有整條序列的最終誤差，也不能判斷整體是否變慢 |
| 保留 acquire e2e，只記耗時 | 使用者路徑的總耗時 | 仍將多層成本混在一起，無法定位某次後端修改 |

使用者對這三項答「A」，選加序列層。這遵循先前要求最終精度、整體沒有變慢，以及刪除 family 前看同型態長序列證據的裁決，見 [Family 去留](../adr/0078-evolution-module-boundary.md#family-去留合格只比耗時)。單段誤差不能代替這些判斷。

原成對工具的 acquire adapter 是僅供參考的設計，不足以推翻使用者的新邊界。固定場景可以重複最核心的 pulse 序列，沿用 [工具不依賴可變 lab 的原則](../adr/0072-module-boundaries.md#使用者空間的依賴原則)。

除錯例外另比較完全禁止 e2e 與保留一次定位路徑。使用者答「我選B」，允許分層都重現不了的問題跑一次 e2e，但不留下腳本或驗收門檻。

## 被否決的選項與理由

- 原 acquire adapter 讓成對差異混入編譯與讀出，不符合分層。
- 只有 kernel 與 dispatch 會遺失序列最終誤差的判定位置。
- 把誤差全交 pytest、工具只量耗時，無法在同一張量表比較誤差與耗時。
- 把獨立參考 solver 搬進 `simulate`，會失去參考實作與 production 演化的獨立性。改採數值 codec 交接，solver 留在 tests，見[凍結數值討論](2026-11-16-frozen-simulator-inputs.md)。

## 留下的問題

Dispatch 查詢的唯讀 history 與 Floquet guard 規則由 [ADR-0078](../adr/0078-evolution-module-boundary.md#黑盒邊界是-simulateevolution) 擁有，不在本篇另定一份。Environment 的注入與各層測試 owner 見 [tests README](../../tests/README.md#evolution-精度矩陣)。

凍結數值 I/O 不代表整個 `simulate` 已封閉。模型聲明、邊界型別與 import contracts 的後續工作不由這次分層取代。

## 依據

- [ADR-0072](../adr/0072-module-boundaries.md) 與 [ADR-0078](../adr/0078-evolution-module-boundary.md) 保存生效的責任與使用者原話。
- [Tools README](../../tools/README.md#simulator-benchmark) 保存量測入口。
- 原始調查、選項比較與挑戰工作檔未入庫。本文寫入當時的規模、選項與否決理由，不要求讀取那些工作檔。

---
date: 2026-11-16
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../../tools/README.md
  - ../../tools/family_pareto.py
  - 2026-11-16-family-pareto-cache-only.md
  - 2026-11-16-tools-standardization.md
  - 2026-11-16-shared-simulator-measurement.md
basis:
  repo_commit: c82f24522e8009defb7567a17abffe677a85821e
  external_sources: none
---

# Exclusion estimator 退休

## 結論

退休 `guard_sweep_cost.py` 的 `measure_engine_cost`、`EngineCost` 與唯一測試，不搬到 tools，也不留相容層。它對 `tools.family_pareto` 的反向 import 一併消失。量表的 reference 改為必填，生效責任見 [ADR-0072](../adr/0072-module-boundaries.md#開發工具的數值交接)。

收窄 guard 或改 family 的實際耗時影響改用成對計時比較前後版本。這是設計者對沒有現存使用者的開發 helper 所作定案，不記為使用者追認。

## 問題

Estimator 對同一個輸入分別量固定 family 與排除 family 的成本。分層要求移除 tests → tools 的反向 import，問題是搬到 tools，還是刪除這個能力。

來源 commit `160301c98` 為一次 guard 邊緣量測提供 helper，該需求已交付。後續 review 發現它與 Pareto 重複 mode、暖機與時計協定，於是改委派 `measure_family`，形成反向 import。這個版本留在 Git commit `e8192521c`。

調查時唯一 caller 是 `test_exclusion_estimator_preserves_policies_modes_and_observed_paths`，沒有非測試 caller。測試只核兩個 policy、三個 mode、每格五筆與路徑，沒有耗時門檻。白名單本身另有 evolution 測試。Helper 約 100 行，專用測試約 21 行，而 Pareto 主檔當時已有 1,291 行。

## 考慮過的選項與比較

| 選項 | 成本與失敗模式 | 可否回頭 |
| --- | --- | --- |
| 原地保留 | 反向 import 不消失，違反邊界 | 可改，但不能作終態 |
| 搬到獨立 tools module | 約 120 行搬遷與測試，多一個只有自己測試的 module；真 Engine 計時還需運算授權 | 可從獨立 module 再刪 |
| 併入 Pareto | 編排職責相近，但超標主檔再增約 100 行，並與計時 cutover 共改一檔 | 可抽回，仍沒現存 caller |
| 放共用計時 owner | 將 scenario／policy 編排塞進只收 prepared session 的 owner | 要再拆回，違反 owner 契約 |
| 退休 | 移除約 125 行與專用測試，不需執行真實計時 | 需要時可從 Git 歷史取回 |

| 先例 | 立場 |
| --- | --- |
| 使用者要求 guard 邊緣量測 | 區別。一次量測已交付，沒有要求常駐此 helper |
| Tools／tests 分工 | 遵循，搬移或退休都移除反向 import |
| 入庫能力需要現存使用者 | 遵循退休，不把一次性 runner 原樣留成通用工具 |
| 速度工作把 DOP853 座標 factory 搬到 tools 的 producer 側 module，使用者追認 | 區別。座標 factory 有 workload 與凍結座標的現存使用者，estimator 沒有 |
| 使用者先前不保留一次性驗收 runner 的裁決 | 遵循，以需求而非是否有 CLI 判斷。入庫範圍的相關比較見[工具標準化](2026-11-16-tools-standardization.md#先例與立場) |
| [t6 channel 移除](2026-11-15-t6-channel-removal.md) | 遵循收窄沒有現存需求路徑的取捨，區別 production 算法與開發 helper |
| [測試契約](../../tests/README.md#放置新增測試) | 遵循移除邏輯就移除專用測試，不新增已刪除斷言 |

獨立挑戰後修正三處比較。第一，不再把沒有 CLI 當刪除理由，改看實際使用者。第二，將併入 Pareto 與新增量表欄位拆開。第三，搬移原案只等場景接縫，不必等計時 owner，兩案的排程成本相近。最後仍以使用需求與真實計時授權的成本選退休。

## 被否決的選項與理由

不原地保留反向 import。不新增只有自己測試的 module，也不為同一個缺少使用者的能力擴大 Pareto。共用計時 owner 不接受場景與 policy 編排。新增 `--exclude FAMILY` 量表欄位會是新功能，當時沒有使用需求。

`reference=None` 的 production 路徑只服務 estimator，沒有為它保留量表分支。成對工具直接經共用計時 owner，不需要這個例外。

## 留下的問題

若要估計「整個 family 不可用」的反事實成本，可用公開 `Engine(families=...)` 加共用計時 owner 寫一次性 script。等兩處工作確實需要，再依入庫工具原則設計。退休 helper 不改白名單與 guard 的公開能力。

## 依據

- [量表 cache-only 遷移](2026-11-16-family-pareto-cache-only.md) 與 [共用計時](2026-11-16-shared-simulator-measurement.md)。
- Git commit `160301c98` 與 `e8192521c` 保留來源與後續委派版本。
- 原始調查與挑戰工作檔未入庫。本文保存使用現況、選項與修正理由，不引用原本的封存工作目錄。

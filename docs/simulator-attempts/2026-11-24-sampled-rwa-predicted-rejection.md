---
date: 2026-11-24
status: paused
commit: 234bf075f043c1cb4e663ff9bf1686ad9772e1ac
modules:
  - lib/zcu_tools/simulate/evolution
  - lib/zcu_tools/simulators/ftf/environment
discussion: none
superseded_by: null
---

# 預測拒收能否跳過 sampled RWA 探針

**讀者：** framework-dev、agent。查哪些探針真的付了積分成本，以及為何不改拒收規則。

## 核心思路

不同載波雙線的同形振幅群先[比價](2026-11-19-shared-multitone-pricing.md)。只有比價選中 sampled RWA 的群才做數值檢查。每個 spot 從固定初態分別算 RWA 與完整驅動 DOP853，取兩個端點密度矩陣的最大元素絕對差。所有 spot 的差都不超過 `0.1 × budget`，群才通過。一次 FTF preparation 內的 [exact memo](2026-11-23-ftf-rwa-probe-memo.md) 可重用完全相同 spot 的比較結果，不重用演化輸出。

想省掉原規則必拒的積分。有兩個可查的方向：用探針前已有的量證明端點必超標，或等一個 spot 真的拒收後，跳過同群其餘 spot。這次只查證據，沒有實作候選。

## 實測

五場離線 FTF Run 各一次，另做一次重建正例的 Engine evolve。量測版為 `234bf075f`。五場共用 block 2 manifest `c2b6b9e0522e` 與 `ftf-demo:initial` fixture。兩批都用 CPU affinity 0–3、非排他、OpenCL 關閉、一個 worker、一個 native thread、seed 123、`time_scale=0`。六個輸入的 `budget` 都是 `1e-3`，端點差門檻為 `1e-4`。

第一批先用空的私有快取跑重建正例，再依序跑 `chevron`、`constructive_phase`、`cz_amplitude`、`rb`。第二批只跑 `rb_heralded`，沿用第一批的快取。`chevron` 和 `rb_heralded` 是各自 runner 的第一場，所以標記 `cold_start=true`。這個標記不代表快取是空的。完整 Run 的 core 與 wall 見[單次 audit](speed-ranking.md#sampled-rwa-probe-audit)。

量測擷取原路由的 spot 判決、積分批次與 memo 命中。探針前的量從實際用來比較的切段輸入補算，沒有重跑探針。下表是 `234bf075f` 上各單次 Run 或正例 evolve 內的總計。探針 CPU 欄用原 `probe_integration_cpu_s`，只計 RWA 與完整驅動參考積分的呼叫執行緒 CPU。41 個新積分批次各有一個 spot，不必把多 spot 批次的成本均攤。正例從空快取開始。若區間內觸發首次編譯，其成本也算在 16.533659 s 裡。原始工作檔未入庫。

| 場景 | 比價跳過 | 新積分 | Memo hit | 拒收／通過 | 探針積分 CPU s 合計 | 端點密度差最小值 | 中位數 | 最大值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `two_qubit/chevron` | 20 | 8 | 0 | 8／0 | 4.421990 | `1.904617911e-4` | `2.317940690e-4` | `2.747055657e-4` |
| `two_qubit/constructive_phase` | 2 | 24 | 2 | 26／0 | 30.015491 | `1.904617911e-4` | `2.515526169e-4` | `2.795145092e-4` |
| `two_qubit/cz_amplitude` | 2 | 4 | 2 | 6／0 | 4.961535 | `1.904617911e-4` | `2.233478721e-4` | `2.562339530e-4` |
| `two_qubit/rb` | 9 | 4 | 0 | 4／0 | 4.423224 | `1.038564462e-4` | `1.042402461e-4` | `1.164139351e-4` |
| `two_qubit/rb_heralded` | 304 | 0 | 0 | 0／0 | 0 | — | — | — |
| `simultaneous_drive_retained24`（重建） | 0 | 1 | 0 | 0／1 | 16.533659 | `9.943252513e-6` | `9.943252513e-6` | `9.943252513e-6` |

五場 Run 的四個拒收場景共留下 44 個拒收比較結果。40 個來自新積分，合計 **43.822241 呼叫執行緒 CPU s**；4 個來自 memo hit，未重積分。這是原積分成本，不是省時或 wall。加上重建正例的一次通過，共 45 個有端點比較結果的 spots。另有 337 個 spots 由原比價跳過，沒有端點誤差可報。單次 `rb_heralded` Run 的 150 次 evolve 沒有 RWA routing telemetry 記錄，但 capture 記下 304 個 `cost_skipped` spots，沒有數值拒收。

### 重建正例

1. 以量測版 `234bf075f` 的 [`tools/family_pareto_scenarios.py`](../../tools/family_pareto_scenarios.py) 為準。複製 `FTF_MODEL`，把 `numerics.truncation.retained_levels` 從 32 設為 24，保留 `numerics.seed=123`。僅在建輸入期間，用副本替代該模組的 `FTF_MODEL`。
2. 以 `simultaneous_drive_input().build()` 建立 source 和 batch。產生器停用 drift，設 `time_scale=0`，用同步 F1/F2 Gaussian 脈衝與共用 gain 掃描。保留原來的 24 點，gain 由 0.01 到 0.1。確認 source 的密度矩陣維度和 batch 點數都是 24。
3. 用 `Engine(1, backend="cpu")` 對這一個 batch 呼叫一次 `evolve`，傳 `budget=1e-3` 和 `telemetry=True`。不要傳 family allowlist。讀 `result.paths.routing["multitone_rwa"].rows` 確認實際採用的 logical rows。若要核對群與 spot 判決，在同一次 evolve 傳 `capture=CaptureBuffer()` 並保留 buffer，從 [capture 的 `sampled_rwa_admissions`](../../lib/zcu_tools/simulate/evolution/README.md#engine-capture) 查 `accepted`，不另跑探針。這次一個 spot 通過，路由選了 24 個 RWA logical rows。

這是從 tracked 產生器重建的正例，不是[歷史 ABBA 輸入](2026-11-18-sampled-multitone-rwa-admission.md#實測)的逐位重播，也沒有量 ABBA 的 before/after。

## 為何採用或否決

在 `234bf075f` 的六個輸入各一次的紀錄中，`omission_phase_rad` 在唯一通過 spot 上是 `2.9983267303e-6` rad，在拒收 spots 中最小是 `9.1083519123e-5` rad，出現在 `two_qubit/rb`。後者是前者的 **30.378×**。這個 proxy 估計省略項的相位量。樣本分開不證明它是原端點密度誤差的下界，也不證明任何切點能保留原拒收判決。既有[中途超標、端點卻通過的反例](2026-11-23-sampled-rwa-probe-early-rejection.md#實測)也排除了用中間態差值代替端點的做法。

按 scene、evolve、group 聚合這六個輸入的 spots，共 382 群，每群只有一個 spot。這不表示每群只有一個 logical row，重建正例就有 24 個採用列。由群大小可知，等到第一個 spot 的端點比較真的拒收後，群內已沒有下一個 spot 可跳過，可少做的比較是 **0 次**。這是從擷取結果推得，沒有執行短路版本。

結論是暫停。保留原估價、逐 spot 端點比較、exact memo、fallback 和 capture。沒有 before/after、候選耗時或實測收益。

## 留下的線索

有方法能證明原路由、輸出逐值及實際 capture 不變，且原 FTF caller 上確有工作可省，才重開。屆時用同一批 FTF 場景與重建正例核對這些條件。若要改求解器、選路、公開診斷或誤差預算，另行決定。這份觀測紀錄不改 `budget` 的含義。

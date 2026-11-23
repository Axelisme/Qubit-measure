---
date: 2026-11-23
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - ../adr/0084-simulation-integer-time-base.md
basis:
  investigation_commit: ffd2a0c6ccb12e3f4cb9de0a7094e246aca237e6
  external_sources: none
---

# FTF sampled RWA 探針的快取壽命

## 結論

使用者選 A。FTF 在一次完整 preparation 內重用精確相同的 sampled multitone RWA 探針證明，只保存各探針的最大 density 差值和通過與否。每次 evolve 仍選路、登記用量、演化並發佈自己的 capture。不同點的物理 span 相同，仍分開計入用量。

2026-11-23，使用者答：「A，我再次聲明，只有可維護性與速度是模擬器的唯一要求，其他的都只是建議」。這項裁決只推翻 [ADR-0078〈損耗漂移下的結構與 preparation lifetime〉](../adr/0078-evolution-module-boundary.md#損耗漂移下的結構與-preparation-lifetime)對 sampled guard 證明跨 evolve 使用的禁令。Engine 不常駐 memo；FTF 在 preparation 完成、取消或失敗時丟棄它。

## 問題

在 `ffd2a0c6` 的 `two_qubit/chevron` 四點離線 Run 中，`_spot_check` 用了 176.36／180.77 caller CPU s（workers=1／7）。其中 workers=1 的 full-drive DOP853 參考積分用了 175.29 CPU s；production DOP853 apply 另用 86.92 CPU s。這些是各一次四點 Run 的 CPU 總和，不是牆鐘。

Fixture 使用 2×2 頻率與長度掃描、gg herald、5795–5805.5 MHz、0.06–0.10 µs、reps=100、rounds=1、seed=123、24 retained levels、projection floor `1e-5` 和 `time_scale=0`。兩輪都用 CPU、單一 native/Numba thread、非排他。128 次探針全拒收，其中只有 48 組不同輸入。80 次重算用去 107.8／117.7 caller CPU s（workers=1／7，同一對 Run）。只比原 prepare key 會漏掉 shifted model 和損耗；不能拿它判定探針輸入相等。

## 選項與比較

| 選項 | 當時的判斷 |
| --- | --- |
| A：暫存精確探針證明 | 同一 preparation 內只省相同輸入的積分，不改通過門檻或路由記帳。 |
| B：記住同類拒收 | 放寬「同類」會略過原來能通過的列。 |
| C：多次拒收後停止探針 | 掃描前段全拒、後段可通過時，會失去後段的 RWA 加速。 |
| K：不改 | 省去快取維護，但保留已量到的 80 次精確重算。 |

B 和 C 改變何時讓 guard 檢查一列，K 不處理重算，所以未採用。裁決保留 [ADR-0072](../adr/0072-module-boundaries.md) 的依賴方向和 [ADR-0084](../adr/0084-simulation-integer-time-base.md) 的整數時基與相位換算，沒有讓完整結果或損耗相關 payload 跨 evolve。和 ADR-0078 的一般成本上限規則不同，這次沒有為 guard 再加斷路器或積分上限。

## 留下的問題

掃描若每列都不同，仍須逐一積分。若之後量到某場景的 guard 成本高於 production，再用該場景證據討論改法。

## 依據

數據來自 `ffd2a0c6` 上對上述離線 fixture 的兩次 warm profile。分析逐列比對探針的模型、損耗、geometry 和控制內容，並用 caller 的 thread CPU 計時。原始量測與裁決工作檔未入庫。

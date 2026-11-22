---
date: 2026-11-22
status: superseded
superseded_by: 2026-11-23-curated-wall-time-scenes.md
related:
  - 2026-11-21-simulator-core-time.md
  - 2026-11-21-wall-time-coverage.md
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - ../adr/0082-component-flux-frame.md
basis:
  repo_commit: 6602f8985376b94a1d5055cb703699bbd5c64c93
  dispatch_candidate_commit: 00a8c3f6adb328b035c68a8acf98fa5ceb8503eb
  external_sources: none
---

# 固定 FTF fixture 缺路由的十列

## 結論

這次保留工具自持的固定 FTF 場景。十個實驗照原 Open／Form 路徑走，未派送的列保留工具實際寫出的 `missing_defaults` 與欄位錯誤。覆蓋摘要逐列標「缺 fixture 路由／校準，未量」，與上游輸出、JPA 無模型和逾時分開。交付只對這十個具名缺口例外，不放寬其他缺值。這些列沒有核心秒數或下界，不參與 180 秒判定，也不做逾時補量。列出缺口不等於接受完整表。[工具說明](../../tools/README.md#實驗-run-牆鐘)記錄判讀方式。

Designer 依 2026-11-21 的使用者授權代裁。使用者尚未覆核此裁決。

## 問題

工具在 `ftf-demo/initial` 上量正常輸入的 Run。調查發現八列缺同一條快磁通線，另兩列缺 MG12 的共用 DC coil。板型有空 generator，只表示能分配一個位置。它不提供 generator 到 loop 的 route、以 Φ0/gain 記的係數、延遲或濾波。共用 coil 是另一條靜態路由，初始 bias 和每個 loop 的校準都會影響所量的工作點。

| 實驗 | 未量的原因 |
| --- | --- |
| `fastflux/distortion/acc_phase` | 缺 Q1 的快磁通線。 |
| `fastflux/distortion/freq` | 缺 Q1 的快磁通線，也缺 Q1 的 `kappa` 或有依據的頻窗。 |
| `fastflux/distortion/phase` | 缺 Q1 的快磁通線。 |
| `fastflux/mist` | 缺 Q1 的快磁通線。MIST 的模型適用域另待澄清。 |
| `fastflux/t1` | 缺 Q1 的快磁通線。 |
| `fastflux/twotone` | 缺 Q1 的快磁通線，也缺 Q1 的 `kappa` 或有依據的頻窗。 |
| `twotone/reset/flux/gain` | 缺 Q1 的快磁通線與固定 `flux_reset_length`。 |
| `twotone/reset/flux/length` | 缺 Q1 的快磁通線與固定 `flux_reset_gain`。 |
| `mux/onetone_flux_global` | 缺 MG12 具名的共用 DC coil，以及 Q1／Q2 對該 coil 的校準。 |
| `mux/twotone_flux_global` | 缺同一條共用 coil 和 Q1／Q2 校準，兩個 tone 也缺 `kappa`。 |

## 考慮過的選項與比較

保留固定場景，讓原工具如實記缺值，能區分未派送和遇到性能瓶頸。這沿用[以目前 FTF point 為基準](2026-11-21-wall-time-coverage.md)的裁決，也遵循 ADR-0072 的自持輸入和 ADR-0078 的完整 Run 計時。原交付方法只允許有原因的上游輸出缺口。這次讓十個路由缺口也能隨報表交付，但不把它們算成已量或豁免。

補上同場景且有來源的 route 是另一個選項。兩個掃頻列還須取得 Q1 linewidth 或可用頻窗，兩個 reset 列仍須拿到固定 pulse 校準。另建工作點也有可能補齊 route，但須先確認它能代表正常輸入，不能和舊表當成同一工作點。

## 被否決的選項與理由

- 啟用 [`low-snr.yaml`](../../zcu_lab/simulation_envs/FTF/low-snr.yaml) 的 F1/gen1 示意 `+0.02 Φ0/gain`。這會改變波形和演化，不是補一個不影響量測的欄位。另做示意診斷也只回答示意輸入的成本。
- 借用 local source、coupler trim 或 six-fluxonium 的 coil。它們不提供 MG12 在本 FTF 場景對 Q1／Q2 的具名 route 和校準。
- 當時就另建校準場景。當時沒有同場景可追溯的校準輸入，新工作點還要獨立驗收。
- 將十列列為 `not_runnable` 或新增報表狀態。前者略過 Open，遮掉欄位錯誤。後者要改 journal 和續跑契約，逐列覆蓋摘要已能指出缺口。

## 留下的問題

取得同一 FTF 場景的可追溯 route、單位、校準、工作點和正常輸入後，才檢查能否補量。若新增共用 DC source，先核對同一 point 的其他實驗是否改了 bias 或初值。補量用新輸入雜湊另開批次，不沿用舊批。MIST 另外需要對照 [guide](../../zcu_lab/v2/fastflux/mist/guide.md) 與 [FTF 磁通路徑](../../lib/zcu_tools/simulators/ftf/environment/README.md#flux-pulses)。可排同時 microwave 與 flux，不代表該掃描的脈衝形狀和幅度已在模型適用域內。

## 依據

- [工具 FTF 設定與 point](../../tools/data/experiment_wall_time/) 有 local DC sources，沒有 Q1 快脈衝 route 或 MG12 共用 coil。快脈衝與靜態路由的設定見 [FTF field guide](../../zcu_lab/simulation_envs/FTF/field-guide.md)，具名 DC 校準見 [ADR-0082](../adr/0082-component-flux-frame.md)。
- 逐列調查以 `6602f8985` 和 `00a8c3f6a` 為依據。原始提問、調查、覆核與派送紀錄未入庫。上表只列可核對的缺口，不把短試跑當這十列的正式計時。

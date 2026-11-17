---
date: 2026-11-16
status: active
superseded_by: null
related:
  - docs/adr/0078-evolution-module-boundary.md
  - lib/zcu_tools/simulate/evolution/README.md
basis:
  - "repo base@5c78022c0e35721503a4721c2e153cc28bd0b77a"
---

# 常數雙 tone 的 secular guard 與唯讀診斷

## 結論

這次在常數雙 tone 的最後 admission 加入 ε_dom²·Ω_ret·T ≤ 1.476e-4 cycle，保留共用的一階 endpoint kick。新增 package-root `multitone_rwa_diagnostic`，讓呼叫者讀到 guard 已算出的最終指標。Kernel guard 與 query 使用同一 admission fact，不各自實作一次判定。現行數值與欄位契約由 [evolution README](../../lib/zcu_tools/simulate/evolution/README.md#constant-pair-rwa-guard-diagnostics) 持有。

這次沒有改價格、sampled-row admission 或其他 family 的門檻。證據只涵蓋固定輸入，沒有完成日常 FTF 常數雙 tone 的 32、288、512 ns、1 µs 驗證。未驗證不是不支援，也不代表算錯。

## 問題

Endpoint kick 消除了被省略 Fourier 邊的一階端點項，沒有消除它們與保留耦合之間隨時間累積的誤差。固定的 ground-state A16 模型在 1600 ns 下仍被放行，完整 density 最大絕對差為 `7.352059944e-3`，高於 `2e-3`。兩級獨立參考解的差為 `3.335506316e-10`，低於 `2e-6`。

只有 acceptance 布林值不足以區分數值拒絕與尚未計算的比較。Restore 改變 Fourier mask 後，前一輪 transfer／phase 值也不能當成最終結果回報。

## 考慮過的選項與比較

保留 kick 而不加最後 guard，會繼續放行上述長脈衝。另寫一套 query 判定則可能在門檻、early return 或 restoration 上與 guard 分歧。採共用 admission fact，能同時供 kernel guard 與公開 query 使用，也能沿既有 WorkPlan 壽命釋放暫存結果。

這次使用 final resolved mask 的 ε_dom 與 Ω_ret。前者是被省略的非零 gain-weighted coupling 與絕對 detuning 比值的最大值，包含對角項；後者是所有保留 tone／sign coupling 絕對值最大值的兩倍。乘上 row-local duration 後以 cycle 比較，不加 `2π`。這是校準代理量，不是任意模型或序列的 density 誤差證明。

診斷沿用 `RWAGuardMetric | RWAUncomputed`。已算出的最終數值即使未通過仍保留，沒算的欄位不偽裝成零或失敗。Restore 後若 dimension 先超限，dimension 仍可見，但未完成的 final-mask omission 與 accumulation 不回報前一輪數值。Same-port 是合法查詢，結果為拒收；未知 port、非有限輸入、負 duration 或非法 budget 才拋 `ValueError`。

這次遵循 [ADR-0078 的公開黑盒與共用數值 owner](../adr/0078-evolution-module-boundary.md)。外部只用 package root，Fourier arithmetic 留在既有 Fourier owner，admission／retention 規則及 facts 留在既有 RWA numeric owner。另遵循 [common-frame 成本討論](2026-11-14-common-frame-step-pricing.md) 的分工：數值 guard 判斷適用域，價格由 dispatch 持有，不將成本補丁混進本次修正。

## 被否決的選項與理由

用既有錯誤輸出當參考會掩蓋近似誤差，因此完整 density 仍對獨立 full-Hamiltonian Lindblad 解比較。把 A16 參考解或保存矩陣放進永久拒絕測試，會讓 guard 契約依賴昂貴求解或固定資料；永久測試改用 test-owned 公開 builder 和 `NoValidKernel`，一次性超標證據另行保留。

沒有把所有常數雙 tone 都禁用，也沒有從固定案例推論日常 FTF 已驗證。沒有為這個 query 新增 generic decision trace、跨呼叫 cache 或 dispatch inspection 介面。

## 留下的問題

常數 guard 內側的兩 phase 案例在 `184.5*(1-1e-6)` ns 下，accumulation 為 `1.475998524e-4` cycle，完整 density 誤差為 `1.014148242e-3`，兩級 reference 差為 `4.751899e-11`。外側 `184.5*1.01` ns 的 `1.49076e-4` cycle 被 query 與 pinned Engine 一致拒絕。這只支持該凍結輸入，不認證所有被 guard 放行的列。

價格未重新校準。日常四個時長、預設選路端到端驗收、較長序列及效能比較仍需各自的輸入與執行授權，不能由這份數值證據代替。

## 依據

- [RWA numeric owner](../../lib/zcu_tools/simulate/evolution/kernels/_rwa_numeric.py) 與 [Fourier owner](../../lib/zcu_tools/simulate/evolution/kernels/_rwa_fourier.py) 持有共享判定與係數計算。
- [Family query](../../lib/zcu_tools/simulate/evolution/kernels/multitone_rwa.py) 定義只讀結果型別與失敗契約。
- [公開診斷契約測試](../../tests/simulate/evolution/contract/test_rwa_diagnostic.py) 涵蓋放行／拒收、same-port、early return、restore dimension 與非法輸入。
- [固定模型 builders](../../tests/simulate/evolution/contract/multitone_cases.py)、[guard-edge builder](../../tests/simulate/evolution/contract/guard_edge_cases.py) 與 [精度矩陣](../../tests/simulate/evolution/contract/test_accuracy_matrix.py) 持有可重新驗證的輸入與完整 density 接縫。

A16 一次性原始工作檔未入庫。本文保留模型來源、ground 初態、時長、超標數字與兩級參考解一致性；模型參數已由上述 test-owned builder 入庫，不依賴本機參數。參考解使用 rtol／atol `2e-10/2e-12` 與 `2e-12/2e-14`。本文沒有引用或使用使用者未提交的 FTF 參數。

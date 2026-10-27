---
date: 2026-10-27
status: active
superseded_by: null
related:
  - zcu_lab/workflows/coherence_fluxdep/form.py
  - lib/zcu_tools/experiment/v2/utils/t1_sampling.py
  - lib/zcu_tools/notebook/analysis/fit_tools/README.md
basis:
  - "repo@a38f18679，含 fit identifiability（關聯矩陣）、硬體時間成本模型、flux offset 模型 v3"
  - "Monte Carlo 與 Fisher 解析原型為未入庫的工作檔（七輪，2026-10-27），結論與必要數字寫在正文"
---

# T1/T2 over flux 量測策略：哪些機制有效，哪些沒有

## 結論

1. 自適應選 flux 點（Bayesian／Ds-optimal 批次貪婪）對六個損耗參數的精度沒有可量到的增益。即使規劃端完全知道真值（oracle），σ 也只到均勻掃描的 0.90–0.96 倍；只有 n_th 例外（0.49–0.66）。
2. 唯一廣泛有效的機制是每條 T1／echo 曲線的 delay 取樣：把約 80% 的 shots 集中在 0、約 1T̂、5T̂ 三個區域，其餘 20% 均勻分布。在 T 估計誤差 50% 下，單曲線 σ 仍為均勻 delay 的 0.73 倍，傳到 A_phi、n_th 0.73、x_qp 0.79、Q_cap 0.80；相當於用同樣硬體時間得到約 1.9 倍的量測量。
3. 因此不需要新的 `bayes_coherence` workflow。advisor 建議只在現有 `coherence_fluxdep` 的 T1／T2 表單加一個 delay 模式（`uniform`／`robust`），其餘需求（interleaved 順序、sweet spot 多配時間、停止、時間估計、關聯矩陣）現有功能已能做。此建議待使用者裁決，並待第 8 輪端到端 MC 確認增益傳到六參數後仍在。

## 問題

Staircase 原本設想一個 `bayes_coherence` workflow：從帶誤差的先驗出發，在有限硬體時間內自適應選 flux 點與量測種類，產出 Q_cap、Q_ind、x_qp、Temp、A_phi、n_th 的擬合值與關聯矩陣，並在模擬上證明它比均勻掃描更準。七輪離線評估要回答：這個設計的增益來自哪裡，值不值得做成 workflow。

評估設定（各輪共用）：fluxonium 正向模型，T1 由 Q_cap、Q_ind、x_qp、Purcell、Temp 決定，echo T2 由 1/f flux noise 的 A_phi 與光子散粒噪聲的 n_th 決定；singleshot 量 Pe 以約束 Temp；flux 範圍 [0.5, 1.0]；預算 43200 s；硬體時間用已 land 的成本模型；60 組真值抽樣；量測端 flux 有殘餘校準誤差；擬合用 repo 的 `analyze_t1`、`analyze_t2echo_fluxdep` 與六參數聯合擬合。

## 考慮過的選項與比較

| 輪次 | 問題 | 結果 |
| --- | --- | --- |
| 1–2 | 兩階段 `bayes_coherence`（先頻率再全量，Ds-optimal 貪婪批次）能否收斂 | 收斂，但均勻掃描的對照組只涵蓋 flux 0.50–0.64，比較不公平 |
| 3 | 修正對照組，五種策略 | 自適應策略與均勻掃描差異不明顯；均勻只用掉 61% 預算 |
| 4 | 對照組用滿預算，成對統計 | 自適應 V0 對 Uniform、Uniform-interleaved 的每回合誤差差值，95% bootstrap 區間全部跨 0：無差異 |
| 5 | 根本原因：三個假說 | H1「模型失配」不成立（校準誤差設 0 仍無差異）；H2「設計空間槓桿小」成立（oracle σ 比 0.90–0.96，n_th 0.66，A_phi 1.11）；H3「估計誤差」不成立（oracle 規劃下仍無差異）。擬合失敗率 0.1–0.4%，不是原因 |
| 6 | 其他可調之處的 oracle 上限（解析 Fisher，c-optimal 與 D-optimal） | flux 點時間分配 0.90–1.00（n_th 0.49）；量測種類比例 0.90–1.03，現行 46/40/14% 已接近最佳 34/48/18%；flux 範圍擴到 [0, 1] 無增益（哈密頓量以 0.5 為中心鏡像對稱）；**delay 取樣 0.63–0.67（max 5T）、0.59–0.64（max 10T）**，且參數間不互相排擠 |
| 7 | 六種 delay 設計在實際條件下（T 估計誤差 s = 0／0.3／0.5；擬合固定單指數；真實衰減為雙指數或 stretched 指數） | 見下表 |

第 7 輪的單曲線 σ 比（相對均勻 0–5T̂，T1 與 echo 相同）：

| 設計 | s = 0 | s = 0.3 | s = 0.5 | 形狀錯配的 χ² 偵測率 |
| --- | --- | --- | --- | --- |
| 均勻 0–5T̂（現行） | 1.00 | 1.00 | 1.00 | echo 100% |
| repo `t1_delay_axis(uniform=False)` | 0.95 | 0.93 | 0.93 | 100% |
| 三點最佳 O3，max 5T̂ | 0.65 | 0.65 | 0.88 | 0.7–3.6%（等於虛警率） |
| 三點最佳 O3，max 10T̂ | 0.61 | 0.59 | 1.09 | 同上 |
| 穩健版 R（80% 三區、20% 均勻） | 0.70 | 0.69 | 0.73 | 99–100% |
| 幾何間距 | 1.06 | 1.09 | 1.06 | 100% |

使用者在第 4 輪後的兩個判斷都被第 5 輪檢驗：「會不會是因為選點假設數據是無偏的？」（H1、H3，不成立）與「預期資訊量都假設一定能測量成功」（失敗率 0.1–0.4%，影響可忽略）。

## 被否決的選項與理由

- **自適應 flux 選點**：設計空間本身沒有槓桿（第 5、6 輪），不是演算法或估計的問題。
- **三點最佳 delay 設計**：只在 T 已知時最佳；T 估計誤差 50% 時失去增益，而且三個點看不出曲線形狀，對雙指數、stretched 指數完全偵測不到。
- **幾何間距 delay**：比均勻差。
- **擴大 flux 範圍到 [0, 1]**：鏡像對稱，沒有新資訊。
- **調整 T1／echo／singleshot 時間比例**：現行比例已接近最佳。
- **新 workflow `bayes_coherence`**：剩下的有效機制只是 delay 陣列的產生方式，現有 workflow 加一個選項即可。

## 留下的問題

- 第 8 輪端到端 MC（U、R-prior、R-interleaved，T̂ 取前一 pass 的擬合值）尚在執行；若增益沒有傳到六參數，則連 delay 模式也不做。
- 第 7 輪每點 SNR 為 50，擬合失敗率全為 0；低 SNR 下各設計的失敗率未量。
- 真實 echo 衰減形狀（單指數或 Gaussian）未由使用者確認。若為 Gaussian 而擬合固定單指數，R 的 T2 偏差小（+2%，均勻為 −8 到 −13%）但散布大（RMSE 為均勻的 2.0–2.6 倍）；保留 20% 均勻點可偵測錯配，之後再切換擬合形狀。
- Destination 的「比均勻掃描更準」與 `bayes_coherence` 名稱待依裁決改寫。

## 依據

- 正向模型與擬合器：`zcu_lab/workflows/coherence_fluxdep/`、`lib/zcu_tools/notebook/analysis/`（關聯矩陣與弱參數標示見 `fit_tools/README.md`）。
- 現有非均勻 delay 取樣：`lib/zcu_tools/experiment/v2/utils/t1_sampling.py`。
- 第 1–7 輪的 Monte Carlo 與 Fisher 解析原型、報告與圖為未入庫工作檔；本文數字取自各輪報告。

---
date: 2026-10-30
status: active
superseded_by: null
related:
  - lib/zcu_tools/program/acquisition.py
  - lib/zcu_tools/program/herald.py
  - docs/discussions/2026-10-29-heralding-post-selection.md
  - docs/discussions/2026-10-29-readout-classifier.md
basis:
  - "repo@97cc8b441"
  - "取數 processor 的巢狀軸設計是 2026-10-30 的使用者裁決，記在未入庫的工作檔；本文需要的內容寫在「取數機制」一節"
  - "沒有外部來源；推導只用兩態高斯混合模型"
---

# Readout 優化指標：用兩次讀出的一、二次矩算 SNR 與 QND

## 結論

1. Readout 參數優化做成一個單 qubit 實驗。每發 shot 依序是製備（I 或 π）、第一次讀出 R1、等待 τ、中間 pulse（I 或 π）、第二次讀出 R2。另有一組分支把 R1 關掉。製備、中間 pulse、R1 開關三個選項總是全開，每組 readout 參數共 8 個分支。
2. 不需要分類參數。取數時 processor 對每個點累加 R1 與 R2 的 IQ 一、二次矩，分析只用這些矩。
3. 指標有七個：SNR = Δ/σ、分離上限保真度 F_sep、熱布居 p、R1 之後的狀態保持 S、QND 保真度 Q = (1 + S)/2、兩個方向的翻轉率 ε↑ 和 ε↓、中間 π 沒翻到的比例 f。f 用來看量測造成的 leakage，以及量測後 π 失準。
4. 優化目標先不自動化。分析畫出 F_sep 與 Q，標出 Pareto 前緣，由人挑參數。
5. R2 用和 R1 同一組參數，掃描時一起變。
6. 這個實驗等新的取數 processor 框架完成後再做。

## 問題

使用者一直想改進 readout 參數優化的指標：「我一直想改進優化readout的指標，苦於沒有snr/qnd的合理指標」。當天在設計 heralding 用的取數 processor 時，決定 Result 存累加總和，總和裡會帶兩次讀出 IQ 的交叉項。使用者因此問：「iq交叉項在沒有分類參數時拿來算關係係數，似乎可以作為優化readout的指標？」接著要求把它展開：「snr與qnd指標怎麼算？以及pulse sequence怎麼安排，假設pi pulse是完美的，除了製備用pi pulse，兩次readout間再放一個開關的pi pulse能不能算出什麼有趣的指標？」

## 取數機制

在 repo@97cc8b441，`acquire_processed` 每個 ADC 各收一個 processor。processor 回傳每發 shot 的值，由框架對 reps 取平均。heralding 的設計討論把它改成下面的形式，本文的指標建立在這個形式上。寫這篇時它還沒有實作。

- 只有一個 processor，它同時看到所有 ADC 每發 shot 的 IQ。
- 每次讀出給每發 shot 一個向量。分類讀出是 k 類的 one-hot；不分類的讀出是 `(1, I, Q)`。
- processor 對每個掃描點輸出所有讀出向量的外積，再對 shot 加總。幾次讀出就有幾個軸，例如 `[IQ(R1), IQ(R2)]` 的每點輸出形狀是 `(3, 3)`。
- 框架只跨 round 累加，Result 存總和原樣。

同一次讀出可以在軸清單裡出現兩次。`[IQ(R1), IQ(R1), IQ(R2), IQ(R2)]` 的每點輸出是 `(3, 3, 3, 3)`，共 81 個數。其中恰有兩個軸不在第 0 格的元素就是全部二次矩，例如 ΣI₁²、ΣI₁Q₁、ΣI₁I₂、ΣQ₂²。第 0 格全取的元素是 shot 數。

## 推導

### 模型

- 每次讀出投影到狀態分離方向上：x = μ_g + Δ·s + n。s = 0 代表 g，s = 1 代表 e；n 是高斯雜訊，在 IQ 平面各方向的標準差都是 σ。
- 製備前的熱布居是 p（e 的比例）。π pulse 完美。
- R1 加上等待 τ 會讓狀態翻轉：g→e 的機率是 ε↑，e→g 的機率是 ε↓。

### SNR：只用 R1

1. m₁ 是 R1 在製備 π 與製備 I 下平均 IQ 的差。它的方向 û 就是分離方向。
2. σ² 是 R1 共變異數矩陣在垂直 û 方向的變異數。這一步用到雜訊各方向相同的假設。
3. v 是沿 û 的變異數減去 σ²，等於 Δ²p(1−p)。兩種製備的 v 相同，因為 p(1−p) 對調 p 與 1−p 不變。
4. 因為 m₁ = Δ(1−2p)、v = Δ²p(1−p)，所以 Δ² = m₁² + 4v。熱布居是 p = (1 − m₁/Δ)/2。

SNR = Δ/σ。兩個高斯在中點切開時，每個狀態的誤判率是 ½ erfc(SNR / 2√2)，所以分離上限保真度 F_sep = 1 − ½ erfc(SNR / 2√2)。F_sep 不含製備誤差，也不含讀出窗口內的衰減。

熱布居 p 是順帶得到的，不需要分類，也不需要另外量。

### QND：加上 R2

1. m₂ 是 R2 在兩種製備下的平均差，投影到 û。
2. S = m₂ / m₁ = 1 − ε↑ − ε↓，表示 R1 加上等待之後，狀態對比剩下幾成。
3. Q = (1 + S)/2。
4. 兩個方向分開算：ε↑ − ε↓ = P₂(I) + P₂(π) − 1，其中 P₂ = (x̄₂ − μ_g)/Δ，μ_g = x̄₁(I) − Δp。

S 有第二種算法：同一個製備內，R1 與 R2 投影的共變異數除以 v。用平均值算的 S 和用共變異數算的 S 不一致，表示模型假設不成立，例如讀出窗口內的衰減讓分布不再是高斯，或有 leakage。這個檢查只能偵測，不能修正。

S 包含 R1 窗口加上 τ 這段時間的 T1 衰減。R1 關閉的分支量的是同樣長度內只有 T1 的對比，S_off = m₂(R1 關) / m₁(R1 開)。量測本身造成的部分近似為 S_on / S_off。

### 中間 π

在 π 完美、只有兩能階時，中間 π 只是把 g 與 e 的標籤對調。m₂ 和共變異數都變號，S_mid = −S，算不出新的翻轉率。

它的用處是檢查 π 有沒有真的翻到。設 π 有 f 的比例沒有作用，則 S_mid = −(1 − 2f)S，所以 f = (S + S_mid)/(2S)。理想值是 0。f 不為 0 有兩個來源：

- R1 讓 qubit leakage 到 f 能階或更高，ge π 移不動這部分布居。
- R1 之後 resonator 的光子還沒散完，AC Stark shift 讓 π 失準。這時「π 完美」在量測之後就不成立，這正是實際最想知道的事。

R1 關閉時的 f 是沒有量測時的 π 基準。f(R1 開) − f(R1 關) 就是量測造成的部分。掃 τ 時，f(τ) 降到平台所需的時間，就是量測後要等多久才能再打 pulse；heralding 讀完 herald 之後要等多久，也由它決定。

## 考慮過的選項與比較

| 題目 | 選項 | 結果 |
|---|---|---|
| 指標形式 | 單一相關係數 ρ；拆成 SNR、Q、p、ε、f | 拆開 |
| 二次矩來源 | `[IQ(R1), IQ(R2)]`；每次讀出列兩次 | 列兩次 |
| 優化目標 | A：畫 F_sep 與 Q 的 Pareto 前緣，由人挑；B：Q 過門檻時取 F_sep 最大；C：合成分數 F_sep × Q | A |
| 參考分支 | 中間 π 與 R1 關閉做成預設關的開關；總是打開 | 總是打開 |
| R2 參數 | A：和 R1 相同，一起掃；B：固定成參考讀出，只掃 R1 | A |

## 被否決的選項與理由

- **單一 ρ 當指標**：ρ 在分離方向上約為 S·v/(v + σ²)，把 SNR、QND 與布居混在一起。qubit 幾乎都在 g 時 p(1−p) 很小，readout 再好 ρ 也低。拆開後每個量各自可解釋。
- **只用 `[IQ(R1), IQ(R2)]`**：外積只有兩次讀出之間的交叉項，沒有 ΣI₁² 這類同一次讀出的二次矩，算不出變異數與 ρ。每次讀出列兩次就有完整的二次矩，代價是每點 81 個數。
- **參考分支預設關閉**：使用者否決，理由是「qnd有個很重要的指標是高能階leakage」。中間 π 是不需要分類就能看到 leakage 的分支，關掉就看不到。
- **R2 固定為參考讀出**：mux 版需要同一個 program 裡有兩組不同的 mux 讀出設定，目前的 mux readout 不支援，heralding 的設計也只先做同設定讀兩次。R2 和 R1 同參數時，兩次讀出的 Δ 與 σ 相同，模型最簡單。

## 留下的問題

- 優化目標 B、C 只是延後，等看過實際 Pareto 前緣的形狀再決定要不要自動選點。
- τ 要固定還是掃描。f(τ) 隨 τ 下降的部分推測來自 resonator 光子、平台來自 leakage，這個分法還沒驗證。
- 模型假設的範圍。相位雜訊或放大器壓縮會讓雜訊在各方向不同，σ 的估法就不成立；讀出窗口內的衰減讓分布不是高斯，Δ 與 σ 變成等效值。
- leakage 狀態在 IQ 平面另成一團時，Δ² = m₁² + 4v 不再成立。要不要改用三類分類，或改成三成分混合模型，還沒討論。
- S_on / S_off 是近似，前提是量測造成的翻轉與 T1 衰減可以相乘分開。
- mux 版本與雙 qubit 的串擾指標還沒討論。

## 依據

- repo@97cc8b441 的 `lib/zcu_tools/program/acquisition.py`：`acquire_processed` 每個 ADC 一個 processor，框架對 reps 取平均。
- repo@97cc8b441 的 `lib/zcu_tools/program/herald.py`：現行 herald 只看單一 ADC 的一次讀出，只保留判為 g 的 shot。
- QICK 0.2.436 的 `qick_asm.py`：每個 round 在 host 用 float64 對逐發 int64 積分取平均，再平均各 round。新框架改成跨 round 累加總和，累加誤差和 QICK 原生平均同一個量級。
- 本文的推導只用上面的兩態高斯混合模型，沒有引用外部文獻。

## 後記

2026-11-01 的實作將 `twotone/ro_optimize/power` 改為八分支 paired-readout gain sweep，保留 R1 開／關與中間 I／π 的比較。分析呼叫 `zcu_tools.analysis.readout_metrics.readout_metrics`，保存 paired moments 並畫各指標與 F_sep／QND-on Pareto 前緣，不自動選 gain。操作與時序限制由 [Power guide](../../zcu_lab/v2/twotone/ro_optimize/power/guide.md) 說明，native 變更見 [格式歷程](../storage-format-history/2026-11-01-readout-power-qnd.md)。此處記錄實作內容，不代表已合併或完成實機驗證。

2026-11-01 的時序修正讓製備 I 與中間 I 都發射同一 qubit cfg 的零增益 pulse，π 使用 cfg 增益。I 不再省略 pulse，時槽由原生編譯器決定，不用 nominal_duration 手算延遲。八個分支的 reset、製備、R1、等待、中間、R2 位置逐一相同，R1 到 R2 的 trigger 間隔一致。R1 off 仍保留零增益 generator 時槽與 ADC trigger，R2 使用共同的掃描 gain。

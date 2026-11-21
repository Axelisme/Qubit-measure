---
date: 2026-11-21
status: active
superseded_by: null
related:
  - ../adr/0078-evolution-module-boundary.md
basis:
  - "本 repo main 0fb47e36f，加上 mist-cavity-reuse 的落地候選 767c478b"
---

# MIST 腔場重用：ADC 擁有 flat span 的結果

## 結論

`mist/flux_dep` 在正常參數下的 FTF Run，從 281.5 s 降到 163.3 s。做法是讓 ADC 既有的腔場 memo owner 也收錄 flat span 求解出的各 state 端點場，不再只收二階矩。FTF 照舊選態、建 Stark／dephasing shift，並把場傳給下一段，不組 ADC 的 key。這放寬了 ADR-0078「末態場不進 memo」的一部分；主分支選擇與 shaped／callable 路徑仍不進 memo，projective 路徑沿用既有的二階矩 memo，不收錄端點場。

這是 designer 依使用者授權（「我授权你的范围除了 push 还有真实硬体之外都可以做」）做的代裁，使用者尚未覆核。

## 問題

在正常參數下（seed 123、101 bias × 101 gain、1000 reps × 10 rounds），`mist/flux_dep` 跑完要 281.5 s，其中讀出 246.1 s，目標是 3 分鐘內。定位結果：

- 既有的 `CavityAverageCache` 在 40,804 次查詢中零命中。key 含 pulse 振幅，而 probe 開始前的兩段 span 在不同 gain 下振幅不同，但這些 pulse 在 span 結束前都還沒作用。
- 不比對這種 solver 讀不到的振幅時，二階矩、端點場與 `SegmentShift` 各有 20,200 次可重用。
- 只重用二階矩，毛省上限是讀出的 37.88%，但要達標需要 41.25%。所以端點場也必須能重用。

ADC 公開的入口只有平均值，FTF 沒辦法在不複製 ADC 規則的情況下判斷兩段 span 是否等價。

## 考慮過的選項與比較

- **A：ADC 給一個 opaque 的等價 key，FTF 在自己的 composition 內重用整段 span 的產物**（平均值、端點場、shift）。ADC 的 memo 契約不用改，但同一份 ADC 數值會由 ADC 和 FTF 兩個 owner 各存一份，而且兩份的總記憶體沒有界線。
- **A′：ADC 擁有 flat span 結果的重用**（採用）。ADC 的單一 owner 在一次求解中記下二階矩和所有 state 的端點場：
  - key 由 ADC 定義，只忽略 span 結束前都未作用的 pulse 振幅，只在初始時間等於 span 起點時收錄；
  - 兩種產品合計 64 MiB，滿額不收錄、不逐出；
  - 壽命與清理沿用既有的一次 acquire composition。

  FTF 只做自己的 chi、選態和 shift。
- **B：只讓既有的二階矩 memo 命中。** 改動最小，但毛上限 37.88% 低於門檻。
- **C：FTF 自己組 ADC 的 key。** FTF 要懂 ADC 的時間邊界、carrier 與切段規則，兩處日後可能不一致。

挑戰時，A 被指出三點：和「單一 owner」的界線沒說清楚；「全段 flat、span 結束前沒有作用中的 drive」只是一個 guard，不是完整的等價證明；新增產品的總記憶體沒有界線。A′ 是挑戰中提出的變體，它讓每個模組只快取自己的數值。

## 被否決的選項與理由

- A：同一份 ADC 產品會由兩個 owner 保留，違反 ADR-0078〈算法內聚，共用數值有單一 owner〉。
- B：毛上限低於門檻，第 6 步達不到。
- C：ADC 的規則會被複製到 FTF。

## 先例與立場

- ADR-0078〈算法內聚，共用數值有單一 owner〉，使用者 2026-10-21「前兩個採用建議」：
  - 遵循的部分：ADC 擁有、lossless key、64 MiB 上限、一次 composition 的壽命、FTF 不組 key、主分支選擇與 shaped 路徑不進 memo。
  - 放寬的部分：「末態場不進 memo」只放寬到 ADC 自己算出的各 state 端點場。
  - 理由：該裁決寫明的理由全部保留，ADR 也沒有寫排除端點場的獨立理由；而只重用二階矩達不到門檻。
- ADR-0078〈先保證正確，再優化瓶頸〉：遵循。驗收要求重用前後同 seed 的輸出一致。

## 結果

落地候選的正式排他重測（clean commit、CPU 0–7 排他、9 份 fixture 的 hash 與原表相同）：

| 秒 | 原表 | 落地候選 |
| --- | ---: | ---: |
| 整個 Run | 281.5087 | 163.3068 |
| Readout | 246.1067 | 127.2003 |
| Evolve | 25.1233 | 25.5915 |

兩次 Run 在不同 commit 上，差異不能全部歸因於快取。審查時補了兩處。第一，不同的實際 carrier 以數值型別與位元編碼分成不同的 key，不會誤命中；它們照常收錄。第二，窄精度時間讓 ringdown 端點不同、無法證明等價時，照舊分別求解、不收錄。

## 留下的問題

- 只在 FTF 模擬器上量測，沒有接硬體。
- 若之後 `SegmentShift` 的建構成為顯著成本，再另外量測，決定 FTF 要不要在自己的 composition 內重用。

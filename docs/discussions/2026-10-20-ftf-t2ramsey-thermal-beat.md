---
date: 2026-10-20
status: active
superseded_by: null
related:
  - docs/adr/draft/simulation-environments.md
  - docs/adr/0071-coherence-units.md
  - lib/zcu_tools/analysis/fitting/base/cos.py
  - zcu_lab/v2/twotone/time_domain/t2ramsey/core.py
  - zcu_lab/simulation_envs/FTF/default.yaml
basis:
  - "repo@4e50be72b：撰寫時的 main HEAD；FTF default.yaml temperature_k 0.05，t2ramsey 分析 fit_phase 預設 False"
  - "FTF 離線模擬 run 20261020T123451Z-02ff7d（twotone/ge/t2ramsey，fringe 模式，相位斜坡 0.5 MHz，delay 0–10 µs）"
---

# FTF 示範環境的 t2ramsey fringe 尾部失準：熱態拍頻與固定相位

## 結論

尾部失準有兩個原因，都不是 T1 基線漂移。

主因是多體熱效應。示範環境溫度 50 mK，旁觀 fluxonium f1 的激發態熱占據 20.5%。目標 qubit 的微波同時驅動 f1 在基態與激發態下的兩條躍遷（0→4 為 5522.803 MHz，2→8 為 5522.755 MHz，矩陣元 0.0147 與 0.0148 幾乎相等），兩條差 58.8 kHz，Ramsey 訊號是兩個頻率的相干疊加，在約 10.4 µs 出現拍頻極小值。單指數 `decaycos` 模型把前 10 µs 的拍頻包絡當成加速衰減，擬出表觀 T2 約 5.5 µs；去掉拍頻後的本徵 T2 約 10.5 µs。雙頻模型使尾部殘差 RMS 從 0.28 降到 0.017。

次因是分析預設鎖相位。`fit_phase=False` 把相位固定在 0° 或 180°，資料起始相位是 325°，優化器只能把頻率扭到 0.508 MHz（偏離 24 kHz）來補相位，RSS 從 3.8 放大到 42.2。

使用者裁決：t2ramsey 分析的 `fit_phase` 預設改 True，固定相位仍可手動選；示範環境的溫度與 twotone 的 reset 預設不動，拍頻是真實物理，留給使用者視需要調整。

## 問題

使用者用 FTF 模擬 qubit 跑 twotone/time_domain/t2ramsey（fringe 模式）時，擬合曲線在長 delay 端系統性偏離資料。Fringe 由第二個 π/2 的相位斜坡產生（名義 0.5 MHz，硬體取整後 0.500093 MHz），不是真的頻率失諧。

## 考慮過的假說與比較

| 假說 | 檢驗 | 結果 |
| --- | --- | --- |
| H1 T1 基線漂移：delay 期間 population 向熱平衡鬆弛，`decaycos` 的常數 y0 吃不下 | 模型加 `B·(1−exp(−t/T1))`，T1 取前序量測 38.2 µs | AIC 只動 0.5，尾部殘差不降；放開 T1 收斂到非物理的 0.79 µs。否定 |
| H2 包絡不是單指數 | 對包絡取 log 看線性度；比較高斯與雙指數 | 高斯略優（AIC −377 對 −338）但解釋不了殘差的週期性。次要 |
| H3 fringe 頻率隨 delay 變化 | 前後半段分段擬合；逐週期峰谷頻率；雙頻模型 | 前半 535.7 kHz、後半 513.4 kHz，逐週期從 526–556 kHz 掉到 500.1 kHz；雙頻模型 f1=0.5225、f2=0.5813 MHz，殘差降 16 倍。成立 |
| 固定相位 | 同資料放開相位重擬 | 相位 324.6°，RSS 42.2→3.8。成立 |

硬體時間與相位取整只貢獻 93 Hz，不是頻率漂移的來源。

## 被否決的選項與理由

- 把示範環境降到 10 mK 或預設啟用 reset 來壓掉拍頻：使用者判斷拍頻是真實的多體熱效應，示範環境應保留，由使用者自己決定是否降溫或加 reset。
- 在 `decaycos` 加 T1 基線項：證據不支持。
- 分析改成依擬合品質自動放開相位：多一層判斷，直接改預設更可預測。

## 留下的問題

- FTF 示範手冊是否要載明：50 mK 且無 reset 時，多體電路會有可見的熱激發，Ramsey 會看到 cross-Kerr 拍頻與表觀 T2 偏低。
- twotone 實驗是否預設帶 reset，另議。

## 依據

- 分析模型：`lib/zcu_tools/analysis/fitting/base/cos.py` 的 `decaycos`；t2ramsey 的投影與相位斜坡：`zcu_lab/v2/twotone/time_domain/t2ramsey/core.py`。
- 模擬去相干：`lib/zcu_tools/simulate/ftf/model.py` 把 `aphi` 轉成 Markov 純去相位速率；溫度與 f1 參數在 `zcu_lab/simulation_envs/FTF/default.yaml`。
- 數字來自顧問派出的唯讀調查（重做投影與擬合、依 secular 模型算該工作點的躍遷頻率與熱占據）。原始調查工作檔未入庫，必要數字已寫在正文。

---
date: 2026-11-14
status: active
superseded_by: null
related:
  - ../../tools/README.md
  - ../../tools/family_pareto_scenarios.py
  - ../../tools/data/family_pareto/random_gate_sequences.json
basis:
  repo_commit: f9ec2fe2e70418c6af2a54e8b41df603c1bdc651
  sequence_source_commit: 46a4291cfdf1b1617bc1fb03499813450b16ed7e
  external_sources: none
---

# RB 型相位序列作為固定量表輸入

## 結論

使用者於 2026-11-14 對新增 constant 與 Gaussian 兩個 RB 型場景的建議答「采用建议」。這兩個場景保留既有 N、S 的模型與 gate 參數，只換相位序列。原有循環場景不變。

Designer 依使用者先前「只複製最核心的pulse序列來模擬」「場景自然是挑選過的」的邊界，選擇離線產生固定相位資料。工具只編譯資料，不帶第二份 runtime Clifford algebra。固定輸入與限制由 [tools README](../../tools/README.md#fixed-scenarios-companions-and-reference-cache) 說明。

## 問題

N 的 X、Y、−X、−Y 循環會讓部分 RWA 誤差抵消。既有 N 表的結果不能代表其他相位次序。當時的離線調查在隨機相位序列看到超過 2e-3 的完整序列誤差，因此需要讓固定量表也涵蓋這種輸入。調查的原始工作檔未入庫，正式誤差由新場景的參考解與量表摘要記錄。

原 RB 程式由 seed 產生 Clifford prefix，再加完整 inverse recovery。它用 virtual Z 追蹤 phase frame，並區分 X90 與 X180 等 physical gates。工具只需這些 gate 的相位次序，不需要執行實驗、讀取校準或維護完整代數。

## 考慮過的選項與比較

採用的方案從使用者原作離線產生 24 點 phase prefix 與 recovery phase，並保存來源 commit、原始軸、seed 及選取 index。20 個 seed 與 9 個 depth 先以 seed-major 展開，再取等距整數 index，包含兩端。Constant 與 Gaussian builder 共用這份資料。

另一方案把 Clifford decomposition、inverse 與 virtual-Z conversion 複製進工具。這能在 runtime 從 seed 重建，代價是維護第二份 RB algebra。固定量表不會在每次量測改 seed，這項能力沒有目前的使用者。

先例的處理是遵循固定 curated workload 的邊界。模型參數已在工具保存固定副本，phase 資料同樣是固定輸入，不是從 lab 動態載入。命名沿用掃描型態，不把實驗名變成工具的 runtime 依賴。

## 被否決的選項與理由

沒有把既有 N 的循環直接換成隨機，避免改變既有摘要的輸入。也沒有只新增 constant 版，因為 Gaussian gate reuse 同樣需要不同 phase order 的觀察。

沒有 import lab、讀取其 YAML 或執行實驗。工具保留自己的固定輸入，lab 的重構不會改變量表。沒有依照新誤差調整 guard、family 行為或 2e-3 門檻。

## 留下的問題

固定 gain 與 pulse length 不是校準過的 Clifford rotations，因此 recovery phase 不保證物理狀態回到 identity。這些場景驗證演化精度，不估計 RB fidelity。換 seed 或 depth 時必須重新產生並審閱資料。

參考解與完整矩陣沿用共用本機位置，摘要才入庫。求解與正式計時分開授權。新增場景不自動改寫 evolution 的 kernel 結論表，後續彙整再讀新摘要。

## 依據

- [固定控制資料](../../tools/data/family_pareto/random_gate_sequences.json) 保存來源與選點，無須本機工作檔即可讀取。
- [場景 builder](../../tools/family_pareto_scenarios.py) 擁有公開 program 編譯與固定模型接線。
- [既有單場景流程](2026-11-14-family-pareto-remeasurement.md) 說明參考解、摘要與量測標註的取捨。
- [程式碼品質](../code-quality.md#依判定用途決定查核與記錄) 規定狀態與誤差等判定數字的單一 owner。

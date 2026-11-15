---
date: 2026-11-15
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - ../../tools/README.md
basis:
  repo_commit: 69473c9fd37f1810b1fee9d1f083c4e07448dbf2
---

# 單段雙 tone benchmark 的量測邊界

## 結論

單段 `dop853` 研究使用專用 Engine input factory，不新增 acquire workload。
兩個 tone 同起同止，每個 tone 都有取樣包絡。
端到端回歸保留既有 acquire 工具。
這份紀錄說明建構與量測的分工，不宣稱場景已取得數值驗收或速度證據。

## 問題

Acquire 同時包含 frontend、編譯、演化與 ADC sampling。
部分重疊脈衝還會產生單 tone、雙 tone 與 idle 等不同段。
這種總時間不能回答一段 sampled 雙 tone 的成本，也不能用來推論單一 kernel 的速度。

## 考慮過的選項與比較

既有 `single_model_input` 經公開 program 編譯掃描，並在末端加 readout。
保留它可以維持既有 family 量表的物理輸入與參考解。
新的 `dop853_segment_input` 只建立公開 timeline、drive track 與 envelope。
它在目標段結尾停止，不附加 readout 或 idle companion。
重複測量把同一段接 N 次。Worker 掃描則使用獨立 points，兩者不混為 rows。

模型與初態仍由 `SingleFluxoniumSource.model_facts` 擁有。
Factory 在建模前驗證 retained levels，再要求 owner 重建模型與 Gibbs 初態。
參考解與 cache key 由 `family_pareto_reference` 擁有。
公開 key 查詢讓量測前的輸入凍結與正式 cache-only 讀取使用同一規則。
Caller 不拼物理矩陣，也不修補 reference 輸出。

遵循 ADR-0072 的固定 benchmark 先例，場景自帶設定，不讀 lab YAML。
區別既有 program-scan 入口，這次研究不需要硬體時基或 readout 邊界。
遵循 ADR-0078 的 evolution 黑盒邊界，只使用 package root 匯出的 Engine 與資料型別。
Factory 不改 dispatch、數值 guard 或環境的預設 worker 數。

## 被否決的選項與理由

不以新增 STIRAP acquire workload 代替單段輸入，因為其總耗時包含其他工作。
不改寫 `single_model_input`，因為既有量表需要保留編譯與 readout 語義。
不切片固定維度矩陣來做能階軸，因為模型投影與初態應由建模 owner 一起重建。
不在 caller 複製 reference key，避免凍結與 cache-only 讀取採用不同契約。

## 留下的問題

同維 oracle 一致只檢查演化誤差，不證明 retained-level 截斷已收斂。
完整 density 的誤差與時間必須使用同一座標。
Wall/process 是整個 Engine.evolve 的時間，包含 routing 與 worker 管理。
Family prepare/apply CPU 是 calling-thread CPU，不能與 wall 混稱裸 kernel 成本。
每個 logical row 的 CPU 攤提值也不是一次 apply 呼叫的成本。
實際速度與截斷範圍仍需另外量測。

## 依據

- [模組定位 ADR](../adr/0072-module-boundaries.md#使用者空間的依賴原則) 擁有固定場景與 lab 的依賴邊界。
- [Evolution ADR](../adr/0078-evolution-module-boundary.md#黑盒邊界是-simulateevolution) 擁有公開演化邊界。
- [工具用法](../../tools/README.md#dedicated-dop853-segments) 說明 factory、reference 與量測方式。
- [場景 owner](../../tools/family_pareto_scenarios.py) 與 [reference owner](../../tools/family_pareto_reference.py) 擁有介面與數值契約。

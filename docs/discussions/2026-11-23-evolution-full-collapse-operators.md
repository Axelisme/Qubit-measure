---
date: 2026-11-23
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - 2026-11-23-frozen-physical-v2.md
basis:
  repo_commit: ed93e276713cd5e50e0630765678306575f078a5
---

# Evolution 保留完整 collapse operators

## 結論

當時採用既有 ModelSource 和 Engine 傳遞完整 jump 矩陣。固定 H 的 CPU `dop853` 在求解座標計算完整 dissipator。沒有完整 jump 的列留在原路徑，也沒把腔模或 Fock 排列交給引擎。決策見 [ADR-0078](../adr/0078-evolution-module-boundary.md#黑盒邊界是-simulateevolution)。

## 問題

方向率只補回人口，不能重現同一 jump 的相干轉移。例如腔由一光子衰減至真空時，完整 `LρL†` 也把一光子 sector 的物質相干送到零光子 sector。若物質與腔解耦，trace 掉腔後物質相干應保持不變。只取 `|Lᵢⱼ|²` 做方向率會漏掉這項回補；換基不能補回已丟棄的交叉項。

## 考慮過的選項與比較

既有 ModelSource 接完整矩陣，可以沿用固定 H 的控制播放、座標變換與 dispatch。不帶 L 的模型也不必遷移。另一個數值軸是直接把獨立 `NumericModel` 接入引擎；它同樣能帶矩陣，但需要界定第二種來源的 shift、準備與重播。固定 photon × matter 的 tensor 表示在 product basis 可減少乘算，卻不能假定換到一般 dressed basis 後仍保有該分解。

無驅動的 Purcell 基礎浴另行驗證，不以固定 H 的數值能力或一光子截斷對照證明瞬時率有效。

## 被否決的選項與理由

沒有擴成第二套數值入口，也沒有全面更換 simulator source。兩者的重播與選路改動超過這一步所需。沒有引入 Fock 專用型別；排列屬於呼叫端。沒有採用 dressed 方向率取代完整 jump；它不能通過解耦相干對照。不交付連續磁通的完整 L。

## 留下的問題

無驅動 Purcell 候選未通過。當時以 `ed93e2767` 的 FTF default 建立 24 個 retained states，取 DC `[0.48, 0.2, 0.49] Φ₀`，將 R2 調至 joint 1↔12 共振，從 `|joint-12,0⟩` 開始。單一 10 ns 觀察點的 `P(12)` 在完整 H＋κ 參考為 0.8702，瞬時方向率為 0.000357。完整解在後段出現激發回流，方向率沒有。這個切片作為不加該近似的證據，並非所有失諧域的判定。原研究腳本與收據未入庫。FTF 驅動段不在本次交付。

## 依據

2026-11-22 由 designer 代裁，使用者後覆核「其他两个是接受。」固定 H 的公開驗收見 [complete jump density](../../tests/simulate/evolution/test_collapse_operators.py)。原取捨調查未入庫。引擎黑盒與模擬器分層沿用 ADR-0072、ADR-0078。

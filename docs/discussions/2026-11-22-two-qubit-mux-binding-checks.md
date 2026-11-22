---
date: 2026-11-22
status: active
superseded_by: null
related:
  - 2026-11-22-two-qubit-identity-and-neutral-positions.md
  - 2026-11-21-form-gain-range.md
basis:
  repo:
    review_base: 85e48d392961e9d990797febc27b761ea4b30fca
    review_candidate: 869a4b20c4ec942e3a207a2fdfd0f92c8183b177
---

# Two-qubit mux 綁定與編輯後的檢查

## 結論

2026-11-22，designer 代裁保留兩次檢查，使用者未覆核。Open／Reset 用固定 capture 驗有序元件對應及已綁定的 readout；表單正常 lowering 後再以同次 capture 驗實際輸出的完整 tone 數。後者不符時在 `readout` 回 `FormErrors`，不交付 cfg。決策見 [ADR-0069](../adr/0069-experiment-core-and-frontends.md#只驗實驗所需的能力)。

## 問題

原候選只驗兩顆 qubit 都在 group 及已綁 readout 等長。只交換 `resonators` 的順序，仍可開表，分類位置卻可能錯。另一例是三成員 group 選前兩個位置。操作者把完整三 tone readout 改成有效的二 tone ref 或 inline，兩個 index 仍合法，core 不知道還有第三個成員。

## 考慮過的選項與比較

在兩個擁有資訊的時點處理。Open／Reset 沿用 [ADR-0072 的有序 mux owner](../adr/0072-module-boundaries.md#mux-group-是實驗室的-component-定義)，逐項比較 `qubits[i].resonator` 與 `resonators[i]`。它檢查元件大類，不恢復舊 FTF 物理子種類限制，也不改 onetone 無 qubits 的 bring-up。Lowering 後才知道實際 ref 或 inline 的 readout；由家族 Form 在公開轉換入口核對，保留一般 lowering 錯誤與可編輯草稿。這遵循 [ADR-0065](../adr/0065-cfg-editing.md) 與 [ADR-0073](../adr/0073-measure-form-derivation.md) 的分工，不需要通用 validator。

只在 adapter.run 檢查能先於 acquisition 拒絕，但現有 RunContext 不帶 Form capture，recipe readiness 也不能提早得知。逐 tone 核對 ADC 路由能找到部分通道交換，卻不能證明該 tone 的元件與校準身分。若要可靠區分同長來源，需要讓 library 與 inline 輸入有明確的 tone owner，會改現有資產與編輯契約。

## 被否決的選項與理由

- 維持候選或只補 Open：不會擋掉三 tone 變二 tone 的有效草稿。
- 鎖定 selector 或 readout slot：拿掉既有手填與換源，沒有解決物理身分的問題。
- 新增共用校驗 hook、Qt 錯誤呈現或資產身分：各有獨立用途，不是這次拒絕錯誤 cfg 的必要條件。

## 留下的問題

等長不代表順序正確，操作員仍須確認。Refresh 以新 capture 重建隱藏位置與檢查，保留既有草稿。Qt 當時能擋 Run，但不能承諾在欄位上顯示所有只在 lowering 才出現的錯誤。這個缺口與本次檢查分開。

## 依據

原始調查與離線 probe 未入庫。調查分別查 [有序 owner](../../zcu_lab/v2/support/measure/README.md#mux-綁定驗證) 和 [家族 Form](../../zcu_lab/v2/two_qubit/_measure.py)。後續以真 joint_readout 的 Form／Resource 查核兩 tone 範本、有明示來源的三成員合成變體及 ref／inline 的拒絕與恢復。未開 GUI、啟動實驗或讀取硬體。

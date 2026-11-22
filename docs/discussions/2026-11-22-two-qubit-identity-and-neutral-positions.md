---
date: 2026-11-22
status: active
superseded_by: null
related:
  - 2026-11-20-two-qubit-recipes.md
  - 2026-11-20-experiment-naming-and-display.md
  - 2026-11-22-two-qubit-mux-binding-checks.md
basis:
  repo:
    before_task: 47fbcf5e8024372cf2655faf28e4325877b0c389
    existing_neutral_roles: dd774f93f6caa61c64f2859c45a07a936cb9906d
---

# Two-qubit 的身分、定位與中性位置名

## 結論

2026-11-20，使用者選定身分、定位與顯示分開。2026-11-21，他確認 two-qubit 不應限用 fluxonium，並追認沿用 repo 的 `qubit1/qubit2`。兩項一起改，12 個 cfg 各升一次 major。規則見 [ADR-0069](../adr/0069-experiment-core-and-frontends.md#身分定位與顯示)。

## 問題

當時的 cfg `roles` 混存實際元件名和兩個 tone index。分類用 index，寫回才需回答是哪個元件；讓分析從 cfg 取圖例，又讓顯示依賴這組身分。舊表單只接收 FTF 的種類組合，但序列與分析沒有以 fluxonium 種類選能階、coupler transition 或 flux point。使用者問：「我還是不懂為什麼core要知道F1/F2的概念，因為誰說是fluxonium的，也可以是transmon啊」。

## 考慮過的選項與比較

身分留在固定 capture，定位以普通數值交 core，圖例則以實驗自己的 analyze options 傳入。如此 Notebook 使用者仍能手動給同樣的輸入，而 adapter 只處理 GUI 來源。使用者指出「其實gui顯示的options與core要的options是解耦的」，不必把圖例標籤另做成 GUI 分析欄位。

位置命名比較了既有的 `qubit1/qubit2` 與新名 `qa/qb`。兩套都能去除種類暗示，但兩份 FTF 模板已綁現有中性名，沿用它省掉再引入一套角色和模板綁定。代價是 `correction_qubit1_deg` 等衍生名變長，使用中的舊 point 仍要更新 CZ slots、library keys 和 phase pair。

這遵循 [ADR-0079](../adr/0079-form-field-transitions.md#remote-與-recipe) 的「不要加一堆難維護的轉換邏輯」。變長的 mux 用 index list；本家族固定兩顆，具名欄位比 `participants[0]` 易讀，也不必為 list 內的 tone index 新增隱藏接縫。[首版 recipe 的 3A](2026-11-20-two-qubit-recipes.md) 保留具名 target，但其 alias 名稱在有種類中性要求後被取代。Cfg 不留舊名 alias，遵循 [2026-11-19 的命名裁定](2026-11-20-experiment-naming-and-display.md)。

## 被否決的選項與理由

- 圖只印位置名：不能從獨立 PNG 直接讀到元件名。把標籤塞進 metadata 或 Plots 則混用來源、呈現與圖形生命週期的責任。
- 保留 FTF 限制或另外做 transmon 家族：沒有物理理由限制，會擋掉可用樣品或重複實驗。
- 把 `f1/f2` 解釋成任意位置：讀者仍可能把 F 當成 fluxonium。Alias 用舊名而 cfg 用新名則要求 recipe 做兩套名稱的對照。
- 全面改用 `qa/qb`：中性但重建一組 repo 已有的角色和綁定。

## 留下的問題

當時未盤點使用者其他 project 的 point。指南記錄一次重綁，不做自動轉換。GUI aliases 與 Notebook 內建角色雖同名，仍是兩套來源。歷史檔案的軸名相容性另見 [native 軸名討論](2026-11-22-two-qubit-native-axis-cutover.md)。

## 依據

原始調查未入庫。它對照了 `cfg.roles` 的分類、寫回與顯示消費位置，以及已有的 [lab 元件與 Notebook 角色](../../zcu_lab/components/README.md)。實作後的操作入口見 [two-qubit guide](../../zcu_lab/v2/two_qubit/guide.md#綁定位置與讀出-tones)。

---
date: 2026-11-23
status: active
superseded_by: null
related:
  - 2026-11-24-swept-drive-cfg-minor.md
basis:
  repo:
    before_task: 2b8f628c31a181dfd9fe187859408c9f4d236c7e
---

# 表單預設引用 library

## 結論

表單用 library ref 開啟。只有表單實際改了 Run 不會覆寫的值，才在 Open 或 Reset 時複製成 inline。試調的操作仍是先按 **Convert to inline**。規則見 [ADR-0073](../adr/0073-measure-form-derivation.md#表單預設引用-library只在改了-run-不覆寫的欄位時複製)。

## 問題

Chevron 等掃頻表單為了把兩條 CZ drive 的 `freq` 設成隱藏的零，先複製 library pulse。Run 又用同一個掃頻點覆寫兩條線的頻率。副本留下的是開啟時的校準快照，重新校準 library 後還要重帶初值。其他表單也複製未改過的 Xπ、Xπ/2 和 CZ pulse。

## 考慮過的選項與比較

預設複製所有值，能在 Open 後直接編輯，但把繼續使用校準值和開始試調混為一談。只處理掃頻入口能去掉零值佔位，卻仍留下其他未改值的副本。全面清查則依 Run 會不會用到被修改的值，區分 ref 和副本。

CZ amplitude 的舊 guide 明寫 Open 會給可編輯的 inline pulse。Constructive phase 也可能需要更換 carrier。兩者皆可先引用 library，試調時才複製。CZ amplitude 的寫回仍取本次 Run 使用的 pulse，不重讀當前 library。

## 被否決的選項與理由

- 預設都複製：對 Run 會覆寫的值多一次改寫，對其他沒改值的 pulse 留下快照。
- 只修掃頻入口：未處理既沒有改值、也無其他理由的副本。
- 保留 CZ amplitude 或兩個校準入口的 inline 預設：每個 ref 都能轉 inline，試調不要求預設就是副本。

## 留下的問題

題內不改通用 Slot、轉 inline 或持久化機制。放寬掃頻 cfg 的 carrier 相等限制後，持久版本另在[掃頻 drive cfg 的值域擴張](2026-11-24-swept-drive-cfg-minor.md)處理。

## 依據

2026-11-23，使用者說：「CZ drive 必須複製，因為表單要把 freq 蓋成隱藏的 0，再由 Run 掃頻這是沒什麼意義的流程，因為不管頻率填什麼都會被覆寫，這是在這個實驗語義下非常合理的行為，類似的問題都要修正。」並要求「連無理由的複製一起查」。2026-11-24 對兩個校準入口答：「form-ref-not-copy 票 02選全部ref」。

沿用 [ADR-0073](../adr/0073-measure-form-derivation.md#form-的形狀與初值) 的作者宣告與固定 capture。以前用副本加隱藏初值，現在保留 ref，inline 選項仍隱藏會被覆寫的欄位。掃頻／固定頻率的不同驗證由 [two-qubit drive cfg](../../zcu_lab/v2/two_qubit/_drive.py) 持有。原始討論未入庫。

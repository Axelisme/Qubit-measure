---
date: 2026-11-22
status: adopted
commit: 60a15e6e6
modules:
  - lib/zcu_tools/simulate/evolution
discussion: none
superseded_by: null
---

# CPU DP5(4) 縮短每個區間的起始步長

**讀者：** framework-dev、agent。查取樣包絡每段重設步長的成本，以及落地改動的範圍。

## 核心思路

CPU DP5(4) 在 sampled envelope 的區間邊界重設自適應 solver 的起始步長。若第一步總是因局部誤差被拒，改從較小步長開始能省掉這次 trial。候選將起始值從 0.1 ns 改為 0.05 ns，再照原規則受區間長度及步長上限約束，不改容差或 tableau。

## 實測

在 `47fbcf5e8` 比較 stock 與未入庫的步長原型。輸入是重建的 heralded RB reference 臂、seed 685721、depth 1、point 1、history `[1]` 的末段，400 ns 雙 tone、d=24、帶固定 segment shift。排他 CPU、native／Numba 單執行緒，單列 apply 中位數從 0.803 s 降到 0.616 s。Stock 的 3,334 次拒絕全發生在各區間第一步，原型沒有拒絕。原型端點對獨立 full-drive 參考的最大密度元素差為 `6.92e-5`。這是單列診斷，不是正式落地版的完整 Run 加速。

落地前的縮樣比較有 plain RB 配對；heralded RB 有逾時，CZ amplitude 有編譯限制。未找齊可逐側核對 commit、輸入及時間範圍的正式收據，因此不列完整 Run 時間比。原型與原始工作檔未入庫。

## 為何採用或否決

`60a15e6e6` 採用較小的 CPU DP5(4) 初始步長，DOP853 與 OpenCL 不變。保留舊容差與選路。這與[先按列選用 DP5(4)](./2026-11-18-short-dop853-dp54.md#核心思路)不同；此處只調整已選 DP5(4) 的第一步。

## 留下的線索

評估新起始步長時，同時記 trial、接受步、重試點和端點密度。減少被拒 trial 也可能增加接受步，總耗時仍要實測。別從單列推估整條投影鏈，也別將本結果當成放寬容差的驗證。

---
date: 2026-11-24
status: rejected
commit: 5dbccaffa0b8f29934ef0c6106971038459e44e3
modules:
  - lib/zcu_tools/simulators/ftf/environment
discussion: none
superseded_by: null
---

# Projective chain 的精確 cell 簽章

**讀者：** framework-dev、agent。查 FTF chain 的編譯控制為何沒有命中精確 cell 重用。

## 核心思路

FTF 在每次投影後清空腔場，並從 DC 能量本徵態續演化。在同一掃描點，若兩個 cell 的控制完全相同，就可以重用從來源態到 populations 的 transition map。診斷在 [FTF projective chain](../../lib/zcu_tools/simulators/ftf/environment/cavity.py) 求解前取得編譯控制。它比較兩種簽章：一種包含絕對區間，另一種只留精確相對區間。兩種都保留 carrier 相位、包絡、flux、post-delay 和初始腔場，不捨入座標或對相位取模。

## 實測

基底為 `5dbccaffa0b8f29934ef0c6106971038459e44e3`。使用 [入庫 manifest](../../tools/data/experiment_wall_time/case-inputs.json) 的 `chain/len_rabi` 和 `chain/t1`，未覆寫表單。每場 101 格，FTF seed 123，離線排他單 worker。兩場各取得首份 preparation 的 1,515 個 cell 控制（101 格 × 15 個）。在同一格內比較，絕對與相對兩種簽章都是零重複。

在同一份收據中，排除 terminal cell 後，`len_rabi` 在每格按兩種交替位置分組，各以第一個 cell 作參考；1,212 次比較的 readout pulse phase 都不同。`t1` 在每格以第一個 cell 作參考，1,313 次比較也都不同。部分比較的 span、drive 或 readout support 也不同。這些是編譯後控制的比較，不是求解後的輸出差。

同次診斷中，`len_rabi` 到 190 s 的 Run 期限仍未完成首輪；外層 runner 在 320 s payload 上限停止時，`t1` 已記控制但未寫 Run 結果。沒有取得可用的 transition-state 求解數、Engine apply rows 或 CPU。原始工作檔未入庫。

## 為何否決

兩種簽章在目標場景都沒有可重用的 cell，因此未實作快取。沒有做逐值驗收，也沒有做速度配對；榜單不更新。結果只否決本次的簽章做法，不證明 cell 在數學上不可能等價。

## 留下的線索

即使模型能消去共同 readout 相位，也不能因此略過 span、drive 或 support 的差異。要再嘗試，先證明現有模型的相位不變性，再查目標場景還有多少完全相同的控制可配對。

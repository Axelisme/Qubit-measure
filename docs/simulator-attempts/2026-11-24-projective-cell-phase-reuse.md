---
date: 2026-11-24
status: adopted
commit: c22947414f7c32495f551a5fdfa2cd35846a48ee
modules:
  - lib/zcu_tools/simulators/ftf/environment
discussion: none
superseded_by: null
---

# Projective cell 共用單音共同相位的轉移列

**讀者：** framework-dev、agent。查找 FTF 哪些讀出單元能共用轉移列，以及縮樣配對量到多少求解。

## 核心思路

[保留絕對相位的簽章](2026-11-24-exact-projective-cell-signatures.md)沒有找到可共用的 cell。FTF 從真空開始 linear cavity propagation。把全部腔入射乘同一相位，會讓每個 state branch 的場同向旋轉。Stark 與 dephasing 讀的是場的二階乘積，旋轉因子在乘積內相消。DC projector、qubit Hamiltonian 與 Lindblad operators 不隨這個旋轉改變。只轉一部分 tone，或保留固定的殘餘腔場，沒有這個保證。

[FTF 的重用資格](../../lib/zcu_tools/simulators/ftf/environment/_cell_reuse.py)只接受同一點、靜態 flux、單 span、單個有效 flat cavity tone，且沒有作用中的 qubit drive。不取相位模數，也不設容差。長度、支撐區間、carrier 和複數 gain 必須精確相同。每層先用自己的機率 floor 選來源態，才查先前實際求出的列。末層要保留完整 density，仍獨立求解。

## 實測

[實驗 Run 牆鐘](../../tools/README.md#實驗-run-牆鐘)比較 `f7a88acef6c39f35b00f304ecd6f527f5e8d3a0f` → `c22947414f7c32495f551a5fdfa2cd35846a48ee`。兩側是離線 `chain/len_rabi`，從[入庫清單](../../tools/data/experiment_wall_time/case-inputs.json)只加 `length.n=11`，其餘輸入相同。衍生清單 SHA-256 為 `589a368693c26cdbef2051597674722696a59b22a5ecec416b0ebc2622e09703`。兩側各一次完整冷 cache Run，11 點、100 reps、100 rounds、FTF seed 123、同一排他 CPU 時段、單 worker 與單原生執行緒。觀察器只保留既有 map 與 Result，在 Run 結束後寫檔，未開 Engine capture。原始收據未入庫。

| 這一對 Run 的比較項 | 前 → 後 |
| --- | ---: |
| 首份 preparation 的來源態求解數，按點、轉移事件、來源態去重，不含 prefix | 3,229 → 2,967 |
| 同一 preparation 的 floor 通過列數 | 3,229 → 3,229 |
| 完整 Run core 秒，合計 compile、evolve、readout、materialize | 96.324 → 96.032 |

每點每事件的 floor 通過列數及 identity weight 相同。有序 initial／transition map 和最終 Result 的數值陣列最大絕對差皆為 0，門檻是 `1e-10`；軸、計數和輸入也相同。

## 為何採用

縮樣確實少解 262 列，不靠提高 floor，也沒有觀察到數值差。這一對 Run 只差 0.292 core s，沒有重複量測，不能當作整輪加速的證據。

## 留下的線索

這次只量 `chain/len_rabi` 的 11 點。完整 101 點與 `chain/t1` 的求解命中、逐值結果和速度都未實測。不把縮樣的 262 列換算成其他場景的秒數。

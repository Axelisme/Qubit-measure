---
date: 2026-11-24
status: rejected
commit: e060dae1f
modules:
  - lib/zcu_tools/simulate/evolution
discussion: none
superseded_by: null
---

# RB 的 exact-key segment map 重用

**讀者：** framework-dev、agent。記錄 RB 序列中昂貴 segment 能共用同一個演化 map 的比例。

## 核心思路

若兩列的物理輸入完全相同，就建一次 segment map，分別套到各自的初始 density，不必逐列重求演化。Key 以浮點位元精度比較，包含 model、DC／flux、誤差預算、segment 時長，以及重疊 drive 的載波、帶符號振幅、相位和 envelope。Clifford 名稱相同不代表 key 相同。既有前綴共用已略過的列不再計入。

## 實測

在 `e060dae1f` 用 `two_qubit/rb` 計時 fixture 開啟 debug capture，輸入為 seed 0、depths 0／1／2／4、24 levels、單 worker。單次診斷 Run 的兩個 acquire 有 70 個實際走 `dop853`、`shaped_sambe` 或 `single_tone_rwa` 的 route rows。在各 evolve 內去重後合計 64 個 key，同一 evolve 只有 6 個重複 apply。跨兩個 acquire 合併比較才找到另外 4 個重複。原始工作檔未入庫。

以同一 `e060dae1f`、同 RB fixture 的單輪 profile 平均 row 成本，假設上述重複 apply 完全免費，同一 evolve 內共用約為 1.03×；讓 map 再跨兩個 acquire 保留，約為 1.07×。這是 evolve 耗時外推，未扣除 map 的建造、儲存或套用成本，沒有實測 map 原型。

## 為何採用或否決

重複的名義閘大多不會命中。CZ 的起點時間會改變載波相位，X pulse 的 frame、重疊 drive 或 pulse 尾段的 offset 也會改變 key。兩個外推值都低於這次衝刺的落地門檻 1.3×，因此不實作 map cache。

## 留下的線索

換 seed 或 depth 後，先用同一套物理欄位重做 route-row 普查。只算現有前綴共用之後實際執行的列，再比較同一 evolve 與跨 acquire 的重複數。

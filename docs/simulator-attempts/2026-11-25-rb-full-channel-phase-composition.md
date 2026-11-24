---
date: 2026-11-25
status: rejected
commit: 13aa91a3c25169d7ed003e0b187e71c1cad5cf85
modules:
  - lib/zcu_tools/simulate/evolution
  - lib/zcu_tools/simulators/ftf/environment
discussion: none
superseded_by: null
---

# Paired RB 的完整通道相位重用未通過 depth 1

**讀者：** framework-dev、agent。查找 FTF 用相位轉換共用完整通道時的誤差與建表成本。

## 核心思路

為每種演化區間建一次完整 density 通道。若改變起點或載波相位只等於共軛變換，就用 joint RWA 的相位電荷將通道轉到各區間的實際相位，再按時序組合。重疊的 qubit drives 仍用同一個聯合通道，idle 也保留；通道作用在完整 density 上，不截掉 leakage。這與[只共用物理輸入完全相同的 segment](2026-11-24-rb-exact-segment-map-reuse.md)是兩種不同的重用條件。

## 實測

在 `13aa91a3c25169d7ed003e0b187e71c1cad5cf85` 量 paired RB 的 reference 和 CZ-interleaved 兩臂，各一次，seed 7、depth 1。輸入為固定初始 DC 的 24 態 FTF。兩條 X90／X180 drive 均為 400 ns 常數波形，載波分別為 297.742 MHz 和 221.720 MHz。32 ns CZ 使用從計時 fixture 衍生的波形，未經實體校準。CPU 計時用 core 3、單 Engine worker、單原生執行緒。原型、輸入快照與量測工作檔未入庫。

兩臂各從邏輯 `|00⟩` 出發。原型把序列經 `build_compiled_request` 和 FTF `plan_template` 轉成演化輸入，以公開的 `Engine.evolve` 作參考。每個區間通道用 576 個合法密度輸入重建成 576 × 576 的矩陣，包含 2 ns 脈衝間 idle 和 1.6056547619 ns 的終端 idle。比較到讀出開始為止，未模擬讀出。

| 同一 depth 1 配對的比較項 | 結果 |
| --- | ---: |
| 兩臂各自的 `|00⟩` population，直接 → 通道組合 | 0.5860257792 → 0.5746312902 |
| 完整 density 的最大絕對元素差 | 0.0113944890 |
| 建四個區間通道的 CPU 總和 | 383.364 s |
| 其中同時 X drive 的通道建表 CPU | 365.366 s |
| 兩臂直接演化的 CPU 總和 | 2.899 s |
| 兩臂套用既有通道的 CPU 總和，不含建表 | 約 0.031 s |
| 建表加模型設定與兩臂套用的 CPU 總和 | 約 386 s |

直接演化與通道組合的 trace 和 Hermiticity 誤差都小於 `4e-14`，但完整 density 差超過 `1e-2` 的驗收門檻。逐區間比對時，兩個 `dop853` 的 X 區間相差 0.00855、0.00604，兩個 `shaped_sambe` 的 CZ 區間相差 0.000854、0.000298。走 `t4` 的 idle 通道完全吻合。

## 為何採用或否決

[Joint RWA phase graph 的 `pruning_admitted`](../../lib/zcu_tools/simulate/evolution/kernels/_rwa_graph.py) 在同一 depth 1 輸入算得 X 和 CZ 的省略項誤差界分別為 0.10123949 和 0.043389215，均超過 `1e-3` 的預算。原型故意套用被拒絕的相位轉換，結果顯示它不能準確重建這些 full-drive 通道。這不是正式路徑的誤差，既有 guard 會保留直接演化。

同一配對在通道建好後，套用約快 94 倍。計入建表與模型設定後，兩臂合計反而約慢 133 倍。這些比值只涵蓋本次到讀出開始的演化，不代表完整 acquire 吞吐量。

## 留下的線索

同一 commit、seed 和 fixture 只編譯 depths 1、11、51、101 時，兩臂合計有 20 個不同區間簽章。只建了 depth 1 的四個通道。更深的 RB、其他 FTF 模型、`cz_optimize` 和 overnight 都未比較。若再試，先在待用的演化 family 驗證相位共變，再比較保留所有 idle 與 leakage 的完整序列，並把冷建表成本計入。

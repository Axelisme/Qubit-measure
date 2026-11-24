---
date: 2026-11-24
status: rejected
commit: 45d465a5a58a68d0d9da36f99a15133a762cdbdd
modules:
  - lib/zcu_tools/simulate/evolution
discussion: none
superseded_by: null
---

# 不同載波雙線的 gate 端點交互作用框架

**讀者：** framework-dev、agent。記錄雙載波的完整基線、gate 端點原型的配對結果與未量的範圍。

## 核心思路

`multitone_gate` 建表時，原本在 lab frame 積分各相位節點的傳播子。原型改在每個四分之一段的局部交互作用框架積分，拿掉生成元中的裸能量對角項。每次 RHS 評估只旋轉兩線的 I/Q 算符一次，讓 5×6 個相位節點共用。每段末端轉回 lab frame，再做原有的極分解投影。包絡 knot 只重啟步長控制器，不重設該段的框架；載波與包絡仍用原列的時間。

原型希望減少裸能量振盪限制的積分步數。它保留完整驅動、DOP853 求解器與容差，不改 guard、選路或直接積分的 `dop853` family。

## 實測

2026-11-24 的基線與配對都用離線 `two_qubit` manifest `c2b6b9e0522e`、同一 fixture、FTF seed 123、`time_scale=0`、1 worker、1 native thread。分類器、Local Z pair 和 CZ arb 是合成值，只供路徑計時。每個掃描軸 11 點，兩場 RB 則用 depth `[0,1,2,4]`、序列 seed `[0]`。各 acquire 10 rounds。Core s／格以整場的 compile、evolve、readout 與 materialize 秒數加總，再除以 Result 格數。Family CPU s 是整場完整 routing 記錄的呼叫執行緒 CPU 時間總和，不是牆鐘，也不是單次 build 耗時。

### 五場完整基線

基線 commit 為 `addd6c8fb57a55bf36742ec01a5a87e51ca55d1a`。CPU affinity 為 0–3，非排他。兩批各跑一次所列場景，第二批沿用第一批的快取。兩批都完成，無逾時。

| 場景 | Result 格數 | Core s／格 | `multitone_rwa` prepare CPU s | `multitone_gate` prepare CPU s | `dop853` apply CPU s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cz_amplitude` | 11 | 5.1668 | 20.8914 | 未列 | 20.4340 |
| `rb_heralded` | 4 | 49.7659 | 未列 | 未列 | 167.2460 |
| `chevron` | 121 | 0.3984 | 4.4420 | 22.1179 | 9.5471 |
| `constructive_phase` | 121 | 1.7864 | 29.4412 | 118.2730 | 39.2976 |
| `rb` | 4 | 3.1657 | 4.3919 | 未列 | 7.4893 |

`rb_heralded` 的最大 family 成本是 `dop853` apply，兩筆 acquire 共 3,040 個邏輯列。`rb` 也是直接積分 apply 最大。`cz_amplitude` 的兩項成本接近，`multitone_rwa` prepare 稍高。`chevron` 和 `constructive_phase` 的最大 family 成本是 `multitone_gate` prepare，因此拿這兩場檢查 gate 原型。

`rb_heralded` 沒有 `multitone_rwa` 記錄，不能判為探針拒收。其餘四場按表順序各完成 4、8、24、4 次 sampled spot comparison，共 40 次。這四場的 `multitone_rwa.rows` 都是 0，表示沒有列用該 family 演化。`multitone_rwa` 的 prepare CPU 同時包含 guard、輸入綁定和探針。[Acquire routing JSONL](../../lib/zcu_tools/simulators/README.md#acquire-telemetry) 沒有存 `probe_integration_cpu_s` 或 `probe_memo_hits`，不能拆出探針積分的成本或 memo 命中數。JSONL 也沒存 `dp54_rows` 或 `dop853_rows`，不能把 `dop853` family 的 apply CPU 當成純 DP5(4) 計時。

用同批的 `(core_s - evolve_s) / point_count` 計算，即使 evolve 耗時歸零且其他階段不變，`rb_heralded` 和 `cz_amplitude` 仍各需 2.9746、0.3018 core s／格。兩者都高於 0.1 s／格目標。這是基線的算術估計，不是改後實測。

### 兩場配對

以下以相同輸入對照 `addd6c8fb57a` 與候選 `45d465a5a58a`。兩側都用乾淨的工作樹、排他 CPU 0–7、關閉 OpenCL，各自從空快取開始。在同一 process 先跑冷首場 `chevron`，接著跑暖態 `constructive_phase`。每側每場各一次完整 Run，均完成，無逾時。這批的 `chevron` 冷首場不當暖態基準，也不和另一批的基線跨批比較。

| 場景／狀態 | Result 格數 | Gate 邏輯列數，每側 | Core s／格，基線 → 候選 | Gate prepare CPU s，基線 → 候選，全場 | Result 最大絕對差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chevron`，冷首場 | 121 | 2,198 | 0.6448 → 0.6774（+0.0326，+5.1%） | 20.6356 → 25.9380（+5.3023） | 0 |
| `constructive_phase`，`chevron` 暖機後 | 121 | 2,088 | 1.7692 → 1.8065（+0.0373，+2.1%） | 117.1628 → 117.4405（+0.2776） | 0 |

兩場的 Result 陣列 `coordinates`、`durations`、`sums` 在各自配對中形狀相同、逐元素相等，最大絕對差為 0，低於 `1e-2` 的驗收門檻。候選 `45d465a5a58a` 另通過三個 CPU 單 worker 的 Engine 契約案例，涵蓋大能差 coherence、混合相位鍵的 density 和耗散鍵的 gate 選路。候選程式碼、離線 fixture 和原始量測及驗證產物均未入庫。

## 為何否決

拒收的是這個 gate 端點原型。兩場結果陣列相同，但兩場的 gate prepare CPU 和 core s／格都沒有下降。每場只有一對 Run，不能把上表的增加比例當成可重現的退步幅度。

`rb_heralded`、`cz_amplitude`、`rb` 只有完整基線，未量候選。也未量 `mux/t1`、`mux/t2echo`、`mux/t2ramsey` 的暖態配對。這個原型沒有修改或單獨量測直接積分的 DP5(4)，不能用這兩場配對否定 DP5(4) 的優化方向。

## 留下的線索

若要再做 DP5(4)，先找新的內部切點，並記錄實際 tableau 端點數與相應的 apply 耗時。[Engine 的 routing telemetry](../../lib/zcu_tools/simulate/evolution/README.md#routing-telemetry) 區分 DP5(4) 和 DOP853 的實際端點數，這次 acquire 紀錄沒有這個粒度。先量候選能省的積分工作，再用同輸入的完整 Run 核對 core s／格，別拿 family 總量當可節省的時間。

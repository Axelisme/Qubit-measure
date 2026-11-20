---
status: draft
---

# Timeline 控制結構的完備性

本草稿保留迴圈的 continue／break 設計。目前沒有使用方，不實作。
共用的編譯期狀態、每發 ADC 讀取數與往後跳轉已實作。規則在 [ADR-0083](../0083-program-macros-and-timelines.md)。

## 已定的範圍

- 作者 label 與跳轉只在 `_body` 最外層，不放寬到 Repeat body 或 Branch case。Agent 於 2026-11-20 依授權保留第一版範圍，待使用者覆核。
- Two-qubit RB 的 plain 與 heralded 兩版仍在板上掃 depth，但不再分塊。每個 seed、arm 編一支 program，PMEM 放不下就報錯。現行裁決見 [ADR-0083](../0083-program-macros-and-timelines.md#量測能力不抵可讀性成本時不加分支)。

跳轉範圍的理由與授權見[往後跳轉的討論紀錄](../../discussions/2026-11-20-timeline-forward-jumps.md)。

## 迴圈的 continue／break（已設計，不實作）

使用者先選定迴圈結構一次設計完整：shot 迴圈與 Repeat 都帶 continue／break 的目標，原話「B, 我傾向一次就設計完整。」（2026-11-20）。設計完成後裁定不實作：「然後continue/break支援，由於目前沒使用方，可以不實作，留在adr draft即可」（2026-11-20）。出現使用方時，從本節開始規劃，並先裁決〈尚未決定〉中屬於本節的三項。

- **Repeat 的 continue。** 目標在 latch：使用者 body 之後、Repeat 每輪的結尾轉移與 QICK 關閉迴圈之前。跳到 continue 完成這一輪，不離開 Repeat。continue 的入邊與正常 fall-through 在 latch 合併，合併結果就是這一輪的迭代轉移，再交給既有的跨迭代 fold。跨迭代的 ADC 與 generator 重疊證明要對每條到達 latch 的路徑都檢查，因為跳過某個 ADC 窗口的路徑可能讓下一輪與上一輪重疊。
- **Repeat 的 break。** 目標在 QICK 關閉迴圈之後、Repeat 出口的合併之前。在第 i 輪（從 0 起算）中途 break 時，出口狀態是「入口 + i × 完整一輪的轉移 + 本輪已執行的前段」，以迴圈計數器在執行期算出，再和正常完成（n 輪）、可能的零次路徑在出口合併。計數器、次數來源與 edge 結果所用的 register 要 pin 住，直到轉移完成。次數 register 沒有有限上界時，只有在沒有任何 timeline 結尾或 shell wait 依賴它的情況下才合法，否則要求 `range_hint` 或先 sync。
- **巢狀跳出。** 跳到外層 Repeat 的 continue／break 或 `shot_continue` 時，由內而外逐層 unwind：每一層用自己的迭代索引、完整一輪轉移與目前的部分狀態，算出到外層目標時的狀態。這是 Repeat 轉移的組合，不是跳過記帳的 raw jump。從 Branch case 跳到外層目標也走同一套 unwind。
- **本發的 continue（`shot_continue`）。** 目標在 `_body` 之後、收尾 sync（`close_adc=True`）之前。跳到這裡只結束這一發的 body：timeline 與 ADC 結尾照樣收尾，照樣執行 shell wait、relax、shot 計數器遞增，其餘 reps 與 sweep 照常進行。
- **沒有本發的 break。** QICK 的累加 buffer、`set_reads_per_shot` 與 acquisition 的每輪收尾，都假定設定好的 reps、sweep 維度與每發固定的 ADC 讀取數，也不回傳實際完成的 shot 數。中途結束 reps 或 sweep 迴圈需要先有「部分結果」的取數契約；在那之前，編譯時拒絕。
- **讀取數。** 跳到 continue、break 或 `shot_continue` 的每條入邊，讀取數都要和目標處相同，與一般 label 的規則一致。所以跨過讀取的 continue／break 會被拒絕。
- **Sync。** 不在 jump 前自動插入 sync。只想在跳走的那條邊上 sync 時，用往後跳的局部 block：條件跳到本地 label，在那裡 sync，再跳到共用目標。各邊執行期結尾不同且沒有在邊上對齊時，照 [ADR-0083 的 label 合併規則](../0083-program-macros-and-timelines.md#往後跳轉與-label-合併)報錯，不另造執行期 phi。
- **介面草案。** Repeat 的 context manager 從回傳 `None` 改成回傳 frozen 的目標 handle，欄位是 `continue_label` 與 `break_label`；program 與 builder 各提供 `shot_continue` handle。`cond_jump(target, arg1, test, op=None, arg2=None)` 沿用 QICK 的參數順序，target 可以是作者 label 或這些 handle。用 handle 而不用字串，是為了讓編譯器產生的目標不會和作者 label 撞名。
- **術語。** continue 的目標對應 C 的 `continue` 與 LLVM 的 latch block，break 的目標對應 C 的 `break` 與 LLVM 的 exit block。Repeat 跨迭代帶著的狀態對應 MLIR `scf.for` 的 `iter_args` 與 `scf.yield`。

選項比較與術語對應見 [Two-qubit RB 在板上掃 depth：提早結束本發與控制結構的完備性](../../discussions/2026-11-20-early-exit-and-control-flow-completeness.md)。那篇紀錄的是改為通用跳轉之前的提案。

## 尚未決定

- Branch 是否改寫成跳轉與 label 上的固定用法：由 selector 分派，加上它自己的 select 合併。Repeat 是往回的邊，要做迭代的 fold，保留自己的結構。
- 迴圈結構出現使用方時，先裁決：本發的 break 是否需要，需要的話，它結束的是 reps 迴圈還是所有 sweep，部分結果怎麼回傳；巢狀跳出是否允許指名任一層外層 Repeat，或只允許最近一層；handle 的名稱與形狀，例如 `loop.continue_label`、`loop.break_label`、`shot_continue`。

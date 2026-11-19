---
status: draft
---

# Timeline 控制結構的完備性

已核准的方向：共用編譯期狀態與每發 ADC 讀取數推算已實作，跳轉尚未實作；迴圈的 continue／break 只有設計，沒有使用方，不實作。現行的 timeline、Branch、Repeat 與模擬器入口規則在 [ADR-0083](../0083-program-macros-and-timelines.md)。

## 方向

2026-11-20，評估 two-qubit RB 在板上掃 depth 時，使用者判斷：「我觉得这算是旧有的机制出现的不完备性，而不是新的需求需要新的机制。所以我们可能要讨论一下如何完善旧机制。」
對三個完善方案，使用者選 K1：「K1，我關注怎麼設計這個共用機制與結構，使其容易維護，可讀性高，有專業術語對應，寫法又自然。」

同日，使用者否決了第一版提案：在 builder 上加專用的 `exit_shot` 區塊。原話是：「首先也是基于 builder 设计语法，但事实上 builder 只是 helper，你应该要基于 timeline program 设计。再来，exit shot 有一点太局部了，呃，而不是通用的资质、机制。真正通用的机制应该是 condition jump 跟 label。」
之後的提案改為下列規則。使用者先接受第 1 條，回覆「可以接受统一只能往后跳」，再回「都采用建议」接受其餘各條。

- **共用的編譯期狀態。** 抽出 `PathState`，收齊後續 placement 依賴的所有狀態：具名 timeline 的 end、QICK channel 的 timestamp 與執行期 tail、source／native 表達式與 unknown ADC、producer 的 defined 與 selection、Repeat frame 的 guard、每個 ADC 已讀取的次數。`TimelineBook` 提供 capture 與 restore。Branch 和 Repeat 依 [每發 ADC 讀取數](../0083-program-macros-and-timelines.md#每發-adc-讀取數) 的規則合併。
- **通用的跳轉。** `TimelineProgramV2` 的 `cond_jump`、`jump`、`label` 改成認得 timeline 的版本，和它改寫 QICK 的 `pulse`、`trigger`、`delay_*` 是同一個做法。
  - `cond_jump(target, ...)` 把當下的 `PathState` 記成 target 的一條入邊；沒跳的路徑帶著同一份狀態繼續往下編。
  - `jump(target)` 記一條入邊。之後到下一個 label 之間的位置不可到達，在那裡放任何 macro 都報錯。
  - `label(name)` 合併所有入邊與 fall-through 的狀態，不可到達的路徑不算入。
  - 這對應 MLIR `cf.br`／`cf.cond_br` 的 block arguments：每條邊把狀態帶給目標 block，由目標 block 合併。
  - 「提早結束本發」的目標是本發迴圈的 continue（`shot_continue`），屬於下節的迴圈結構，不實作。作者在 `_body` 最後放自己的 label 就能達到同樣效果。Builder 只是 helper，不另立專用結構。
- **第一版規則：**
  1. 只能往前跳：目標 label 必須在 jump 之後。往回跳等於迴圈，照舊用 Repeat 或 sweep。
  2. Jump 與 label 只能放在 shot body 的最外層，不能跨進或跨出 Repeat body 或 Branch case。
  3. Label 的合併規則分兩種。編譯期已知的值取最大，和 Branch `align_timeline=True` 相同。執行期才知道、且各邊不同的值，直接報錯，要作者在 jump 前加 sync。之後若要放寬，可以在邊上寫 register（phi move）。
  4. 一個 label 的所有入邊必須有相同的逐 ADC 讀取數。合併時比對既有的 `PathState.read_counts`。
  5. 模擬器多認兩種有限的 metadata：帶目標的 JUMP，以及 LABEL。跳走的 lane 暫停，到目標 label 才恢復。沒有標記的 raw jump 照舊拒收，延續 ADR-0083〈Compiled snapshot 與 simulation 入口〉只擴充有限 metadata 的做法。
- **順序：** 先修 Branch 還原 case 時遺失 unknown ADC 的缺陷，再抽 `PathState` 並推算每發讀取數，最後加跳轉。

## 迴圈的 continue／break（已設計，不實作）

使用者先選定迴圈結構一次設計完整：shot 迴圈與 Repeat 都帶 continue／break 的目標，原話「B, 我傾向一次就設計完整。」（2026-11-20）。設計完成後裁定不實作：「然後continue/break支援，由於目前沒使用方，可以不實作，留在adr draft即可」（2026-11-20）。出現使用方時，從本節開始規劃，並先裁決〈尚未決定〉中屬於本節的三項。

- **Repeat 的 continue。** 目標在 latch：使用者 body 之後、Repeat 每輪的結尾轉移與 QICK 關閉迴圈之前。跳到 continue 完成這一輪，不離開 Repeat。continue 的入邊與正常 fall-through 在 latch 合併，合併結果就是這一輪的迭代轉移，再交給既有的跨迭代 fold。跨迭代的 ADC 與 generator 重疊證明要對每條到達 latch 的路徑都檢查，因為跳過某個 ADC 窗口的路徑可能讓下一輪與上一輪重疊。
- **Repeat 的 break。** 目標在 QICK 關閉迴圈之後、Repeat 出口的合併之前。在第 i 輪（從 0 起算）中途 break 時，出口狀態是「入口 + i × 完整一輪的轉移 + 本輪已執行的前段」，以迴圈計數器在執行期算出，再和正常完成（n 輪）、可能的零次路徑在出口合併。計數器、次數來源與 edge 結果所用的 register 要 pin 住，直到轉移完成。次數 register 沒有有限上界時，只有在沒有任何 timeline 結尾或 shell wait 依賴它的情況下才合法，否則要求 `range_hint` 或先 sync。
- **巢狀跳出。** 跳到外層 Repeat 的 continue／break 或 `shot_continue` 時，由內而外逐層 unwind：每一層用自己的迭代索引、完整一輪轉移與目前的部分狀態，算出到外層目標時的狀態。這是 Repeat 轉移的組合，不是跳過記帳的 raw jump。從 Branch case 跳到外層目標也走同一套 unwind。
- **本發的 continue（`shot_continue`）。** 目標在 `_body` 之後、收尾 sync（`close_adc=True`）之前。跳到這裡只結束這一發的 body：timeline 與 ADC 結尾照樣收尾，照樣執行 shell wait、relax、shot 計數器遞增，其餘 reps 與 sweep 照常進行。
- **沒有本發的 break。** QICK 的累加 buffer、`set_reads_per_shot` 與 acquisition 的每輪收尾，都假定設定好的 reps、sweep 維度與每發固定的 ADC 讀取數，也不回傳實際完成的 shot 數。中途結束 reps 或 sweep 迴圈需要先有「部分結果」的取數契約；在那之前，編譯時拒絕。
- **讀取數。** 跳到 continue、break 或 `shot_continue` 的每條入邊，讀取數都要和目標處相同，與一般 label 的規則一致。所以跨過讀取的 continue／break 會被拒絕。
- **Sync。** 不在 jump 前自動插入 sync。只想在跳走的那條邊上 sync 時，用往後跳的局部 block：條件跳到本地 label，在那裡 sync，再跳到共用目標。各邊執行期結尾不同且沒有在邊上對齊時，照第一版規則報錯，不另造執行期 phi。
- **介面草案。** Repeat 的 context manager 從回傳 `None` 改成回傳 frozen 的目標 handle，欄位是 `continue_label` 與 `break_label`；program 與 builder 各提供 `shot_continue` handle。`cond_jump(target, arg1, test, op=None, arg2=None)` 沿用 QICK 的參數順序，target 可以是作者 label 或這些 handle。用 handle 而不用字串，是為了讓編譯器產生的目標不會和作者 label 撞名。
- **術語。** continue 的目標對應 C 的 `continue` 與 LLVM 的 latch block，break 的目標對應 C 的 `break` 與 LLVM 的 exit block。Repeat 跨迭代帶著的狀態對應 MLIR `scf.for` 的 `iter_args` 與 `scf.yield`。

選項比較與術語對應見 [Two-qubit RB 在板上掃 depth：提早結束本發與控制結構的完備性](../../discussions/2026-11-20-early-exit-and-control-flow-completeness.md)。那篇紀錄的是改為通用跳轉之前的提案。

## 尚未決定

- Branch 是否改寫成跳轉與 label 上的固定用法：由 selector 分派，加上它自己的 select 合併。Repeat 是往回的邊，要做迭代的 fold，保留自己的結構。
- 第一個使用者 two-qubit RB 是否改成在板上掃 depth。
- 一般跳轉的 label 是否放寬到任一層 region 內（同一個 region、只能往後）。完整設計的調查建議放寬；第一版規則 2 仍限定 shot body 最外層。規劃跳轉時裁決。
- 迴圈結構出現使用方時，先裁決：本發的 break 是否需要，需要的話，它結束的是 reps 迴圈還是所有 sweep，部分結果怎麼回傳；巢狀跳出是否允許指名任一層外層 Repeat，或只允許最近一層；handle 的名稱與形狀，例如 `loop.continue_label`、`loop.break_label`、`shot_continue`。

前兩項在跳轉實作完成後帶回使用者裁決。

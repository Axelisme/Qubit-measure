---
status: draft
---

# Timeline 控制結構的完備性

已核准的方向：共用編譯期狀態與每發 ADC 讀取數推算已實作，跳轉尚未實作。現行的 timeline、Branch、Repeat 與模擬器入口規則在 [ADR-0083](../0083-program-macros-and-timelines.md)。

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
  - Shell 在本發收尾的 sync 之前提供一個 label，作為「提早結束本發」的目標。Builder 只是 helper，不另立專用結構。
- **第一版規則：**
  1. 只能往前跳：目標 label 必須在 jump 之後。往回跳等於迴圈，照舊用 Repeat 或 sweep。
  2. Jump 與 label 只能放在 shot body 的最外層，不能跨進或跨出 Repeat body 或 Branch case。
  3. Label 的合併規則分兩種。編譯期已知的值取最大，和 Branch `align_timeline=True` 相同。執行期才知道、且各邊不同的值，直接報錯，要作者在 jump 前加 sync。之後若要放寬，可以在邊上寫 register（phi move）。
  4. 一個 label 的所有入邊必須有相同的逐 ADC 讀取數。合併時比對既有的 `PathState.read_counts`。
  5. 模擬器多認兩種有限的 metadata：帶目標的 JUMP，以及 LABEL。跳走的 lane 暫停，到目標 label 才恢復。沒有標記的 raw jump 照舊拒收，延續 ADR-0083〈Compiled snapshot 與 simulation 入口〉只擴充有限 metadata 的做法。
- **順序：** 先修 Branch 還原 case 時遺失 unknown ADC 的缺陷，再抽 `PathState` 並推算每發讀取數，最後加跳轉。

選項比較與術語對應見 [Two-qubit RB 在板上掃 depth：提早結束本發與控制結構的完備性](../../discussions/2026-11-20-early-exit-and-control-flow-completeness.md)。那篇紀錄的是改為通用跳轉之前的提案。

## 尚未決定

- Branch 是否改寫成跳轉與 label 上的固定用法：由 selector 分派，加上它自己的 select 合併。Repeat 是往回的邊，要做迭代的 fold，保留自己的結構。
- 第一個使用者 two-qubit RB 是否改成在板上掃 depth。

跳轉實作完成後，再把以上兩項帶回使用者裁決。

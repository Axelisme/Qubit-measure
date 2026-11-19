---
status: draft
---

# Timeline 控制結構的完備性

已核准、尚未實作的方向。現行的 timeline、Branch、Repeat 與模擬器入口規則在 [ADR-0083](../0083-program-macros-and-timelines.md)。

## 方向

2026-11-20，評估 two-qubit RB 在板上掃 depth 時，使用者判斷：「我觉得这算是旧有的机制出现的不完备性，而不是新的需求需要新的机制。所以我们可能要讨论一下如何完善旧机制。」
對三個完善方案，使用者選 K1：「K1，我關注怎麼設計這個共用機制與結構，使其容易維護，可讀性高，有專業術語對應，寫法又自然。」

- Branch 與 Repeat 改用一個共用的編譯期狀態物件，提供存、還原、合併三個操作，行為不變。
- 補上第四種結構「提早結束本發」：沒跳走的路徑從進入區塊前的狀態接著編，跳走的路徑在本發收尾匯合。
- 模擬器多認一種有限的結構 metadata，延續 ADR-0083〈Compiled snapshot 與 simulation 入口〉只擴充有限 metadata、不採 generic payload 的做法。
- 提早結束路徑的每發讀取數由 lib 計算，作者不手動宣告。

選項比較與術語對應見 [Two-qubit RB 在板上掃 depth：提早結束本發與控制結構的完備性](../../discussions/2026-11-20-early-exit-and-control-flow-completeness.md)。

## 尚未決定

- 共用狀態物件的介面與命名。
- 提早結束的第一版限制：exit 區塊長度在編譯期可知、只放在 shot 最外層、跳走前自動 sync。
- 第一個使用者 two-qubit RB 是否改成在板上掃 depth。

開始實作前，先把以上各項帶回使用者裁決。

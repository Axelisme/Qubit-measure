---
date: 2026-11-20
status: active
superseded_by: null
related:
  - 2026-11-20-early-exit-and-control-flow-completeness.md
  - 2026-11-20-timeline-path-state.md
  - 2026-11-20-timeline-read-counts.md
basis:
  repo:
    integration: 5051ab6f8
  external: []
---

# Timeline 往後跳轉與 two-qubit RB 分塊

## 結論

Timeline program 用 `cond_jump`、`jump` 與 `label` 表達往後跳轉，builder 只轉交同名動作。跳轉限於 `_body` 最外層。Label 合併可到達路徑的編譯期狀態。模擬器驗證兩種固定標記，再按 sweep 點執行各自的路徑。現行規則見 [ADR-0083](../adr/0083-program-macros-and-timelines.md#往後跳轉與-label-合併)。

2026-11-20，使用者把介面從 builder 移回 Timeline program，並否決專用的 `exit_shot`：「真正通用的机制应该是 condition jump 跟 label。」同日，使用者回答「可以接受统一只能往后跳」，再以「都采用建议」接受其餘第一版規則。迴圈設計完成後，使用者裁定不做：「然後continue/break支援，由於目前沒使用方，可以不實作，留在adr draft即可」。

同日，使用者授權：「STDO:我去睡了，我看過目前規劃方向沒有問題，請你之後幫我代裁設計與推進任務，都不需要問我，你自己決定」。以下保留第一版範圍、補定合併與模擬細節、採用 RB 分塊的決定，都是 agent 依這項授權代裁，待使用者覆核。

Two-qubit RB 要不要實際改成板上掃 depth，原留給使用者在跳轉做完後裁決。Agent 代裁採用：plain 與 heralded 兩版一起改；每個 seed、arm 各自分塊，每塊在板上掃，PMEM 放不下就對半切。這項決定待使用者覆核。

## 問題

當時 two-qubit RB 每個 depth、seed、arm 各編一支 program。2026-11-19，使用者說：「但我希望two_qubit版本也改成硬體掃描depth」。
同一 seed 和 arm 的淺序列可重用深序列的 random Clifford 前綴。到了指定深度，程式要離開前綴，播放該點自己的 recovery，再讀取。Repeat 與 Branch 執行完仍會回到後續前綴，無法做這個跳轉。

2026-11-20，使用者把問題定為現有結構不夠用：「我觉得这算是旧有的机制出现的不完备性，而不是新的需求需要新的机制。所以我们可能要讨论一下如何完善旧机制。」
因此先補 timeline 的控制流，再用同一套機制寫 RB。

QICK 的原生跳轉能改變執行路徑，卻不會替 timeline 保存狀態。若編譯器照原順序累計，被跳過的動作也會影響結尾、ADC 重疊檢查與每發讀取數。模擬器也無法從沒有標記的 JUMP 得知它屬於哪一種控制結構。

## 考慮過的選項與比較

- 沿用第一版的最外層限制。RB 的檢查點、recovery 與共用讀取都在 `_body` 最外層，因此用不到區域內的跳轉。作者需要提早結束 body 時，可以跳到最後一個作者 label，讓 shot 外殼照常收尾。這沿用 [ADR-0075 的按需加法原則](../adr/0075-agent-interface-layers.md#先做簡單設計再按需加法)，等出現巢狀使用方再增加範圍。
- 沿用 [PathState 的共用擷取與還原](2026-11-20-timeline-path-state.md)，由 label 定義自己的合併。編譯期已知結尾取最大，與 `Branch(align_timeline=True)` 一致。逐 ADC 讀取數必須在會合處相同，延續[每個 Branch 出口對齊讀取數](2026-11-20-timeline-read-counts.md)的裁決。讀取 label consumer 不追跳轉邊，所以共用 trigger 放在所有跳轉範圍之後。
- 沿用 [ADR-0083 的編譯快照契約](../adr/0083-program-macros-and-timelines.md#compiled-snapshot-與-simulation-入口)。編譯器只附 `CONTROL_JUMP` 和 `CONTROL_LABEL`，frontend 核對標記與指令後才送出 request。跳走的 lane 等到目標才繼續，其他 lane 照常執行。模擬器不從 raw JUMP 猜結構，也不增加量測回饋。
- 改成板上掃 depth，但不要求整份 depth 放進一支 program。[先前的 HW3 spike](2026-11-20-early-exit-and-control-flow-completeness.md#hw3-spike-的結果)用 `ir.analyze_timing` 比對 646 條路徑、399,037 個事件，與逐 depth 程式相同。Seed 7 的 depth 0–100 全掃與 0–200 每 10 取一點都放得下。0–200 的 201 點全掃則分別用了 4348 與 4743 words，超過 profile 的 4096 words。這次比對核對的是編譯出的排程，沒有測實板的 tProc latency。

RB 每塊先播共用前綴。除最後一點外，各點在自己的檢查點跳到對應的 recovery。走完整段前綴的路徑進入最後一點的 recovery。各路徑在進入共用的終端讀取前，先用 `delay_by_timeline` 同步到自己的 default 結尾。否則，共用 label 取最大值會把較短路徑的讀取延後到最長路徑的結尾。Heralded 與 plain RB 共用序列與排程，兩版一起改可避免保留兩種取數路徑。

分塊先嘗試整份 depth，只有 PMEM 超量才按原順序切成兩半重試，點數為奇數時前半多一點。最小是一個 depth 一支 program。主機取數順序改為 seed、arm、塊、round。Result 的軸、形狀與 dtype 不變。

## 被否決的選項與理由

- Builder 專用的 `exit_shot` 只解提早結束本發。使用者在 2026-11-20 指出：「事实上 builder 只是 helper，你应该要基于 timeline program 设计。」因此用 program 上的通用跳轉，builder 不另立控制結構。
- 允許每個 Repeat body 或 Branch case 有自己的往後跳轉。即使只准跳到同一區域，編譯器也要逐層追蹤可到達性，模擬器則要在巢狀迴圈與分支內切換 lane mask。RB 的跳轉都在最外層，不需要這些規則。往回跳還涉及迴圈的狀態轉移，留給 Repeat 與 sweep。
- 用 Branch 的執行期 select 合併不同的 runtime end。Branch 有單一 sweep selector，任意 label 的入邊卻可能來自互不相關的 TEST。要選到正確的值，編譯器得在各邊寫入共用 register，並管理它的存活範圍。第一版要求作者先對齊結尾，不同就報錯。也不自動在條件跳轉前同步，因為那會同時移動未跳走的路徑。
- 直接使用原生 JUMP，再手抄 timeline 狀態。Spike 用這個做法驗證了時序，但它要求作者自行保存每一項隨路徑改變的狀態。正式介面由編譯器記帳。在可到達的位置，編譯器不新增對 raw ASM 跳轉的檢查，避免改變既有寫法。模擬器仍拒絕無標記的跳轉。
- 整份 depth 放不下就全面退回主機掃描，或要求操作者減少 depth。前者會保留兩種取數路徑，且 201 點的案例完全用不到板上掃描。後者會讓原本可以逐 depth 執行的 cfg 失敗。分塊仍能把放得下的多個點放在同一支 program。

## 留下的問題

- 使用者需覆核 agent 的代裁，尤其是原留給使用者的 RB 板上分塊決定。本文的 repo 基準含跳轉與模擬器實作，尚未包含 RB 分塊實作。
- Branch 是否改用往後跳轉實作仍未決定。Repeat 的迭代 fold 保留自己的結構。
- 有使用方後，再討論本發是否需要 break、部分結果怎麼回傳、巢狀跳出可指名哪一層，以及目標 handle 的介面。方案留在 [continue／break 草稿](../adr/draft/control-flow-completeness.md#迴圈的-continuebreak已設計不實作)。
- 實板的 tProc latency 與每次取數的往返時間仍未量測。

## 依據

- [前一篇控制結構討論](2026-11-20-early-exit-and-control-flow-completeness.md)保留 HW3 原型、PMEM 數據與最初的提早結束本發提案。
- Repo `5051ab6f8` 的 [跳轉與合併](../../lib/zcu_tools/program/v2/_timeline_jumps.py)、[compiled request builder](../../lib/zcu_tools/program/v2/simulation/README.md#compiled-request-builder) 與 [快照格式](../../lib/zcu_tools/program/v2/ir/README.md#pre-optimization-compile-snapshot)記錄已實作的契約。
- 原始設計與裁決工作檔未入庫。上文保留採用的規則、取捨與數字。

## 後記

2026-11-20，使用者覆核後保留板上掃 depth，取消 PMEM 對半分塊。RB 改成每個 seed、arm 一支 program，將 recovery 放在各點的 Clifford 前綴後，只跳一次到共用 readout；最深點不跳。比較與新裁決見[板上 RB 程式改寫](2026-11-20-rb-depth-program.md)。通用往後跳轉與 label 合併規則不變。

2026-11-20，使用者覆核本文的代裁。agent 提議全部確認，使用者回覆「確認」。保留第一版範圍（作者 label 與跳轉只在 `_body` 最外層）以及補定的合併、拒收與模擬細節，都維持原決定。RB 分塊已由上一段的裁決取代。

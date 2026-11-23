---
date: 2026-11-23
status: active
superseded_by: null
related:
  - "CLAUDE.md#使用者決策的引用"
  - docs/adr/README.md
  - docs/discussions/2026-11-23-heralded-conditional-spectroscopy.md
basis:
  - "repo@3e86bba6d8c9e3e9cacf25599c7bb82566a82050"
  - "PEP 1: https://peps.python.org/pep-0001/"
  - "PEP 659: https://peps.python.org/pep-0659/"
---

# 哪些取捨要寫進 ADR

## 結論

使用者採用三條判準，任一條成立就寫進 ADR：

- 約束或可能約束多個底層模組。
- 決定子系統內所有部件共同遵守的架構。
- 難以反悔，或需要留下否決其他方案的理由。

其餘只有擁有者自己用、可在一個 task 內反悔的取捨，寫進擁有它的 README。個別元件的閾值與校準常數寫在模組 README 或程式旁。團隊用 Python PEP 作類比：內部的重大架構改變也值得記錄，小改進不必。規則寫在 [CLAUDE.md〈使用者決策的引用〉](../../CLAUDE.md#使用者決策的引用)，[ADR 索引](../adr/README.md)保留摘要和例子。

## 問題

落地[獨立 heralded 條件頻譜](2026-11-23-heralded-conditional-spectroscopy.md)時，designer 原擬把為它保留入口的例外加進 [ADR-0074](../adr/0074-flat-readout-module.md)。使用者問「0074有必要進adr嗎？他有跨模塊嗎？」並裁定「搬到README，adr只放跨模塊決策」。他接著說明「我說的跨模組是指可能橫跨多個底層模組，剛剛的herald只有自己要用，那就不進adr」。

照這個說法，Evolution 內部各 kernel 共用的規則也會被排除。使用者指出「但從一體性來看，evolution與內部kernel規則確實值得進adr，你覺得到底哪些要進adr哪些不要，可以參考python的pep規則」。

## 考慮過的選項與比較

最窄的做法是只收跨底層模組的決策。第二種做法也收子系統內部的共同架構，讓讀者知道各部件必須遵守什麼。採用的第三種做法再收難以撤回、或日後需要查明為何否決替代方案的取捨。

外部參照有兩種寫法：直接說明 repo 的三條判準，或加上 Python PEP 作為讀者熟悉的例子。[PEP 1](https://peps.python.org/pep-0001/)說明 PEP 可提供社群資訊，也可描述 Python、其流程或環境的新作法。[PEP 659](https://peps.python.org/pep-0659/)記錄 CPython 直譯器的特化設計，是內部但影響架構的例子。這次只借用類比，不引入 PEP 的編號、狀態或審查流程。

## 被否決的選項與理由

只收跨模組決策會把 Evolution 內部共用的 kernel 規則移出 ADR，讀者就少了一處能查到整個子系統共同約束的紀錄。只用前兩條則無法收錄單點但難以撤回、或必須保留方案比較的取捨。不提 PEP 雖不改變判準，卻少了使用者提議的對照例。

## 留下的問題

這次未普查既有 ADR。對既有條文要留、要移或要縮小，使用者指示「先分類再決定」，另題處理。

## 依據

2026-11-23，使用者在提出 Evolution 的反例後，對三條判準回覆「採用建議，要不要提及pep?」。加入 PEP 參照後，使用者對最終文字答「可以」。原始決策工作檔未入庫，當時的判斷理由與必要原話保留在本文。

## 後記

後續按這三條收錄判準普查了 28 篇現行 ADR，共 297 個決策單位。不計索引、草稿與已退役文件。分類工作檔未入庫。

- 284 個單位留在原 ADR。
- 9 個局部單位移到 7 份 owner README 與 [`notebook/adapter.py`](../../lib/zcu_tools/notebook/adapter.py) 的註解。原 ADR 留下指標。
- 4 段已被取代的原文封存至 [`docs/adr/retired/`](../adr/retired/)。原段落連到接替決策與封存原文。

從 [ADR 索引](../adr/README.md) 可查搬移後的位置與封存條目。這次只調整位置與指標，沒有改動規則。

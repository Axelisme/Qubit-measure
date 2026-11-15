---
date: 2026-11-15
status: active
superseded_by: null
related:
  - ../../tools/README.md#document-relations
  - ../../tools/doc_relations.py
  - ../../tests/tools/test_doc_relations.py
  - ../adr/0072-module-boundaries.md#模組之間的合作
basis:
  repo_commit: 3182ad615c469629ff5d51f297d744815d764690
  external_sources: none
---

# Doc relations 略過 fenced code 示例連結

## 結論

本次在 checker owner `tools/doc_relations.py` 修正示例連結誤報。`check_references` 略過 fenced code block 裡的連結，正文連結仍照原行號檢查。當前契約由 [tools README](../../tools/README.md#document-relations) 持有。

修改沒有更動公開簽名、回傳型別或 CLI。ADR 引用在 fenced code 內仍照常檢查。討論快照與退役 ADR 的略過規則也沒有改變。本次沒有新增依賴、另一個 checker 或跨模組決策。

## 問題

`doc_relations.py --check` 曾把 `zcu_lab/v2/measure-authoring.md` 的 fenced Markdown 示例 `[...](pulse.png)` 當成正文連結，回報目標未入庫。該段展示本機圖片語法，並未要求 repo 提供這張圖片。原 checker 逐行搜尋連結，沒有區分正文與 fenced code。

相同誤報已在不同交付中重複出現。只核對本次新增診斷，能暫時讓其他交付繼續，但每次仍須辨認同一筆既有錯誤。

## 考慮過的選項與比較

- 改示例，避開 checker 的連結比對。改動小，但會讓教學配合誤判，其他文件仍會遇到相同問題。
- 單次採零增量例外，保留 raw exit 1，確認沒有新增診斷。這能解除個別交付的阻擋，卻沒有修正 checker。
- 在既有 `check_references` 排除 fenced 示例連結。這直接修正判讀位置，正文缺失連結與行號仍可從公開介面驗證。本次採用此項。

先例採以下立場。遵循 [ADR-0072 的模組合作規則](../adr/0072-module-boundaries.md#模組之間的合作)，由能力 owner 修正，不在 authoring 文件繞過檢查。遵循既有工具的可 import 函式與 `main(argv)` 接縫，以合成 Markdown 驗證。區別先前針對同一示例的單次零增量驗收，本次修正必須讓 raw `--check` 回傳 0，不把例外變成通用規則。

## 被否決的選項與理由

本次沒有改 authoring 示例，因為問題不在示例內容。也沒有把單次零增量例外延續為本次驗收，因為這次的目的就是移除誤報。

本次沒有加入 Markdown parser 依賴或新的 CLI。局部 fence 辨識已能處理這次的示例，並保留既有連結、ADR 與 snapshot 判準。

## 留下的問題

這次只修連結在 fenced code 的判讀，不改其他 Markdown 語法或 ADR 引用的意義。測試從 `check_references` 與同 process 的 `main(argv)` 觀察結果，不固定 repo 文件的措辭或行號。這份紀錄不代表模擬器正確性或性能驗證。

## 依據

- Base 為 `3182ad615c469629ff5d51f297d744815d764690`。當時調查與裁定的工作檔未入庫，必要事實與選項已寫在本文。
- 工具的行為與操作見 [Document relations](../../tools/README.md#document-relations)，程式由 [`tools/doc_relations.py`](../../tools/doc_relations.py) 持有。
- 合成輸入的 regression tests 位於 [`tests/tools/test_doc_relations.py`](../../tests/tools/test_doc_relations.py)。

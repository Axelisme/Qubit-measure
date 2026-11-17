---
date: 2026-11-17
status: active
superseded_by: null
related:
  - 2026-11-17-global-point-parameter-types-and-document-set.md
  - ../adr/0063-persistence-ownership.md
  - ../adr/0081-point-data-disk-sync.md
  - ../../lib/zcu_tools/resources/entry/README.md
basis:
  repo: ed552a79b287db464bce43c1414252f221f16fdb
  external: none
---

# Aliases 綁檔物件的讀取、refresh 與候選准入

## 結論

`entry.aliases(point=None)` 回傳綁單一 `aliases.yaml` 的物件（現名 `AliasFile`）。它持有一次觀察，只有 refresh 讀檔：

- Opener 只讀自己的檔一次，做寬鬆觀察。缺檔得到空表、revision None，不建檔。解碼、header、schema 錯誤延後放進觀察的 `error`，revision 是無效 bytes 的 hash。其他 I/O 錯誤由 opener 拋出。
- `read()` 回持有觀察的深拷貝，`document_revision(obj)` 回它的 revision，兩者都不讀磁碟。
- `refresh(strict=False)` 重做寬鬆觀察並換上，包括帶錯誤的觀察。`refresh(strict=True)` 遇格式錯誤拋出，狀態不變。I/O 錯誤兩種模式都拋出、狀態不變。物件只驗檔案格式，不驗 session 的 alias 語法。
- Refresh 回傳的 bool 與 revision 分開。它比有序表與格式有效性；兩次觀察都無效時 bytes 不同即 True。缺檔與有效空表互換、只改註解或排版回 False。
- `save()` 原子寫入、無鎖、不先讀舊檔，所以壞檔可以直接覆寫修復。成功後持有輸入的深拷貝與 writer 回傳的 revision，不事後 refresh。
- Session 不對已安裝的 aliases 物件呼叫 refresh。開 entry、切 point 與 strict reload 都用 opener 開候選。Strict reload 對格式錯與語法錯拋出，通過才把物件與 session 的表一起換上，再發 fact。

這是 designer 依使用者 2026-11-17 的授權代裁，待使用者追認。現行行為由 [resources.entry README〈Global 與 point alias 表〉](../../lib/zcu_tools/resources/entry/README.md#global-與-point-alias-表) 與物件 docstring 擁有。

## 問題

型別與文件組收斂（見 [Global／point 參數型別收斂與文件組](2026-11-17-global-point-parameter-types-and-document-set.md)）定了 aliases 用一個綁檔物件取代 4 個函式，三種檔一律 `refresh(*, strict=False) -> bool` 與 `document_revision(x)`。
沒定的是：opener、read、refresh 各做不做 I/O；bool 比什麼；strict 失敗對內容與 revision 的效果；物件與 session 語法驗證的界線。
實作者提案 A：read 每次讀檔並更新物件的 revision。

調查（基準 `ed552a79b`）查到的事實：

- 原本只有 4 個每次讀寫磁碟的函式：寬鬆讀、兩個嚴格讀與無鎖存檔。已安裝狀態只在 session。
- Alias 語法（識別名、不重複、只能向前連結）由 session 的 `AliasTable` 驗證，約 40 行純資料檢查。依 ADR-0063，語法 owner 是 session。
- 兄弟物件 `ParameterView`、`ModuleLibrary` 開檔讀一次，查詢回深拷貝、不讀磁碟，refresh 才讀，候選通過自身驗證才換上。
- 若 session 在已安裝物件上原地 refresh，物件只驗得到格式。語法驗證失敗時物件已前進、session 保留舊表，`document_revision`、FileSync 與下一次 refresh 的答案互相矛盾。既有測試 `test_invalid_candidate_retains_memory_revision_and_bytes` 要求失敗時兩邊都不變。
- 產品碼沒有地方讀 refresh 的 bool。Aliases 的產品呼叫點 7 處，測試 15 檔；Notebook 不讀寫 aliases。

## 考慮過的選項與比較

- **A（實作者提案）**：read 每次讀檔並更新物件狀態。
- **Safe-A**：read 每次讀檔，只在候選物件上用。
- **B-stage（採用）**：物件快取一次觀察，只有 refresh 讀檔；session 以 opener 開候選，通過語法才與表一起換上。
- **B-cand**：物件加 `candidate()` 方法產生候選。
- **B-hook**：opener 收 `validate=` 注入語法驗證，三種檔都在原地 refresh。
- **Keep／S**：無狀態綁檔物件，不提供 refresh。
- **Natural**：語法規則搬進 resources 的解析函式，物件完全照兄弟，session 原地 refresh。

B-stage 不加公開方法或 hook，session 的 strict reload 形狀與原本相同（先準備區域候選、通過才安裝），三種 scope（GLOBAL、POINT 作用中、POINT 下的 global）的安裝、reload、save 成敗都成立。
代價是每個 scope 有物件觀察與 session 表兩份表述，靠「一起換」維持一致；session 仍能誤對已安裝物件呼叫 refresh，只靠 docstring 擋。

先例與立場：

- ADR-0063（一次 I/O 固定觀察、session 驗語法、caller 不重讀）：遵循。
- ADR-0081（已安裝快取、先安裝再發 fact、reload 失敗保留原狀）：遵循。
- ADR-0067（非作用中寫入與 reload 的通知分流、dirty draft 保留原 revision）、ADR-0065（替換不改舊 capture）：遵循。
- 三種檔統一 refresh 與 `document_revision` 的使用者裁定（見上一篇）：遵循，並保留「aliases 語法由 session 驗證才安裝」「aliases 存檔無鎖」兩項差異。
- 兄弟物件的快取 handle：遵循。Bool 的比較對象依 aliases 格式區別：順序參與比較，因為 aliases 只能向前連結，順序就是內容；header 不參與，因為 aliases 的內容只有 `aliases` mapping。
- ContextService「先準備候選、成功才發布」：遵循。

## 被否決的選項與理由

- A：read 之後的 refresh 看不到同一次外部修改；對已安裝物件 read 就等於原地 refresh。
- Safe-A：read、`document_revision` 與 refresh bool 的基準分家，偏離兄弟的快取查詢，opener 後 read 讀兩次。
- B-cand：多一個公開方法，對已安裝物件誤用 refresh 的問題仍在；session 已有 entry 與 point，opener 就能開獨立候選。
- B-hook：hook 只有 session 一個使用者。
- Keep／S：推翻三種檔統一 refresh 與 `document_revision` 的使用者裁定。
- Natural：唯一消除成因的選項，成本約搬 40 行並改 3 處 session 程式；但它推翻 ADR-0063 的語法 owner 與「aliases 語法由 session 驗證」的既定差異，需要使用者裁定。追認時一併呈給使用者。

## 留下的問題

- 出現第二個需要長壽 aliases 物件、並在原地帶語法檢查 refresh 的使用者時，改考慮 B-hook。
- 使用者願意把語法規則搬到 resources 時，改採 Natural。
- 本裁定待使用者追認；推翻時改 status 並指向新紀錄。

## 依據

- 調查基準 `ed552a79b`（main 歷史中）。調查與 challenge 的工作檔未入庫，結論與事實寫在本文。Challenge 補上的項目：候選介面的兩個替代（B-cand、Safe-A）、可變資料隔離、三種 scope 的 save 與安裝、bool 邊界。
- 現行行為：[resources.entry README](../../lib/zcu_tools/resources/entry/README.md)、`lib/zcu_tools/resources/entry/aliases.py` 的 docstring。
- 相關 ADR：[ADR-0063](../adr/0063-persistence-ownership.md)、[ADR-0081](../adr/0081-point-data-disk-sync.md)。

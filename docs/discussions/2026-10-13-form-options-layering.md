---
date: 2026-10-13
status: active
superseded_by: null
related:
  - docs/adr/draft/pydantic-form-tool.md
  - docs/adr/0073-measure-form-derivation.md
  - docs/discussions/2026-10-13-measure-form-redesign.md
basis:
  - main@007a30ccc（撰寫時）
  - measure-form integration@3dba3f5dd（討論時讀的表單核心、生成器與 editor 程式碼）
---

# 表單選項的分層：生成器給選項，widget 照清單畫

## 結論

決策寫在 [Pydantic 表單 ADR draft](../adr/draft/pydantic-form-tool.md) 的「選項清單與分層」。本篇記推導過程與被否決的路。

- 任何「從幾個選項挑一個」的欄位，input schema 都直接列出選項清單。每個選項有顯示文字、選了要寫入的值，以及物件值的子表單欄位。
- 帶實驗語義的選項由 Form 生成器產生：module 與波形欄位的 `None`、依 kind 過濾的 library keys、每個 inline kind 一項。表單核心只提供通用的選項與指令機制，不知道 module 或 library。Widget 與 MCP 照清單畫、照清單寫，不解讀 union 結構。
- 換分支一律用新分支的預設值，不繼承舊分支的同名欄位，也不看切換前的值。
- 「從 library key 繼承再覆寫」用明確的選項指令「Convert to inline」：把該 key 當下的內容複製成 inline。Widget 只列出目前選項帶的指令並呼叫它，不知道指令的意思。
- 未標 tag 的 `str | Model`（例如 pulse 的 `waveform`）由生成器換成有 tag 的 ref／inline，`to_cfg` 再換回持久 Spec。核心從此只處理一種 key-or-inline。
- 掃描的 `ExplicitList` 刪除，掃描只用 `LinRange`。

## 問題

起點是一個外觀需求。使用者要把 library key、inline 與 `None` 放進同一個下拉：「將ref/inline factory/None都集中在一個dropbox對使用者來說很直觀簡潔」。接著補了一點：「注意Inline可能有很多個，每個對應不同的kind」。

當時的做法是 widget 從 input schema 推回選項。它讀 `x-form-key-or-inline`、`x-form-library-keys`、`x-form-discriminator-values` 與 null 分支，自己拼出下拉。要加入 `None` 與多個 kind，就得在 widget 裡再補規則。

使用者要求不受現有設計限制，從「資訊該從哪裡取得、widget 該如何建構」重新討論。過程中又提出兩條原則：

- 「這些資訊應該是由Form生成器負責將帶有實驗語義轉變成有顯示偏好的格式」。
- 「我不要看沉默成本，我要看維護成本，有些設計屬於個人偏好，但犧牲實作複雜度與維護難度」。

## 畫一個下拉需要的資訊

| 資訊 | 例子 | 來源 |
| --- | --- | --- |
| 可不可以留空 | `reset` 可以，`readout` 不行 | core 型別有沒有 `\| None` |
| 允許哪些 kind | readout 的 direct、pulse | core 型別的 union 成員 |
| 列哪些 library key | 只列 kind 符合的 entry | 開分頁時捕捉的 library，依上一列過濾 |
| 目前選哪一項 | `ro_main` | 表單狀態 |
| 選了之後的初值 | 範本有指定該 kind 就用範本，否則用型別預設 | 範本與型別 |
| 子表單畫哪些欄位 | 該 kind 的欄位，不含 kind 本身 | 型別 |

除了「目前選哪一項」，其他資訊都來自型別、範本與捕捉的來源。這些都是生成器已經擁有的語義。所以 widget 不需要判斷任何事。

## 考慮過的選項與比較

### 選項清單由誰產生

- X：沿用現狀，widget 解讀 schema 結構後自己拼選項。約半天，但判斷會分散在核心與 widget 兩邊，agent 也看不到整理好的清單。
- Y：核心直接產生選項清單，widget 只畫。使用者選 Y：「FormInput/FormModel與Widget間是簡單的關係……在不同widget/app中使用，不一定是為了Module/Library」。
- 範圍再往前推一步：選項不只用在 module。Literal、Enum、執行時的 `choices(env)` 與一般 tagged union 都用同一種清單。帶實驗語義的部分歸生成器，核心只負責通用部分。

### 從維護成本找出最難維護的部分

討論一度從「切換 kind 的機制已經做好了」出發。使用者指出這是沉默成本。改用最近一週的修正 commit 找出真正難維護的地方，表單核心的 `_structure.py` 與 `_schema.py` 各約 11 個修正，集中在兩處：

1. 把 Pydantic 的 union 翻成給 widget 推論的 schema：具名 union、nullable alias、根 model 的固定 tag、branch 與 literal owner 的對應，約 12 個修正。根本原因是 widget 和 MCP 要從 schema 推回結構，核心只好一路補標記。
2. 舊分頁的相容還原：未標 tag 的 union 要猜分支，約 5 個修正。

兩處的共同放大器是 key-or-inline 有兩種表示法：未標 tag 的 `str | Model`，以及有 tag 的 `ModuleRef | ModuleInline`。

因此有兩項簡化：

- S1：schema 不再暴露 union 結構，改成「選項＋子表單」。第 1 處的修正來源大多消失。
- S2：生成器把 `str | Model` 換成有 tag 的形狀，核心不再處理未標 tag 的 union。第 2 處的「猜分支」消失。

使用者同意 S1，聽完 S2 的細節後也同意，兩項一起在表單核心凍結前完成。

### 動態切換要不要保留

使用者問：切換 inline kind 就改變子欄位格式，是否必然需要 FormInput 或 FormModel 帶 factory hook？

不需要新的 hook，前提是形狀與選項只看型別和捕捉的來源，初值只看型別、範本與來源。所有分支的形狀在產生表單時就已確定。切換只寫入新的值，核心再套用新分支的預設值。

只有「依切換前的值決定新內容」才需要帶狀態的 hook，例如切換時複製 library 內容，或沿用同名欄位。這兩件都不做，這條界線也寫進決策。

依這條界線比較四種「形狀會隨值改變」的情況：

| 情況 | 處理 |
| --- | --- |
| key 或 inline | 保留，這是 library 引用的本質 |
| 可留空 | 保留，core 型別允許 |
| inline 有多種 kind | 保留。使用者的偏好，只要分支機制還在，額外成本只有生成器列出選項 |
| 掃描用 Range 或 List | 刪除。使用者：「我目前也沒有要用list的實驗與workflow」 |

### 繼承 library module 再覆寫

使用者補了一個先前沒想到的場景：recipe 常會先引用 library module，再覆寫部分欄位。現有做法是 recipe 先呼叫 `use_inline` 複製，再寫入覆寫值。使用者提議在 ref 下拉的右鍵選單加「轉成 inline」。使用者再問：「widget層需要有ref/inline的認知嗎？」

最後的做法是通用的選項指令。選項可以帶 `commands`，widget 右鍵列出目前選中選項的指令，按下就呼叫 `FormState.run_command`。生成器在每個 key 選項上宣告 `to_inline` 並提供實作。recipe 的 `use_inline` 改成呼叫同一個指令。Widget 與核心都不知道 ref、inline 或 library。

## 被否決的選項與理由

- 選項清單由 widget 拼出：判斷分散，agent 看不到，下一個欄位類型又要補 widget。
- 選項只做 key-or-inline，其他下拉維持原樣：核心凍結後就不能再擴，兩套讀法會長期並存。
- 下拉切到 Inline 時自動複製 key 的內容：切換規則要看切換前的值，核心也得知道 library。改成明確指令。
- 每個 key 選項預先帶完整內容，widget 直接寫入：不必加指令機制，但 schema 會帶著所有 library 內容，MCP 每次讀表單都要搬。
- 真正的繼承，也就是 ref 加覆寫並保留和 library 的連結：價值是 library 更新會自動傳到使用處，但要新增輸入形狀，核心、生成器、widget 與 recipe 都要改。目前不做。
- 刪除多 kind inline 以拿掉分支機制：要在開表單時把 kind 縮窄成固定一種，MCP 的 `use_inline` 遇到不同 kind 的 entry 會衝突，Inspect 也不能直接換 inline 的 kind。省下的不多，還多了兩條規則。
- ref 選中時顯示被引用 entry 的唯讀內容：input 只有 key，要顯示就得讓選項帶 library 內容。使用者：「如果實作困難ref就不用支援只讀顯示」，下拉「只顯示選單名就可以」。
- 還原舊分頁改成完全不猜（S3）：S2 已經消除猜分支，剩下的還原規則不複雜，先不改。

## 留下的問題

- 真正的繼承（ref 加覆寫）如果之後需要，另開設計。
- 「分頁開著時 library 已變更」的提示目前不做。使用者選擇維持「只在開分頁與 Reset 時捕捉」。
- 選項指令目前只有 `to_inline`，之後的指令由生成器宣告，核心與 widget 不需要改。

## 依據

- 表單核心、生成器與 editor 的程式碼，讀自 measure-form integration@3dba3f5dd：`gui/form/_structure.py`、`gui/form/_schema.py`、`gui/form_adapter/_modules.py`、`gui/widgets/form/_schema_view.py`、`_controls.py`、`editor.py`。
- 修正次數取自 2026-10-06 之後觸及 `gui/form` 的 commit 標題。

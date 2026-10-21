---
date: 2026-10-13
status: active
superseded_by: null
related:
  - docs/adr/0073-measure-form-derivation.md
  - docs/adr/retired/0012-cfgbuilder-value-layer-fluent-assembly.md
  - docs/adr/0065-cfg-editing.md
  - docs/adr/draft/module-boundaries.md
basis:
  - main@bc244c544（討論起點，2026-10-12）
  - main@553ac802f（撰寫時）
---

# Measure 表單重設計：從 builder 重抄到 core cfg 推導

## 結論

決策見 [measure-form ADR draft](../adr/0073-measure-form-derivation.md)，實作驗收後轉正式 ADR。本篇記推導過程與被否決的路。

- 表單形狀從 core 的 pydantic cfg 推導，adapter 不再用 builder 重述。
- 欄位說明用 `Annotated[..., FieldMeta]` 放 core；起始值與偏差放每個實驗目錄的 `form.py`，是一棵對齊 cfg 的樹。
- module 節點只有 `Slot` 與 `Template`，加 `fallback`；角色表退化成「新建 module」的範本目錄，以元件 alias 為參數。
- `dev` 袋子換成 core 的具名裝置欄位，adapter 把 alias 解析成 direct 名稱當起始值。
- 對外格式與 core 的差異用 `extras` 加具名 `lower` 方法明寫。

## 問題

討論從 lab-boundary 票 02 的三個歸屬接縫開始（seeds 怎麼拆、notebook flux_dep、singleshot 的擬合與畫圖）。使用者把第 1 題的前提改掉：seed 與 builder 是一體兩面，可以帶實驗認知；真正的困擾是「生成的辦法不直觀，從實驗模組看不出表單長什麼樣」。

調查結果：

| 事實 | 數字 |
| --- | --- |
| 有 `cfg_definition` 的 adapter | 52 |
| 對外格式真的與 core 不同的 adapter | 5，全是 `super()` 後改一個欄位 |
| 用本地 `_blank_*` 工廠重寫角色預設的 adapter | 23，共 20 個函式 |
| `custom(...)` seed 中真的讀 context 的 | 0，全部回傳運算式字串 |
| core cfg 的欄位型別種類 | 約十種，全部可對到既有 spec 型別 |
| builder、seeds、spec_helpers 的測試 | 1285 行，隨機制一起退役 |

## 考慮過的選項與比較

### 改善 builder（方向 A 到 C）

可預覽的 `describe()`、收斂 23 個 raw `.field` 宣告、預設值資料化。能做，但仍是兩份形狀宣告。使用者：「我覺得你的建議有機會，繼續推進」指向更進一步的推導。

### 形狀從 core 推導（方向 D 的前提）

卡住的點是 adapter 的黑盒子理念。使用者：「adapter內對外是黑盒子……但現實是往往內部格式都與外部極其接近」。5 個覆寫全是同一棵樹加一個欄位的翻譯，所以把預設翻過來：預設等於 core，偏差明寫。

### label 放哪

core 的 `Annotated` 或 adapter 的 overlay。使用者：「我想放core會比較簡單，而維護實驗的作者都是已一整個實驗為單位」。判準定為「MCP 的 schema 描述用不用得到」。`FieldMeta` 不能放 `experiment.cfg_editing`，它間接 import `gui.cfg` 會違反 C15；放 `zcu_tools.cfg_model`，`program` 層也能用。

### 裝置欄位

使用者指出 `dev` 是舊設計：device snapshot 已獨立存 `/device`，儀器改為直接操作 driver。第一版提案讓 core 宣告角色、`RunContext` 以角色為 key；使用者否決：「core不該有屬於gui/mcp/notebook/...的認知，就是單純的實驗宣告，應該是adapter負責包裝」，並要求「GUI的flux_dev初始值也要以direct的格式」。改為 core 具名 `str` 欄位。

### 角色表怎麼交給 Form（設計槓桿）

使用者：「3可以詳細討論，這是設計槓桿」。三個可動的點：shape 在推導後多餘；表由組合根注入而非每個 adapter 傳；seeds 以 `{qubit}`、`{resonator}` 為參數以備多 qubit。`Pulse`、`Ro` 載體先改成通用 path 對應，使用者覺得「有點難讀」，改回 zcu_lab 的 `pulse()`、`ro()` 小工具產生對應，trig_offset 政策成為可見預設。

### G、F、D 三個方向

概念數到了十五個，而且仍用字串路徑指欄位。我提了三條不同的路：

- G：起始值也上 core 的 `Annotated`。使用者：「G我擔心的是，實驗專屬的參數覆寫的需求，比如寫某個module的欄位」。
- F：角色退化成槽位加範本，去掉 init 模式。使用者：「F目前看起來最有希望。可以試試」。
- D：起始值是資料檔。使用者：「考慮到一個實驗是core.py+adapter.py+...，D有相當大的吸引力，但特製解析邏輯有點過度工程的味道」。

覆寫需求成了篩選器，它偏好樹。最後定案是 F 的語意加樹狀 `form.py`，用 Python 字面值寫；D 留作日後換語法的選項。

### 壓力測試

使用者要求三項：複雜預設值（singleshot 的 confusion_matrix）、`dev` 預設值、是否獨立 form.py。結果：confusion_matrix 是分析時從參數讀的，不是表單預設值；所有 `custom` seed 都是運算式字串，`Expr` 涵蓋 100%；裝置三種既有樣式對應 `str`／`str | None` 加 `AliasName`。使用者：「採用，更新草稿，form.py慣例也記進去」。

## 被否決的選項與理由

- 每個 adapter 傳角色表：52 處重複參數，改由組合根注入。
- 角色退化後「庫裡沒有就 Invalid」：使用者「我認為需要fallback」，加回 `fallback=Template(...)`。
- lab-boundary 票 02 為 builder 寫合成測試：1285 行將隨機制刪除，縮成機械搬移。使用者：「採用建議」。
- `dev` 搬家寫 migration：「目前沒有實驗在改版後使用，所以不需要處理遷移邏輯」。
- 裝置改 run 參數：provenance 要另外記。

## 留下的問題

- `list[...]` 型別的推導規則，連同 mux 欄位的多 qubit 顯示方式。
- `Slot(copy=True)` 是複製 library 模組再套 `set`，與今天從角色 seeds 建 blank 不同；遷移時逐實驗確認。
- 範本目錄接進現有 `TemplateCatalog`，以及 `Slot` 掛 library 槽位的實作，原型未碰。

## 依據

- 原型（一次性工作檔，未入庫）：用真實的 `PROGRAM_SHAPES` 與 spec 型別推導 twotone/freq 與 onetone/flux_dep，diff 舊 builder 只剩已知三項。
- 現況程式：`zcu_lab/v2/_support/measure/`（schema_builder、seeds、defaults/role_table）、`lib/zcu_tools/experiment/cfg_editing/catalog.py`、`lib/zcu_tools/gui/app/measure/adapter/analyze_params.py`（`Annotated` 推導的先例）。
- 相關 ADR：[0012](../adr/retired/0012-cfgbuilder-value-layer-fluent-assembly.md)、[0009](../adr/retired/0009-spec-value-fluent-and-literal-lock.md)、[0065](../adr/0065-cfg-editing.md)。

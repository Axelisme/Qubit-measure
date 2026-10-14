---
status: draft
---

# Measure 表單從 core cfg 推導，起始值寫在 form.py

**狀態：** draft。使用者已採用設計方向（2026-10-13），由 measure-form task 實作；實作完成並驗收後轉正式 ADR 並編號，完成前 [[0012]] 的 builder 仍是現行機制。
**關聯：** [[0012]] 的 context-free definition（實作完成後由本篇接替）、[[0009]] 的角色 default 表（由本篇的範本目錄接替）、[[0065]] 的 cfg 編輯與 `dev` 對應、[[0069]] 的核心與前端分工、[[0051]] 的 program shape catalog、[module-boundaries](draft/module-boundaries.md) 的分層。

## 問題

measure adapter 的表單由 `cfg_definition()` 用 builder 逐欄宣告。core 的 pydantic cfg 已經宣告過同一棵樹，adapter 再抄一次：52 個 adapter 都有兩份形狀宣告，而只有 5 個 adapter 的對外格式真的和 core 不同，且全是「同一棵樹加一個欄位的翻譯」。預設值包在 `Seed` 閉包裡，要追進三層函式才知道一個 sweep 從哪裡來，而 52 個 adapter 的 `custom(...)` seed 沒有一個真的讀 context，全部只回傳運算式字串。23 個 adapter 用本地 `_blank_*` 工廠命令式地重寫角色表已有的預設。表單的 label、section、鎖定、小數位散在五處，建好的 definition 無法印出「最後表單長什麼樣」。

使用者 2026-10-12：「目前生成的辦法不是那麼的直觀，也不太能從實驗模塊看出來生成的表單會長什麼樣子」。

## 決策

### Adapter 黑箱與捕捉來源

使用者於 2026-10-14 定案：「感覺怪怪的，理論上對框架來說，adapter內是黑盒子，不該知道裡面是實驗」「兩個都同意，adaper就是一個黑盒子」。

框架只要求 adapter 的 `make_form(ctx)` 回傳 `Form[C]`，並把該 Form 驗證後的輸出原封不動交給 `run` 與 `validate_run_request`。泛型連起 Form 輸出與 consumer 參數。框架不知道這份輸出是實驗 cfg，不檢查具體型別，也不重新驗證。`ExpCfg_cls` 與 `BaseAdapter.build_exp_cfg` 的重複驗證退休。Adapter 自己處理輸出與 extras。框架保存快照時只把輸出當可序列化模型。

`form.py` 統一提供 `make_form(ctx)`，取代固定 `FORM`。`ctx.has(ref)` 只查詢捕捉的 library 或 alias 參照是否存在。缺來源、缺 key 或未綁 alias 回 False，來源故障仍 raise。Ctx 只持有這個查詢需要的來源，不帶 adapter 名稱，也不提供 has_value。函式使用一般 Python if／else 決定初值與形狀，取代 Slot fallback 節點。

Make_form 照常回傳 `Form(Cfg, tree)`，不把 ctx 傳進 Form。框架在 Open 與 Reset 捕捉來源、呼叫 make_form，再綁定範本目錄並驗證。Edit、preview 與 Run 使用既有捕捉，不呼叫 make_form 或 refresh。啟動與 registry register 不做範本預檢。Reset 可同時換形狀與初值，memento 不符時沿既有 best-effort 規則丟棄。

持久範本目錄仍是固定資料。`bath_reset.pi2_cfg` 使用 `Slot("{qubit}.module.x90")`，有 x90 時保留引用，缺 x90 時留空並標 Invalid。使用者於 2026-10-14 接受這個差異：「A是理想，但目前沒有用的話就C」。目前不加入函式型範本，第二個需要依 ctx 選來源的範本出現時再討論。

### 一個實驗三個檔

| 檔案 | 內容 | 依賴 |
| --- | --- | --- |
| `core.py` | cfg 欄位與其 `FieldMeta` 說明、run、analyze | `zcu_tools.experiment`、`program` |
| `form.py` | 匯出 `FORM`：一棵對齊 cfg 的樹，給起始值與表單偏差 | 只 import Qt-free 的節點型別 |
| `gui.py` | adapter：run、analyze、writeback、guide；`form = FORM` | adapter 框架 |

`form.py` 是純資料。MCP 與 notebook 讀起始值時不載入 Qt。它和表單 dump 出的文字同形。日後若改成資料檔，只是換語法。

使用者 2026-10-13：「採用，更新草稿，form.py慣例也記進去」。

### 表單形狀從 core cfg 推導

生成器以 `FormModel.options(env)` 宣告 module／waveform 的語義選項，核心投影為 `x-form-options`。Widget 與 MCP 只消費 label、value、可選 fields／commands，不解讀 union 或 library。持久 child key 在 input 邊界變成 tagged ref／inline，轉回 cfg 時去 wrapper，保留 key。普通換分支呼叫 `defaults(env)` 建 fresh 範本／型別初值。`to_inline` 指令才複製固定 library 快照，recipe 的 use_inline 共用同一指令，不另寫 measure 複製政策。表單、host guard 與版本邊界見 [Pydantic 表單 draft](pydantic-form-tool.md)。

adapter 對外的 cfg 格式預設等於 core 的 pydantic cfg。lib 的共用 Form 生成器 `gui.form_adapter` 從型別推導 spec，adapter 不重述形狀，只宣告起始值、鎖定、隱藏與偏差。黑盒子仍成立。對外契約仍是 adapter 發布的 spec，只是沒有差異的部分不必再說一次。

使用者 2026-10-12：「adapter內對外是黑盒子，也就是說adapter可以曝露與內部使用不同的cfg格式，但現實是往往內部格式都與外部極其接近」。

推導規則：

| core 型別 | spec |
| --- | --- |
| `int`、`float`、`bool`、`complex`、`str` | 純量 |
| `T \| None`（純量） | 純量，optional |
| `Literal[...]`、enum | 純量，choices |
| `SweepCfg`、`SweepCfg \| list[...]` | sweep；list 分支只給 notebook |
| 單一 module 型別（`PulseCfg`、`PulseReadoutCfg`、各 reset） | module reference，一個 shape |
| `ReadoutCfg`、`ResetCfg`（`Annotated[Union, Field(discriminator)]`） | module reference，allowed 為 union 成員的 shape |
| `X \| None`（module） | 同上，optional |
| `ConfigBase` 子類 | section，遞迴 |

先剝 `Annotated` 再攤平 union；shape 的 discriminator 值取自各 class 的 `type: Literal[...]`，不另建對照表。`float | QickParam` 忽略 `QickParam` 成員。`required` 採「pydantic 無預設值」，空值在表單就拒絕。欄位順序是實驗自己宣告的欄位依宣告順序，繼承的 program 欄位在後。`list[...]` 型別尚無推導規則，相關欄位暫列隱藏。

### 欄位說明放 core 的 `Annotated`

顯示名稱、單位、小數位、一句說明是前端中立的實驗知識，寫在 core 欄位的 `Annotated[..., FieldMeta(...)]`。`FieldMeta` 放 `zcu_tools.cfg_model`：它是 Qt-free kernel，`program` 層的 `ProgramV2Cfg` 也能用它標 `reps`、`rounds`、`initial_delay`、`relax_delay`。判準是「MCP 的 schema 描述用不用得到」。用得到的放 core。提到前端機制的留 adapter，例如 device 候選清單來源、分組順序、widget。沒標的欄位用欄位名自動產生。`Annotated` 不改序列化，沒有格式版本問題。

module 與 waveform cfg 分第二步加 `FieldMeta`，讓 `experiment.cfg_editing` 手寫的欄位 dict 改由同一個推導器產生；`ProgramSpecPolicy`、inherit hook 與 shape 登記表留在 cfg_editing。`mux_*`、`mask`、`outsel`、`ro_ch` 暫列隱藏，不是永久：它們預計用於多 qubit 同時量測，顯示方式定了再逐個顯示。

使用者 2026-10-12：「我想放core會比較簡單，而維護實驗的作者都是已一整個實驗為單位」；2026-10-13：「其實mux_freqs、mux_gains、mask、outsel、ro_ch是有計劃顯示的，但目前還沒好的顯示方式，預計用於同時多qubit量測的實驗」。

### form.py 的樹

樹對齊 cfg 的結構，巢狀 dict 對應 section。純量節點：字面值、`Expr("...")`（運算式字串）、`Range.around(center, halfwidth, expts)`／`Range.between(start, stop, expts)`、`AliasName("flux_dev")`、`Locked(value)`、`Hidden()`。module 節點只有兩種：

- `Slot(path, copy=False, set={}, lock={}, fallback=None)`：掛 library 的槽位，例如 `"$qubit.module.readout"`。`copy=True` 從該模組複製一份 inline 再套 `set`。庫裡沒有時，`fallback=Template(...)` 補一份新鮮模組；沒給 fallback 就是 Invalid。`set`、`lock` 的 path 相對 module 根。
- `Template(id, set={}, lock={})`：從範本目錄建新鮮 inline 模組。

沒有 init 模式。鎖定與覆寫掛在所屬節點上，不用跨層路徑字串。這來自原型，兩個實驗推導出的 spec 與舊 builder 相等：

```python
FORM = Form(FreqCfg, tree={
    "modules": {
        "reset": Slot("$qubit.module.reset"),
        "init_pulse": Hidden(),
        "qub_pulse": Slot("$qubit.module.probe", copy=True, fallback=Template("qub_probe"),
                          set={"gain": 0.1, "waveform.length": 5.0}, lock={"freq": 0.0}),
        "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
    },
    "sweep": {"freq": Range.around("$qubit.freq", halfwidth="1.5 * $qubit.kappa", expts=301)},
    "relax_delay": 1.0, "reps": 1000, "rounds": 100,
    "initial_delay": Hidden(),
})
```

`Range`、`Expr` 由 lib 擁有，alias 路徑由實驗填；lib 不寫死任何 alias 名稱。

使用者 2026-10-13：「F目前看起來最有希望。可以試試」；「我認為需要fallback」。

### 範本目錄取代角色表

`TemplateDef(module, label, seeds)` 是 lib 的通用格式：module 型別、「新建 module」下拉選單的名字、以及相對 module 根的 path 到純量或運算式。範本表在 zcu_lab，就是今天新建 module 用的那份；seeds 與 slot 以 `{qubit}`、`{resonator}` 為元件佔位符，`str.format` 展開，預設綁定 `$qubit`、`$resonator`；多 qubit 實驗寫 `Template("readout", qubit="$q2", resonator="$r2")`。產生 seeds 的 `pulse()`、`ro()` 是 zcu_lab 自己的小工具，trig_offset 政策是函式簽名上可見的預設參數。lib 不認得任何 shape。

範本目錄由組合根注入：`register_all(..., templates=...)`。註冊 adapter 時走訪 `form.py` 的 `Slot`／`Template`，範本 id 不存在或 module 型別與欄位不相容即啟動失敗。adapter 不再逐個傳角色表。

### 裝置是 core 的具名欄位

`dev: Mapping[str, str]` 刪除。core 用具名 `str` 欄位宣告裝置，例如 `flux_dev: str`，可選裝置為 `str | None = None`；取 driver 仍是 `context.devices[cfg.flux_dev]`，`RunContext.devices` 維持以名稱為 key。角色、alias、`$flux_dev` 都是 session 與 GUI 的概念，core 不出現。adapter 負責包裝。欄位走一般推導，`choices_source` 給候選清單。起始值由 `AliasName` 在建立 cfg 時解析成具體名稱，表單值是 direct 字串。必填欄位解析不到是 Invalid，可選欄位留空。`flux_unit` 留在 cfg，它是 sweep 數值的單位。MCP recipe 的 `flux_device` 參數不變，只有 recipe helper 改寫這個欄位。

受影響實驗的 `cfg_schema_version` 升 major，不寫 migration、不加 before validator。[[0065]] 描述 `dev` 對應的那一段在實作完成後由本節取代。

使用者 2026-10-12：「core不該有屬於gui/mcp/notebook/...的認知，就是單純的實驗宣告，應該是adapter負責包裝」；「GUI的flux_dev初始值也要以direct的格式，我不太喜歡alias形式的Enum欄位」；2026-10-13：「目前沒有實驗在改版後使用，所以不需要處理遷移邏輯」。

### 偏差明寫

對外格式與 core 不同時，adapter 宣告 `extras`（GUI 專屬欄位的小 model）加一個具名的 `lower(extras, cfg) -> cfg`。既有的五個偏差都落在這裡：`earlystop_snr` 非正數代表關閉、`detune_ratio` 換算成 `detune`、`homophasal` 子區段依取樣模式送入、`method` 固定為常數。核心不知道 extras；載入舊資料只有 cfg，不需要 extras。

### 依賴方向

`FieldMeta` 與樹的節點型別只依賴標準庫與 `zcu_tools.cfg_model`，不 import `zcu_tools.gui`；使用者 core import 它們時，既有 contract C15 仍成立（C15 檢查間接 import）。推導器與 `Form` 在共用的 Qt-free 前端框架 `zcu_tools.gui.form_adapter`，measure、workflows、library 與 writeback 的表單定義共用，不 import `gui.app` 或 `zcu_lab`；contract `form-adapter-no-host` 執行這條邊界。範本目錄的格式在 lib、表在 zcu_lab，C16 不變。

使用者 2026-10-13 裁決把生成器從 measure adapter 框架移到共用位置，稱它為「gui adapter 工具」，名稱「都行」。

### 驗收

每個遷移後的 adapter，推導出的 spec 等於舊 builder 的 spec，起始值在同一個合成 context 下產生相同的值樹。遷移完成後刪除 builder、seeds、spec_helpers、角色表與其測試。

## 取捨與被拒絕方案

- **維持每個 adapter 各自宣告，只改善 builder 可讀性。** 仍是兩份形狀宣告，漂移來源不消失。
- **label 放 adapter overlay。** 用字串路徑指欄位，打錯到執行期才知道；core 作者要跨檔維護。
- **起始值也上 core 的 `Annotated`。** 覆寫一多就變成壓在型別註記裡的小語言，flux_dep 的 readout 要覆寫四個欄位；使用者 2026-10-13：「G我擔心的是，實驗專屬的參數覆寫的需求」。
- **角色表帶 blank／ref 雙重身分與三種 init 模式。** `Slot` 加 `Template` 加 `fallback` 表達同樣的語意，少三個概念。
- **起始值寫成資料檔。** 使用者 2026-10-13：「D有相當大的吸引力，但特製解析邏輯有點過度工程的味道」。樹的 schema 相同，日後只是換語法。
- **裝置改為 run 參數不進 cfg。** provenance 要另外記，各實驗 run 簽名不同。
- **`dev` 搬家加 before validator。** 相容性邏輯；目前沒有需要重讀的資料。

## 後果

- core.py 一個檔看到欄位與顯示，form.py 一個檔看到起始值與偏差；gui.py 只剩 adapter 邏輯。
- 起始值是資料：可印、可 diff、不需 session 就能測；MCP 的 guide 文字可從它推導。
- program cfg 新增欄位時表單自動出現，不顯示要明寫 `Hidden()`。
- 範本與槽位錯誤在啟動時失敗，不等到開 tab。
- 代價：52 個 adapter 與 9 個 core 要遷移；9 個實驗升 major 版；`experiment.cfg_editing` 的 shape catalog 第二步改推導。

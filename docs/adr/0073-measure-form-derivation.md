---
status: accepted
---

# ADR-0073 — Measure 表單從 core cfg 推導

本篇接替 [[0012]] 的 measure definition／seed 契約與 [[0009]] 的角色 default 表。
Library、writeback 與 Inspect 的 CfgSchema 契約仍見 [[0065]]。
實驗核心與前端分工見 [[0069]]。

## 問題

Builder 要 adapter 再宣告一次 core Pydantic cfg 已有的形狀。初值藏在 Seed 閉包與
本地 blank 工廠，作者需要追多層函式才能知道表單會長什麼樣。欄位說明、鎖定與初值
分散在不同檔案，兩份形狀也容易漂移。

使用者 2026-10-12：「目前生成的辦法不是那麼的直觀，也不太能從實驗模塊看出來生成的表單會長什麼樣子」。

## 決策

### Adapter 是黑箱

使用者於 2026-10-14 定案：「感覺怪怪的，理論上對框架來說，adapter內是黑盒子，不該知道裡面是實驗」「兩個都同意，adaper就是一個黑盒子」。

Adapter 的 `make_form(ctx)` 回傳 `Form[C]`。框架把該 Form 驗證後的輸出原封不動
交給 `run` 與 `validate_run_request`，不檢查具體型別，也不重新驗證。
`BaseAdapter` 以泛型連起 Form 輸出與 consumer；Registry 保持非泛型。
`ExpAdapterProtocol` 的 make_form、run 與 validate_run_request 用 Any 表示同一 adapter
自己的不透明輸出，不把這個例外擴到其他資料結構。保存快照時框架只把輸出當可序列化模型。
`ExpCfg_cls` 與 `build_exp_cfg` 的重複驗證不再存在。

Adapter 可以公開不同於 core 的輸入。需要 extras 時，Form 接受小型 Pydantic model
與具名 `lower(extras, cfg)`，兩者成對宣告。Adapter 自己處理轉換與 Run-only 資訊。
框架不猜測哪些欄位屬於實驗。

### 一個實驗的檔案分工

| 檔案 | 責任 |
| --- | --- |
| `core.py` | cfg 型別、前端中立的 FieldMeta、run 與 analyze |
| `form.py` | Qt-free `make_form(ctx)`，用一般 Python 選初值與形狀 |
| `gui.py` | 連接 make_form、run、analysis、writeback 與 operator guide |

使用者 2026-10-13：「採用，更新草稿，form.py慣例也記進去」。

Core 的 `Annotated[..., FieldMeta(...)]` 保存 label、unit、decimals 與 description。
沒有標記的欄位由生成器使用欄位名。FieldMeta 不改序列化格式。
前端的 widget 選擇與分組不放 core。使用者 2026-10-12：「我想放core會比較簡單，而維護實驗的作者都是已一整個實驗為單位」。

### Form 的形狀與初值

`gui.form_adapter.Form(core_cfg, tree)` 從 Pydantic 型別推導 FormModel、起始 input
與輸出轉換。純量、nullable、Literal／enum、巢狀 model 與 module／waveform
由共用生成器處理。SweepCfg 使用 LinRange input，count 葉名為 n。
持久 Spec 保留 child key，to_cfg 不從 library materialize 或改寫 key。
未知 seed、受屏蔽 seed、缺必要 discriminator 與不支援的選項在入口報錯。

Tree 對齊 model 的欄位。字串是 literal，`Expr` 才是 expression。
`Range.around`／`Range.between` 宣告掃描，`AliasName` 產生 direct 名稱，
`Locked` 與 `Hidden` 宣告表單偏差。Module 節點分為兩種。

- `Slot(path, copy=False, set={}, lock={})` 使用捕捉的 library 槽位。Copy 產生 inline input。
- `Template(id, set={}, lock={})` 從注入的持久範本建 fresh inline input。

缺 Slot 來源時，optional 欄位停用且有效，required 欄位留空且 Invalid。
沒有 Slot fallback 或 init 模式。需要選來源時，make_form 明寫 if／else。

```python
def make_form(ctx: FormContext) -> Form[FreqCfg]:
    return Form(FreqCfg, {
        "modules": {
            "readout": Slot("$qubit.module.readout")
            if ctx.has("$qubit.module.readout") else Template("readout"),
        },
        "sweep": {"freq": Range.around("$qubit.freq", "1.5 * $qubit.kappa", expts=301)},
        "reps": 1000,
    })
```

初值 expression 在建立 fresh input 時求值，成功後保存 direct 值。
缺來源或求值失敗不補數字 fallback；必要欄位維持未填。
使用者寫入的 expression 保留原式。換分支用 fresh defaults，不繼承 previous input。
`to_inline` 是明確複製固定 library 內容的指令。Widget 與 MCP 只消費 core 提供的
Option label、value、fields 與 commands，不猜 union 或 library 結構。

Module／waveform 使用持久 Spec 與 FieldMeta。硬體專屬的不支援欄位由生成器屏蔽；
必填 ADC ro_ch 保留。List 推導與多 qubit 顯示不在本設計內。

### 捕捉來源與使用時驗證

`FormContext` 只提供 `ctx.has(ref)`，回答捕捉的 library／alias 參照是否存在。
缺來源、缺 key 或 alias 未綁回 False；來源故障 raise。它沒有 adapter 名稱或 has_value。
Make_form 回 `Form(Cfg, tree)`，不把 ctx 傳進 Form。

`TabFormResource` 在 Open 與 Reset 捕捉來源，呼叫 make_form，然後綁定並驗證
Registry 的範本目錄。Register 不執行 factory，也不預檢範本。
Edit、preview、result 與 Run 使用同一捕捉，不重建 Form 或刷新來源。
Reset 可以同時換形狀和初值；失敗保留原 Form。Memento 保存 Form input，
不相容的葉按既有 best-effort 規則丟棄，不加舊版雙讀或 native cfg 反向回填。
Load 只載分析／writeback 所需結果，不覆蓋表單。

資源擁有 mutation guard、通知與撤銷。Editor 只讀借用的 state，write 與 run_command
成對交回資源。Subscriber 故障記 log 後繼續，不回滾已提交 input。
Measure 不再使用 CfgSchema 的跨 tab refresh／Unavailable 恢復模型。
詳細 editing、Run admission 與 wire 契約見 [[0065]]。

### 持久範本目錄

`cfg_model.TemplateDef` 宣告持久 Spec 型別、label 與 seeds。表在 `zcu_lab.templates`，
組合根以 `Registry(templates=TEMPLATES)` 注入。TemplateCatalog 使用同一份表建立
Library 新建選單，reload candidate 從 current Registry 取得表，不重建角色表。
`{qubit}`／`{resonator}` 佔位符預設展開成 `$qubit`／`$resonator`，caller 可明填綁定。
範本 seeds 疊在 Spec defaults 上。Trig_offset 政策由實驗室的範本工具參數宣告。

範本目錄是固定資料。`bath_reset.pi2_cfg` 使用 x90 Slot，有來源時保留引用，
缺來源保持 Invalid，不改回 pi2_pulse 或數字初值。使用者 2026-10-14：「A是理想，但目前沒有用的話就C」。
函式型範本沒有現存需求，第二個需要依 ctx 選來源的範本出現時再討論。

### Core 使用具名裝置欄位

Core 用 `flux_dev: str` 等具名欄位取代 `dev: Mapping[str, str]`。
可選裝置使用 `str | None`。Core 不認得 alias 或 session，透過
`context.devices[cfg.flux_dev]` 取得 driver。Form 用 `AliasName("flux_dev")`
在 Open／Reset 把 alias 解析為 direct 字串；既有 input 不隨 alias 表更新。
缺名稱或不在捕捉的候選清單時由表單報錯，不使用靜態裝置 fallback。

Flux_unit 獨立保存在 cfg。Recipe 的公開參數與來源政策不變，helper 寫 caller
指定的具名欄位。它只比 cached unit，不換算、不連裝置。受影響 cfg 升 major，
不加 dev migration 或 before validator。

使用者 2026-10-12：「core不該有屬於gui/mcp/notebook/...的認知，就是單純的實驗宣告，應該是adapter負責包裝」「GUI的flux_dev初始值也要以direct的格式，我不太喜歡alias形式的Enum欄位」。
使用者 2026-10-13：「目前沒有實驗在改版後使用，所以不需要處理遷移邏輯」。

### Owner 與驗證邊界

`cfg_model` 擁有 Qt-free 節點與 TemplateDef，不依賴 GUI。C11 保護這個方向。
`gui.form_adapter` 擁有生成器，不 import gui.app 或 zcu_lab，
由 `form-adapter-no-host` 保護。C15 保護實驗 core 不間接依賴 GUI，C16 禁止框架反向 import zcu_lab。

Builder、Seed wrapper、spec_helpers、角色表與舊 measure tab CfgSchema 資源退休。
Library、writeback、Inspect 保留 CfgSchema、cfg_editing、TemplateCatalog 與公開
resolve_library_reference；這些能力不是 measure Form 的相容路徑。

Framework 測試用 synthetic adapter 驗 Form、Open／Reset、固定 capture、Run、
通知與存還原。實驗定義以真 Registry validate、正式 reload 與必要 Open／Reset
檢查接線，offline Invalid 不等於 cfg 有效。不把逐實驗 parity、Run 或 analysis
當成持續驗收要求。初值差異依明確裁決接受，其餘可疑數值保值待審。

使用者 2026-10-14：「我預期能大幅簡化預設值與格式的定義代碼，目前預設值有些不太合理，可能是建立時打錯的，遷移時可以選擇：保留目前行為，但記錄下來之後讓我看。或是目前行為遷移後會需要比較複雜的寫法，可以直接問我或你。」

## 取捨

維持 builder 仍要抄兩份形狀。把初值全放 Annotated 則讓實驗覆寫變成型別註記內的小語言。
Make_form 保留普通 Python 分支，也讓作者在 form.py 直接查看來源。

使用者 2026-10-13 對資料檔方案的取捨：「D有相當大的吸引力，但特製解析邏輯有點過度工程的味道」。
目前不引入資料檔解析器。Label 放 core，前端偏差放 form.py，代價是作者需同時維護
core 的中立說明與 Form 的來源政策。範本錯誤延至使用時報錯，Registry validate
只保證 adapter 契約，不保證每個離線表單都有完整校準來源。

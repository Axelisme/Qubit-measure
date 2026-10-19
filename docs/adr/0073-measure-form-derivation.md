---
status: accepted
---

# ADR-0073 — Measure 表單從 core cfg 推導

本篇接替 [[0012]] 的 measure definition／seed 契約與 [[0009]] 的角色 default 表。
Library、writeback 與 Inspect 也使用共用 Form，持久 Spec 與提交邊界見 [[0065]]、[[0008]]。
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

Core 的 `Annotated[..., FieldMeta(...)]` 保存 label、unit、decimals、description、group、readonly 與 element_label。
group 是 MCP 單層欄位描述可用的前端中立分類名稱 hint，因此與其他欄位描述一起放在 core。
前端決定實際的 widget、排版與折疊行為。
沒有標記的欄位由生成器使用欄位名。FieldMeta 不改序列化格式。
使用者 2026-10-12：「我想放core會比較簡單，而維護實驗的作者都是已一整個實驗為單位」。

### Form 的形狀與初值

欄位切換與固定 model list 的新契約見 [[0079]]，已採用但尚待實作。以下切換與還原條款依此描述目標設計。

`gui.form_adapter.Form(core_cfg, tree)` 從 Pydantic 型別推導 FormModel、起始 input
與輸出轉換。純量、nullable、Literal／enum、巢狀 model 與 module／waveform
由共用生成器處理。SweepCfg 使用 Range／Center 與 Expts／Step 的四個 tagged branches。Count 葉名為 n，步長葉名為 step。新表單選 Range + Expts。每個分支只保存 active 座標與採樣輸入。Center 的 span 是帶正負號的完整寬度，不保存另一份端點快取。
持久 Spec 保留 child key，to_cfg 不從 library materialize 或改寫 key。
未知 seed、受屏蔽 seed、缺必要 discriminator 與不支援的選項在入口報錯。

Tree 對齊 model 的欄位。字串是 literal，`Expr` 才是 expression。
`Range.around`／`Range.between` 宣告掃描，`AliasName` 產生 direct 名稱，
`Hidden` 宣告隱藏欄位。需要顯示但不能改的值，使用樹上的初值與 `Display(readonly=True)` 或 core 欄位的 `FieldMeta(readonly=True)`。
生成器把中立 metadata 翻成 Display，由 Form core 強制唯讀，不生成單值 Literal。

`LogPoints(max, points, include_zero=True)` 為指定的整數清單欄位宣告 Log 初值，`cfg_model` 不計算清單。
`form_adapter` 擁有 Log／List 分支、換算與清單產生。Core 只接收 `list[int]`，普通 list 與 SweepCfg 不變。
LogPoints 與 Range.between 使用相同的樹上初值邊界，依 advisor 2026-10-19 裁決。
預設 Log 必含 0 與 max，含 0 的 points 至少為 2，geometric samples 的末點固定為 max。
依 advisor 同日的正值分支裁決，include_zero=False 使用 `geomspace(1, max, points)`，首尾固定為 1 與 max，再四捨五入去重。
此時 max 至少為 1；max>1 的 points 至少為 2，max=1 接受 points≥1 並產生 `[1]`。List 只接受正整數，保留順序與單點。
兩種政策的 List→Log 都依最大值與長度回填，不補點；不符合所選政策的結果維持 invalid draft。

Module 節點分為兩種。

- `Slot(path, copy=False, set={})` 使用捕捉的 library 槽位。Copy 產生 inline input。
- `Template(id, set={})` 從注入的持久範本建 fresh inline input。

Slot／Template 不提供 lock 參數，也不提供 `Locked` 節點。List 各位置可有不同的唯讀初值。

範本保留各自的 input projection／model，承載非 lock 初值、Expr requiredness 與 branch ancestry，不在其中生成單值 Literal 來實作 lock。範本初值、依 branch ancestry 的 inline 切換初值與 `input.model.default_input(env)` 重算維持既有行為，不改用通用 defaults。

Module 子欄位唯讀時仍可選 library key。Module 節點本身標唯讀則固定整個 module 子樹，包含 ref／inline 選擇與所有子欄位，不論初值是 ref 或 inline。不另設 selector-only 宣告，也不自動強制 inline。使用者於 2026-10-19 裁定：「我會選B，lock應該被認知為只對子樹作用。」

只為組實驗 cfg 存在的固定值不放進 Form 或實驗 cfg 型別，由 `run()` 補齊，例如單發的 reps／rounds=1。Mux tone name 需要給使用者看，保留並宣告唯讀。使用者於 2026-10-19 裁定：「有一些是為了方便轉換成experiment cfg而塞入的，其實可以改成直接不曝露欄位。run再補齊。」

FTF roles 的 f1、f2、coupler、mux、tone index 與 tone_count 供驗證、離線 analysis 與 writeback 使用，不屬於只為組 cfg 存在的固定值。這些欄位保留唯讀，初值仍來自 captured group，cfg／result／保存契約不變。

Bath reset freq 的 `tested_reset.pi2_cfg.phase` 移除舊 lock，由 Template 的 `set` 給初值 90，欄位可編輯。
`run()` 組 program 時使用 cfg 的 phase，不覆寫為固定值。Guide 註明 90 是建議的 tomography offset。
現有 Template 沒有 nested Hidden／readonly 作者入口，且只有這一處需要，因此不新增 nested metadata API。
不複製共用 module 型別，也不另建局部 cfg 型別。

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
使用者寫入的 expression 保留原式。選項切換由欄位的一個 converter 決定子表單；沒有 converter 時只改 selection，保留原結構與 input，不補 fresh defaults。Inline 的明確繼承政策由 gui.form_adapter 集中宣告，見 [[0079]]。
Sweep 的顯式指令是保值換算，不套 fresh defaults。座標換算只求值座標，保留 n／step 原稿。
採樣換算求值座標與採樣欄位，成功原子安裝 direct 數字，失敗保留整組 input。
未填參與欄位不補數字。非零 span 配 n=1、零 span 配多點都不能藉切換修成其他掃描。
Step 的 floor count、相對容差與有效端點由共用 Form 驗證層擁有，契約見 [[0065]]。
Adapter 集中座標與 step metadata 的精度，lowering 使用有效端點、點數與原 step。
Qt 明示原稿與有效值的差異，MCP 讀同一份 values。兩者不換算，讀取也不寫回 input。
`to_inline` 是明確複製固定 library 內容的指令。Widget 與 MCP 只消費 core 提供的
description 的 Option label、value 與 commands，不猜 union 或 library 結構。
Option.fields 仍是作者宣告，不進選單回覆；查所選 path 才列直接 children。
單一普通 wrapper 攤平一層，但完整 model-field path 不變。
FormEditor 首次展開才建子表單，Inspect／New Entry／writeback 與 workflows 使用相同描述。

Module／waveform 使用持久 Spec 與 FieldMeta。硬體專屬的不支援欄位由生成器屏蔽；
Readout 的 `gen_ch` 與 `ro_ch` 必填且可編輯，`freq` 是唯一 generator／ADC 頻率。
Readout 的 waveform 仍可引用 key，不推導通道或補值，見 [[0074]]。
例如 `Template("readout", set={"gain": 0.1, "ro_length": 2.0})` 直接覆寫平鋪路徑。
固定具名 model list 的推導、picker 與獨立 participants 清單由 [[0079]] 擴充。Framework 不接管 mux group 關係或 lab runtime 的名稱到原始 index 轉換。

### 捕捉來源與使用時驗證

`FormContext` 只提供 `ctx.has(ref)`，回答捕捉的 library／alias 參照是否存在。
缺來源、缺 key 或 alias 未綁回 False；來源故障 raise。它沒有 adapter 名稱或 has_value。
Make_form 回 `Form(Cfg, tree)`，不把 ctx 傳進 Form。

`TabFormResource` 在 Open 與 Reset 捕捉來源，呼叫 make_form，然後綁定並驗證
Registry 的範本目錄。Register 不執行 factory，也不預檢範本。
Edit、preview、result 與 Run 使用同一捕捉，不重建 Form 或刷新來源。
Reset 可以同時換形狀和初值；失敗保留原 Form。Memento 保存 selection／branch 與 Form input，
由可信宣告重建，還原不執行 converter。唯讀欄位保留存檔值，不從宣告重算。型別或來源 identity 不匹配明確報錯，不靜默丟葉，
不加舊版雙讀或 native cfg 反向回填，見 [[0079]]。
Load 只載分析／writeback 所需結果，不覆蓋表單。

資源擁有 mutation guard、通知與撤銷。Editor 只讀借用的 state，write 與 run_command
成對交回資源。Subscriber 故障記 log 後繼續，不回滾已提交 input。
Measure 不再使用 CfgSchema 的跨 tab refresh／Unavailable 恢復模型。
詳細 editing、Run admission 與 wire 契約見 [[0065]]。

### 持久範本目錄

`cfg_model.TemplateDef` 宣告持久 Spec 型別、label 與 seeds。表在 `zcu_lab.templates`，
組合根以 `Registry(templates=TEMPLATES)` 注入。Library 新建選單使用同一份表，
reload candidate 從 current Registry 取得表，不重建角色表。
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

單一實驗的初值政策留在該實驗的 form.py。已定案且供多個 Form 共用的初值政策，
可集中在 zcu_lab 的具名 domain helper，目前包括 T1 時間尺度選源，以及 Rabi 的
來源、有效性與首次初值。各 Form 仍明確選擇倍率或場景，並組合結構。
這些實驗初值政策不進入 generic framework，不擴及 writeback 或實驗順序。
共用型別間的 inline 繼承則由 gui.form_adapter 擁有，不是實驗初值政策，見 [[0079]]。
這個分工讓同類 Form 使用同一政策，避免改了一個、漏改另一個。
跨 owner 的依賴方向仍由下述 C15、C16 與 form-adapter-no-host contracts 保護。

使用者 2026-10-17：「同意，這判斷標準沒問題」

`cfg_model` 擁有 Qt-free 節點與 TemplateDef，不依賴 GUI。C11 保護這個方向。
`gui.form_adapter` 擁有生成器，不 import gui.app 或 zcu_lab，
由 `form-adapter-no-host` 保護。C15 保護實驗 core 不間接依賴 GUI，C16 禁止框架反向 import zcu_lab。

Builder、Seed wrapper、spec_helpers、角色表與舊 measure tab CfgSchema 資源退休。
Library、writeback 與 Inspect 使用相同的 Form 生成器與持久 Spec，不保留
CfgSchema、cfg_editing 或舊 reference resolver。TemplateDef 目錄由 Registry 注入。

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

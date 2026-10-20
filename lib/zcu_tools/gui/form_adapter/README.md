# gui/form_adapter — 從 core cfg 生成 Form

**Last updated:** 2026-10-20，局部欄位排序與 mux tones JSON leaf

本 package 是 GUI 的 Qt-free 前端框架。它把 core cfg 宣告與作者寫的起始樹翻譯成 `gui.form` 的 `FormModel`，再把表單結果轉回驗證過的 core cfg。它供 measure、workflows、library 與 writeback 的表單定義共用。Measure 的註冊、protocol 與分頁 Form resource 使用 `Form`，workflows 的手寫 plan 表單使用 `ModuleInputField`。

主要生成入口如下：

- `Form(core_cfg, tree, *, extras=None, lower=None, fixed_elements=(), field_order=())`：生成 FormModel 子類別、defaults hook、choices、options 與 `to_cfg`。Form 的 cfg 結果型別具協變性，具體 `Form[SampleCfg]` 可交給 generic-free host，`to_cfg` 仍回傳 SampleCfg 或 FormErrors。Core annotations 決定形狀與驗證；樹決定起始值、唯讀屬性、隱藏欄位與 module 預設。Expr 初值欄位在生成的 model 中不帶靜態預設，算不出時保持未填。Range 的 Expr start／stop 也沿用這個規則，不回落 LinRange 的 0／1。Range 端點為 None 時顯示空白且必填，對應 input key 缺席也不使用 0／1。Literal 端點與未宣告的掃描仍保留原有預設。
- `module_form(spec_type)`：為一個具體持久 module 或 waveform Spec class 建 Form。生成的 model 可當 FormState root，也可當 `ModuleInline` 的 module 欄位；host 的選項沿用自己的 `choices(env)`。`to_cfg` 回傳同一個 Spec，child library key 保持字串，不讀 library、asset，也不轉 native cfg。`Form.input_from_spec(spec, env)` 把既有 Spec（library entry 或 writeback proposal）投影成 input；它只接受這類 Form 的同型 Spec。
- `FormContext(parameters)`：實驗 `make_form(ctx)` 使用的固定來源查詢。parameters 是捕捉的 ParameterContext，缺來源時為 None。`has(ref)` 接受 library root 或 point module slot，含 `$alias` 前綴。缺 key、未設定或懸空的 slot、缺或未綁 alias 回 False；語法錯誤、其他 alias 解析錯誤與來源故障仍 raise。它不 materialize module、不讀 asset、不 refresh。
- `ModuleValues`：`Form.read_module` 回傳的單一 module 欄位實際值、模式與來源。
- `ModuleInputField(family, *, nullable=False, ref_family=None)`：把具體持久 ModuleSpec class 或其 union 生成為手寫 FormModel 可用的 tagged 欄位。`annotation` 是 runtime Pydantic 型別，`options(env)` 是同一欄位的選項與複製指令。`family` 決定 inline 家族，`ref_family` 只限制 captured ref keys，省略時沿用 family。`nullable` 單獨決定 None，不在 family 再放 None。空、不支援或超出 inline family 的 ref 宣告在建構時拒絕。`to_spec(value)` 接收該欄位的成功 FormState.result 值，ref 回 literal key，inline 回 detached、驗證過的持久 Spec，nullable None 保 None。它保留 child keys、不讀來源或 materialize，屏蔽欄位由 Spec defaults 補回。其他欄位的 generated 值或不合法 Spec 明確拒絕。

手寫表單在 `options(env)` hook 將 `ModuleInputField.options(env)` 放到欄位路徑，並在 `converters(env)` 的同一路徑宣告 `ModuleInputField.converter(env)`。兩個 hook 使用 host 的同一份固定 capture。宣告不讀來源，也不儲存共享 mutable capture。既有最深 local choices 規則照舊。Python 型別檢查器不能反射 runtime `annotation`，靜態別名可用 `ModuleRef | ModuleInline` 作 nominal value view，nullable 時另含 None。這個 view 只供保存結果與傳給 `to_spec`，不表示 generated `.module` 已是持久 Spec。Workflows 三個 plan 表單共用這個入口，先取持久形式再交既有 core/native factories；校準 policy 仍屬 workflows。

`AliasName` 的 devices 選項包含 core 欄位的靜態字串預設，空字串顯示為 None。Description 與驗證共用同一份 choices。缺 alias 時 input 保持缺席，有 core 預設就用該預設，無預設的 required 欄位仍 Invalid。綁定 alias 照舊帶入裝置名，其他未列出的值仍被拒絕。

生成器擁有兩條翻譯政策，host 與 `gui.form` 不重做：

- `FieldMeta` 的 label、unit、decimals、description、group、readonly 與 element_label 轉成 `Display`，core 欄位與 module 欄位共用同一個翻譯。唯讀不把欄位改成單值 Literal，原有數值型別與精度保留。Sweep 的 decimals 傳到 start／stop、center／span 與 step float 欄位，不套外層 model 或 n。
- 生成 module 欄位時，選填的 `desc` 不在 model、input、description 與寫入路徑。`to_cfg` 驗證成 Spec 時由 Spec 預設補回。從 library 複製成 inline 或 `input_from_spec` 時，原 description 被丟棄，所以編輯後會變回預設。Readout 的必填通道照常可編輯；Pulse cfg 不保存 `ro_ch`、`mux_*`、`mask` 或 `outsel`。範本 seed 屏蔽欄位時，註冊入口拒絕，訊息寫出 template id 與 seed 路徑；`to_cfg` 只由 Spec 預設補回。其他欄位的字串 seed 照舊是字面值。目前沒有露出覆寫機制。

生成的 `options(env)` 宣告可留空欄位的 None、依持久 Spec family 篩選的 captured library keys，以及每個可編輯 Inline kind。只有結構 tag 的 kind 不列為 Inline。對明確預設為 None 的 nullable module，生成器補入停用值，讓選單顯示 None 而非空白。只允許 None、卻沒有 None 預設的 required 欄位不自動停用。通用 Form 的 static defaults 仍省略 None，不影響 validator 推導值。所有 key input 使用 `{"kind": "ref", "name": key}`，inline 使用 `{"kind": "inline", "module": content}`，可留空時也接受 null。Options 的 value 只帶 key 或所選 kind 的 tag。fields 指向該 kind 的 wrapper 子表單，describe 所選欄位才列 children，不重複固定 tag。Ref 選項不帶 fields，也不顯示 entry 內容。切換由 `_converters.py` 集中處理，依 [ADR-0079](../../../../docs/adr/0079-form-field-transitions.md#共用-inline-政策) 共用同名同型別的 raw 繼承。

新型別的每個欄位若在舊型別有同名且相同的可編輯 Form 型別，就沿用原始輸入；否則用目標初值。比較沿生成器去除 QickParam 的規則，不依 raw 值的 Python 型別或有效性判斷。各次生成的同一持久 model 視為同型，巢狀逐層套用。Waveform 與 Pulse、PulseReadout、MuxReadout、各 Reset 都用這條規則。Expression、缺席與 invalid 值原樣保留，超出目標限制由一般驗證報錯，不求值或修正。同型 ref→inline 與 inline→inline 保留全部可編輯欄位，ref 讀固定 Spec，inline 不重新讀來源。Gauss 切 FlatTop 保留 length，raise_waveform 用初值，不把舊 waveform 包進去。Pulse 切 PulseReadout 保留 freq／gain／waveform，但 ch 與 gen_ch 不同名，不搬移。Core 不加入 domain 規則或補 defaults，converter 提供完整目標 input。

每個 key 選項帶 `to_inline` 指令，label 是 Convert to inline。`FormState.run_command` 用固定 FormEnv 的 library 複製持久 Spec，保留 child key、丟棄屏蔽欄位，以一次原子安裝提交快照。來源失敗不改 input。明確複製使用 `to_inline` 指令或 `Form.prepare_inline_copy`，普通 defaults 只建 fresh 範本／型別初值，不提供隱含複製或缺來源 fallback。

持久 Spec 與範本 seed 路徑仍使用原欄位名稱，不含 wrapper。生成器在 input 邊界加入 wrapper，`to_cfg` 去掉 wrapper，保留 child key 字串。Nested ref 使用本 package 的 `KeyRef`，不限制 choices，也不新增 captured source 檢查。Core module ref 仍使用 `ModuleRef` 並檢查固定 library。兩種 ref 的 name 都只接受 literal key，不求 Expr。

`readout/pulse` 直接編輯平鋪的 generator 與 ADC 欄位。`gen_ch` 與 `ro_ch` 必填且可編輯；一般 Pulse 沒有 `ro_ch`。單一 `freq` 同時供兩者使用，不從其他欄位推導通道或補值。Waveform 仍可保留 child library key。

`readout/mux` 沿用 module 的 ref／inline 與 Convert to inline。Scalar 欄位分別編輯，tones 是整個 list 的 JSON leaf，包含每筆 tone 的 label。生成器只在 `MuxReadoutSpec.tones` 加上 `InputPolicy(indexed=False)`，保留原 typed annotation 與驗證。空 tones 可直接輸入 JSON，不靠 picker 新增元素。其他 model list 與 FixedElements 保留 indexed editing。它沒有 waveform 或逐 tone 引用。

Core-cfg 表單可用 `field_order=("participants", "tones", ...)` 把指定頂層欄位放在前面。未列欄位維持原相對順序，nested model 不變。空 tuple 沿用原序，tree 的寫入順序不控制畫面。未知或重複欄位在建構時以 ValueError 拒絕。持久 Spec 的非空 field_order 也以 ValueError 拒絕，保留繼承的欄位順序與 validators。Extras 若有宣告，仍在最後。

`Slot(copy=True).set` 的 dotted Spec 路徑若穿過 captured child key，生成器只把路徑上的 child Spec 複製成 inline，再套用覆寫。未涉及的 child key 與來源保持不變，既有 input 不隨後續來源更新重填。查詢不 materialize 或讀 asset；缺 child、錯 family 與來源故障直接回報，None 覆寫沿用必填空白語意。

Template／Slot 節點在 host 呼叫 `Form.bind_templates(catalog, adapter=...)` 後才解析。Measure 在 Open／Reset 捕捉來源，呼叫 adapter 的 `make_form(ctx)`，再綁定 Registry 的範本目錄。註冊與 reload 不呼叫 factory，也不預檢範本。範本目錄的格式在 `cfg_model`，真實範本表在 `zcu_lab`。
通用佔位符只使用呼叫端明填的 `Template.bindings`，例如 `bindings={"qubit": "$qubit1"}`。
FormEnv 只提供來源，不提供模板綁定預設值。
綁定 catalog 時保留語法、family 與 seed 路徑檢查，真正建立 input 時才展開佔位符並拒絕缺 binding。

Spec Form 的 `template_input(Template(id), catalog, env)` 直接建立範本的 fresh input，不需先 bind。它沿用 Expr 初值與宣告唯讀規則。缺 Slot 來源時，required 欄位留空並標 Invalid；alternate 來源由 make_form(ctx) 決定。已綁定 Slot 保留 child library key，`to_cfg` 回持久 Spec，不展開 native cfg。Core cfg Form 拒絕這個入口。

各範本保留自己的 input projection／model，承載非 lock 初值、Expr requiredness 與 branch ancestry，不在其中生成單值 Literal 來實作 lock。範本初值、依 branch ancestry 的 inline 切換初值與 `input.model.default_input(env)` 重算維持既有行為，不改用通用 defaults。用回傳 input 建新 `FormState`，換範本則建新 state。範本目錄與 captured sources 不因這項操作改變。

Module、seed、Display 與 annotation 翻譯放在私有模組 `_modules.py`、`_seeds.py`、`_display.py`、`_types.py`。`_sections.py` 集中遞迴 model 生成、defaults 與 native 值投影；`form.py` 提供公開 Form 操作，`module_input.py` 提供手寫欄位入口。

唯讀欄位以初值與 `Display(readonly=True)` 或 `FieldMeta(readonly=True)` 表示，不使用 `Locked`、`Slot.lock` 或 `Template.lock`。限制只作用於被宣告欄位的子樹。還原保留唯讀欄位的存檔值，不從宣告重算。

Module 子欄位唯讀不移除 library key 選項。Module 節點本身標唯讀則固定整個 module 子樹，包含 ref／inline 選擇與所有子欄位，不論初值是 ref 或 inline。不另設 selector-only 宣告，也不自動強制 inline。

只為組實驗 cfg 存在的固定值不進 Form 或實驗 cfg 型別，由實驗的 `run()` 補齊。供驗證、analysis 或 writeback 使用的固定輸入仍可保留為唯讀欄位。具體分類由 [ADR-0073](../../../../docs/adr/0073-measure-form-derivation.md#form-的形狀與初值) 定義。

## Hidden 填入值

`Hidden(value)` 隱藏欄位並提供明確的 literal。沒有 core default 的必填欄位也可使用，
但填入值仍受 core／Spec 的型別、範圍與交叉欄位驗證。普通 core tree 在 `to_cfg` 供值；
module 的 generated input 使用該 literal 作為隱藏欄位的初值與 default，不新增 readonly。
`Hidden()` 不帶值時保留普通 core 欄位的省略語意，由 core default 還原；必填且無 default 時
在 Form 建構報錯。Module seed 必須明填值，`Hidden(None)` 表示 literal None，不是省略。

實驗 form 在自己的 `Template.set`／`Slot.set` 以 dotted Spec 路徑標出 Run 會覆寫的欄位，
例如 `Template("readout", bindings={"resonator": "$resonator"}, set={"freq": Hidden(0.0)})`。
巢狀 waveform 路徑寫
`"waveform.length"`，不含 generated input 的 `module` wrapper。Slot 的 `set` 仍只作用於
`copy=True` 的 inline 初值；未複製的 ref 仍使用固定 library 的完整有效值，不改 library。
Inline 選項共用該欄位的隱藏政策。真正 user-owned 欄位照常顯示與驗證。

生成器不推導來源，不新增 sweep→module 關係或另一份 hidden_paths 表。Run 照原樣覆寫，
core input 型別不變。只有經 Run 確認會覆寫的欄位才能這樣宣告，不能以隱藏規避無效設定。
Bath reset phase 維持可編輯。決策見 [ADR-0073](../../../../docs/adr/0073-measure-form-derivation.md#form-的形狀與初值)。

## 固定元素宣告

`FixedElements(list_path, name_field, participants_path)` 指定 root-relative typed paths 與元素的 str 名稱欄位。手寫表單繼承 `FixedElementsForm`，以 `fixed_elements` tuple 列出宣告；生成器接受 `Form(..., fixed_elements=(... ,))`。Fresh `default_input` 先呼叫 Core，再檢查宣告與捕捉名稱唯一性。重名錯誤列出名稱及原始 indices。這個共用能力集中名稱與 participants 政策，Core、widgets 和各 host 不再各自維護。

`FixedElements.identify(data, path)` 依同份 input 的完整原始 index 回傳 `ElementIdentity`，不讀 participants。Logical field observation 同時攜帶該 identity，供 Run capture 保留當次名稱；巢狀固定 list 使用最深的宣告。

普通 mutation 比較目前與候選的有序 raw 名稱，增刪、重排或改名以 `FormInputRejected` 原子拒絕。元素的一般 invalid 值仍可保存。Participants 在一般 Pydantic 驗證檢查非空、無重複且名稱存在，錯誤保留欄位與 index，輸入順序不變。未參與元素照常驗證。還原不比新 capture，存檔 list 成為普通 mutation 的邊界；Reset 才建 fresh input。

宣告者另外提供 `Display(element_label=name_field)` 作 picker label，名稱欄位的 `Display(readonly=True)` 則由 Form core 強制唯讀。同值寫入通過，不同值拒絕；父層或批次寫入不能繞過唯讀名稱。名稱若不接受 expression，可用既有 `InputPolicy(expression=False)` 明訂 literal input。

`Form.read_elements(input, path, env)` 以一次 preview 回傳所有原始位置的 authored input、可用 canonical leaf values 與定向錯誤。名稱取自 FixedElements 宣告，普通 model list 則用 description label。未參與元素仍在結果中。Refs 保留欄位值，明確 materialization 使用支援 indexed path 的 `Form.read_module`。Tab resource 提供同名 `read_elements(path)`，不觸發 converter、capture 或 UI focus。可選的 `fields=(("drive", "gain"), ("sweep", "start"))` 指定元素相對 logical paths。Owner 在同一份 input/env 中回傳每個元素的 `fields` observations，順序與請求相同，並附 FixedElements identity。Refs 仍讀固定 captured Spec，不 materialize。Unknown path 直接拒絕，invalid leaf 保留 value_status 與定向 errors。

### Native model lists

`Form` 遞迴生成 `list[ConfigBase]` 的元素 model。起始樹的 list 按位置提供初值，scalar list 也可直接提供初值。省略 list tree 時，生成器把 native defaults 中的 module 與 sweep 投影成可編輯輸入。`to_cfg` 遞迴還原 native model list，不改 participants 順序。

所有位置共用元素宣告，各位置可提供不同的唯讀初值。位置 tree 直接放 `Hidden` 仍在建立 Form 時拒絕。
位置的 `Template.set`／`Slot.set` 可用 `Hidden(value)`。所有位置必須宣告相同的 Hidden Spec 路徑，
`bind_templates` 才把隱藏政策套到共享元素 model；路徑不同時，錯誤列出 list 路徑與不一致的 indices。
共享 model 只取得隱藏路徑，不取得第一個位置的值、來源、bindings 或其他 defaults。
各位置仍從自己的 Template／Slot 建 input，Hidden literal 也各自保留。Native 欄位用 `FieldMeta(readonly=True)` 宣告 Form core 的唯讀限制，用 `FieldMeta(element_label="name")` 提供 picker label，不需 import GUI metadata。

```python
from typing import Annotated
from zcu_tools.cfg_model import ConfigBase, FieldMeta, PulseCfg, Range, Slot
from zcu_tools.gui.form_adapter import FixedElements, Form
from zcu_tools.program.v2.sweep import SweepCfg

class Tone(ConfigBase):
    name: Annotated[str, FieldMeta(readonly=True)]
    drive: PulseCfg
    sweep: SweepCfg

class Group(ConfigBase):
    tones: Annotated[list[Tone], FieldMeta(element_label="name")]
    participants: list[str]

form = Form(Group, {
    "tones": [{"name": "Q1", "drive": Slot("Q1.pulses.drive"),
               "sweep": Range.between(4000, 4100, expts=3)}],
    "participants": ["Q1"],
}, fixed_elements=(FixedElements(("tones",), "name", ("participants",)),))
# form.default_input(env) uses the caller's fixed capture containing the slot.
```

## Sweep 表示

SweepCfg 生成 Range／Center × Expts／Step 四個 tagged branches，各自只保存 active 座標與 n 或 step。Fresh input 選 Range + Expts。Range 的 None 端點與求值失敗的 Expr 初始端點保持未填且必填，缺少 input key 也不補 0／1。`gui.form_adapter.sweep` 擁有四種數值 model 與換算。指令掛在 sweep row，以共用 Form 的 readonly snapshot、局部 evaluate 與原子安裝執行，不由葉欄位修改 parent。Step 的有效端點與 floor count 由這些 model 的驗證產生，`to_cfg` 使用這份結果與原 step，不另算步長或截短。既有 SweepCfg 仍驗證單點與點數一致性。Memento 保存 FormInput，因此模式與新 expression 不依賴 Qt view，read／preview／result 不暗改原稿。

完整的 sweep container logical edit 可以一次安裝 Range + Expts，不必讓舊 Center draft 可換算。Recipe 的完整 set_sweep 使用這條路徑；部分編輯仍只接受目前 active paths。

Step 驗證保留 start 與 step，以 floor count 投影實際 stop／center／span。接近整數的商使用 1e-9 相對容差，不普遍 round。步長大於非零區間、零步長或方向錯誤保留錯誤草稿，不能縮成單點。Input 保存原稿，preview／result 提供有效端點，不回寫 input。不提供 explicit-list 分支。

座標 command 不要求 n／step 有效，採樣 command 要求座標與目前採樣值有效。換算只在成功時將參與 expression 改為 direct 值；未參與 input 保留原文與缺席狀態。缺值、expression 或換算失敗不安裝部分結果。手寫表單可使用同一個 `SweepInput`；annotation 只接受部分分支時，description 與 execution 不提供其他分支的 command。

## Log／List 整數清單

`Form(Cfg, {"depths": LogPoints(max=128, points=9)})` 為 `list[int]` 欄位選用 Log／List 輸入。`LogPoints` 從 `cfg_model` 匯入。未標記的 list 與 SweepCfg 不變。

Log 產生 0 加 `geomspace(1, max, points-1)`，末點固定為 max，再四捨五入、排序並去重。points 包含 0，至少為 2。兩點產生 [0, max]。去重後 preview 的 points 是實際點數，原稿保留請求值，Editor 顯示 Effective 值。max=0 只有 [0]，請求點數仍至少為 2。單元素 List 切到 Log 時，回填的 points=1 保持為 invalid draft，不自動增加點數。

`LogPoints(..., include_zero=False)` 選用正值政策。Log 的 max 至少為 1；max>1 時 points 至少為 2，max=1 時 points 至少為 1。它產生 `geomspace(1, max, points)` 並四捨五入去重，首尾固定為 1 與 max；max=1 得到 `[1]`。List 同樣只接受正整數，保留輸入順序與單點。`PositiveLogIntegerPoints(max=..., points=...).values` 提供相同產生政策給 recipe 等前端 caller，不各自重算。

Log→List 帶入實際清單，List→Log 回填最大值與清單長度。切換經 `FormState.select(path, {"kind": "list"}, env)` 或 `{"kind": "log"}`，使用共用 converter 與原子安裝；無效來源或 expression 失敗不改草稿。List 的 `values` 是非空非負整數清單，不重新取樣，額外實驗限制由 core 驗證。`to_cfg` 只輸出整數清單，不保存輸入模式。

## Logical 操作

`KeyRef`、`ModuleRef`、`ModuleInline` 與 `ModuleInput` 從本 package import，不再由 form core export。`KeyRef.name` 以通用 `InputPolicy` 宣告 literal input。它不要求 captured key 存在，持久 Spec 保存 child key 的政策不變。`ModuleRef.check_sources` 承接 core module 的 captured source admission。

生成器在 core cfg、Spec 與手寫 `ModuleInputField.annotation` 附上 `ModuleSelector` metadata。它記錄 wrapper 名稱、inline/ref families 與 module/waveform namespace，沒有 env 或 mutable state。Logical paths 只略過這些生成 wrapper，真實名為 `module` 的欄位保留。Canonical Form/widget paths 與持久 JSON 不變。

`Form.read_logical_field(input, path, env)` 回 `LogicalFieldObservation`。Description、children 與 errors 使用 logical paths。Authored input、普通值、available/missing/unavailable、selector mode/key、最近 captured source 與 `readonly` 各自有欄位。`LogicalFieldObservation.readonly` 表示穿過 ref 祖先，不表示宣告唯讀，也不授予寫入權限。穿 ref 只讀 captured Spec，以 core 的 field-local preview 求值，不轉 native cfg、不載 asset samples，也不寫 input/base/library。目標 ref 缺來源時仍可觀察選擇與 unavailable errors。穿 ref 祖先讀 descendants 或執行 copy 缺來源時，raise `LogicalSourceError`，不補 defaults。

`prepare_logical_edits` 先按呼叫開始時的 active structure 解析整批 paths，再經 core prepare_write 一次產生候選。重複、重疊、不可達路徑與 ref 祖先整批拒絕，`LogicalEditError` 保留原 edit index。普通 model-invalid 值仍是可安裝的 draft。Generated selector 本身只接受專用的 `prepare_library_selection`、`prepare_inline_copy` 與 `prepare_module_disable`。Library selection 檢查 actual key/family 與宣告限制；copy 保留 child keys；disable 只允許 nullable selector。Inline copy 的 no-op 回 changed=False。Read 不會讓 ref descendants 變成可寫。

候選保留範本的 input projection／model、requiredness 與唯讀宣告，不以單值 Literal 實作 lock。Adapter 不取得 host 控制權，也不 install 或發布 state。Host 在同一 guarded turn 以目前 input/env prepare，再 install。`logical_errors` 只翻譯已有 directed errors，保留真實同名欄位與 container indexes，不求值或新增診斷。

## 依賴方向

本 package 依賴 `cfg_model`、`program`、`resources`、`gui.form` 與 `gui.session.parameters`。它不 import `gui.app`、widgets、remote、experiment、mcp、Qt 或 `zcu_lab`；`.importlinter` 的 `form-adapter-no-host` contract 與 C16 執行這條邊界。

`gui.form` 是通用 Pydantic 輸入核心，不認得 core cfg 推導。推導與 module materialization 放在本 package，不放回 `gui.form`。

設計背景見 [ADR-0073 measure 表單推導](../../../../docs/adr/0073-measure-form-derivation.md) 與 [`gui/form` README](../form/README.md)。

Readout 的跨 owner 契約見 [ADR-0074](../../../../docs/adr/0074-flat-readout-module.md)。

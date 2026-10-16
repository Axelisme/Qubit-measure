# gui/form_adapter — 從 core cfg 生成 Form

**Last updated:** 2026-10-16，Logical 路徑、純候選與 captured Spec 唯讀投影

本 package 是 GUI 的 Qt-free 前端框架。它把 core cfg 宣告與作者寫的起始樹翻譯成 `gui.form` 的 `FormModel`，再把表單結果轉回驗證過的 core cfg。它供 measure、workflows、library 與 writeback 的表單定義共用。Measure 的註冊、protocol 與分頁 Form resource 使用 `Form`，workflows 的手寫 plan 表單使用 `ModuleInputField`。

主要生成入口如下：

- `Form(core_cfg, tree, *, extras=None, lower=None)`：生成 FormModel 子類別、defaults hook、choices、options 與 `to_cfg`。Form 的 cfg 結果型別具協變性，具體 `Form[SampleCfg]` 可交給 generic-free host，`to_cfg` 仍回傳 SampleCfg 或 FormErrors。Core annotations 決定形狀與驗證；樹決定起始值、鎖定、隱藏欄位與 module 預設。Expr 初值欄位在生成的 model 中不帶靜態預設，算不出時保持未填。Range 的 Expr start／stop 也沿用這個規則，不回落 LinRange 的 0／1。
- `module_form(spec_type)`：為一個具體持久 module 或 waveform Spec class 建 Form。生成的 model 可當 FormState root，也可當 `ModuleInline` 的 module 欄位；host 的選項沿用自己的 `choices(env)`。`to_cfg` 回傳同一個 Spec，child library key 保持字串，不讀 library、asset，也不轉 native cfg。`Form.input_from_spec(spec, env)` 把既有 Spec（library entry 或 writeback proposal）投影成 input；它只接受這類 Form 的同型 Spec。
- `FormContext(parameters)`：實驗 `make_form(ctx)` 使用的固定來源查詢。parameters 是捕捉的 ParameterContext，缺來源時為 None。`has(ref)` 接受 library root 或 point module slot，含 `$alias` 前綴。缺 key、未設定或懸空的 slot、缺或未綁 alias 回 False；語法錯誤、其他 alias 解析錯誤與來源故障仍 raise。它不 materialize module、不讀 asset、不 refresh。
- `ModuleValues`：`Form.read_module` 回傳的單一 module 欄位實際值、模式與來源。
- `ModuleInputField(family, *, nullable=False, ref_family=None)`：把具體持久 ModuleSpec class 或其 union 生成為手寫 FormModel 可用的 tagged 欄位。`annotation` 是 runtime Pydantic 型別，`options(env)` 是同一欄位的選項與複製指令。`family` 決定 inline 家族，`ref_family` 只限制 captured ref keys，省略時沿用 family。`nullable` 單獨決定 None，不在 family 再放 None。空、不支援或超出 inline family 的 ref 宣告在建構時拒絕。`to_spec(value)` 接收該欄位的成功 FormState.result 值，ref 回 literal key，inline 回 detached、驗證過的持久 Spec，nullable None 保 None。它保留 child keys、不讀來源或 materialize，屏蔽欄位由 Spec defaults 補回。其他欄位的 generated 值或不合法 Spec 明確拒絕。

手寫表單在自己的 `options(env)` hook 將 `ModuleInputField.options(env)` 放到欄位路徑；既有最深 local choices 規則照舊。Python 型別檢查器不能反射 runtime `annotation`，靜態別名可用 `ModuleRef | ModuleInline` 作 nominal value view，nullable 時另含 None。這個 view 只供保存結果與傳給 `to_spec`，不表示 generated `.module` 已是持久 Spec。Workflows 三個 plan 表單共用這個入口，先取持久形式再交既有 core/native factories；校準 policy 仍屬 workflows。

`AliasName` 的 devices 選項包含 core 欄位的靜態字串預設，空字串顯示為 None。Description 與驗證共用同一份 choices。缺 alias 時 input 保持缺席，有 core 預設就用該預設，無預設的 required 欄位仍 Invalid。綁定 alias 照舊帶入裝置名，其他未列出的值仍被拒絕。

生成器擁有兩條翻譯政策，host 與 `gui.form` 不重做：

- `FieldMeta` 的 label、unit、decimals、description、group 轉成 `Display`，core 欄位與 module 欄位共用同一個翻譯。被鎖定的欄位是單值 Literal，不帶 decimals。Sweep 的 decimals 只傳到生成 LinRange 的 start／stop float 欄位，不套外層 model 或 n。
- 生成 module 欄位時，選填的 `desc` 不在 model、input、description 與寫入路徑。`to_cfg` 驗證成 Spec 時由 Spec 預設補回。從 library 複製成 inline 或 `input_from_spec` 時，原 description 被丟棄，所以編輯後會變回預設。Readout 的必填通道照常可編輯；Pulse cfg 不保存 `ro_ch`、`mux_*`、`mask` 或 `outsel`。範本 seed 屏蔽欄位時，註冊入口拒絕，訊息寫出 template id 與 seed 路徑；`to_cfg` 只由 Spec 預設補回。其他欄位的字串 seed 照舊是字面值。目前沒有露出覆寫機制。

生成的 `options(env)` 宣告可留空欄位的 None、依持久 Spec family 篩選的 captured library keys，以及每個可編輯 Inline kind。只有結構 tag 的 kind 不列為 Inline。所有 key input 使用 `{"kind": "ref", "name": key}`，inline 使用 `{"kind": "inline", "module": content}`，可留空時也接受 null。Options 的 value 只帶 key 或所選 kind 的 tag。fields 指向該 kind 的 wrapper 子表單，describe 所選欄位才列 children，不重複固定 tag。Ref 選項不帶 fields，也不顯示 entry 內容。普通換分支使用 `defaults(env)` 的 fresh 範本／型別初值，不接收 previous、不複製來源，也不保留其他 kind 的草稿。

每個 key 選項帶 `to_inline` 指令，label 是 Convert to inline。`FormState.run_command` 用固定 FormEnv 的 library 複製持久 Spec，保留 child key、丟棄屏蔽欄位，以一次 write 安裝快照。來源失敗不改 input。明確複製使用 `to_inline` 指令或 `Form.prepare_inline_copy`，普通 defaults 只建 fresh 範本／型別初值，不提供隱含複製或缺來源 fallback。

持久 Spec、範本 seed 與 lock 路徑仍使用原欄位名稱，不含 wrapper。生成器在 input 邊界加入 wrapper，`to_cfg` 去掉 wrapper，保留 child key 字串。Nested ref 使用本 package 的 `KeyRef`，不限制 choices，也不新增 captured source 檢查。Core module ref 仍使用 `ModuleRef` 並檢查固定 library。兩種 ref 的 name 都只接受 literal key，不求 Expr。

`readout/pulse` 直接編輯平鋪的 generator 與 ADC 欄位。`gen_ch` 與 `ro_ch` 必填且可編輯；一般 Pulse 沒有 `ro_ch`。單一 `freq` 同時供兩者使用，不從其他欄位推導通道或補值。Waveform 仍可保留 child library key。

`readout/mux` 沿用 module 的 ref／inline 與 Convert to inline。Scalar 欄位分別編輯，tones 是整個 list 的 JSON leaf，包含每筆 tone 的 label。它沒有 waveform 或逐 tone 引用。

Template／Slot 節點在 host 呼叫 `Form.bind_templates(catalog, adapter=...)` 後才解析。Measure 在 Open／Reset 捕捉來源，呼叫 adapter 的 `make_form(ctx)`，再綁定 Registry 的範本目錄。註冊與 reload 不呼叫 factory，也不預檢範本。範本目錄的格式在 `cfg_model`，真實範本表在 `zcu_lab`。

Spec Form 的 `template_input(Template(id), catalog, env)` 直接建立範本的 fresh input，不需先 bind。它沿用 Expr 初值與 literal lock 規則。缺 Slot 來源時，required 欄位留空並標 Invalid；alternate 來源由 make_form(ctx) 決定。已綁定 Slot 保留 child library key，`to_cfg` 回持久 Spec，不展開 native cfg。Core cfg Form 拒絕這個入口。

回傳的 `input.model` 可以是 `Form.model` 的專用 subclass，讓不同範本的 required 欄位與 lock 互不影響。用回傳 input 建新 `FormState`。同一範本重算初值用 `input.model.default_input(env)`，換範本則建新 state。原 Form.model、範本目錄與 captured sources 不變。

Module、seed、Display 與 annotation 翻譯放在私有模組 `_modules.py`、`_seeds.py`、`_display.py`、`_types.py`，由 `form.py` 與公開欄位入口 `module_input.py` 組合。

## Logical 操作

`KeyRef`、`ModuleRef`、`ModuleInline` 與 `ModuleInput` 從本 package import，不再由 form core export。`KeyRef.name` 以通用 `InputPolicy` 宣告 literal input。它不要求 captured key 存在，持久 Spec 保存 child key 的政策不變。`ModuleRef.check_sources` 承接 core module 的 captured source admission。

生成器在 core cfg、Spec 與手寫 `ModuleInputField.annotation` 附上 `ModuleSelector` metadata。它記錄 wrapper 名稱、inline/ref families 與 module/waveform namespace，沒有 env 或 mutable state。Logical paths 只略過這些生成 wrapper，真實名為 `module` 的欄位保留。Canonical Form/widget paths 與持久 JSON 不變。

`Form.read_logical_field(input, path, env)` 回 `LogicalFieldObservation`。Description、children 與 errors 使用 logical paths。Authored input、普通值、available/missing/unavailable、selector mode/key、最近 captured source 與 readonly 各自有欄位。穿 ref 只讀 captured Spec，以 core 的 field-local preview 求值，不轉 native cfg、不載 asset samples，也不寫 input/base/library。目標 ref 缺來源時仍可觀察選擇與 unavailable errors。穿 ref 祖先讀 descendants 或執行 copy 缺來源時，raise `LogicalSourceError`，不補 defaults。

`prepare_logical_edits` 先按呼叫開始時的 active structure 解析整批 paths，再經 core prepare_write 一次產生候選。重複、重疊、不可達路徑與 ref 祖先整批拒絕，`LogicalEditError` 保留原 edit index。普通 model-invalid 值仍是可安裝的 draft。Generated selector 本身只接受專用的 `prepare_library_selection`、`prepare_inline_copy` 與 `prepare_module_disable`。Library selection 檢查 actual key/family 與宣告限制；copy 保留 child keys；disable 只允許 nullable selector。Inline copy 的 no-op 回 changed=False。Read 不會讓 ref descendants 變成可寫。

候選保留 template specialization 的 model、locks 與 requiredness。Adapter 不取得 host 控制權，也不 install 或發布 state。Host 在同一 guarded turn 以目前 input/env prepare，再 install。`logical_errors` 只翻譯已有 directed errors，保留真實同名欄位與 container indexes，不求值或新增診斷。

## 依賴方向

本 package 依賴 `cfg_model`、`program`、`resources`、`gui.form` 與 `gui.session.parameters`。它不 import `gui.app`、widgets、remote、experiment、mcp、Qt 或 `zcu_lab`；`.importlinter` 的 `form-adapter-no-host` contract 與 C16 執行這條邊界。

`gui.form` 是通用 Pydantic 輸入核心，不認得 core cfg 推導。推導與 module materialization 放在本 package，不放回 `gui.form`。

設計背景見 [ADR-0073 measure 表單推導](../../../../docs/adr/0073-measure-form-derivation.md) 與 [`gui/form` README](../form/README.md)。

Readout 的跨 owner 契約見 [ADR-0074](../../../../docs/adr/0074-flat-readout-module.md)。

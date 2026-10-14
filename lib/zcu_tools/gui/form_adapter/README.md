# gui/form_adapter — 從 core cfg 生成 Form

**Last updated:** 2026-10-14，captured factory 與使用時範本綁定

本 package 是 GUI 的 Qt-free 前端框架。它把 core cfg 宣告與作者寫的起始樹翻譯成 `gui.form` 的 `FormModel`，再把表單結果轉回驗證過的 core cfg。它供 measure、workflows、library 與 writeback 的表單定義共用。Measure 的註冊、protocol 與分頁 Form resource 使用 `Form`，workflows 的手寫 plan 表單使用 `ModuleInputField`。

公開 root 有五個名字：

- `Form(core_cfg, tree, *, extras=None, lower=None)`：生成 FormModel 子類別、defaults hook、choices、options 與 `to_cfg`。Form 的 cfg 結果型別具協變性，具體 `Form[SampleCfg]` 可交給 generic-free host，`to_cfg` 仍回傳 SampleCfg 或 FormErrors。Core annotations 決定形狀與驗證；樹決定起始值、鎖定、隱藏欄位與 module 預設。Expr 初值欄位在生成的 model 中不帶靜態預設，算不出時保持未填。Range 的 Expr start／stop 也沿用這個規則，不回落 LinRange 的 0／1。
- `module_form(spec_type)`：為一個具體持久 module 或 waveform Spec class 建 Form。生成的 model 可當 FormState root，也可當 `ModuleInline` 的 module 欄位；host 的選項沿用自己的 `choices(env)`。`to_cfg` 回傳同一個 Spec，child library key 保持字串，不讀 library、asset，也不轉 native cfg。`Form.input_from_spec(spec, env)` 把既有 Spec（library entry 或 writeback proposal）投影成 input；它只接受這類 Form 的同型 Spec。
- `FormContext(parameters)`：實驗 `make_form(ctx)` 使用的固定來源查詢。parameters 是捕捉的 ParameterContext，缺來源時為 None。`has(ref)` 接受 library root 或 point module slot，含 `$alias` 前綴。缺 key、未設定或懸空的 slot、缺或未綁 alias 回 False；語法錯誤、其他 alias 解析錯誤與來源故障仍 raise。它不 materialize module、不讀 asset、不 refresh。
- `ModuleValues`：`Form.read_module` 回傳的單一 module 欄位實際值、模式與來源。
- `ModuleInputField(family, *, nullable=False, ref_family=None)`：把具體持久 ModuleSpec class 或其 union 生成為手寫 FormModel 可用的 tagged 欄位。`annotation` 是 runtime Pydantic 型別，`options(env)` 是同一欄位的選項與複製指令。`family` 決定 inline 家族，`ref_family` 只限制 captured ref keys，省略時沿用 family。`nullable` 單獨決定 None，不在 family 再放 None。空、不支援或超出 inline family 的 ref 宣告在建構時拒絕。`to_spec(value)` 接收該欄位的成功 FormState.result 值，ref 回 literal key，inline 回 detached、驗證過的持久 Spec，nullable None 保 None。它保留 child keys、不讀來源或 materialize，屏蔽欄位由 Spec defaults 補回。其他欄位的 generated 值或不合法 Spec 明確拒絕。

手寫表單在自己的 `options(env)` hook 將 `ModuleInputField.options(env)` 放到欄位路徑；既有最深 local choices 規則照舊。Python 型別檢查器不能反射 runtime `annotation`，靜態別名可用 `ModuleRef | ModuleInline` 作 nominal value view，nullable 時另含 None。這個 view 只供保存結果與傳給 `to_spec`，不表示 generated `.module` 已是持久 Spec。Workflows 三個 plan 表單共用這個入口，先取持久形式再交既有 core/native factories；校準 policy 仍屬 workflows。

`AliasName` 的 devices 選項包含 core 欄位的靜態字串預設，空字串顯示為 None。Schema 與驗證共用同一份 choices。缺 alias 時 input 保持缺席，有 core 預設就用該預設，無預設的 required 欄位仍 Invalid。綁定 alias 照舊帶入裝置名，其他未列出的值仍被拒絕。

生成器擁有兩條翻譯政策，host 與 `gui.form` 不重做：

- `FieldMeta` 的 label、unit、decimals、description、group 轉成 `Display`，core 欄位與 module 欄位共用同一個翻譯。被鎖定的欄位是單值 Literal，不帶 decimals。Sweep 的 decimals 只傳到生成 LinRange 的 start／stop float 欄位，不套外層 model 或 n。
- 生成 module 欄位時，選填的 `desc`、`mask`、`outsel`、`ro_ch` 與 `mux_*` 不在 model、input、schema 與寫入路徑。`to_cfg` 驗證成 Spec 時由 Spec 預設補回。從 library 複製成 inline 或 `input_from_spec` 時，這些欄位的原值被丟棄，所以編輯後會變回預設。必填且沒有預設的欄位，例如 `DirectReadoutCfg.ro_ch`，照常可編輯。範本 seed 這些欄位時，註冊入口拒絕，訊息寫出 template id 與 seed 路徑；`to_cfg` 只由 Spec 預設補回。其他欄位的字串 seed 照舊是字面值。目前沒有露出覆寫機制。

生成的 `options(env)` 宣告可留空欄位的 None、依持久 Spec family 篩選的 captured library keys，以及每個可編輯 Inline kind。只有結構 tag 的 kind 不列為 Inline。所有 key input 使用 `{"kind": "ref", "name": key}`，inline 使用 `{"kind": "inline", "module": content}`，可留空時也接受 null。Options 的 value 只帶 key 或所選 kind 的 tag。fields 指向該 kind 的 wrapper 子表單，schema 不重複固定 tag。Ref 選項不帶 fields，也不顯示 entry 內容。普通換分支使用 `defaults(env)` 的 fresh 範本／型別初值，不接收 previous、不複製來源，也不保留其他 kind 的草稿。

每個 key 選項帶 `to_inline` 指令，label 是 Convert to inline。`FormState.run_command` 用固定 FormEnv 的 library 複製持久 Spec，保留 child key、丟棄屏蔽欄位，以一次 write 安裝快照。來源失敗不改 input。`to_inline` 是唯一公開複製入口，普通 defaults 只建 fresh 範本／型別初值，不提供隱含複製或缺來源 fallback。

持久 Spec、範本 seed 與 lock 路徑仍使用原欄位名稱，不含 wrapper。生成器在 input 邊界加入 wrapper，`to_cfg` 去掉 wrapper，保留 child key 字串。Nested ref 使用 core 的 `KeyRef`，不限制 choices，也不新增 captured source 檢查。Core module ref 仍使用 `ModuleRef` 並檢查固定 library。兩種 ref 的 name 都只接受 literal key，不求 Expr。

Readout 的 ADC 推導由 `PulseReadoutSpec.derive_input_fields` 擁有。生成器將 input 投影回持久形狀後呼叫它，保留其他 invalid leaves。既有 Spec validator 使用同一份推導。

Template／Slot 節點在 host 呼叫 `Form.bind_templates(catalog, adapter=...)` 後才解析。Measure 在 Open／Reset 捕捉來源，呼叫 adapter 的 `make_form(ctx)`，再綁定 Registry 的範本目錄。註冊與 reload 不呼叫 factory，也不預檢範本。範本目錄的格式在 `cfg_model`，真實範本表在 `zcu_lab`。

Spec Form 的 `template_input(Template(id), catalog, env)` 直接建立範本的 fresh input，不需先 bind。它沿用 Expr 初值與 literal lock 規則。缺 Slot 來源時，required 欄位留空並標 Invalid；alternate 來源由 make_form(ctx) 決定。已綁定 Slot 保留 child library key，`to_cfg` 回持久 Spec，不展開 native cfg。Core cfg Form 拒絕這個入口。

回傳的 `input.model` 可以是 `Form.model` 的專用 subclass，讓不同範本的 required 欄位與 lock 互不影響。用回傳 input 建新 `FormState`。同一範本重算初值用 `input.model.default_input(env)`，換範本則建新 state。原 Form.model、範本目錄與 captured sources 不變。

Module、seed、Display 與 annotation 翻譯放在私有模組 `_modules.py`、`_seeds.py`、`_display.py`、`_types.py`，由 `form.py` 與公開欄位入口 `module_input.py` 組合。

## 依賴方向

本 package 依賴 `cfg_model`、`program`、`resources`、`gui.form` 與 `gui.session.parameters`。它不 import `gui.app`、widgets、remote、experiment、mcp、Qt 或 `zcu_lab`；`.importlinter` 的 `form-adapter-no-host` contract 與 C16 執行這條邊界。

`gui.form` 是通用 Pydantic 輸入核心，不認得 core cfg 推導。推導與 module materialization 放在本 package，不放回 `gui.form`。

設計背景見 [measure 表單推導 ADR 草稿](../../../../docs/adr/draft/measure-form-derivation.md) 與 [`gui/form` README](../form/README.md)。

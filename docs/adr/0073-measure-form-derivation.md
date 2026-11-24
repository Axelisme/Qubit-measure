---
status: accepted
---

# 0073 — Measure 表單從 core cfg 推導

本篇接替 [[0012]] 的 measure definition／seed 契約與 [[0009]] 的角色 default 表。Library、writeback 與 Inspect 也使用共用 Form，持久 Spec 與提交邊界見 [[0065]]、[[0008]]。實驗核心與前端分工見 [[0069]]。

## 局部取代範圍

已由 [[0079]] 取代，見[〈決策狀態與問題〉](0079-form-field-transitions.md#決策狀態與問題)；[原文](retired/measure-form-derivation-replacement-scope.md#局部取代範圍)僅供追溯。

## 問題

舊 builder 讓 adapter 重抄 core Pydantic cfg 的形狀，兩份宣告容易不一致。初值藏在 Seed 閉包與本地 blank 工廠，欄位說明、鎖定與初值又分散在不同檔案。作者得追多層函式，才知道表單的形狀。

使用者 2026-10-12：「目前生成的辦法不是那麼的直觀，也不太能從實驗模塊看出來生成的表單會長什麼樣子」。見 [Measure 表單重設計](../discussions/2026-10-13-measure-form-redesign.md)。

## 決策

### Adapter 是黑箱

**規則。** Adapter 的 `make_form(ctx)` 回傳 `Form[T_Input]`。框架把驗證過的輸出原樣交給 `run(inp, *, context)`，不檢查具體型別、不重新驗證，也不認得 adapter 內部的 Experiment。每個 adapter 自己宣告 `run`，把輸入轉成實際實驗 cfg、呼叫實驗，再回傳該 cfg 與結果。Form 不提供 `extras` 或 `lower`。

**例子。** `T_Input` 預設等於實驗 cfg 型別。沒有轉接需求時，adapter 仍用本地 `run` 呼叫實驗並回傳 `RunRecord(cfg=actual_cfg, result=...)`，不曝露實驗的 bound run，也不使用共用 run factory。Form 輸入與實驗 cfg 不同時，轉接也只在這個 `run` 做。

**邊界。** Adapter 的七個泛型槽連起實驗 cfg、結果、兩層分析 options／result 與 `T_Input`，只有 Registry 抹除異質型別。Host 可附加 acquisition metadata，不再次包裝 record，也不以 Form input 取代作者回傳的 cfg 或 `None`。Save 只存 record 的實驗 cfg 與結果。Load 依 spec 還原 record，不做反向輸入轉換。Run 與 Load 後的分析及 writeback 都只讀 record，不用目前 Form 補歷史。分析需要的資訊由 adapter 的 `run` 放進實驗 cfg，或由實驗放進結果。相關契約見 [Form adapter README](../../lib/zcu_tools/gui/form_adapter/README.md) 與 [[0069]]。

**依據。**
- 2026-10-14，使用者：「感覺怪怪的，理論上對框架來說，adapter內是黑盒子，不該知道裡面是實驗」「兩個都同意，adaper就是一個黑盒子」。設計背景見 [Measure 表單重設計](../discussions/2026-10-13-measure-form-redesign.md)。
- 2026-10-24，使用者：「run只吃cfg?這完全是錯誤的想法，adapter就應該在run這個函數轉接cfg，不是直接曝露實驗run，不然我為什麼不直接曝露exp_cls就好？就是讓adapter能夠轉接cfg格式」「Form那個很有問題，為什麼要綁定adapter對外的Form必須是內部實驗的Cfg，這本來就該是adapter負責轉接的。」「你認為record該放實驗cfg還是表單cfg? 理論上是實驗cfg，但怎麼處理load的cfg呢」。同日選擇每個 adapter 本地宣告 `run`，並指出「方向能接受，但我覺得record_run的寫法不好懂」。原始討論未入庫，見 Git `980e0edda`。

### 一個實驗的檔案分工

**規則。** `core.py` 擁有 cfg 型別、前端中立的 `FieldMeta`、run 與 analyze。Qt-free 的 `form.py` 宣告 `make_form(ctx)`，用一般 Python 選初值與形狀。`gui.py` 接上 make_form、run、analysis、writeback 與 operator guide。

**例子。** `core.py` 用 `Annotated[..., FieldMeta(...)]` 宣告欄位的 label、unit、decimals、description、group、readonly 與 element_label。`group` 可供 MCP 的單層欄位描述使用，所以也放在 core。沒有 label 時，生成器使用欄位名。欄位清單見 [cfg_model README](../../lib/zcu_tools/cfg_model/README.md)。

**邊界。** `FieldMeta` 不改序列化格式。實際的 widget、排版與折疊由前端決定；前端特有的初值與形狀留在 `form.py`。

**依據。** 2026-10-12，使用者：「我想放core會比較簡單，而維護實驗的作者都是已一整個實驗為單位」。2026-10-13，使用者：「採用，更新草稿，form.py慣例也記進去」。比較見 [Measure 表單重設計](../discussions/2026-10-13-measure-form-redesign.md)。

### Form 的形狀與初值

**規則。** `gui.form_adapter.Form(input_type, tree)` 從 Pydantic 輸入型別推導 FormModel、起始 input 和輸出轉換。Adapter 可使用實驗 cfg 或自訂輸入型別。共用生成器處理純量、nullable、Literal／enum、巢狀 model 與 module／waveform。Tree 對齊欄位，普通字串是 literal，`Expr` 才是 expression。初值 expression 在建立 fresh input 時求值，求不出時不補數字，必要欄位保持未填。使用者寫入的 expression 保留原式。初值的 raw 表示見 [Form 與固定來源](../../zcu_lab/v2/measure-authoring.md#form-與固定來源)。需要條件或聚合邏輯時，adapter 提供 `Form(Cfg, tree, initial_values=factory)`。Factory 收到 `model.defaults(env)` 新建、尚未求值的 raw 初值，可原地修改並回傳，或回傳新的完整樹，不自行重算宣告初值。Form 深複製回傳值並整份採用，不與宣告初值合併。Factory 只在 Open／Reset 的固定捕捉邊界算初值，不由 widget、編輯、preview 或 Run 重算。

**例子。** 單 qubit 的頻譜表單用 lab helper `source(ctx, "$resonator.module.readout", Template("readout"))` 選 readout 初值。`$resonator` 由 `resonator: $qubit.resonator` alias 解析，readout 與 slot 的 resonator 所有權見 [[0072]]。可解析時 helper 回傳 `Slot(path)`；缺席、懸空或 alias 未綁時才用傳入的 Template，不重寫路徑。頻率掃描用 `Range.around("$qubit.freq", "1.5 * $qubit.kappa", expts=301)`，`reps` 初值為 1000。其他作者入口及例子見 [Form adapter README](../../lib/zcu_tools/gui/form_adapter/README.md)：
- `Range` 的字串端點轉成 `Expr`，數字與 `None` 不變。[Sweep 表示](../../lib/zcu_tools/gui/form_adapter/README.md#sweep-表示)列出四個 tagged branches。Range／Center 與 Expts／Step 各只保存 active 欄位，分別以 `n` 或 `step` 保存採樣，新表單用 Range + Expts。Center 的 span 是帶正負號的完整寬度，不另存端點。
- `LogPoints(max, points, include_zero=True)` 宣告一個整數清單的初值。`cfg_model` 只保存宣告，`form_adapter` 擁有分支、換算與產生清單的規則。Core 只接收 `list[int]`。LogPoints 只用於整數次數；浮點的對數間距掃描不在此列，有實驗需要時再連同 core 的清單掃描一起設計。兩種 Log 的首尾、點數限制、去重與 List→Log 的 invalid draft 行為見 [Log／List 整數清單](../../lib/zcu_tools/gui/form_adapter/README.md#loglist-整數清單)，不影響普通 list 或 SweepCfg。
- 唯讀限制作用於整棵被標記的子樹。Module 子欄位標唯讀不影響 library key 的選擇；module 本身標唯讀則固定 ref／inline 與所有子欄位。Designer 的實作選擇是不另設 selector-only 宣告或強制 inline。只為組 cfg 存在的固定值由 `run()` 補齊，例如單發的 `reps`／`rounds=1`，不放進 Form 或實驗 cfg 型別。要顯示的 mux tone name 保持唯讀。
- Two-qubit 與 heralded 表單的 [`captured_qubits`](../../zcu_lab/v2/support/measure/paired_qubits.py) 從固定 capture 解析元件名。元件名不進表單欄位或 cfg，見 [ADR-0069 的 adapter 身分契約](0069-experiment-core-and-frontends.md#綁定身分留在-adapter)。表單以 `Hidden` 將按 `$mux.qubits` 順序推得的 `qubit1_tone`、`qubit2_tone` 填入 cfg。`TwoQubitForm` 提交時從同一次 capture 取 group 的 `resonators` 長度，核對 readout 的 tone 數。Tone 數不是表單欄位。
- Run 會覆寫的必填欄位由作者宣告 `Hidden(合法值)`。普通 core tree 和 `Template.set`／`Slot.set` 的 dotted Spec 路徑共用這個節點，填值仍受驗證。相反地，Bath reset 的 `tested_reset.pi2_cfg.phase` 由 Template 的 `set` 給可編輯的初值 90。`run()` 使用填入的 phase，不覆寫成 90，guide 將 90 說明為建議的 tomography offset。逐 tone 的 Hidden 路徑相同才可共享元素 model；各位置的值與來源仍獨立。`Hidden()` 的省略語意、共用路徑檢查與普通位置 tree 的限制見 [Hidden 填入值](../../lib/zcu_tools/gui/form_adapter/README.md#hidden-填入值)及 [Native model lists](../../lib/zcu_tools/gui/form_adapter/README.md#native-model-lists)。
- Readout 使用平鋪欄位，例如 `Template("readout", set={"gain": 0.1, "ro_length": 2.0})`。`gen_ch`、`ro_ch` 必填且可編輯，generator 與 ADC 共用 `freq`，waveform 可保留 child key。生成器屏蔽不支援的硬體欄位，不推導通道、不補值，見 [[0074]] 與 [Form adapter README](../../lib/zcu_tools/gui/form_adapter/README.md)。

**邊界。**
- `Slot(path, copy=False, set={})` 指向捕捉的 library 槽位，copy 才建 inline；`Template(id, set={}, seed_metadata={})` 從注入的持久範本建 fresh inline input。框架不替 Slot 選 fallback。作者可用 lab 的 `source` helper 或明寫 if／else，不增設 init 模式。可解析但 family 不相容就報錯，不回退。未選 Template 而 Slot 缺來源時，必填欄位留空並標 Invalid，可留空且預設 `None` 的欄位停用而維持有效。Seed metadata 只合併實際採用的 Template，與 `Form.with_seed_metadata` 共用 root-relative canonical paths，同路徑以顯式 Form metadata 優先。`to_cfg` 保留持久 Spec 的 child key，不 materialize 或改寫。未知或受屏蔽的 seed、缺必要 discriminator 或不支援的選項在入口報錯。範本各自的 input projection／model、Expr requiredness、branch ancestry 與 `input.model.default_input(env)` 的邊界見 [Form adapter README](../../lib/zcu_tools/gui/form_adapter/README.md)。
- Designer 的實作選擇是 `Slot`／`Template` 不提供 `lock`，也沒有 `Locked` 節點。唯讀初值用 `Display(readonly=True)` 或 `FieldMeta(readonly=True)`，Form core 強制唯讀，不生成單值 Literal。List 可按位置給不同唯讀初值。生成器不從欄位同名或 sweep 與 module 的關係推導 Hidden，不增設 hidden_paths、nested readonly、共用型別的複本或局部 cfg 型別，也不改 Run。
- 普通選項切換由該欄位的 converter 決定；沒有 converter 只改 selection，不補 fresh defaults。Inline 繼承由 `gui.form_adapter` 集中宣告，欄位轉換與固定具名 model list 的設計見 [[0079]]。固定元素 picker 與 participants 清單分開，lab 在 Open／Reset 把捕捉到的 alias 名稱轉成原始 index，框架不接管 mux group 關係。
- 顯式 Sweep 指令保值換算，不套 fresh defaults。座標換算只求值座標，採樣換算另求值採樣欄位；成功才原子安裝 direct 數字，失敗保留整組原稿。點數或區間無效時不靠切換修成另一個掃描。Step 的 floor count、相對容差、有效端點、精度與 lowering 見 [Sweep 表示](../../lib/zcu_tools/gui/form_adapter/README.md#sweep-表示)及 [Measure tab 的 input 與使用](0065-cfg-editing.md#measure-tab-的-input-與使用)。Qt 顯示原稿與有效值的差異，MCP 讀同一份 values，都不另算，也不寫回 input。
- `to_inline` 明確複製固定 library 內容。Widget 與 MCP 用共用 description 的 Option label、value 與 commands，不猜 union 或 library 結構。直接 children 的查詢、wrapper 攤平、完整 model-field path 與首次展開才建子表單的契約見 [Measure tab 的 input 與使用](0065-cfg-editing.md#measure-tab-的-input-與使用)。Inspect、New Entry、writeback 與 workflows 共用這套描述。

**依據。**
- 2026-10-19，使用者：「我會選B，lock應該被認知為只對子樹作用。」「有一些是為了方便轉換成experiment cfg而塞入的，其實可以改成直接不曝露欄位。run再補齊。」同日 advisor 代裁，讓 LogPoints 與 Range 使用相同的樹上初值邊界，並另定正值分支。使用者 2026-11-21 追認 LogPoints 這兩項決定：問過「第三个代材只限于 integer 吗？它不能是 floating 吗？」後，選維持只限整數，浮點的對數間距掃描等有實驗需要時再設計。原始討論未入庫，見 Git `c499c06d9`、`d2911c048`、`eccae1221`。
- 2026-10-20，使用者：「應該有更簡單的做法，比如就隨便填一個合法值，或是給adapter生成時說明哪些路徑不重要。」同日補充「不只是onetone/freq有這個問題」。各實驗依自己的 Run 覆寫路徑宣告 Hidden。原始討論未入庫，見 Git `ada1f6895`、`51a3a474b`。
- 2026-10-30，使用者答「9A」；其中 Template 自有 metadata 的部分仍適用。2026-11-01，使用者以「B，如果要省下重複名字，那就一個factory helper就好」決定由 lab helper 選來源。舊 Slot fallback 的原話與規則未有入庫討論，見 Git `304ba12ef`、`a967ae052`。
- 2026-11-02，使用者：「我以為生成Form預設值的form_adapter就能處理這種if/else的複雜邏輯，這樣Form就只要呈現就好，或是搭配command或convert或即時計算」，並對 factory 接收宣告初值答「A」。同日問「Factory不能回傳新的樹嗎？他可以原地修改default並回傳吧？」，再以「採用建議」確認由 Form 整份採用回傳樹。原始討論未入庫，見 Git `a8c7eb66a`。
- 2026-11-18，使用者過目輸入型別的推導、Hidden 與切換及描述規則：「我大概看了一遍，基本上是没问题」，並說明這類規則「并不算是裁决，而算是建议」。這些是 agent 的選擇，僅供參考。原始討論未入庫，見 Git `1f4b16aa0`。

#### 表單預設引用 library，只在改了 Run 不覆寫的欄位時複製

**規則。** 表單的 `Slot` 預設保留 library 引用。只有表單改了 Run 不會覆寫的欄位，才設定 `copy=True`，並在程式旁寫下複製的理由。為了隱藏 Run 會覆寫的欄位，不複製 library 模組；可在 ref 的欄位宣告使用 `Hidden`，只影響 inline 欄位，不改引用值。需要試調時，操作員先按 **Convert to inline** 再編輯；可能要試調不是預設複製的理由。

**例子。** Chevron、Conditional phase 和 Conditional spectroscopy 引用兩條 CZ drive，Run 用掃頻座標覆寫 `freq`。CZ amplitude 和 Constructive phase 也預設引用兩條 drive。Bath reset 的 Xπ/2 初值設定 `phase=90`，Amplitude Rabi 設定 `waveform.length`，部分 mux readout 填入 tone 初值。這些值會進入 Run，所以有來源時保留副本。

**邊界。** [掃頻 drive cfg](../../zcu_lab/v2/support/paired_drive.py) 不要求兩條 library carrier 相等；固定頻率入口保留等頻檢查。既有合法的 inline cfg 仍可載入。

**依據。** 2026-11-23，使用者說：「CZ drive 必須複製，因為表單要把 freq 蓋成隱藏的 0，再由 Run 掃頻這是沒什麼意義的流程，因為不管頻率填什麼都會被覆寫，這是在這個實驗語義下非常合理的行為，類似的問題都要修正。」「連無理由的複製一起查」。2026-11-24 裁定：「form-ref-not-copy 票 02選全部ref」。見[表單預設引用 library 的討論](../discussions/2026-11-23-form-ref-not-copy.md)。

#### 掃描初值的點數

**規則。** 每個等距 Range 掃描軸的初值至少 31 點，多軸掃描也逐軸適用。以參考值為中心的掃描用奇數，讓中心值成為一個取樣點。

**例子。** Mux onetone 掃描 ±10 MHz、101 點；constructive phase 掃描 −180° 到 180°、51 點。`raman_gf` 的 detuning 和 gain 由 11×10 點改為 31×31 點。受影響的雙軸掃描預設格點數約為原來的 2 到 9 倍。每點採集設定不變時，每輪逐點採集的時間按格點數增加。模擬速度按每格點比較，見 [ADR-0078〈模擬器速度目標〉](0078-evolution-module-boundary.md#模擬器速度目標)。

**邊界。** 只管初值。Form 不因操作者填偶數或少於 31 點而拒絕輸入，也不重寫已存 cfg 或開著的 tab。單邊 ramp 沒有要採樣的中心，奇數只是一種方便的選法，31 點下限仍適用。CPMG pulse count、RB 的 `LogPoints` depth，以及雙 qubit 普通與 heralded RB 的 depth／seed 清單是 protocol 決定的離散序列，不在此規則內。`Range(0, 0, expts=1)` 的待填佔位維持不變。

**依據。** 2026-11-23，使用者：「可以加一個小慣例，掃描點數建議是奇數，這樣可以包含中心點」。看到 10 點的 ramp 初值時說「10點太少了，建議至少31點」，確認多軸也逐軸適用時答「都對」，對排除 protocol 的離散序列答「A」。見[掃描初值至少 31 點、用奇數](../discussions/2026-11-23-scan-defaults-min-31-odd.md)。

### 捕捉來源與使用時驗證

**規則。** `TabFormResource` 擁有來源捕捉及 Form 狀態。Open／Reset 捕捉來源後呼叫 `make_form`，再綁定並驗證 Registry 的範本目錄。`FormContext` 只提供 `ctx.has(ref)` 查捕捉到的 library／alias，缺來源、缺 key 或 alias 未綁回 False，來源故障原樣拋出。`make_form` 回傳 `Form(Cfg, tree)`，不把 `ctx` 傳進 Form。Editor 只讀借用的 state，`write`、`run_command` 交回資源。Mutation guard、通知與撤銷由資源負責。

**例子。** 從歷史檔開 tab 時，只有 `open_tab_from_file` 隔離 `adapter.make_form` 失敗：tab 照常 Load，表單區顯示原始錯誤，Run 停用並顯示同一原因。Reset 再捕捉來源及呼叫 factory，成功才裝入正常 Form。原本無 Form 時，若 Reset 的 capture、make_form、範本綁定或 defaults 失敗，表單與 Run 更新為本次的錯誤，Reset 仍原樣拋出。Load 的清理邊界見 [[0067]]。

**邊界。** Register 不執行 factory 或預檢範本。Edit、preview、result、Run 不重建 Form 或刷新來源。Refresh 保留 input 與 seed metadata 快照，不重算初值；Reset 重抓來源並重算初值與 seed metadata，可以同時換形狀與初值，失敗保留既有 Form。歷史 Load 不回填 Form，無 Form 時也不造空表單或補來源。範本綁定、defaults、Load 及一般 new-tab 的錯誤仍傳出。Memento 保存 input 與 selection，按 [[0079]] 的還原設計重建，不跑 converter 或重算唯讀欄位。舊版格式不雙讀，也不以 native cfg 反填。Subscriber 故障只記 log，不回滾已提交 input。Refresh、Reset 與 wire 細節見 [Seed diagnostics](../../lib/zcu_tools/gui/form_adapter/README.md#seed-diagnostics)及 [[0065]]。

**依據。** 2026-10-24，使用者：「只在 open_tab_from_file 路徑隔離 make_form 失敗，tab 照常建立並 Load，表單區與 Run 顯示 factory 原文錯誤，Reset 走一般 fail-fast；一般 new-tab／Reset 不變」。同日補充無 Form 的 Reset 失敗也要更新診斷，但仍按失敗傳出。原始討論未入庫，見 Git `6db415d26` 與 [[0067]]。

### 持久範本目錄

**規則。** `cfg_model.TemplateDef` 宣告持久 Spec 型別、label 與 seeds。`zcu_lab.templates` 持有固定目錄，由組合根以 `Registry(templates=TEMPLATES)` 注入。Library 新建選單與 reload candidate 讀同一目錄，reload 不重建角色表。Seeds 疊在 Spec defaults 上，trig_offset 政策由 lab 的範本工具參數宣告。

**例子。** 呼叫端用 `Template.bindings` 明填 `qubit → $qubit` 或 `qubit → $qubit1`。模板展開只使用這份替換，不解析 alias。建構先驗宣告語法與結構，展開時缺 binding 才報 `ValueError`。詳見 [cfg_model README](../../lib/zcu_tools/cfg_model/README.md)。

**邊界。** Session、startup 與 FormEnv 不提供模板綁定預設值。`bath_reset.pi2_cfg` 用 x90 Slot，有來源就保留引用，缺來源保持 Invalid，不改回 pi2_pulse 或數字初值。沒有函式型範本；第二個需要依 ctx 選來源的範本出現時再討論。Lab Form factory 需要讀範本 seeds 時，由 `zcu_lab/definitions.py::register_all` 從注入的 `registry.templates` 綁定，不 import 全域 `TEMPLATES`；第三個這種 factory 出現時，改評由 `FormContext` 提供 catalog。

**依據。** 2026-10-14，使用者：「A是理想，但目前沒有用的話就C」。2026-10-20，使用者：「你搞錯了吧，模板/alias是不同的東西，後者才是startup填的」「模板是form adapter提供的工具的語法，讓呼叫端可以填需要的名子，比如$qubit或$qubit1，來展開成使用不同alias的表達式」。原始討論未入庫，目錄的設計背景見 [Measure 表單重設計](../discussions/2026-10-13-measure-form-redesign.md)。2026-11-21，factory 的 catalog 綁定是 designer 代裁，使用者未覆核；依據是使用者同日「一般实验用简单的做法就好了，有需求再做多余的设计」。見 [readout 時序預設](../discussions/2026-11-21-readout-timing-defaults.md)。

### Core 使用具名裝置欄位

**規則。** 實驗 core 宣告 `flux_dev: str` 等具名欄位，可選裝置用 `str | None`。Core 不認得 alias 或 session，透過 `context.devices[cfg.flux_dev]` 取得 driver。Measure 用 alias 選元件，不用 `flux_dev`、`coupler_flux_dev` 或 `jpa_rf_dev` 的 alias 列選裝置。Form 在 Open／Reset 從捕捉的元件欄位取得可覆寫的 direct 名稱，之後不隨 alias 或元件參數更新。

**例子。** 單 qubit 用 `$qubit.flux_dev`，mux 全域線圈用 `$mux.flux_dev`，JPA 用 `$jpa.flux_dev` 與 `$jpa.rf_source`。Mux 的 record 同時保留 global coil 與各 local 線，分析呼叫端明示使用哪台裝置作座標。Adapter 自行組裝 Run 的完整裝置快照、分析純量與來源；框架只 append JSON，不解析或補入裝置欄位。見 [Record policy](../../zcu_lab/v2/measure-authoring.md#record-policy)與 [[0063]]。

**邊界。** 缺來源時必填欄位留空並由表單報錯，可選欄位保留 `None` 的原有語意，不用靜態 fallback 或從 cache 猜裝置。新 entry 的範本只保存元件焦點 aliases。`resonator` alias 與 readout 推導不變，既有 `aliases.yaml` 不遷移。`flux_unit` 獨立存在 cfg。Recipe 的公開參數與來源政策不變，helper 寫 caller 指定的具名欄位，只比 cached unit，不換算或連裝置。受影響的 cfg 升 major，不加 `dev` migration 或 before validator。

**依據。**
- 2026-10-12，使用者：「core不該有屬於gui/mcp/notebook/...的認知，就是單純的實驗宣告，應該是adapter負責包裝」「GUI的flux_dev初始值也要以direct的格式，我不太喜歡alias形式的Enum欄位」。2026-10-13，使用者：「目前沒有實驗在改版後使用，所以不需要處理遷移邏輯」。見 [Measure 表單重設計](../discussions/2026-10-13-measure-form-redesign.md)。
- 2026-10-23，使用者：「我認為flux_dev應該是由qubit/resonator推導，而不是意圖，因為沒有對象是以device為目標的」。原始討論未入庫，見 Git `28fdd8eb7`。
- 2026-10-25，使用者：「2mux版的比較複雜，有兩個flux儀器都可能貢獻，我的想法是，兩個儀器都紀錄，其餘都同意」。其後由使用者裁定框架保持中立，見 [Record policy](../../zcu_lab/v2/measure-authoring.md#record-policy)與 [[0063]]。

### Owner 與驗證邊界

**規則。** 單一實驗的初值政策留在自己的 `form.py`。已有定案的政策若為多個 Form 共用，可交給 `zcu_lab` 的具名 domain helper。`cfg_model` 擁有 Qt-free 節點與 `TemplateDef`，不依賴 GUI，由 C11 保護。`gui.form_adapter` 擁有生成器，不 import `gui.app` 或 `zcu_lab`，由 `form-adapter-no-host` 保護。C15 阻止實驗 core 間接依賴 GUI，C16 阻止框架反向 import `zcu_lab`。Library、writeback 與 Inspect 共用 Form 生成器與持久 Spec，範本目錄由 Registry 注入。

**例子。** T1 時間尺度選源、Rabi 的來源、有效性與首次初值可共用 helper；各 Form 仍明選倍率或場景並組合結構。見 [校準初值政策](../../zcu_lab/v2/measure-authoring.md#校準初值政策)。

**邊界。** 實驗初值政策不進通用框架，也不擴及 writeback 或實驗順序。共用型別的 inline 繼承則由 `gui.form_adapter` 擁有，見 [[0079]]。框架用 synthetic adapter 驗 Form、Open／Reset、固定 capture、Run、通知及保存還原。實驗定義用真正的 Registry validate、正式 reload 與必要的 Open／Reset 檢查接線。Offline Invalid 不等於 cfg 有效，不要求逐實驗持續驗 parity、Run 或 analysis。初值差異依明確裁決接受；其餘可疑數值保留待審。

**依據。** 2026-10-17，使用者：「同意，這判斷標準沒問題」，見 [Measure Form 的校準初值與 sweep 模式切換](../discussions/2026-10-17-measure-form-initial-values-and-sweep-mode.md)。2026-10-14，使用者：「我預期能大幅簡化預設值與格式的定義代碼，目前預設值有些不太合理，可能是建立時打錯的，遷移時可以選擇：保留目前行為，但記錄下來之後讓我看。或是目前行為遷移後會需要比較複雜的寫法，可以直接問我或你。」原始討論未入庫，見 Git `c66b4eb5f`。

### 自動推導點到為止

**規則。** 一般實驗的表單初值用簡單固定的公式，不為了讓每個 point 的初值都能執行而加分支、讀更多來源或依板子調整。由參數推導的初值有兩種寫法：建立 Form 時算好並填入數值，或把推導式寫成 expression。推導結果可能超出不依板子的固定範圍時，兩種寫法都讓初值留在範圍內。填數值就先截斷，寫 expression 就用 `min` 或 `max` 截斷。缺值、無效值與來源故障按[校準初值政策](../../zcu_lab/v2/measure-authoring.md#校準初值政策)分流。

**例子。** Length Rabi 沿用捕捉的 x180，掃描上限是 x180 長度的四倍。FTF 測試 point 用 0.48 µs 的 cosine，上限 1.92 µs 已超過 drive generator 的 envelope 容量，公式仍不改。`singleshot/reset_check` 的 Gain Stop 取 π pulse gain 的四倍，上限 1.0。表單寫成 `min(1.0, 4.0 * $qubit.module.x180.gain)`，例如 0.125 得 0.5，0.617 得 1.0。作者也可填入事先算好並截斷的數值。見 [Length Rabi guide](../../zcu_lab/v2/singleshot/len_rabi/guide.md)及 [Reset Check guide](../../zcu_lab/v2/singleshot/reset_check/guide.md)。

**邊界。** 截斷只用於不依板子的固定上下限，例如正規化 gain 的 [-1, 1]。依輸入浮動的導出值照公式帶入，Form 不再 clamp，由後續驗證報錯。依板子的 envelope 容量、單一 pulse 可編碼長度不在初值裡截斷，見[板子限制](#板子限制先不加程式檢查)。這是慣例，不是禁令：作者有具體需求時，例如生成初值要偵測環境，可以加需要的邏輯，並在 guide 寫出初值怎麼來；沒有需求就用簡單做法。

**依據。** 本條由 designer 依下列裁決整理；使用者 2026-11-21 覆核措辭並定為慣例：「这四处措辞基本没问题，不过它是作为惯例。如果作者有需求，比如说生成初值的时候需要一些特别的环境侦测逻辑是没有问题的。一般实验用简单的做法就好了，有需求再做多余的设计。」2026-11-21，使用者對 Length Rabi 答「……2a，簡單比複雜好，自動推導的正確性點到為止即可」；「2a」即沿用 x180 的形狀。同日對 reset_check 說：「關於4gain超過1.0使於是預設值設計不良，應該套一層min來在expression中截斷」。見 [2026-10-17 的初值討論](../discussions/2026-10-17-measure-form-initial-values-and-sweep-mode.md)及[板子限制討論](../discussions/2026-11-21-form-board-limits.md)。

2026-11-23，使用者說：「1兩種選擇，推導後填入數值，或是將推導式填入expression」。Designer 依使用者同日授權「你繼續幫我代裁」，定下兩種寫法的截斷方式，使用者未覆核。此次原始問答未入庫。

### 預設值的瑕疵就改預設值

**規則。** 表單初值在一般 point 上就違反實驗自己的規則時，修初值公式，不加檢查。新公式對所有 point 一致，guide 說明公式。Form 不因個別 point 的算出值另做修正，Run 的既有檢查照常拒絕手動填錯的值。

**例子。** CPMG 的 Minimum time 用 16 ×（x90 長度 + x180 長度），見 [CPMG guide](../../zcu_lab/v2/twotone/time_domain/cpmg/guide.md)。兩個 0.48 µs 的脈衝得 15.36 µs，通過既有 interval 檢查；手填 15.35 µs 仍被拒。

**邊界。** 違反的是依板子的限制時不改初值，留給 Run 報錯，見[下一節](#板子限制先不加程式檢查)。例如 `singleshot/t1_tone` 缺 T1 時仍用 150 µs 作掃描終點，即使 `qick_216_v2` 的 gen 0 單一 pulse 上限約 109 µs。

**依據。** 本條由 designer 依下列裁決整理，使用者 2026-11-21 覆核措辭，定為慣例（原話見 ADR-0073〈自動推導點到為止〉）。2026-11-21，使用者選擇修 CPMG 初值：「1c，没错，这个是 这个是预设值的瑕疵，这种类似的东西都是改预设值就好了。」同日以「核准」接受 t1_tone 不改初值的邊界。2026-11-23，使用者以「2A」確認保留 guide 對公式的說明，不要求 seed metadata 記錄公式。舊初值的失敗與公式比較見[板子限制討論](../discussions/2026-11-21-form-board-limits.md)。

### 板子限制先不加程式檢查

**規則。** 表單初值超出板子的 envelope 容量、單一 pulse 可編碼長度等限制時，不加表單檢查、Run 前 preflight 或 Run 開頭檢查。Run 內原有的 program build、upload 與實驗 core 檢查照常報錯。判讀與改法放 `measure_knowledge/` 的 judge 條目，不寫在 guide，分工見 [[0075]]〈判讀依據放知識庫〉。

**例子。** `singleshot/len_rabi` 在 FTF 測試 point 以初值完成 23 點後，第 24 點報錯：`generator 2: buffer length must be 8192 samples or less.`。操作員或 agent 按訊息查 judge 條目，換算容量，再縮短 `Length → Stop` 或改用 flat_top。

**邊界。** 操作員或 agent 常在按 Run 後碰到同一類限制時，再考慮加檢查。若要加，按 [[0067]] 放在 `adapter.run` 開頭。Gain 這類不依板子的固定範圍不適用，見[表單的簡單檢查](#表單只做現成機制做得到的簡單檢查)。

**依據。** 本條由 designer 依下列裁決整理，使用者 2026-11-21 覆核措辭，定為慣例（原話見 ADR-0073〈自動推導點到為止〉）。2026-11-21，使用者改判：「上一題的一我改變想法，我想先補guide就好」。同日再指定「然後guide請不要包含取決於板子資訊的內容，其應該是中性的說明，要提供錯誤解析，應該放知識沉澱的疑難排解」。原先在 Run 開頭報錯的方案、改判與離線案例見[板子限制討論](../discussions/2026-11-21-form-board-limits.md)。

### 表單只做現成機制做得到的簡單檢查

**規則。** 不依板子、只看單一值的固定範圍，用現成的欄位 annotation 在表單驗證。不為掃描端點加專用型別、端點投影、邊界特例或其他警示機制。需要新增並長期維護的檢查交給 Run 內既有的檢查，或依內容寫進 guide 或知識庫。

**例子。** Runtime `PulseCfg.gain`、`PulseReadoutCfg.gain` 用 `cfg_model` 的 `Gain`，範圍為 [-1, 1]。填 1.6 時表單標紅該列，不交付 cfg。掃描仍用一般 `SweepCfg`。`singleshot/reset_check` 的 Gain Stop 填 1.6 時，表單接受，Run 編譯 FPGA 掃描時由 program 的 `add_pulse` 拒絕。Host 掃描不逐點建整份 cfg。`onetone/power_dep` 每點用 `PulseReadout("readout", cfg.readout, freq=freq, gain=step.value)` 建 readout，套用 gain 覆寫時由該 module 自己的 pulse cfg `Gain` 拒絕超界值。兩條路徑的超界點都不進 acquisition，但報錯位置與訊息不屬契約。建構契約見 [Program modules 的建構覆寫](../../lib/zcu_tools/program/v2/modules/README.md#constructor-overrides)。

**邊界。** 跨欄位計算或依板子的限制留在 Run，例如 CPMG 的 interval 和 CZ 的 `scale × gain`。Mux Amp Rabi 仍在 cfg 建構時由既有 participants validator 檢查參與的 tone，不擋未參與 tone 的草稿。防止「程式照跑卻默默產生錯誤資料」的守門不受本條限制，仍要加。掃描端點的投影、錯誤列與浮點特例為何不採用，見 [Gain 固定範圍討論的選項比較](../discussions/2026-11-21-form-gain-range.md#被否決的選項與理由)。

**依據。**
- 2026-11-21，使用者：「2檢查固定範圍這種簡單的機制可以在form做，太複雜的保留在run抱錯」。同日選擇只留純量：「那就采用建议吧，能够简单实做的机制，那就可以实做。如果需要加上一些需要维护的机制，那就算了。检查这种东西简单就好，对我来说，维护的负担才是重点」。隨後核准原則與邊界：「例子没错，边界也对。简单来说，不影响行为的检查应该以自然、简单即可。」見 [Gain 固定範圍討論](../discussions/2026-11-21-form-gain-range.md)。
- Mux 保留 participants validator 是 designer 代裁，使用者 2026-11-21 追認：「啊，其余大部分都没有问题啊，整批追认。」
- Host 掃描的拒絕位置是 designer 代裁，使用者 2026-11-21 追認行為並更正用詞：「零零九五的用词也有点怪，呃，行为上是没错，但是什么叫做主该点的 config？我记得目前实作惯例上并没有主 config 的概念吧。因为它是把 config 拆成多个需要用的子 config，然后按需使用，而不是每个 host 点都建一个 config。」兩項追認的原始紀錄未入庫。

## 取捨

形狀與中立欄位說明留在 core，初值與前端偏差留在 `form.py`。作者要維護這兩處，但可以直接在 `make_form` 用 Python 查看來源與分支。Registry validate 只驗 adapter 契約；範本與來源等到實際使用時才驗，不能靠註冊保證每份離線表單都有完整校準來源。

使用者 2026-10-13 對資料檔方案的取捨：「D有相當大的吸引力，但特製解析邏輯有點過度工程的味道」。不引入專用資料檔解析器。Builder 重抄形狀、把初值塞進 `Annotated` 與資料檔方案的比較見 [Measure 表單重設計](../discussions/2026-10-13-measure-form-redesign.md)。

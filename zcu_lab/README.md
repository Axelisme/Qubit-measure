# zcu_lab

**Last updated:** 2026-10-23，DC 局部線與 global coil 校準

**讀者：** lab-author、agent。查找實驗室定義的套件邊界、註冊入口與操作文件。

`zcu_lab` 只放實驗定義、實驗室宣告與各 workflow 自己的小工具。`zcu_tools` 提供實驗與前端框架，不反向 import 這個套件。使用者 GUI 入口在 `scripts/`，agent MCP 組合根在 `mcp_servers/`，取得定義後顯式注入框架。

使用者擴充入口見 [tutorial](tutorial/README.md)，程式碼規則與檢查方式見 [POLICY](POLICY.md)。

預設的 session 還原腳本屬於框架，位於 `lib/zcu_tools/gui/app/measure/default_startup.py`，行為見 [ADR-0064](../docs/adr/0064-process-startup.md#保存與關閉)。這裡只放使用者自己的 startup 腳本（`simulation_envs/`、`kickstart/`）；自訂腳本要先還原上次 session 時，import 並呼叫 `zcu_tools.gui.app.measure.default_startup.startup`。`--clean` 略過全部腳本，`--script` 完全取代預設腳本，兩者互斥。建立 entry 與 alias 的步驟見 [startup 教學](tutorial/02-startup-and-entry.md)。[Six-fluxonium kickstart](kickstart/six_fluxonium/README.md) 建立六顆 fluxonium 的 entry，不連硬體、不載模擬。

`v2/` 放各實驗的 core 與前端附件，`recipes/` 放量測流程定義。`workflows/` 的 `coherence_fluxdep/`、`mist_fluxdep/`、`overnight/` 各自擁有流程、表單與科學工具，`common/` 只放真正共用的實驗步驟。`workflows/registry.py` 提供有序註冊清單。Window、App、RPC、MCP、表單協調、layout 與 input 保存都由 lib 宿主負責。實驗室宣告位於 `components/`、`definitions.py` 與 `templates.py`。[`simulation_envs/`](simulation_envs/FTF/README.md) 放 GUI／Notebook 共用的模擬參數檔與 startup 範例，不提供離線建庫入口。[Six-fluxonium](simulation_envs/six_fluxonium/README.md) 提供六顆無耦合 fluxonium、單一 global DC source 與 mux 讀出。六顆各自設定 flux offset 與週期，不含 fast-flux pulse。

GUI 啟動時 alias 表是空的，沒有 preset 檔。Alias 只存在各 point 的 `aliases.yaml`：開 entry 清空 alias，選 point 載入該 point 的表；Inspect 編輯目前表的草稿，Apply 才保存至目前 point。要在啟動時有初值，由 startup 腳本先建立 point，再用 `ctx.context.apply_alias_rows` 初始化並保存其表（simulation_envs 與 kickstart 的腳本都這樣做）。框架不選裝置，也不從 cache 綁定 flux_dev。

`templates.TEMPLATES` 是「新建 module／waveform」的持久範本來源。組合根以 `Registry(templates=TEMPLATES)` 注入一次；Library 新建直接讀 registry 的範本，不另建 catalog。`definitions.register_all` 註冊 adapters，並可用 `components` 傳入共用 ComponentRegistry。Reload 省略此 receiver，保留既有範本與元件註冊。Import 不執行註冊，重複 bootstrap 按 registry 規則報錯。

`components/` 擁有具體元件 model、wiring、module 槽與內建角色。`base.py` 放共用欄位與基礎元件；`qubit.py` 放共用 qubit 欄位與 wiring。Fluxonium、transmon、resonator、JPA 與 mux 各有獨立子模組。`__init__.py` 提供公開匯出與明確註冊入口。Model 自行宣告 ext 與 extra 策略，普通字串引用在 resolve 時解析。GUI 與 MCP 組合根明確 bootstrap；notebook caller 可呼叫 `components.register_all(component_registry)`。框架只提供容器、kind 查表、角色解析與來源記錄。

`MuxGroupSchema` 以 `group/mux` 註冊於同一 ComponentRegistry。必填 resonators 保存 1 至 8 個有序名稱。選填 qubits 必須等長且不重複。`flux_source` 保存 global coil 的 device 名稱；`coil_flux_half`／`coil_flux_period`／`coil_flux_int` 依 qubits 順序保存對 coil 的校準，使用 coil device 的原生單位。整欄或個別元素可 null。非 null 校準 list 要求有 qubits 且等長，沒有 qubits 時不接受空 list。提供 qubits 時，lab Form 在 Open／Reset 以固定 ParameterContext 檢查兩邊 component 存在、kind 正確，並逐項要求 qubits[i].resonator 等於 resonators[i]。錯誤指名欄位、index 與實際值。Resonators 不去重；沒有 qubits 的 onetone 保留手填 readout 與缺來源的既有政策。

Scalar alias `mux=MG12` 讓 mux adapter 從固定 capture 的 `$mux.module.readout` 取得 Readout 初值。完整 tones[i] 綁定 resonators[i]。active_tones 只選擇量測哪些原始序號，不改綁定或保存 resonators。Module owner 是 group，不是第一個 resonator；重排不會改 owner。Framework 不知道 group schema，也不需要 list alias。定位與 writeback capture 邊界見 [ADR-0072](../docs/adr/0072-module-boundaries.md#mux-group-是實驗室的-component-定義)。

## Component 欄位單位

元件型別別名在既有 Field 附靜態 `json_schema_extra` 單位資料。Frequency 用 MHz，Duration 用 us，Energy 用 GHz，PumpPower 用 dBm。Qubit 的 flux_half／flux_period／flux_int 描述自己的 DC 線，依該 qubit 的 `flux_unit` 解讀。Mux group 的 coil 校準使用 coil device 的原生單位，不標固定單位。Global flux 原值仍可存於 `general.ext.flux_value`。`Q1.flux_unit` 與 JPA 的 `J1.flux_unit` 各自維持原契約。Flux 不標固定單位。未宣告單位不代表無因次。Resources 以公開路徑查詢提供固定欄位的靜態 unit。GUI 的 WB 列只用查詢結果加標籤，不換算。Ext 與 flux 類欄位不顯示單位；run context 不消費這些標記。

`recipes.RECIPES` 明列 `@recipe(name, description)` 產生的宣告。作者寫 `async def (session, inputs: FormModel) -> Report`，框架由 inputs model 產生 MCP JSON schema，不手寫 schema 或 summary declarations。Recipe inputs 只作 JSON schema／pydantic 驗證，不渲染 GUI Form。MCP 組合根把清單注入 framework，import 不執行量測或註冊。Recipes 保留 target 輸入，以 `aliases={"qubit": target}` 交給 `RecipeSession.open_tab`，省略時沿用 Reset 的全域 aliases。GUI adapters 的 cfg 與 Run 使用該 tab 的固定 capture，後續全域 alias 變動不追改既有 Run。

Measure 的 `form.py` 提供 `make_form(ctx)`，在 Open／Reset 依固定來源選初值與形狀。
`onetone/flux_dep` 與 `twotone/fluxdep` 的 core 用具名 `flux_dev: str`；
JPA 使用具名 `jpa_flux_dev`。Form 以 `AliasName("flux_dev")` 解析 direct 裝置名稱。
Alias 變更不更新既有 input；Open／Reset 才讀新名稱。缺列、未綁定或不在捕捉的
候選清單時保持 Invalid，不使用靜態名稱 fallback。Core 透過 RunContext.devices
取得 driver，不認得 alias。Analyze／WB／Save 使用 Run 的固定來源，不重讀目前表。
具名欄位與範本契約見 [ADR-0073](../docs/adr/0073-measure-form-derivation.md)。

Framework 的 `RecipeTab` 提供通用 Form read／set／select／apply、RPC call，以及 Run／Analyze／問答／同步 Save。Session 的 `context.read` 讀 session 值，不需要 tab；tab.read 讀固定 Form capture。
`recipes.helpers` 使用這些公開介面，擁有 frequency／flux 的來源順序、校準範圍與 device unit policy。明填 frequency 優先，其次是有效 library leaf，再讀 caller 指定的 calibration。`FrequencySweepCalibration` 明列 frequency／linewidth 路徑與 full-span 倍率。三個 resonator spectrum 的預設倍率為 freq 3、power 3、flux 2，點數沿 GUI 的 301／201／101。Helper 讀目前數值，集中算 span 與 numeric start／stop。無效 linewidth 在 sweep edit／Run 前拒絕，不從 raw expression 抽倍率。Twotone 的 center_mhz／span_mhz／points 與 flux survey 的 flux_range／flux_points 必填且不能為 null。Flux helper 只比 snapshot unit，成功後寫 cfg 的 device identity 與獨立 flux_unit。它不換算 native 座標，也不連裝置。

Recipe 的 Report 由作者選欄位，不另宣告 SummaryElements 或 estimates 路徑。框架回 `RecipeOutcome(status, report, runs)`；runs 保留各次 Run 與 Save 收據，full query 另保留 Run 前條件、owner-native analysis summary／invalid 與明確捕捉的寫回提案。元素位置與名稱由 owner 提供，不從 participants row 推測。操作與來源界線見 [mux recipe guide](recipes/mux-guide.md)。

所有 lab async def 的 await 只允許 env／tab／session 呼叫或靜態解析到 lab 定義的 async helper，helper 可以再呼叫 helper。Quality gate 全掃本文，不分 recipe、workflow 或 helper。明列 asyncio.*、create_task、gather、wait_for、TaskGroup 的直接呼叫不論有無 await 都拒絕，其他 async with／async for 不另禁。這是 AST guardrail，不追蹤 assignment aliases、動態分派或包裝，不是完整 type checker。作者仍須遵守序列執行契約，詳見 [recipes](recipes/README.md)。

Run 子層取消回 partial，作者可繼續。根取消使下一個 Run／Analyze／Ask await raise 普通 ScriptCancelled，不是 CancelledError。進行中的 Run 先 drain 並保留 run_id 與 partial facts，再 raise；partial 留在 tab 交 agent 決定是否保存。既有 Run 在 try 外，該 await 被根取消時不進後續 Save finalizer。已進入 try 的流程仍在 finally 同步 Save，不新增取消分支或 Save finalizer。取消終態回 cancelled、report=null 與 runs。

Analyze 預期擬合失敗回 AnalysisResult(result=None, status="failed", reason=...)，任意程式錯誤仍 raise。Recipe 每次 Analyze 後先檢查 result，失敗即回含 analysis_failure 的 report，不 Ask、不接受 writeback、不跑相依 Post。Post 失敗保留 Primary 估計；成功時 analysis_failure=None。

[`v2`](v2/README.md) 是實驗的使用者 namespace。每個實驗的 core 與可選 GUI、notebook 附件放在同一資料夾。Core 不依賴 GUI。

有擬合的分析在 summary 附 `fit_quality`，建議使用 `zcu_tools.analysis.fitting.compute_fit_quality`。實驗定義者負責選擇各 fit 的資料、參數及 covariance；GE 的 joint、ground、excited 可分開回報。這是實驗定義者的約定，框架不檢查、不特別解讀品質，也不修改 optimizer 或 writeback 政策。

本套件的實驗定義不寫個別測試。Framework owner 用最小合成 workflow／adapter 驗證 runtime、NotebookAdapter、GUI、MCP 與共用工具，不 import 本套件。刪掉 `zcu_lab/` 後，整套 pytest 仍須通過，包含 slow；實驗定義錯誤由 GUI 啟動或載入直接回報。Import-linter C16 禁止 `zcu_tools` 反向依賴本套件。

Predictor 使用 entry 的 `predictors/*.json` 或手動參數，不從實驗室 component 轉換輸入。

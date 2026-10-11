# zcu_lab

**Last updated:** 2026-10-11，component predictor 參數預填

`zcu_lab` 擁有具體實驗與其前端附件。`zcu_tools` 提供實驗與前端框架，不反向 import 這個套件。組合根取得定義，再顯式注入框架。

`definitions.register_all` 接受 caller 的 measure Registry。Startup 可用 `templates` 傳入 TemplateCatalog，並用 `components` 傳入框架共用的 ComponentRegistry。Reload 省略這兩個參數，保留既有範本與元件註冊。Import 不執行註冊，重複 bootstrap 按 registry 規則報錯。

`components.py` 擁有具體元件 model、wiring、module 槽與內建角色。Model 自行宣告 ext 與 extra 策略，普通字串引用在 resolve 時解析。GUI 與 MCP 組合根明確 bootstrap；notebook caller 可呼叫 `components.register_all(component_registry)`。框架只提供容器、kind 查表、角色解析與來源記錄。

元件型別別名在既有 Field 附靜態 `json_schema_extra` 單位資料。Frequency 用 MHz，Duration 用 us，Energy 用 GHz，PumpPower 用 dBm。Qubit 的 flux_half／flux_period／flux_int 依 `general.ext.flux_unit` 解讀。Global flux 原值存於 `general.ext.flux_value`。Local 的 `Q1.flux_unit` 與 JPA 的 `J1.flux_unit` 各自維持原契約。Flux 不標固定單位。未宣告單位不代表無因次。Resources 以公開路徑查詢提供固定欄位的靜態 unit。GUI 的 WB 列只用查詢結果加標籤，不換算。Ext 與 flux 類欄位不顯示單位；run context 不消費這些標記。

`storage_migration.build_mapping` 擁有舊 key、waveforms/modules envelope 裡的 module 候選與 R1/Q1/J1 profile。Kind 由 caller 明確指定，數字維持工作單位。舊 `flux_unit` 對應 `general.ext.flux_unit`，`cur_A` 對應 `general.ext.flux_value`。缺 flux 單位只 pending，不從舊 key 名稱推測單位。`migration_experiments` 明列 core 的 native declarations，以歷史 source_tag 與 cfg_type 配對 generic schema、concrete cfg validation 與 typed reader，另明示 canonical native_tag；Callback 重用 datafile 的完整 cfg 格式規則，再驗證隔離副本，不把派生欄位或 normalized model 回填歷史快照。不建構實驗 instance、不註冊或做動態 discovery。常駐 CLI 在 `tools/migrate_storage.py`，正常 runtime 不使用 legacy fallback。

`recipes.RECIPES` 明列使用者的 generator definitions。每個 recipe 模組同時擁有流程、手寫 input_schema 與 summary declarations。MCP 組合根把清單注入 framework，import 不執行量測或註冊。Recipes 保留 target 參數，明確以 `aliases={"qubit": target}` 交給 generic `RecipeSession.open_tab`。Target 省略時不覆寫。GUI adapters 不宣告 parameter roles，cfg 與 Run 直接讀固定的全域 alias 表。

[`v2`](v2/README.md) 是實驗的使用者 namespace。每個實驗的 core 與可選 GUI、notebook 附件放在同一資料夾。Core 不依賴 GUI。

有擬合的分析在 summary 附 `fit_quality`，建議使用 `zcu_tools.analysis.fitting.compute_fit_quality`。實驗定義者負責選擇各 fit 的資料、參數及 covariance；GE 的 joint、ground、excited 可分開回報。這是實驗定義者的約定，框架不檢查、不特別解讀品質，也不修改 optimizer 或 writeback 政策。

測試在 `tests/zcu_lab`，目錄對應本套件。通用契約逐一走過 measure registry，驗證 adapter conformance、context-free spec、成品 cfg validation 與 fresh schema。新增或刪除實驗不需要維護另一份測試清單。

共用 domain helpers 與 template startup policy 有各自的接縫測試。個別實驗不另寫行為測試，除非使用者要求特殊邏輯；目前保留 AllXY、ZigZag 與 ZigZagScan 的 D142 案例。共同 runtime、NotebookAdapter 與 GUI framework 行為由框架 owner 的測試驗證。Import-linter C16 禁止 `zcu_tools` 反向依賴本套件。

`components.params_for(component)` 讀 fluxonium 元件目前值，回傳 simulate 的 FluxoniumParams。
Notebook 傳入 point.Q1；GUI composition 傳入隔離 capture 的 component model。
缺項保留 None，ext.flux_bias 不代入預設值。參數與 Calibrate 不會回写元件。

# zcu_lab

**Last updated:** 2026-10-06，元件定義與顯式 bootstrap

`zcu_lab` 擁有具體實驗與其前端附件。`zcu_tools` 提供實驗與前端框架，不反向 import 這個套件。組合根取得定義，再顯式注入框架。

`definitions.register_all` 接受 caller 的 measure Registry。Startup 可用 `templates` 傳入 TemplateCatalog，並用 `components` 傳入框架共用的 ComponentRegistry。Reload 省略這兩個參數，保留既有範本與元件註冊。Import 不執行註冊，重複 bootstrap 按 registry 規則報錯。

`components.py` 擁有具體元件 model、wiring、module 槽與內建角色。Model 自行宣告 ext 與 extra 策略，普通字串引用在 resolve 時解析。GUI 與 MCP 組合根明確 bootstrap；notebook caller 可呼叫 `components.register_all(component_registry)`。框架只提供容器、kind 查表、角色解析與來源記錄。

`autofluxdep_catalog.build_catalog` 明列 Autofluxdep 的 measurement Builders。組合根將 catalog 注入 app，不在 import 時建立全域 registry。

`recipes.RECIPES` 明列使用者的 generator definitions。每個 recipe 模組同時擁有流程、手寫 input_schema 與 summary declarations。MCP 組合根把清單注入 framework，import 不執行量測或註冊。

[`v2`](v2/README.md) 是實驗的使用者 namespace。每個實驗的 core 與可選 GUI、autofluxdep、notebook 附件放在同一資料夾。Core 不依賴 GUI。

測試在 `tests/zcu_lab`，目錄對應本套件。通用契約逐一走過 measure registry，驗證 adapter conformance、context-free spec、成品 cfg validation 與 fresh schema。新增或刪除實驗不需要維護另一份測試清單。

共用 domain helpers 與 template startup policy 有各自的接縫測試。個別實驗不另寫行為測試，除非使用者要求特殊邏輯；目前保留 AllXY、ZigZag 與 ZigZagScan 的 D142 案例。共同 runtime、NotebookAdapter 與 GUI framework 行為由框架 owner 的測試驗證。Import-linter C16 禁止 `zcu_tools` 反向依賴本套件。

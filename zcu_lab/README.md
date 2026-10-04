# zcu_lab

**Last updated:** 2026-10-05，Autofluxdep catalog 組合入口

`zcu_lab` 擁有具體實驗與其前端附件。`zcu_tools` 提供實驗與前端框架，不反向 import 這個套件。組合根取得定義，再顯式注入框架。

`definitions.register_all` 接受 caller 的 measure Registry，startup 可另外傳入 RoleCatalog。import 不執行註冊。此入口目前沒有 measure declarations。

`autofluxdep_catalog.build_catalog` 明列 Autofluxdep 的 measurement Builders。組合根將 catalog 注入 app，不在 import 時建立全域 registry。

`v2` 是實驗的使用者 namespace。每個實驗的 core 與可選 GUI、autofluxdep、notebook 附件放在同一資料夾。Core 不依賴 GUI。

測試在 `tests/zcu_lab`，目錄對應本套件。共同框架行為由框架 owner 的測試驗證；使用者套件保留通用契約測試。

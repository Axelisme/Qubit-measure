# zcu_lab 擴充教學

這份教學寫給會 Python 與 pydantic、要擴充 `zcu_lab` 的使用者。你會新增 component、用 startup 建立 entry、註冊實驗與表單，再接上 MCP recipe，不需要讀 `lib/` 原始碼。先完成前一篇的註冊與資料準備，再往下操作。FTF kickstart 不安裝模擬環境，範例值也不代表已完成校準或可以安全 Run。

## 閱讀順序

1. [新增 component](01-components.md)：宣告與註冊新的 kind，建立元件，讓表單以 `$role.field` 讀取校準值。
2. [用 startup script 建立 entry 與 alias](02-startup-and-entry.md)：指定自己的 startup 與 entry-template，建立 point、保存 alias rows，並確認重開時保留哪些資料。
3. [新增實驗與 GUI 表單](03-experiment.md)：複製 `twotone/power_dep`，逐檔修改 core、Form、adapter 與 operator guide，最後從 GUI New Tab 開啟新實驗。
4. [為實驗加入 MCP recipe](04-recipe.md)：為新 adapter 註冊 recipe，串接 Run 與保存，再用 `recipe_guide` 讀取 operator 文件。

開始修改前先讀 [POLICY](../POLICY.md)，逐條核對寫法與檢查方式。各篇會連到 authoring 文件的對應段落，查契約時沿連結讀，不必先通讀所有框架文件。

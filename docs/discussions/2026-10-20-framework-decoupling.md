---
date: 2026-10-20
status: active
superseded_by: null
related:
  - docs/adr/0063-persistence-ownership.md
  - docs/adr/0064-process-startup.md
  - docs/adr/0067-gui-application.md
  - docs/adr/0072-module-boundaries.md
  - docs/adr/0073-measure-form-derivation.md
  - docs/discussions/2026-10-13-simulation-environment-ftf-preset.md
basis:
  - "repo@24b58fe9d：框架 alias、template 與 lab helper 去具體化落地"
  - "repo@4cd9a0900：point 各自的 alias 表與明填 template 綁定落地"
  - "repo@72da0677a：entry 層 predictor 檔落地"
  - "repo@0e0cb61ad：共用 session 移除 chip／qubit 政策"
  - "repo@a501b9352：撰寫時的 main HEAD"
---

# 框架去具體化：alias、模板、predictor 與一晶片一 qubit 假設

## 結論

使用者對框架與具體實作的分界定了四條原則：

- 具體 alias、儀器與物理模型的認知放 `zcu_lab`，框架不認。「zcu_lab可以有alias的具體認知，框架不行。」
- 模板是 form adapter 的語法工具，綁定只由呼叫端明填；alias 是 setup／point 各自的資料容器，由 startup 或 Inspect 填。兩者不是同一層。「模板/alias是不同的東西，後者才是startup填的。」
- 框架內建幾種支援能力時用內聚子模組，不以注入為原則；腳本為了獨立性集成演算法也可以。「C太多注入也不是很好維護，有些可以是框架提供支援的幾種功能，已內部內聚子模塊實現，而不是用注入的方式。script有些是為了獨立性而集成算法。」
- 「一個晶片一個 qubit」的 session 假設要消除；分析工具改用 entry＋component，舊資料由 `migrations/` 轉換，工具本身不相容舊格式。

落地結果：

| 項目 | 結果 | 現行規則 |
| --- | --- | --- |
| 儀器 helper 在 lib | 搬到 `zcu_lab`；`DeviceInfo` 聯集留在 `device`；MCP value patch 改為依欄位的通用規則 | ADR-0072、ADR-0068 |
| 寫死 alias（`$flux_dev`、`$qubit.kind`） | sample writeback 的 device alias 由 zcu_lab adapter 宣告；workflow 預檢移除 | 程式碼與模組 README |
| 模板佔位名只接受 qubit／resonator | `Template.bindings` 由呼叫端明填任意名稱；沒有 session／startup 預設值 | ADR-0073 |
| 規格清單手寫 | `library_entry` 從 `ModuleSpec`／`WaveformSpec` union 推導 | 程式碼 |
| predictor 綁 component 與 `params_for` | predictor 讀 `result/<entry>/predictors/<名稱>.json`，不從 component 轉換 | ADR-0063 |
| alias 一個 entry 一份 | setup 與每個 point 各一份 `aliases.yaml`，建立時複製、不繼承 | ADR-0063、ADR-0067 |
| session 的 chip／qub／res 欄位 | 從共用 session 刪除；分析 GUI 的專案模型移到 `gui/app/analysis_project` 過渡 | ADR-0067（session 欄位） |
| MCP 說明寫死 recipe 規則 | 移到各 recipe guide，框架說明只留通用規則 | MCP measure 說明文字 |
| T1 命名殘留 | workflow 快取檔名與 RPC server 名去 T1，不做舊快取相容 | 程式碼 |

## 問題

GUI 對模擬無知的工作完成後，使用者要求全面檢查框架層還有哪些地方依具體名稱分支。三位調查員分別掃 `gui/`、`experiment/`＋`mcp/`＋`notebook/`＋`scripts/`、能力模組（`device`、`program`、`simulate`、`resources`、`analysis`、`cfg_model`、`utils`），共列 54 項。顧問把它們分成四類：

- **A. 框架依具體名稱分支**：A1 儀器（`experiment/utils/device.py` 的 `isinstance` 四儀器分支、`DeviceInfo` 四儀器聯集、MCP 寫死 YOKO 必帶 `value_unit`）；A2 寫死 alias（sample writeback 固定讀 `$flux_dev`、workflow 表單要求 `$qubit.kind`）；A3 寫死 component 種類（模板展開只接受 `{qubit, resonator}`、`library_entry` 手寫 7 種 ModuleSpec 與 6 種 WaveformSpec）；A4 predictor 寫死 fluxonium／transmon；A5 MCP 寫死實驗類型（`mcp/fluxdep` 的 onetone／twotone 分支、measure MCP 說明寫死 recipe 名稱）；A6 T1 命名殘留。
- **B. 專案結構假設**：session 的 `chip_name`／`qub_name`／`res_name` 與 `result/<chip>/<qubit>` 兩級目錄。Measure 與 Workflows GUI 已改用 `ResultEntry`，這些欄位在執行期永遠是 `unknown_*`，但仍被 writeback 與 MCP 回覆廣播；fluxdep／dispersive GUI 與 `ResultScopeManager` 仍靠兩級目錄找 `params.json`。
- **C. 具體實作放錯 package**：FTF 環境在 `experiment/ftf`、flux-pick 外掛在 gui 框架、`experiment/utils` 的 t1_sampling／single_shot／snr、notebook 下的 fluxonium 物理分析、能力模組互相依賴（program → resources、simulate → QubitParams、analysis 直接讀寫 h5py）、scripts 內的可重用演算法。
- **D. 刻意的專用工具**：fluxdep GUI、dispersive GUI、dispersive MCP 本身就是 fluxonium 工具。

## 考慮過的選項與比較

### A1 儀器 helper

`experiment/utils/device.py` 的五個 helper 只有 `zcu_lab/v2/jpa/auto_optimize` 一處呼叫，lib 內沒有使用者。使用者：「A1我覺得有點尷尬，如果是zcu_lab的工具就沒問題，但偏偏是lib內。」結論是整檔搬到 zcu_lab。`DeviceInfo` 聯集留在 `device`：device package 列自己的儀器不算框架耦合。

MCP 的 YOKO 規則有三個去處：

| 選項 | 內容 | 結果 |
| --- | --- | --- |
| Info 要求明填 unit | 破壞 `YOKOGS200Info` 由 mode 推導 unit 的契約；merged cfg 已含 snapshot 的 unit，規則形同失效 | 否決 |
| 新增 partial-validation seam | 為一條規則新增跨層介面 | 否決 |
| 規則留在 MCP patch 層，改通用 | Info model 有 `value_unit` 欄位時，patch 含 `value` 卻不含 `value_unit` 就拒絕；MCP 不比對儀器名 | 採用 |

規則的目的是避免 agent 以為單位是 A、snapshot 卻是 V。這是 patch 語意（呼叫端是否明說單位），屬 MCP patch 層。副作用是 FakeDevice 同樣要求明填，規則對所有有單位的儀器一致。

### A3 模板佔位名

最初的方向是「框架初始為空，由 startup 填」（使用者：「A3我認為初始應該是空的，讓startup填。」）。顧問代裁了一版：cfg_model 提供 `PlaceholderRegistry`，由 session 組裝、startup 經 `ctx.placeholders` 登錄預設值，展開時與 `Template.bindings` 合併。

使用者審閱時推翻：「你搞錯了吧，模板/alias是不同的東西，後者才是startup填的」「模板是form adapter提供的工具的語法，讓呼叫端可以填需要的名子，比如$qubit或$qubit1，來展開成使用不同alias的表達式」。

最終：模板沒有任何 session／startup 層的預設值。呼叫端在 `Template.bindings` 明填每個佔位名要換成的 alias 名，缺 binding 在展開時報錯。startup 填的是 alias。

### alias 的歸屬

模板與 alias 分開後，alias 放哪一層：

| 選項 | 內容 | 代價 |
| --- | --- | --- |
| 甲：一個 entry 一份 | alias 與元件分處兩層；換 point 要手改，改一次波及舊 point | 否決 |
| setup＋point 繼承覆寫 | 兩層查找，值的來源不明 | 否決 |
| 乙：setup 與每個 point 各一份 `aliases.yaml` | 與 `point.yaml`、`module_cfg.yaml` 並列；新建 point 複製 setup 的表，clone 複製來源 point 的表，之後各自修改；缺檔視為空表 | 採用 |

理由：alias 指向元件，元件屬於 point，與元件、module_cfg 走同一條複製路徑。使用者：「採用乙，alias. yaml的獨立檔案，找不到就視內容為空，與其他兩個資料容器並列」。既有 point 缺檔時不自動複製 setup 的表，使用者在 Inspect 自行填。預設 startup 不寫 alias，恢復 point 就自然帶回；`ctx.context.apply_alias_rows` 只用於模擬場景的腳本初始化。

### A4 predictor

Predictor dialog 原本依 component kind 前綴與 alias `qubit` 選對象，再經注入的 `params_for` 把 component 欄位轉成框架內建的 fluxonium／transmon 參數。第一輪方向是比照 device adapter，每個模型一個內聚模組。使用者隨後提出更簡單的路：「框架內支援的fluxonium/transmon是理論上的兩種qubit，有自己定義的參數，與用戶定義的component解偶」「我認為還有一個最簡單的，不轉。讓fluxdep的工具寫preditor要用的格式檔，而不是component的欄位。」

採用：fluxdep 工具寫 `result/<entry>/predictors/<名稱>.json`，格式含 model、params、`flux_unit` 與 source；dialog 從該目錄選檔載入，載入後可手改。移除 `params_for` 注入、component 清單、alias 偏好與預設校準式。檔案放 entry 層而不是 point 層：「1先entry吧」。

### B 一晶片一 qubit

調查給出三個替代模型：

| 選項 | 內容 | 評估 |
| --- | --- | --- |
| A：全面統一到 ResultEntry＋component | 單層目錄；多 qubit 由 point 的 components 承載；session 與 wire 清掉死欄位；fluxdep／dispersive 改開 entry 並選目標 component | 改動面最大，與 ADR-0063／0067 一致 |
| B：兩級路徑泛化為 namespace／sub-entry | 目錄維持兩級但解除物理意義；`ResultEntry` 支援階層路徑 | 與 entry 內的 `points/` 層重疊，仍未解一個 entry 多 qubit 的映射 |
| C：先淨化 session 核心，分析工具局部隔離過渡 | 立即刪 session 死欄位與 wire 污染；`ProjectInfo`／`ProjectDialog`／`ResultScopeManager` 移出共用 session，由分析 GUI 自己持有 | 改動小，不動既有資料 |

使用者：「B我認為要安排任務處理掉這種認知。」裁決「1B可以，2要，3讓migrations裡支援遷移，工具本身不兼容」：先走 C 清 session 核心，再以 A 為目標改分析工具，舊 `result/<chip>/<qub>/params.json` 由 `migrations/` 轉成 entry。

### C 實作擺放

調查報告對 C 類多建議「改由外部注入」或「抽象成 Protocol 讓呼叫端傳入」。使用者否決這個方向：框架常用的幾種支援能力直接內建為內聚子模組，scripts 為了獨立執行集成演算法也可以。C 類不開 task、不進 backlog。例外：`scripts/create_entry.py` 的 `materialize_entry` 被測試 import，違反「scripts 不放被依賴的實作」，隨 framework-decoupling 刪除；FTF 環境薄包裝另依模擬環境討論移除。

### A5 MCP

`mcp/fluxdep` 的 onetone／twotone 分支維持：「mcp/fluxdep就是專門處理onetone/twotone fluxdep的，這個沒問題。」measure MCP 說明寫死的 recipe 規則移到各 recipe guide：「4改，這種地方偷偷偶合最難維護了，偶合的東西就該集中起來，5甲」。

## 被否決的選項與理由

- 儀器 helper 改為 capability Protocol 由各 driver 實作：呼叫者只有 zcu_lab 一處，搬走即可，不需要新抽象。
- `DeviceInfo` 改 `BaseDeviceInfo` 或 TypeVar：device package 列自己的儀器是它的職責。
- session 擁有的 placeholder registry：與 alias 重複一層對照，讓 startup 碰到模板。
- predictor 比照 device adapter 做模型模組：比「不轉、讀自己的格式檔」多一層轉換。
- 兩級目錄只改中性名稱：與 entry 內 points 層重疊，未解多 qubit 映射。
- C 類以注入或 Protocol 解耦：注入太多不好維護。

## 留下的問題

- D：fluxdep GUI、dispersive GUI 與 dispersive MCP 是否維持為 fluxonium 專用工具，使用者尚未裁決。
- C 類的擺放債務（notebook 下的 fluxonium 分析、`experiment/utils` 的分析演算法、能力模組互相依賴）維持現狀；ADR-0072 的債務清單與 `.importlinter` 的 `ignore_imports` 是現行記錄。
- 分析工具改 entry＋component 的 task 尚未開工。

## 依據

- 三份調查報告（gui、experiment／mcp／notebook／scripts、能力模組）為未入庫的工作檔；本文的分類與項目清單取自其中。
- 現行規則：ADR-0063（alias 表與 predictor 檔）、ADR-0067（ContextService 與 session 欄位）、ADR-0072（模組定位、內建支援優先於注入）、ADR-0073（模板綁定）、ADR-0068（MCP value patch 與 recipe guide）。
- 模擬環境改由啟動腳本組裝的脈絡見 [模擬量測環境、低能投影與 FTF preset 的討論](2026-10-13-simulation-environment-ftf-preset.md)。

---
status: draft
---

# 模組定位與依賴邊界

## 脈絡

`zcu_tools` 有 15 個頂層 package，過去只有 7 條 import-linter contract，其餘邊界靠「責任明確」這類原則維持。agent 會把既有耦合當成先例，也不會主動寫 contract。

例子是 `experiment/base.py` 的 `PersistableExperiment`。它呼叫 `datafile`，卻自己實作 Labber 版面、讀回驗證、通道型別與 comment 編碼，這些都屬於 `datafile` 的責任。另一個例子是能力模組之間的兩個循環：`resources → program → simulate → resources`，以及 `resources → program → analysis → simulate → resources`。

本篇規定每個 package 負責什麼、可以依賴誰，並把規則投影成 `.importlinter` contract。定位由使用者在 2026-10-04 的討論中決定。本篇是目標設計：分層與 contract 已生效，債務清單列出尚未達成的部分。

## 決策

### 模組定位

| package | 負責 | 不負責 |
|---|---|---|
| `datafile` | 量測資料檔的儲存工具：Labber、`data.h5`、streaming、grouped。包括版面慣例、讀回時依 schema 驗證、通道型別與 metadata 編碼 | 存檔位置與批次（D46）；實驗語意 |
| `device` | 各 pyvisa 儀器的定義與控制介面 | 實驗流程 |
| `program` | 控制 ZCU216 與底層 QICK 的工具和類別，包括模擬板 | 參數與資產的保存 |
| `analysis` | 分析工具 | 檔案讀寫；畫圖 |
| `plotting` | 畫圖 | 分析計算 |
| `simulate` | 數值模擬 | 讀取資料容器；呼叫端傳入數值 |
| `resources` | 各實驗領域的資料容器，以及其 YAML、JSON、CSV、npz 格式 | 量測資料檔；program 的執行機制 |
| `cfg_model` | 共享核心，見下節 | 執行行為與 I/O |
| `utils` | 無法歸入任何其他類別的雜項 | 任何能歸入其他類別的東西 |
| `progress_bar`、`qick_remote` | 照字面 | — |
| `experiment` | 實驗框架：實驗介面、AxesSpec、RunRecord、RunContext、runtime 與註冊機制。舊格式的具體實驗暫留於此 | 檔案格式；具體實驗的新格式（在 `zcu_lab`） |
| `gui` | 各 GUI app 的框架 | 具體實驗的 adapter（D75） |
| `notebook` | Notebook 特化的工具集合 | 通用分析；具體實驗 |
| `mcp` | 各 MCP 的框架 | 具體 recipe（D72、D80） |
| `zcu_lab`（repo 根目錄） | 具體實驗、recipes 與其 gui、notebook 附件（D75、D80） | — |

放置新程式碼時，先找最貼近的類別；所有類別都不合，才放 `utils`。

### 分層

由下往上五層。上層可以依賴下層，下層不知道上層。

1. 能力模組：`datafile`、`device`、`program`、`analysis`、`plotting`、`simulate`、`resources`、`cfg_model`、`utils`、`progress_bar`、`qick_remote`
2. 實驗框架：`experiment`
3. 前端框架：`gui`、`notebook`、`mcp`
4. 使用者套件：`zcu_lab`
5. 組合根：`scripts/` 與 MCP server 入口

`zcu_tools` 不 import `zcu_lab`（D72）。定義由組合根注入（D73）。

### 能力模組的兩類

| 類別 | 模組 | 規則 |
|---|---|---|
| 穩定工具與值型別 | `cfg_model`、`utils`、`analysis`、`simulate`、`plotting`、`datafile`、`progress_bar` | 任何模組都可以選用。工具之間可以互相依賴，但不得循環 |
| 有狀態的能力模組 | `program`、`resources`、`device`、`qick_remote` | 彼此不得 import，只透過 `cfg_model` 的型別與 Protocol 溝通 |

依賴朝向穩定的一方：被很多模組依賴的模組必須少變，常加功能的模組不被別人依賴。純工具接收數值、回傳數值，不自己讀容器、硬體或檔案以外的狀態。

### 共享核心 `cfg_model`

`cfg_model.py` 擴充成 package `zcu_tools.cfg_model/`。

- **內容**：`ConfigBase`；module 與 waveform 的 cfg schema（欄位、驗證、`set_param`）；有狀態模組之間溝通用的 Protocol，例如 `ModuleResolver`（依名稱取 module 或 waveform cfg）與 `WaveformAssetInfo`（依名稱查波形長度）。
- **依賴**：只依賴外部函式庫（pydantic、numpy、qick 型別）與 `utils`。
- **准入**：型別只有在兩個以上有狀態模組都需要時才放進來。
- **改動**：核心 schema 會寫進檔案，例如每個工作點的 `module_cfg.yaml`。改核心等於改格式，要處理格式版本。

`program` 以 `type`／`style` 對應到執行用的 Module builder，取代 cfg 上的 `build()`。`resources` 的 module library 以核心 schema 作為 DocumentStore 的 model，並實作核心的 Protocol。驗證時的參照解析與波形資產查詢都透過驗證 context 注入。

### 模組之間的合作

- 需要另一個模組負責的能力，而它沒有提供時，擴充那個模組或停下來問。不在呼叫端重做。
- 兩個有狀態模組需要合作時，由需要的一方定義最小 Protocol，由組合根注入實作；或把組合邏輯放到上一層。
- 呼叫端驗證被呼叫者的輸出格式、組被呼叫者的內部版面、或修補被呼叫者回傳的資料，都表示責任放錯位置。

### 可執行投影

`.importlinter` 的 C8–C13 投影本篇規則：

| contract | 類型 | 內容 |
|---|---|---|
| C8 | `layers` | 前端框架 > `experiment` > 能力模組 |
| C9 | `acyclic_siblings` | `zcu_tools` 的頂層 package 之間沒有循環 |
| C10 | `independence` | `program`、`resources`、`device`、`qick_remote` 彼此獨立 |
| C11 | `forbidden` | `cfg_model` 只依賴 `utils` |
| C12 | `forbidden` | 穩定工具不依賴有狀態模組 |
| C13 | `forbidden` | `mcp` 只使用 `gui.remote` 與各 app 的 wire spec |

尚未建立的投影：`zcu_tools` 不 import `zcu_lab`，等 `zcu_lab` 存在後加入 root packages 再建立。

債務以確切的 module 對逐條列在 `ignore_imports`，不用萬用字元，每組註解寫明「債務，由誰移除」。萬用字元只用於合法例外，例如 C1 的 `v2_gui`，註解寫明「不是債務」。`ignore_imports` 只減不增。

## 債務

| 債務 | 移除方式 | 負責 |
|---|---|---|
| `program → resources` | `program` 改依賴核心 Protocol（第 1 階段） | backlog |
| `resources → program` | cfg schema 抽到核心，`build()` 改為 program 端 builder（第 2 階段） | storage-redesign 的 module library；`zcu_lab` 搬家 |
| `simulate → resources` | `predict` 改由呼叫端傳入數值 | backlog |
| `experiment/base.py` 的格式處理 | 移入 `datafile` 的 reader／writer | storage-redesign（D53、#15） |
| `experiment → notebook`（`make_sweep`） | `make_sweep` 移到 `experiment` 框架 | backlog |
| `notebook/analysis`、`notebook/experiments` | 通用分析移到 `analysis`；具體實驗移到 `zcu_lab` 附件 | `zcu_lab` 搬家 |
| `utils` 的 `shot_classification`、`tomography` | 移到 `analysis` | backlog |
| `utils/datasaver/` | 空目錄，刪除 | backlog |
| `analysis/fluxdep/io.py`、`search.py` 直接用 h5py | 檔案讀寫移到 `datafile` | backlog |
| `experiment/v2`、`v2_gui` | 移到 `zcu_lab`（D80） | `zcu_lab` 搬家 |
| `mcp → gui.logging_setup` | logging 設定改由 `mcp` 自己或共用的非 GUI 模組提供 | backlog |

## 後果

- 增減功能時依賴圖的形狀不變：工具之間的邊很少變，有狀態模組之間沒有邊。兩個有狀態模組要合作，做法固定是 Protocol 或上移組合。
- 新增跨 package 依賴會被 contract 擋下或在 review 中出現，不會靜默成為先例。
- 第 2 階段的 cfg 抽離牽涉約 77 處 `.build(` 呼叫，大多在要搬到 `zcu_lab` 的實驗，所以跟搬家一起做。

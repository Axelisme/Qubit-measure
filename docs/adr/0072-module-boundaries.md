# 0072 — 模組定位與依賴邊界

## 脈絡

`zcu_tools` 有 15 個頂層 package，過去只有 7 條 import-linter contract，其餘邊界靠「責任明確」這類原則維持。agent 會把既有耦合當成先例，也不會主動寫 contract。

例子是 `experiment/base.py` 的 `PersistableExperiment`。它呼叫 `datafile`，卻自己實作 Labber 版面、讀回驗證、通道型別與 comment 編碼，這些都屬於 `datafile` 的責任。另一個例子是能力模組之間的兩個循環：`resources → program → simulate → resources`，以及 `resources → program → analysis → simulate → resources`。

本篇規定每個 package 負責什麼、可以依賴誰，並把規則投影成 `.importlinter` contract。定位由使用者在 2026-10-04 的討論中決定。分層與 contract 已生效，債務清單列出仍以 `ignore_imports` 記帳或尚未搬移的部分。

## 決策

### 模組定位

| package | 負責 | 不負責 |
|---|---|---|
| `datafile` | 量測資料檔的儲存工具：Labber、`data.h5`、streaming、grouped。包括版面慣例、讀回時依 schema 驗證、通道型別與 metadata 編碼 | 存檔位置與批次（見 ADR-0063）；實驗語意 |
| `device` | 各 pyvisa 儀器的定義與控制介面 | 實驗流程 |
| `program` | 控制 ZCU216 與底層 QICK 的工具和類別，包括模擬板 | 參數與資產的保存 |
| `analysis` | 分析工具 | 檔案讀寫；畫圖 |
| `plotting` | 畫圖 | 分析計算 |
| `simulate` | 數值模擬 | 讀取資料容器；呼叫端傳入數值 |
| `resources` | 實驗領域資料容器與 YAML、JSON、CSV 的讀寫；waveform 資產的命名、formula 語意與 NPZ 實體 I/O | 量測資料檔；program 的執行機制；重複 waveform codec |
| `core` | 共用的無狀態型別與格式。`core.waveform_format` 驗證 samples、編解碼 NPZ 與 opaque JSON recipe | 檔案 I/O、formula 語意、GUI、資源 scope 或硬體 |
| `cfg_model` | 配置 schema 與注入依賴契約，見下節。搬進 core 尚未排定 | 執行行為與 I/O |
| `utils` | 無法歸入任何其他類別的雜項 | 任何能歸入其他類別的東西 |
| `progress_bar`、`qick_remote` | 照字面 | — |
| `experiment` | 實驗框架：實驗介面、AxesSpec、RunRecord、RunContext、runtime 與註冊機制 | 檔案格式；具體實驗 |
| `gui` | 各 GUI app 的框架 | 具體實驗的 adapter |
| `notebook` | Notebook 特化的工具集合 | 通用分析；具體實驗 |
| `mcp` | 各 MCP 的框架 | 具體 recipe |
| `zcu_lab`（repo 根目錄） | 具體實驗、recipes 與其 gui、notebook 附件 | 離線遷移的對照與宣告（放 `migrations/`） |

放置新程式碼時，先找最貼近的類別；所有類別都不合，才放 `utils`。

### 分層

由下往上五層。上層可以依賴下層，下層不知道上層。

1. 能力模組：`datafile`、`device`、`program`、`analysis`、`plotting`、`simulate`、`resources`、`core`、`cfg_model`、`utils`、`progress_bar`、`qick_remote`
2. 實驗框架：`experiment`
3. 前端框架：`gui`、`notebook`、`mcp`
4. 使用者套件：`zcu_lab`
5. 組合根：使用者入口在 `scripts/`，agent MCP 入口在 `mcp_servers/`

- `zcu_tools` 不 import `zcu_lab`。定義由組合根注入，框架不反向載入使用者實驗。
- tests 不依賴 `zcu_lab`，不 import、不使用其 fake 實驗、不綁定實驗流程或預設值。

框架測試使用最小合成 workflow／adapter，不綁定實驗流程或預設值。實驗定義錯誤由 GUI 啟動或載入時回報，不另維護個別實驗測試。使用者於 2026-10-12 裁定：「我給你一個判斷方法，如果我刪除zcu_lab測試會報錯，那測試就是寫錯了。」

使用者於 2026-10-16 改定驗收方式：「算了，我認為不用真的測試ZCU_LAB移開後會不會出現錯誤。這中東西直接看有沒有import就好，偷偷做的在review也會被抓到，感覺是浪費時間測試」。Tests 不依賴 `zcu_lab` 的原則不變，由 review 檢查 import，必要時以 grep 輔助。不再暫移 `zcu_lab/` 跑 pytest，也不新增 `ZCU_LAB` 環境變數或 import hook。

### 使用者元件與模擬環境的放置

`zcu_lab.components` 是 package。共用欄位型別、通用 wiring 與基礎元件放在 `base.py`；qubit 共用定義與參數轉換放在 `qubit.py`。Fluxonium、transmon、resonator、JPA 與 mux 各自擁有一個子模組。Package 入口保留明確的 `register_all()` 與公開匯出，不作 import-time 註冊。

GUI／Notebook 共用的範例環境、FTF YAML 與相鄰離線建庫入口放在 `zcu_lab/simulation_envs/`，不放在 Notebook 文件目錄。框架仍以 caller 傳入的路徑載入可信 Python，不自動探索這個目錄；C14 與 C16 繼續禁止框架反向 import 使用者定義。

2026-10-16，使用者要求：「比如components我希望除了基礎原件，其他的都是獨立的子模塊檔。」對上述拆分與環境搬移確認：「可以移到zcu_lab/simulation_envs/中嗎？2同意」。

### Library 分組與 module slot

Library 的設計是「元件 → 分組 → Pydantic model」。分組是擴充點，不是永久封閉的種類清單。現行格式仍只接受既有四分組；有第五種需求時才調整容器與 key 驗證，不把任意分組寫成目前已支援。

原裁決日期未記錄，使用者於 2026-10-16 map 結案時確認沿用，原話「都採用建議」。原裁決原話為「我無法知道元件下會需要什麼欄位，我認為元件->xxx->pydantic model比較適合，不規定只有四種」及「我選D，但暫時沒有超過這四種的需求」。這裡選擇的是元件下自由命名分組的設計方向，不是當時就改格式。

Component 的 module slot 指向下游實驗要使用的模組，不用來為每一種校準方案保存一個 slot。參數累積、哪些實驗只提參數、哪些收尾實驗建立或更新模組，都由實驗作者決定。Framework 不提供通用的齊全檢查或累積機制。具體欄位、reset 名稱及 pair 方向留在實驗作者文件，不提升成框架規則。使用者於 2026-10-16 以「都採用建議」確認這個歸檔範圍。

### Mux group 是實驗室的 component 定義

`zcu_lab.components.MuxGroupSchema` 宣告 `group/mux`，沿既有 ComponentRegistry 註冊。
它保存必填的 `resonators`，含 1 至 8 個有序名稱，以及選填的 `qubits` 和一般 module／ext。
Group 自己擁有 readout module，重排 resonators 不改 owner。
使用者於 2026-10-16 對專屬 group component 方案裁定：「我選B」。

`resonators` 取代原 members，不提供舊欄位相容讀取。Tone i 對應完整 `resonators[i]`。
選填 qubits 讓沒有 qubit 的 resonator bring-up 繼續使用同一種 group，不從 resonators 反推 qubit。
使用者於 2026-10-18 裁定：「或是改寫mux_group這個原件的欄位吧，qubits分resonators」。
對 resonators 必填、qubits 選填並逐項檢查，以及改名不留相容層的細節，裁定：「12都同意」。

Model 驗證名稱、qubits 與 resonators 等長及重複 qubit。
提供 qubits 時，lab Form 在 Open／Reset 以固定 ParameterContext 檢查兩邊 component 存在與 kind，
並逐項要求 `qubits[i].resonator == resonators[i]`。缺失或 null 同樣拒絕。
錯誤指名欄位、index 與實際值，在 Run 前拒絕來源。不排序、猜配對或自動修正，不額外拒絕重複 resonator。
Framework resources、session 與 program 不擁有這些關係。既有表單不隨全域來源編輯刷新。

Scalar alias `mux=MG12` 已足以讓 adapter 經固定 ParameterContext 讀取 `$mux.resonators`，
Readout 初值使用 `$mux.module.readout`。Framework 不認得 group/mux 或 resonators，
也不新增 list alias、group registry、RPC 或保存格式。
使用者於 2026-10-16 對只用 scalar mux alias 的方案裁定：「B」。

完整 readout 的 tones[i] 綁定 resonators[i]，cfg 不保存 writeback 目標或 group snapshot。
active_tones=None 量全部；指定非空、唯一且有效的原始 index list，則依輸入順序投影 subset
給 program。Program 不知道 MG 或 resonators。RunRecord 保存完整 readout 與 selection，
資料軸保留原始序號，不重新編號。使用者於 2026-10-16 提議：「一個MG component類，內部屬性綁定到R1, R2，gui中預設指定MG，寫回目標也是從MG的綁定取」，並補充「為了能只讀特定幾個R，可以考慮MG外cfg另外提供mask之類的」。對 active_tones 方案裁定：「可以」。

[[0079]] 採用但尚待實作的 lab 表單契約使用完整具名 tones list 與獨立 participants。元素位置是原始 tone index，name 不可編輯；participants 的順序是執行順序，未參與元素同樣驗證。Lab runtime 邊界一次解析名稱到原始 index，再交給既有 program／raw active_tones 接口。這不把 group snapshot 或 writeback 目的地塞進 cfg，也不讓 framework 解讀 group。具體實驗 cfg 與 recipe 接線由 lab 交付。

Program cfg 驗證錯誤沿既有宿主 result／Run 診斷定位 `active_tones` 並拒絕 Run，不保證欄位變紅。
FormState 的 preview／describe 不執行 lowering，也不新增 FormEditor 外部錯誤投影。
此分界沿用 [[0065]]。使用者於 2026-10-16 裁定：「報錯我選B」。

Writeback 由 adapter 經 Run／Load 既有 capture 讀取 `$mux.resonators`，以完整原始序號對應成功 active tones。
Onetone 沒有 qubits 時保留手填 readout。缺 capture 或 resonators 數不等於完整 N 時只提 module，並顯示原因。
有 qubits 時仍須通過上述逐項檢查。新 qubit 實驗另要求 resonators 數等於完整 tone 數，驗證後才投影 active_tones。
新 qubit 實驗的成功 tone i 寫回 `qubits[i]`，onetone 則仍寫回 `resonators[i]`。
目的地只從該次 Run／Load capture 解析，沒有 capture 時不得猜目的地或讀目前 alias。
固定 module target 是 `$mux.readout.readout_rf`，與 group 當前 slot 所指的 key 無關。
使用者於 2026-10-16 裁定：「應該參考其他使用alias是怎麼寫回的」。
這些 lab 定義沿第 4 層依賴下層的方向，C15、C16 保護 core 與 framework 邊界。

### 能力模組的兩類

| 類別 | 模組 | 規則 |
|---|---|---|
| 穩定工具與值型別 | `core`、`cfg_model`、`utils`、`analysis`、`simulate`、`plotting`、`datafile`、`progress_bar` | 任何模組都可以選用。工具之間可以互相依賴，但不得循環 |
| 有狀態的能力模組 | `program`、`resources`、`device`、`qick_remote` | 彼此不得 import，只透過 `cfg_model` 的型別與 Protocol 溝通 |

依賴朝向穩定的一方：被很多模組依賴的模組必須少變，常加功能的模組不被別人依賴。純工具接收數值、回傳數值，不自己讀容器、硬體或檔案以外的狀態。

Package 初始化不把專用依賴帶進無關工具。`analysis` 在選用 `fluxdep` 時才載入它，
讓 fitting 不要求互動 line picker。`utils` 的 QICK import 留在選用的 `format_obj`
轉換函式內。`datafile` facade 按公開名稱載入 owner，讓 storage 使用 path helpers 時
不載入 HDF5。這些調整保留公開名稱及計算、保存契約，不改變 C8、C9、C12 的依賴方向。

### 共享核心 core 與 cfg_model

`zcu_tools.core` 只依賴標準庫、外部函式庫與 repo 的 utils。`core.waveform_format` 擁有 WaveformSamples、NPZ codec 及單一 WaveformFormatError。它只驗 recipe 的 JSON 語法。Resources 解讀 formula 與管理檔案，program/sim 播放 path 指定的 samples。

`core.numeric_scalar.normalize_numpy_scalar` 擁有 AnalysisDocument 與 AcceptedWrite 共用的無狀態 numpy scalar 表示轉換。
它不遞迴，finite 與 strict JSON 驗證仍由各邊界負責。
Datafile／resources → core 的方向由 C8、C9、C12 約束。此函式只依賴 numpy，符合 C19。
使用者於 2026-10-12 裁決：「採 1：core.numeric_scalar.normalize_numpy_scalar 放 core（同 complex_codec 的無狀態表示轉換），finite/strict JSON 驗證留在兩個 boundary，core README 補一行。」

`cfg_model` 與頂層 `format_version.py` 的搬遷排在 waveform 接線後，目前仍保留原位置。

- **內容**：`ConfigBase`；module 與 waveform 的 cfg schema（欄位、驗證、`set_param`）；有狀態模組之間溝通用的 Protocol，例如 `ModuleResolver`（依名稱取 module 或 waveform cfg）與 `WaveformAssetInfo`（依名稱查波形長度）。
- **依賴**：cfg_model 只依賴外部函式庫、utils 與 core.waveform_format，不讀寫檔案。
- **准入**：型別只有在兩個以上有狀態模組都需要時才放進來。
- **改動**：核心 schema 會寫進檔案，例如每個工作點的 `module_cfg.yaml`。Library 的 format／format_version 只描述 header 與容器結構，不描述個別 module／waveform Spec 的欄位。新增、刪除或修改 Spec 欄位不自動升 library 版本。目前 library 維持 2.0，不遷移、不加相容讀取；entry schema 仍依現行定義驗證，不能以 forward minor 放寬 extra 策略。

使用者於 2026-10-15 裁決：「只是改Module而已真的需要升Library版本嗎？就像加一個Component原件定義，不應該讓舊版本不能讀對吧」「不要一次驗證，而是best-effort」「1可以，2A，3保留2.0」。Library 逐條驗證既有資料，合法條目不因另一筆 Spec 過時而失去可讀性。壞條目的備份與讀取清理例外由 [[0063]] 規定。

`program` 以 `type`／`style` 對應到執行用的 Module builder，取代 cfg 上的 `build()`。`resources` 的 module library 以核心 schema 作為 DocumentStore 的 model，並實作核心的 Protocol。驗證 context 注入固定 library 與 asset capability。Program 的 ModuleCfgFactory.from_core 集中轉換成 native cfg，透過 assets.path_for 取得絕對路徑。Native arb 只保留 data key 與必填 path，不保留 provider，不查 global DB。

### 模組之間的合作

- 需要另一個模組負責的能力，而它沒有提供時，擴充那個模組或停下來問。不在呼叫端重做。
- 兩個有狀態模組需要合作時，由需要的一方定義最小 Protocol，由組合根注入實作；或把組合邏輯放到上一層。
- 呼叫端驗證被呼叫者的輸出格式、組被呼叫者的內部版面、或修補被呼叫者回傳的資料，都表示責任放錯位置。

### 可執行投影

`.importlinter` 的 C8–C13、C15、C16 與 C19 投影本篇規則：

| contract | 類型 | 內容 |
|---|---|---|
| C8 | `layers` | 前端框架 > `experiment` > 能力模組 |
| C9 | `acyclic_siblings` | `zcu_tools` 的頂層 package 之間沒有循環 |
| C10 | `independence` | `program`、`resources`、`device`、`qick_remote` 彼此獨立 |
| C11 | `forbidden` | `cfg_model` 只依賴 `utils` 與 `core.waveform_format` |
| C12 | `forbidden` | 穩定工具包含 core，不依賴有狀態模組 |
| C13 | `forbidden` | `mcp` 只使用 `gui.remote` 與各 app 的 wire spec |
| C15 | `forbidden` | 使用者實驗的 core 不依賴 GUI framework |
| C16 | `forbidden` | `zcu_tools` 不 import `zcu_lab` |
| C19 | `forbidden` | `core` 在 repo 內只依賴 `utils` |

`zcu_tools`、`zcu_lab`、離線 `migrations` 與 repo 的 `mcp_servers` 都是 import-linter 的 root packages。只為離線遷移存在的程式與對照表放在 `migrations/`，`zcu_lab` 與 `zcu_tools` 不 import 它，遷移完成後可整個刪除。使用者於 2026-10-14 裁定：「我自己會預期，我遷移完後刪除migrations就可以了」。組合根顯式注入 measure registry、workflow 註冊清單與 MCP recipes，framework 不反向載入使用者定義。Tests 的 lab 獨立性由 review 檢查 import，必要時以 grep `zcu_lab` 輔助，不新增 tests import contract。

債務以確切的 module 對逐條列在 `ignore_imports`，不用萬用字元，每組註解寫明「債務，由誰移除」。舊 cfg_editing → gui.cfg 例外隨兩個模組退休而移除，不是可仿照的依賴。`ignore_imports` 只減不增。

## 債務

| 債務 | 移除方式 | 記帳位置 |
|---|---|---|
| `program → resources` | `program` 改依賴 `cfg_model` 的 Protocol | C10 `ignore_imports`：`program.v2.modules*` → `resources.context` |
| `simulate → resources` | `predict` 改由呼叫端傳入數值 | C10、C12 `ignore_imports`：`simulate.fluxonium.predict` → `resources.qubit_params` |
| `notebook/analysis` | 通用分析移到 `analysis`；實驗專屬 helpers 已由 `zcu_lab` 附件擁有 | 尚未搬移 |
| `utils` 的 `shot_classification`、`tomography` | 移到 `analysis` | 尚未搬移 |
| `analysis/fluxdep/io.py` 直接用 h5py | 檔案讀寫移到 `datafile` | 尚未搬移 |

已清除的債務不再列出：`resources → program`、`experiment/base.py` 的 Labber 格式處理（已改用 `datafile`）、`utils/datasaver/`、`mcp → gui.logging_setup`。

## 後果

- 增減功能時依賴圖的形狀不變：工具之間的邊很少變，有狀態模組之間沒有邊。兩個有狀態模組要合作，做法固定是 Protocol 或上移組合。
- 新增跨 package 依賴會被 contract 擋下或在 review 中出現，不會靜默成為先例。

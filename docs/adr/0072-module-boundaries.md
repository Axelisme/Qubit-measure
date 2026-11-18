---
status: accepted
---

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
| `program` | 控制 ZCU216 與底層 QICK 的工具和類別，包括由呼叫端 profile mapping 建 QickConfig 的 `build_soccfg` 與模擬 SoC binding；時脈、channel 型別與數量等硬體 profile 數值不留在 lib | 參數與資產的保存 |
| `analysis` | 分析工具；模型自己的公開 renderer，使用與客製化由 caller 決定 | 檔案讀寫；實驗的圖形政策 |
| `plotting` | 具名 Figure、owner 更新機制與共用診斷圖 | 分析計算；lab 的 live 工具與格式 |
| `simulate` | 數值模擬、通用數值輸入與參考解 codec | 讀取量測資料容器、裝置佈局或使用者環境參數；模型數值由呼叫端提供 |
| `simulators` | 具體物理模擬器（FTF、single_fluxonium、six_fluxonium）的模型與 environment | 通用演化、ADC、bath、元件 Hamiltonian 等機制，留在 `simulate` |
| `resources` | 實驗領域資料容器與 YAML、JSON、CSV 的讀寫；waveform 資產的命名、formula 語意與 NPZ 實體 I/O | 量測資料檔；program 的執行機制；重複 waveform codec |
| `core` | 共用的無狀態型別與格式。`core.waveform_format` 驗證 samples、編解碼 NPZ 與 opaque JSON recipe | 檔案 I/O、formula 語意、GUI、資源 scope 或硬體 |
| `cfg_model` | 配置 schema 與注入依賴契約，見下節。搬進 core 尚未排定 | 執行行為與 I/O |
| `utils` | 無法歸入任何其他類別的雜項 | 任何能歸入其他類別的東西 |
| `progress_bar`、`qick_remote` | 照字面 | — |
| `experiment` | 實驗框架：實驗介面、AxesSpec、RunRecord、RunContext、runtime 與註冊機制 | 檔案格式；具體實驗 |
| `gui` | 各 GUI app 的框架 | 具體實驗的 adapter |
| `notebook` | Notebook 特化的工具集合 | 通用分析；具體實驗 |
| `mcp` | 各 MCP 的框架 | 具體 recipe |
| `apps`、`mcp_servers`（repo 根目錄） | GUI 與 MCP 的組合根，注入 lab 定義與 GUI module 名稱 | framework 與 lab 的可重用實作 |
| `zcu_lab`（repo 根目錄） | 具體實驗、recipes 與其 gui、notebook 附件 | 離線遷移的對照與宣告（放 `migrations/`） |

放置新程式碼時，先找最貼近的類別；所有類別都不合，才放 `utils`。

FTF 與 single_fluxonium 的除錯資訊由 debug capture 提供：開啟 `numerics.debug_capture` 時，manifest 記錄 Engine 輸入、逐列 routes、seed、每輪 DC、model identity 與該 identity 當時的有效設定。除錯需要新的資訊時，擴充 manifest 的內容。2026-11-18，使用者說：「你需要的资讯应该本身就在 dump 的结果里了。如果你会需要 replace 多余的资讯、多余的算法，那就是你 dump 的资料不够多嘛。」出處：[移除 private replay](../discussions/2026-11-18-remove-private-replay.md)。

### 框架機制與模型參數

2026-11-01，使用者要求：「框架應該只有機制本身以及最簡單的範例，比如white_noise，絕對不可以hardcode參數」。

板子 profile 由呼叫端 YAML 提供獨立設定。使用者對此來源選項的原話是「b，模擬器就是要能設定模擬參數」。

具體物理模型放在新增的 `zcu_tools.simulators` package。使用者對此位置選項的原話是「A」。

### 模擬器是封閉的子模塊，模型由使用方聲明

`simulate` 不 import `zcu_tools` 的其他 package，其他 package 可以依賴它。
前端與 environment 共用的 acquisition 邊界型別定義在 `simulate.boundary`，包括 `TimeBase`、`AcquireRequest`、`RoundRequest`、timeline 與 slot、signals、statistics 與 `ChargeDriveRequirement`。
Numba 機器碼快取的根目錄由 `simulate` 自己解析，規則與 `utils.cache.user_cache_root` 相同，所以快取目錄不隨封閉搬家。
`.importlinter` 的 `simulate-closed` forbidden contract 以 `zcu_tools.simulate` 為 source，明列其他所有頂層 package 與 `format_version` module，間接 import 也算，不設 `ignore_imports`。
新增頂層 package 時，同步把它加進這份清單。
C19 照舊禁止 `core` import `simulate`；舊位置 `core.simulation` 已刪除，不保留 re-export。

`simulate` 提供聲明模型的介面，具體裝置模型由使用方經這個介面聲明。
例：`simulate.evolution.ModelSource` 收使用方給的 `build: Callable[[FluxLevel], SecularModel]`，這是依賴注入。
只收物理參數、不綁裝置佈局或接線的元件屬於 `simulate`，例如 `simulate.fluxonium.SingleFluxoniumModel`（`EJ`、`EC`、`EL`、DC flux、截斷、溫度）、`simulate.transmon` 與 `simulate.resonator_mode` 的 resonator 模態 builder。
裝置怎麼組合屬於使用方，例如 FTF 的耦合與 resonator 配置、six 的路由；這些今天在 `simulators`，定位見〈模組定位〉。
三個 environment 不共用基底類別或 acquire 流程，因為 FTF 的腔記憶與 six 的 scatter 會變成基底上的可選 hook；它們只共用中性的 `simulators.telemetry` writer 與欄位型別，以及 `simulate.evolution` 的錄製 buffer，見 [ADR-0078〈模擬器帶通用調試介面〉](0078-evolution-module-boundary.md#模擬器帶通用調試介面)。
模擬器的量測工具與測試只依賴模擬器，收模型聲明當輸入；量真實模型時由使用方那側的入口注入，工具本身不 import 具體模型。

邊界：測試用的合成模型（例如兩能階模型）是照介面聲明的範例，可以和 `simulate` 自己的測試放在一起。
這條規則只管 `simulate` 的依賴方向與元件放置，不改 `simulators` 的定位，也不改〈使用者空間的依賴原則〉中入庫工具與測試不讀使用者環境參數的規則。

2026-11-15，使用者說明 package 方向：「先告訴你，我預計將模擬器功能獨立出zcu_tools成獨立的package，後者依賴前者，目前作為封閉子模塊可以」。
對模型放在哪一側，使用者答：「C，模擬器應該提供一個方法讓使用方透過繼承或是組合或是依賴注入的方式讓使用方聲明模型」。
對 `SingleFluxoniumModel` 放哪，使用者答「1A」，即只收物理參數的元件留在模擬器。
對原則與預測（量真實 FTF 時由使用方把模型聲明注入通用量測工具，工具不 import FTF），使用者答「A」。
整包獨立成 package 是尚未建的方向，見[模擬器獨立成 package](draft/simulator-standalone-package.md)。
討論與選項見[模擬器封閉成子模塊](../discussions/2026-11-15-simulator-closed-submodule.md)；搬遷的切票順序見[封閉 simulate 的搬遷順序](../discussions/2026-11-17-simulate-closure-expand-order.md)。

### 真實 acquire 與模擬邊界

真實硬體的 acquire 路徑不執行任何模擬模組，包含 `simulate` 與 `program.v2.simulation` 的排程器。共用時間成本計算放在中性的 `program` 工具，不讓真實 acquire 為了估時走模擬排程。2026-10-26，使用者裁決：「我不太希望真实 acquire 还会走模拟的模块。」

Charge admission 是 caller 對 generator 必須綁定 charge drive line 的明確要求。Caller 以 `simulate.boundary.ChargeDriveRequirement` 傳入診斷名稱與 generator index，不查 environment settings 或判斷 physical membership。名稱不是 role registry 的 key。Program 的模擬入口只把需求放進 `AcquireRequest.charge_requirements`；`simulators.replay` 保存同一個 boundary record，不重做物理驗證。

知道實體線路的 environment 擁有驗證。Single-fluxonium 每次 acquire 都在 preflight 檢查 `drive_lines`，早於 waveform 建構、preparation reuse 與 evolution。Generator 已綁 flux 或 readout 也不能滿足 charge 需求。這不改變其他合法 fast-flux 事件的路由。

硬體入口取走這個 keyword，不轉送 QICK、不建立 simulation request，也不執行此物理驗證。Operator 負責核對實機線路。`white_noise` 沒有物理線路模型，各 generator 只產生雜訊，因此接受需求而不驗證。未支援 charge admission 的物理環境收到非空需求時，在 evolution 前明確拒絕；six-fluxonium 與 FTF 目前採此行為，空需求維持原有流程。不以清空需求、猜測 waveform 角色或擴充共用 `QubitWiringSchema` 代替驗證，也不新增通用 protocol 或 `RunContext` capability。

2026-10-30，advisor 依使用者代理授權「可以按需排task，我授權你代理」裁定上述分工。Advisor 的技術裁定原話為：「驗證 owner 是 single_fluxonium 環境，在既有拒絕 unbound generator 的 preflight 位置一起檢查，早於任何 evolution。」

邊界 record 沿 program／experiment／simulators → `simulate.boundary` 的方向傳遞；`zcu_lab` 的 STIRAP 實驗也從這裡取 `ChargeDriveRequirement`。
真實硬體上的實驗因此會 import `simulate.boundary` 的 record 型別；import 型別不執行模擬，`simulate` 的 package 初始化只載入 numpy。
C8 與 C9 維持分層與無循環，C12 禁止 `simulate` 依賴有狀態模組，`simulate-closed` 禁止 `simulate` 依賴其他任何 `zcu_tools` package，見〈模擬器是封閉的子模塊，模型由使用方聲明〉。
`program-simulation-entry` contract 限制 program classes 直接依賴 simulation frontend、timeline 與 adapter internals，只能經模擬入口使用。
這些 contract 是依賴檢查，不是 runtime 隔離的完整保證；真實 acquire 不執行模擬模組，仍須在審閱執行路徑時確認。

### Mux PFB channel 查詢與 carrier selection 的 owner

`program.v2.mocksoc.pfb_channel` 對呼叫端提供的 QickConfig 查 PFB channel，包裝 QICK 公開的 `calc_ro_freq` 並取其 `pfb_ch`。FTF 只做 event／carrier selection，不組 QICK register 結構，也不複製 channel folding 算式。這讓 QICK 的 channel assignment 留在 program owner，而非由 experiment 重做。

依賴方向沿用 C8、C9 的 `experiment` → `program`，不新增或放寬 import contract。

2026-10-31，使用者核准 mux 濾波：「Mux的可以」。上述 owner 邊界的細節由 advisor 依使用者「1代理模式就都給你決定，我事後看就好」的代理授權裁定，不是使用者逐字指定的技術方案。

### Manifold 衰減模與分析診斷的 owner

`simulate.fluxonium.coherence` 負責組合 directional rate matrix 並計算衰減模。
Manifold forward 每個 flux 點回傳所有衰減模，型別為 `list[list[Mode]]`。
`Mode` 只有 `t1_ns` 與 `amplitude`，清單按 signed amplitude 降序排列。
Simulate 不選主模，不回傳 secondary fraction，也不套用非單指數標記門檻。

`notebook.analysis.t1_curve` 選振幅最大的模作為 model T1，計算次模佔比並標記圖表。
這些是分析端的選取與呈現政策，不放進數值工具。兩種 T1 的物理定義與共同單位見 [[0071]]。
依賴方向沿用 import-linter C8 的 notebook → simulate，無環由 C9 檢查；不新增或放寬 contract。
局部輸出與失敗契約由 [coherence README](../../lib/zcu_tools/simulate/fluxonium/coherence/README.md) 與
[t1_curve README](../../lib/zcu_tools/notebook/analysis/t1_curve/README.md) 擁有。

2026-10-28，使用者提出「我覺得工具就應該保持單純」。Advisor 復述為 simulate 回傳所有模、
t1_curve 負責主模選取與診斷，使用者回答「對」。這個分工避免讓呼叫端的分析政策成為數值工具的回傳契約。

### 模擬損耗漂移的數值、設定與 truth owner

純 log-OU 數值類別 `LogOU` 位於 `simulate.drift`：只接 mean、sigma、tau_s 與 RNG，依 caller 提供的時間算精確轉移，不讀 YAML、clock 或檔案。
漂移設定型別 `DriftParameter` 與 schema `CoherenceDrift` 由 `resources.single_fluxonium.schema` 擁有，single、six 與 FTF 共用；resources 不依賴 experiment。
設定編排、acquisition 開始時的取值凍結與 truth log I/O 留在 experiment：`experiment.single_fluxonium.drift` 的 `LossRow` 與 `write_truth` 是 single、six、FTF 共用 truth JSONL 的唯一格式與寫入 owner，
FTF 呼叫它而不複製格式；coupler 沒有的 q_ind／xqp 以 null 表達。每個環境自己組 effective settings 與 loss streams，FTF 不建 single settings 的 adapter。
Truth 是模擬端 acquisition 的診斷資料，不進 Run record、GUI、MCP 或 Labber metadata。FTF 與 single-fluxonium 開啟 `numerics.debug_capture` 時，manifest 的 `metadata.effective_settings` 按已觀察的 round model identity 記錄當時的有效設定，包含取樣後的損耗。發布使用 round snapshot 對應的記錄。
依賴方向沿用 C8、C9 的 experiment → simulate／resources，不新增 package edge。
物理前提見 [[0078]]。使用者於 2026-10-27 提出漂移需求：「另外我想引入跨acquire的T1/T2漂移，這樣overnight量測才有意義」。
Owner 分工是 advisor 於 2026-10-27 與 2026-10-28 依本篇 simulate 與 experiment 定位、以及使用者授權「B你可以自己開task，不用我看過」訂定。

### 實驗計算與 GUI 接線

core.py 的 Experiment 擁有 run 與 analyze 的計算；gui.py 只接線、解析來源、決定 writeback 與 operator guide。
分析計算包含擬合、逐 tone 迴圈、失敗隔離與結果型別。
GUI 為 writeback 決策做的有限值檢查與摘要整理不算分析計算。
使用者於 2026-10-24 裁決：「怎麼會有分析寫在gui ,這是違規吧」「分析修正可以做」。
mux/onetone 的逐 tone 數值計算、outcomes 與單 tone 失敗隔離由 core 擁有。
gui.py 保留來源與 route prior 解析，以及 writeback。
此規則以直接審閱確認，不新增靜態內容測試；core 不依賴 GUI 的方向仍由 C15 保護。

Analysis 回傳值是 lab 實驗交給 measure GUI 的摘要，大小最多隨掃描點數 O(log N)。
隨掃描點數或 shot 數成長的陣列只在 `analyze` 內畫圖用，不回傳。
`summarize_analysis` 在分析完成、發布結果前檢查；任一 ndarray、list 或 tuple 攤平後超過 64 個元素就 raise `ValueError`，不截斷。
64 是檢查上限，不代表小於 64 的 O(N) 回傳值就符合摘要契約。
數值 helper 回傳完整計算結果，core `analyze` 用完整結果畫圖後只回小摘要，例如 `RabiJointFitSummary`、`Centroid3Summary`。
摘要由數值 helper 的 owner 定義，GUI 不另設一層摘要型別，也不替 core 重算或重跑 fitter。
2026-11-02，使用者對大陣列進 JSON 指出「為什麼會把很大的數據用json.dumps？這不合理」，並裁定「summarize_analysis可以檢查結果的大小，太大就報錯。summarize請不要放隨著掃描參數成O(N)以上的結果。最多O(logN)」。
對 64 元素上限與修改 core 的選擇，使用者回覆「64可以，CPMG選a，core」。

各實驗在自己的 core 取數，不呼叫另一個實驗的 `run`。
Mux flux 各 leaf 自行建立 program、執行 flux 取數迴圈，不以帶旗標的共用 run 分派不同實驗。
共用工具可處理數值、設定 flux 與組裝 Result，不隱藏各 leaf 的量測迴圈。
2026-11-02，針對「各自取數」或「維持呼叫 spectrum 實驗」的選擇，使用者裁定「a，我想討論的是initial_values與Form.default_input的定位問題」，採用各實驗獨立取數。

Workflow 的替代壽命模型比較與疊圖由 `zcu_lab/workflows/common` 擁有。Mux T1／T2echo core 只回本次選定模型的擬合，不為 workflow 額外比較模型。使用者於 2026-10-30 選擇「18A」，將這項診斷留在使用它的 workflow。

Mux onetone core 只接收每 tone 的 edelay 種子並回傳擬合結果。Adapter 決定手動值、capture prior 與全域搜尋的優先序，記錄來源並判斷 writeback。使用者於 2026-10-30 指出「這不對勁，聽起來像是adapter的邏輯漏進core」，並選擇「A」。

各實驗獨立完成 analyze 與畫圖，不組假 source 呼叫另一個實驗的 analyze。實驗可共用數值分析工具，不共用分析圖的畫法。使用者於 2026-10-30 裁定「不能獨立重複分析嗎？可以共用分析工具，但畫圖就有點超過了，不利於客製化」。Mux GE 因而逐 tone 呼叫既有數值工具並自行畫圖。

2026-11-12，使用者確認以下分工：「確認」，並說明「實驗間重複算是偶然耦合而不是實際耦合。」
含 `core.py` 的實驗目錄不 import 其他實驗或其他家族的家族層模組。自己家族的 helper、`support` 與家族 `__init__.py` 匯出子實驗不受此限制。
Cfg、分析選項、結果型別、小訊號轉換、GUI 選項與 review 文字由各實驗宣告，欄位只留自己使用的。宣告放在同一實驗內使用它的檔案。繼承其他實驗 cfg 也改成本地宣告，保留原本經繼承生效的 validator，除非另有行為變更裁決。
多個實驗共用的資料數值分析放 `zcu_tools.analysis`。認得 lab 型別的邏輯放家族私有模組，跨家族時放 `zcu_lab/v2/support`。只有單一家族使用的序列產生與查表留家族層，不視為資料分析。
即時圖也由各實驗的 `run` 擁有。Helper 換 owner 後，所有呼叫者直接 import 新位置，不留舊路徑 re-export。

2026-11-12，使用者對數值與繪圖分離及公開 renderer 的例外裁定：「接受」。
`support.resonator.fit_resonance` 只回傳數值擬合結果。`onetone/freq` 與 `mux/onetone` 各自在 `analyze` 持有 Figure，決定何時畫圖與是否使用模型 renderer。
不共用分析圖，指不在 lab／support 放替多個實驗建圖與畫圖的 helper。Lib 擬合模型自己的公開 renderer 是模型介面，可以使用，不複製它的內容，也不呼叫模型私有方法。

2026-11-12，使用者對 GE confusion 的 owner 與介面裁定：「可以」。
GE confusion 的數值計算由既有 `zcu_tools.analysis.singleshot` 擁有，與分類及 population correction 放在一起。Lab 各自從自己的分析結果組成 `GECalibration`，不把 lab cfg 傳入數值層。
分析選項的預設值、lab 結果與存檔格式、GUI 文字及 confusion 圖留在各實驗。Lib 只收明填的選項，介面細節由 [analysis README](../../lib/zcu_tools/analysis/README.md#子模組) 與函式 docstring 擁有。

以上分工沿用 lab → analysis／plotting 的依賴方向。Framework 不反向 import lab 由 C16 保護，core 不依賴 GUI 由 C15 保護，分層與無循環由 C8、C9 檢查。

Local flux reset 沿用 `reset/pulse`，由 lab 的 library 範本與 `twotone/reset/flux/{gain,length}` 實驗組裝及校準。
所需脈衝已能由既有 reset 模組播放，因此框架不新增 reset 型別。驗收重用既有 reset check，不另建專用檢查實驗。
範本名稱與參數留在 lab 範本及 [gain](../../zcu_lab/v2/twotone/reset/flux/gain/guide.md)、[length](../../zcu_lab/v2/twotone/reset/flux/length/guide.md) 操作文件，不成為框架規則。
2026-10-30，使用者選擇：「1A 2同意 3感覺復用reset/check就好？倒是reset/可以加一個flux_reset，4同意」，並對上述組裝與校準方式回覆「確認」。

### Two-tone 廣掃的選峰與窗口

`twotone/freq` 只暴露 Model type、Expected center 與 Min SNR，預設 SNR 為 5。
數值選峰及窗口由 `analysis.fitting.spectrum.fit_spectrum` 的明填 `local_peak=True` 提供。
Lab 保留選項宣告、IQ 投影與實際窗口的圖形標示，不讓數值工具讀 lab options。
有指定中心時取附近最高局部極大值，不以全譜 range 比例拒絕弱峰；空白時取最顯著峰。
窗口左右各取六個半高寬，鄰峰中點可裁切，但至少保留最近六點。
例如廣掃同時包含強 transition 與高度不到它 0.2 的弱 transition 時，Expected center 可指定弱峰。
Min SNR 仍可拒絕弱峰，失敗時回報候選數及門檻，不改擬合遠處強峰。
使用者 2026-11-18 過目此條：「我大概看了一遍，基本上是没问题」，並說明這類規則「并不算是裁决，而算是建议」；它是 agent 的選擇，引用時視為僅供參考的先例。

此規則只適用 `twotone/freq`。`SpectrumFitOptions` 保留原欄位，`twotone/chain` 保留最近候選與中點窗口。
Notebook fluxdep 與 workflow calibration 的獨立分析不改行為。
這些 caller 若需要別的門檻或窗口，仍自行明選，不繼承 freq 的 GUI 選項。
依賴方向沿用 lab → analysis，由 C8、C9、C15、C16 保護，不新增依賴邊。

2026-11-16，使用者指出：「你twotone/freq的算法不好，對於廣掃，多transition peaks是很常見的，gui控制應該只需要model type/expected center兩個就好。」
對窗口與門檻補充：「取6個半高寬。其他門檻，如果預設可能會導致找不到峰，那要考慮曝露。」
使用者對保留 Min SNR、移除相對 prominence 與固定其餘兩門檻為零的提案回答「A」。
本次小改動依使用者要求不另建討論快照，操作契約見 [frequency guide](../../zcu_lab/v2/twotone/freq/guide.md#peak-selection)。

### 即時圖工具與機制

使用者於 2026-10-24 裁定：「框架與機制留在lib可以，live_line的定位應該是工具，可按需求擴充，但放在lib裡很難改。以及liveplot模塊應該可以考慮刪除，我以為plotting已經取代他了。」

Lib 保留 Plots、FigureCollection 與 PlotHost。2026-11-01，通用更新已遷移至 owner decorator。作者用不帶參數的
`@plots.on_plot_thread` 包裝原生 matplotlib 更新函式，呼叫時先檢查 lifetime，再 deepcopy
資料，並在 host owner 執行本體。本體回傳改了的 artists，Figure 也可回傳。框架先驗證
所有 artists 的歸屬，再對已由此 Plots 呈現的根圖去重 refresh。None 回傳 raise TypeError，
外來圖 raise ValueError，finish／release 後拒絕呼叫。本體失敗不 refresh，也不回滾修改。

即時更新與一次性改動共用這個 decorator。圖顯示後的改動一律經此入口。
`fig.show()` 顯式呈現；show 前的更新只保留 artist 修改，不 refresh、不載入 ipympl。
之後 show 或 finish(present=True) 呈現最新內容。混合回傳只 refresh 已呈現的圖，
未呈現的外來圖仍拒絕。Notebook 與 Qt host 的 refresh 前置條件不變。
2026-11-01，全部實驗遷移完成，LiveUpdate、字串圖名的 draw 與舊 helper 已退休。框架不決定資料格式，也不反向 import lab。

使用者於 2026-10-30 裁定：「畫圖採用建議，但名字叫live不太好懂」、
「@plots.on_plot_thread好了，隱藏執行線程的機制資訊可能在某些時候造成難以發現的bug」。

三個 live helpers、typed 工具 handle 與線圖、熱圖、非均勻熱圖、scatter 格式放在 lab
共用支援套件的直接層。實驗作者可以直接修改或擴充，不另建 plotting 子套件。
`live_line(ax)`、`live_heatmap(ax)` 與 `live_scatter(ax)` 是本地選用工具，只掛在給定 axes，
在 caller thread 修改並回傳 changed artists。不建圖、不設標籤與標題、不切換執行緒、
不 refresh、不 present。作者可在 decorator 本體內回傳 `line.set(xs, signals)`。
這些工具與 decorator 不登記 axes，不檢查彼此的佔用。
多個 helper 可共用 axes，由作者協調 artists、autoscale 與色階。
新版面由作者用 subplots 排。2026-11-01，舊 helper 遷移已完成。
使用者於 2026-10-30 裁定：「我也傾向後者，live_line只是本地執行緒的」。
Singleshot 分類圖與 joint-fit 診斷圖歸 lab 的 singleshot 家族，分析計算不隨圖搬移。
獨立 liveplot plotter／backend、reference marker 與動畫錄影退休。
Fluxdep 所需非均勻背景由自己的私有 renderer 擁有，不依賴 lab 工具。

這個分工讓 decorator 使用 host 的 owner 執行與 lifetime 機制，
新 helper 不綁 Plots lifetime。實驗格式仍可在 lab 修改。2026-11-01，全部實驗已完成此分工的遷移。
依賴方向是 lab → plotting；framework 不 import lab 由 import-linter C16 保護，
能力層與使用者層的方向沿用 C8，core 不依賴 GUI 沿用 C15。

### 分層

由下往上五層。上層可以依賴下層，下層不知道上層。

1. 能力模組：`datafile`、`device`、`program`、`analysis`、`plotting`、`simulate`、`resources`、`core`、`cfg_model`、`utils`、`progress_bar`、`qick_remote`
2. 實驗框架：`experiment`
3. 前端框架：`gui`、`notebook`、`mcp`
4. 使用者空間：`zcu_lab`、`scripts`。使用者撰寫內容並安排結構。
5. 組合根：GUI 在 `apps/`，agent MCP 在 `mcp_servers/`。兩者同層且互不 import，可依賴 `zcu_tools`；對 lab 只直接 import `zcu_lab.assembly`，以 `lab_definitions()` 取得宣告。MCP 以注入的 module 字串啟動 GUI，不 import GUI 組合根。使用者在 2026-10-25 選擇「乙」的問題脈絡與決策見 [[0068]]。

`tools/` 是開發工具，對 lab 同樣只直接 import `zcu_lab.assembly`；量表工具 `tools/family_pareto*.py` 不 import lab。
`migrations/` 轉換 lab 實驗保存的舊資料，可以直接 import 實驗與元件，不受 assembly 入口限制。

`scripts/` 的每個 Python 檔案都是可獨立執行的使用者入口，不放被其他入口依賴的實作套件。
可重用實作放在 `lib/zcu_tools/`，依上表定位；rclone 外部資料夾傳輸不屬於量測資料檔格式或實驗容器，
放在 `utils.result_transfer`，由 `scripts/transfer_results.py` 注入 checkout 的本地結果根目錄。
`lib` 不反向依賴 `scripts`，由 import-linter C21 保護。
使用者於 2026-10-19 裁定：「不要把依賴放在scripts, 後者的每個檔案都該被理解為獨立腳本入口，需要用到的內容可以放lib裡」。

- `zcu_tools` 不 import `zcu_lab`。定義由組合根注入，框架不反向載入使用者實驗。
- 框架 package 不得 import `simulators`，由 import-linter `framework-no-simulators` forbidden contract 保護。
- tests 不依賴使用者空間 `zcu_lab` 與 `scripts`，不 import、不綁定實驗流程或預設值。`tests/scripts/` 既有的腳本 `main` import 是待移除債務。

框架測試使用最小合成 workflow／adapter，不綁定實驗流程或預設值。Tests 不 import `zcu_lab`，也不增加個別實驗的 durable 測試。使用者於 2026-10-12 裁定：「我給你一個判斷方法，如果我刪除zcu_lab測試會報錯，那測試就是寫錯了。」

使用者於 2026-10-16 改定驗收方式：「算了，我認為不用真的測試ZCU_LAB移開後會不會出現錯誤。這中東西直接看有沒有import就好，偷偷做的在review也會被抓到，感覺是浪費時間測試」。Tests 不依賴 `zcu_lab` 的原則不變，由 review 檢查 import，必要時以 grep 輔助。不再暫移 `zcu_lab/` 跑 pytest，也不新增 `ZCU_LAB` 環境變數或 import hook。

實驗定義（`zcu_lab`）的改動，包含畫面改動，使用 `experiment` 層的共用離線 probe 驗收。`experiment.probe` 提供三個獨立階段 `run_stage`、`save_stage`、`analysis_stage`，各回具名結果。run 與 analysis 回 PNG bytes，不寫磁碟。`save_stage` 寫 native `data.h5` 並 restore。三階段無 CLI。run 與 analysis 分開驗收，不組 e2e。合成輸入用 `synthesize`（任何回 ArrayLike 的模型）或 `sample_iq_shots`（raw IQ）。Agent 看圖並記錄判讀，probe 腳本與結果留在 task scratch，不 commit 成 durable 測試。框架 UI 改動仍使用 ComponentDriver 情境。操作與完成條件由 [UI 驗證](../../tests/README.md#ui-驗證) 擁有。

2026-10-30 使用者提出：「關於測試實驗定義的方式，用ui跑e2e有點太重量級了，我認為可以修改規則，透過建立context的方式測試run與analysis」。當時比較 A（只改驗收規則）、B（改規則並提供共用 probe）、C（再加 `zcu_lab` 自有的 durable 測試）。使用者回答：「B」。採用共用 probe，讓各實驗只需提供實驗與 cfg，不再各自組裝離線驗收流程；durable lab 測試不在此決定範圍。

2026-11-01 使用者「run跟analysis本來就該分開測試，不要有e2e的測試」，stage-tests（main 2f5e933c3）實作。

### 使用者空間的依賴原則

GUI、MCP 組合根與開發工具需要 lab 宣告時，只直接 import `zcu_lab.assembly`。
例如 `apps/measure.py` 呼叫 `lab_definitions()` 取得宣告，不直接 import `zcu_lab.v2.twotone.freq.core`。
這讓 lab 可以在 assembly 後方調整實驗、元件、recipe 與 workflow 的佈局。
`migrations/` 是例外，因為它必須認得舊資料所屬實驗的格式。若遷移工具改成供不同 lab 共用的框架，改用 assembly 消費宣告，不保留此例外。
這條規則不保證刪除 lab 後現有 lab 組合根仍能啟動，也不引入位置字串或 Python entry point 註冊。

2026-11-14，使用者說：「其实应该是 B， 因为我对这个 repo 的期望是，我把 [zcu_lab] 删掉，也可以正常运作。而 [zcu_lab] 是使用者自由的、控制的、写的、安排结构的模块。」
接著澄清：「我的意思是 ，Script 的定位与 [zcu_lab] 比较像，而 Tool 的定位是用于开发。我不确定该不该让它依赖 Lab。MCP 需要发现、实验，而也确实是需要一种正规的方法来宣告你有哪些实验。我不确定要怎么好的结合两者，让 MCP 尽量不依赖[zcu_lab]。」
使用者對註冊機制的取捨是：「但是呢，我又不想把注册逻辑搞得太复杂，导致实作行数太多。有简单的方法吗？还是我就放弃？我就是让它可以依赖，但是只依赖局部的部分的档案。」
使用者對 assembly 入口限制與 migrations 例外回答：「好，甲」。引文中的 `[zcu_lab]` 代替語音輸入的 lab 名稱。

量表工具的場景是挑選過、固定的一組，核心 pulse 序列與模型參數放在工具內，不 import lab，也不讀可變的 lab YAML。
`tools/family_pareto_scenarios.py` 宣告固定序列，經 `tools/family_pareto_scene_builders.py` 的公開 program 編譯與資源生命週期，凍結完整 batch。
固定裝置模型值由 `tools/family_pareto_settings.py` 持有。
`tools/family_pareto_kernel_scenes.py` 則用公開 `SingleFluxoniumModel` 直接宣告單段數值控制，不經前端。
通用 codec 歸 `simulate.evolution`，數值資料不包含 solver 或裝置 builder。
凍結場景的數值不入庫。2026-11-16，使用者要求：「檔案大就請不要入庫，用類似快取的方式來減少重複計算，記得考慮跨worktree也能復用的方案」；對範圍與釘住方式答「1a 2a」，即所有凍結場景都不入庫，Git 只留名稱與 hash 的小清單，重建結果與清單不符就停，只有明確更新清單才算數；對存放提案答「R」。
數值放在 repo 共用快取，每筆 record 一個單筆檔，各 worktree 共用，位置與檔名見 [Tools README](../../tools/README.md#freeze-numeric-scenes)。`simulate` 擁有清單與單筆檔的讀寫、hash 核對與原子發布，只收顯式路徑；Git 位置的解析與覆寫留在 tools。Git 只留 `tools/data/family_pareto/frozen-synthetic.index.json` 與 `frozen-real.index.json` 兩份清單。
量測工具與 tests 側參考解 producer 只讀清單與快取，缺檔就列出 producer 指令後失敗，不自建，因為自建會讓量測工具 import 場景 producer 與 `program`。每個凍結場景都要有入庫的 builder。
Native thread 設定會改浮點末位，所以快取遺失而重建無法逐位元重現時，只能明確更新清單並重算受影響的參考解。
選項比較見[凍結場景存放](../discussions/2026-11-16-frozen-scene-storage.md)。
Report 身份使用每筆 name、physical_sha256、oracle_sha256 與 metadata_sha256，不隨存放方式改變。
Producer 自持具名 `Scene`，不依賴 tests 的案例型別或清單。獨立參考解 producer 留在 tests 側，經檔案交接，契約見 [[0078]]。
此處接受與 lab 重複，讓讀工具的人可以直接看懂場景；lab 序列改動不會自動更新已凍結的量表輸入。
這個取捨只適用於固定的 optimizer benchmark。需要真實 lab 宣告的品質檢查仍經 assembly 取得宣告，不複製一份。

2026-11-12，使用者說：「我其實不樂意看到tool依賴zcu_lab，我認為你要測試，那就只複製最核心的pulse序列來模擬啊，又不需要整個experiment」。
對工具內建場景的方案，使用者回答：「你這個工具的目的是對優化器做benchmark吧？那場景自然是挑選過的」，並補充：「而且這樣tool也能提高可讀性，我寧可重複也要提高可讀性」。
2026-11-15，使用者進一步說：「但腳本也不應該依賴於我的模擬器環境參數設定，而是自己為每個場景設定的固定環境」，並界定範圍：「研究的话就没差，都可以用。但是如果你要持久的工具的话，麻烦请不要依赖于作者可能会改到的东西。」
因此入庫、會持續使用的模擬器量測工具與測試不讀 `zcu_lab/simulation_envs/` 下的環境參數，每個場景的模型值由工具或測試自己固定持有；一次性研究可以讀已提交的環境參數。這條只管入庫的工具與測試，不改「Lab 只經 assembly 入口被依賴」的規則。
兩次裁決的比較與先例立場見 [Lab 依賴邊界討論](../discussions/2026-11-14-lab-dependency-boundary.md)。

### 開發工具的數值交接

`tools.family_pareto_reference` 持有量表選定輸入的整批 cache-only preflight。
它呼叫 `simulate.evolution.reference_key` 與 `load_reference`，不另寫 key 公式，也不求解 miss。
每筆缺失的名稱與 key 一次列出，全部載入成功前不開始 policy 或 Engine。
`family_pareto` 的 check 與量測共用此載入函式。獨立 solver 與 cache 生產留在 tests 側，見 [[0078]]。

`tools.family_pareto_results` 持有結果型別、誤差、資格與 dominance 規則。
Runner、report 與 summary 讀同一個 owner，不由 report 反向 import 執行入口。
選取、reference preflight 與 check 的結束點留在 caller，計時協定只消費已備好的 session。
`tools.simulator_measurement` 是 mode transition、暖機、history 策略與 wall/process clock 的共同 owner。
它只接 caller 準備好的 session callbacks，不 import 場景 producer、solver 或 tests。
`family_pareto.measure_family` 必須提供獨立 reference，並將 owner 的 observations 轉回既有量表 schema。
計時只包一次 `Engine.evolve`，建構、history reset、暖機、誤差與 telemetry 整理都在 timer 外。
Elapsed 秒數須有限且非負。Clock 未在解析度內前進時，0 是合法讀數，原樣記錄，不補 epsilon 或重取。
負值、NaN 與 infinity 在 sample 建構時失敗，process CPU 不必小於 wall。
共同 owner 不處理跨 worktree 協定或統計，兩側比較須先含相同 committed 工具。
`tools.paired_benchmark` 擁有 controller、child、ABBA 統計及 telemetry 校準。
它直接管理 child，每側 warm 長駐，first/cold 每筆新 process，不設轉送層。
每個 first child 只量一筆，不套 Pareto 的慢 family 協定，也不增加 owner API。
Controller 經公開來源接縫與 solver-free codec 取得輸入，重用 prepared-session owner。
Child 沿用外層獨占 runner 的 process group 與鎖，不另取鎖。
兩側、兩個 clock 的 telemetry overhead 都在各自 noise IQR 閉區間內才共同開啟。
輸出保留原始區塊、reference 證據及完整 density 誤差，額外診斷不混入計時比值。
統計與 process 取捨見[成對 Engine 計時討論](../discussions/2026-11-16-paired-engine-timing.md)。
取捨與介面收窄見[共用計時討論](../discussions/2026-11-16-shared-simulator-measurement.md)。

`family_pareto.select_inputs` 是公開來源／身份接縫，回傳工具的 `Input` 與原 `FrozenRecord`。
它用 `simulate.evolution.digest(record.metadata)` 算 metadata hash，不另寫 hash 公式。
接縫經公開 codec 讀入庫 index 與 repo 共用 scene cache，存放位置不參與 record 身份。
共用清單的成員不等於 Pareto 決策集合。未指定 index 或 scenario 時，接縫要求並選取旁邊宣告的 19 個 Pareto 名稱。
它先核對跨清單重名與必要名稱，再讀選定 payload。顯式 index 不帶 scenario 時全選，scenario 可嚴格單選非 Pareto 列。
Profile 列不作 family 去留證據。新增 Pareto 場景須同時加名稱宣告與清單列。
場景 producer 只在明確 `--update-index --scenario NEW` 時，接受 inventory 已宣告且所有清單皆無的新列。
它先建構及寫快取，再加到該 category 尾端，保留其他列與順序。完整 prepare 仍核對 exact name set，不推導刪列或換組。
選取與新增列的取捨見[共用清單與 Pareto 選集討論](../discussions/2026-11-17-pareto-selection-and-scene-append.md)。
量測 consumer 不再寫 records 副本，也不保留舊 freeze helper 或 producer import。
計時 owner 只接已備好的 session；paired caller 透過這個接縫取得同一組每筆身份。

測試側 exclusion estimator 與唯一 caller 測試已退休，不保留替代 module 或相容層。
其 tests → tools 反向 import 隨之移除。
取捨與介面退休見 [量表 cache-only 遷移](../discussions/2026-11-16-family-pareto-cache-only.md) 與 [Exclusion estimator 退休](../discussions/2026-11-16-exclusion-estimator-retirement.md)。

### Lab 定義與組合根

2026-10-28 使用者裁決：「認同這個建議」。Lab 的宣告與模組佈局由 lab 擁有，組合根不逐項重組這些知識。

`zcu_lab.assembly.lab_definitions()` 是唯一的 lab 組裝入口，回傳 frozen dataclass `LabDefinitions`。
九個欄位是 `register_components`、`register_experiments`、`templates`、`workflow_attachments`、
`recipes`、`catalog_reload_policy`、`logging_namespaces`、`fluxdep_export_planner` 與
`component_flux_context_provider`。呼叫此函式只取得宣告，不註冊 components、不開 entry，也不啟動 GUI 或 MCP。
Package roots 由 lab 從已安裝的 `zcu_lab` 與 `zcu_tools` 定位；reload 邊界與額外 logging namespaces 不寫在組合根。

Framework 的四個入口各自在自己的 package 宣告最小 Protocol，只讀該入口消費的欄位：

| 入口 | Protocol owner | 欄位 |
| --- | --- | --- |
| Measure GUI | `gui.app.measure.lab.MeasureGuiLab` | `register_components`、`register_experiments`、`templates`、`workflow_attachments`、`catalog_reload_policy` |
| Fluxdep GUI | `gui.app.fluxdep.lab.FluxDepGuiLab` | `register_components`、`fluxdep_export_planner`、`component_flux_context_provider` |
| Dispersive GUI | `gui.app.dispersive.lab.DispersiveGuiLab` | `register_components` |
| Measure MCP | `mcp.measure.lab.MeasureMcpLab` | `register_components`、`recipes` |

上述 owner 路徑都以 `zcu_tools` 為前綴。`LabDefinitions` 與 framework 測試的合成 lab 物件以結構符合這些 Protocol。
C16 禁止 framework import `zcu_lab`，所以入口不依賴具體 dataclass。
`lab-assembly-only` forbidden contract 限制 `apps`、`mcp_servers` 與 `tools` 不直接 import assembly 以外的 lab 子模組。
`allow_indirect_imports = True` 保留經 assembly 間接載入 lab 的路徑。新增 lab 直接子模組時，同步加入 contract 的 `forbidden_modules`。
C13 限制 MCP 只使用 GUI remote 契約，不能為共用 lab 型別新增 GUI import。
若把九欄 dataclass 放在 framework 的跨 GUI／MCP package，就會把 measure catalog、workflow 與 fluxdep policy 一起帶給不使用它們的入口。
因此具體描述留在 lab，消費契約留在各入口，不設共用的 framework lab package。
`CatalogReloadPolicy` 則由 `gui.app.measure.lab` 擁有，讓真實 lab 與合成 lab 共用 measure 的 reload 值契約。

三個 GUI 組合根解析 CLI、取得 lab 描述，並以 `functools.partial` 綁定 behavior 的明確參數。
`launch_gui_runtime(spec, options, *, behavior_factory, logging_namespaces)` 先初始化 logging，再呼叫 zero-arg factory。
這個順序讓註冊與 behavior 建構失敗寫入 session log；factory 不是另一個 lab 組裝入口。
`logging_namespaces` 直接取 lab 的 tuple，不另設 logging Protocol。
Startup 腳本選擇、project root、control port 與 log 路徑描述本次程序，仍由組合根決定，見 [[0064]]。

兩個 analysis behavior 明確接收 lab 與 `AnalysisSelection`，先註冊 components，再以 `AnalysisTarget.open(selection)` 開啟已選 entry。
`gui.app.analysis_project` 擁有 selection 的 `project_root`、`entry`、`component` 與純 args 轉換。
未選 entry 時不開 target；只有 component 沒有 entry 時報錯。註冊與開啟錯誤原樣傳遞。
Target 選擇不藏在 callback，也不保留舊 `project` keyword。

Measure MCP 組合根呼叫 `zcu_tools.mcp.measure.server.main(lab, *, gui_module)`，由 server 註冊 components 並取用 recipes。
Fluxdep／dispersive MCP 的 roots 與 server 只接 GUI module，不接未消費的 lab，也不註冊 components。
它們明確啟動的 GUI subprocess 才取得 lab，由 GUI behavior 註冊 components。

### 使用者元件與模擬環境的放置

`zcu_lab.components` 是 package。共用欄位型別、resonator wiring 型別與基礎元件放在 `base.py`；qubit 共用定義與參數轉換放在 `qubit.py`。Fluxonium、transmon、resonator、JPA 與 mux 各自擁有一個子模組。Package 入口保留明確的 `register_all()` 與公開匯出，不作 import-time 註冊。

2026-10-23，元件 wiring 依 kind 宣告。`BuiltinComponent` 只提供 `ext`、`module` 與 unsupported-field migration。Resonator 自宣告 `WiringSchema(ch, ro_ch)`，qubit 經 `QubitBaseSchema` 保留 `QubitWiringSchema(ch, lo_flux_ch)`。JPA 與 current source 沒有通道讀取者，不宣告 wiring。

Mux group 自宣告 `MuxWiringSchema`，以 `gen_ch` 表示 mux generator，以 `ro_ch[i]` 表示 `resonators[i]` 的 ADC。`ro_ch` 為 None 時表示尚未接線，有值時才驗證兩列表等長。這些 ADC 不從 resonator 的單 tone `wiring.ro_ch` 推導。

使用者指出：「mux group的屬性沒有寫好，simulation/FTF 的startup忘了寫mux group的gen_ch，而ro_ch沒有意義，因為有多個ro_ch」，並要求：「為什麼BuiltinComponent有那麼多無用的欄位，他不該是……儘量空的嗎？」

GUI／Notebook 共用的模擬參數檔與 startup 範例放在 `zcu_lab/simulation_envs/`，不放在 Notebook 文件目錄。FTF 沒有離線建庫入口。GUI 以 caller 選定的路徑執行可信 startup 腳本，不自動探索這個目錄；Notebook 可直接呼叫公開 `experiment.ftf.ftf(...)`。C14 與 C16 繼續禁止框架反向 import 使用者定義。

2026-10-16，使用者要求：「比如components我希望除了基礎原件，其他的都是獨立的子模塊檔。」對上述拆分與環境搬移確認：「可以移到zcu_lab/simulation_envs/中嗎？2同意」。

Workflow 的取數 helper 放在 `zcu_lab/workflows/` 側。共用 helper 放 `common/`，專屬 helper 放各 workflow 目錄。Helper 自己組 program、自己呼叫 runner，存檔沿用 core 的 Cfg／Result 型別與 `AXES_SPEC`，不借用其他實驗的取數或分析流程。客製化需求留在 workflow，不加到 core、lib 或其他不需要它的位置。

2026-10-21 使用者曾裁決 helper 與 core 並列：「可以是獨立檔案與core並列來方便維護」、「不要每個實驗，workflow需要的實驗就好」。2026-11-01 使用者改為 workflow 獨立實作：「所以我才不想要一個實驗使用另外一個實驗，twotone freq與workflow用的helper應該要獨立實現需要的流程」，並確認位置與理由：「3a可以，b是根據一個設計理念，耦合的邏輯就該放在同一個位置，這樣改邏輯的修改範圍才局部」。

Recipe 以 `@recipe` 裝飾函式並由 `zcu_lab/recipes` 的顯式清單註冊，與 `@workflow` 同形；三種腳本（startup、recipe、workflow）的作者慣例見 `zcu_lab/POLICY.md`（待實作 task 補寫）。

Kickstart 腳本放在 `zcu_lab/kickstart/<env>/`，只建 entry、元件參數與 alias，不連 SoC 與 device。每個 startup 腳本（模擬環境與 kickstart）各自擁有同目錄的 `entry-template.yaml`，腳本之間不共用範本：改一份範本只影響用該腳本新建的 entry。2026-10-21，使用者裁決：「3我希望startup腳本間獨立，而不是過早共用template」；目錄名：「既然都叫kickstart了，那我認為可以將envs/FTF改成kickstart/FTF」。

### Library 分組與 module slot

Library 的設計是「元件 → 分組 → Pydantic model」。分組是擴充點，不是永久封閉的種類清單。現行格式仍只接受既有四分組；有第五種需求時才調整容器與 key 驗證，不把任意分組寫成目前已支援。

原裁決日期未記錄，使用者於 2026-10-16 map 結案時確認沿用，原話「都採用建議」。原裁決原話為「我無法知道元件下會需要什麼欄位，我認為元件->xxx->pydantic model比較適合，不規定只有四種」及「我選D，但暫時沒有超過這四種的需求」。這裡選擇的是元件下自由命名分組的設計方向，不是當時就改格式。

Component 的 module slot 指向下游實驗要使用的模組，不用來為每一種校準方案保存一個 slot。參數累積、哪些實驗只提參數、哪些收尾實驗建立或更新模組，都由實驗作者決定。Framework 不提供通用的齊全檢查或累積機制。具體欄位、reset 名稱及 pair 方向留在實驗作者文件，不提升成框架規則。使用者於 2026-10-16 以「都採用建議」確認這個歸檔範圍。

單音 readout module 與其 `module.readout` slot 由 resonator 擁有，module root 例如 `R1.readout.readout_rf`，slot 例如 `R1.module.readout`。Qubit 經自己的 `resonator` 欄位與 session 的 `$resonator` alias 取得讀出；alias 宣告為 `resonator: $qubit.resonator`。單 qubit 表單不讀取 qubit 上的舊 readout slot，resonator 未設定 slot 時改用 readout 範本。Component 的 `module` 是可擴充的 `dict[str, str]`，新增 slot 不需修改 component schema。本次不遷移既有 point。`twotone/ro_optimize/*` 將最佳化 module 寫到 `$resonator.readout.readout_dpm`，並提出 `$resonator.module.readout` 指向該 root 的更新。Apply module 與 slot 後，之後所有使用該 resonator 的實驗都沿此 slot 取得最佳化讀出。`$qubit.ext.best_ro_freq`、`best_ro_gain`、`best_ro_length` 保留原有位置與用途。

使用者於 2026-10-25 裁定：「目前onetone/freq的readout module writeback是寫到qubit裡，但我認為應該是寫入到resonator componenet裡。而qubit要使用的readout應該是去綁定的resonator中找才對」。方案選擇與遷移政策的原話為「選甲，不遷移，應該說，目前module欄位下應該沒有限制型態吧？應該可以擴充吧？」。使用者同日以「Readout採用建議」確認最佳化讀出也寫入 resonator，並取代其 readout slot 指標。

### Mux group 是實驗室的 component 定義

`zcu_lab.components.MuxGroupSchema` 宣告 `group/mux`，沿既有 ComponentRegistry 註冊。
它保存必填的 `resonators`，含 1 至 8 個有序名稱，以及選填的 `qubits`、`flux_dev`、`mixer_freq`、`time_of_fly`、mux wiring（文件層可未知，表單／Run 才必填）和一般 module／ext。
`mixer_freq` 是有限的 mux generator LO 設定值，單位 MHz；None 表示未知。無 readout slot 的 mux Form 從 `$mux.mixer_freq` 取初值，缺值留空必填，不由 tone 頻率或 κ 推算。
Mux group 與 resonator 各自宣告頂層 `time_of_fly`，保存各自量測路徑的有限非負 flight time，單位 µs；None 表示尚未量測。不從 ext 讀取，也不自動搬移既有 entry。
Mux onetone 的無 slot trigger 初值為 `$mux.time_of_fly + 5/(2π·min κ)`，group TOF 或任一 κ 缺值時，trigger 與 pulse length 留空必填。Resonator 的 pulse 路徑 TOF 不供 mux 使用。
2026-10-23，使用者要求：「mixer frequency屬性請加入到mux group中，並用於生成預設值。」並要求：「讓mux group有自己的time_of_fly屬性，也用於生成預設值。」後續裁定：「將time_of_fly轉正吧。不在放ext裡了。」time_of_fly 因而轉為 mux group 與 resonator 的頂層欄位。
2026-10-23，使用者指出：「但kickstart應該是生成欄位讓用戶填，必填的話無法做到」。
文件允許 `wiring: {}`；沒有 ADC wiring 時，各 tone 的 `ro_ch` 在表單中留空必填。
Group 自己擁有 readout module，重排 resonators 不改 owner。
使用者於 2026-10-16 對專屬 group component 方案裁定：「我選B」。

`resonators` 取代原 members，不提供舊欄位相容讀取。Tone i 對應完整 `resonators[i]`。
選填 qubits 讓沒有 qubit 的 resonator bring-up 繼續使用同一種 group，不從 resonators 反推 qubit。
使用者於 2026-10-18 裁定：「或是改寫mux_group這個原件的欄位吧，qubits分resonators」。
對 resonators 必填、qubits 選填並逐項檢查，以及改名不留相容層的細節，裁定：「12都同意」。

Qubit 的 `flux: {offset, period: {<device>: ...}}` 保存零輸出時的 flux 與各儀器 signed periods，見 [[0082]]。Mux group 的 `flux_dev` 只保存預設 global coil 名稱，不要求提供 qubits，也不保存各 qubit 的校準。Global fluxdep 分開提出 `<qubit>.flux.period.<device>` 與預設勾選的 `<qubit>.flux.offset`，同時 append 原始擬合 record。保存沿用 [[0063]] 的單文件 mapping-leaf 合併與交易，不擴充 framework。

Model 驗證名稱、qubits 與 resonators 等長及重複 qubit。
提供 qubits 時，lab Form 在 Open／Reset 以固定 DocumentSetCapture 檢查兩邊 component 存在與 kind，
並逐項要求 `qubits[i].resonator == resonators[i]`。缺失或 null 同樣拒絕。
錯誤指名欄位、index 與實際值，在 Run 前拒絕來源。不排序、猜配對或自動修正，不額外拒絕重複 resonator。
Framework resources、session 與 program 不擁有這些關係。既有表單不隨全域來源編輯刷新。

Scalar alias `mux=MG12` 已足以讓 adapter 經固定 DocumentSetCapture 讀取 `$mux.resonators`。
一般 mux 與 FTF 表單在 `$mux.module.readout` 存在時引用 library，保留其完整 tones 與 scalar 設定。
Mux 缺 slot 時用 blank 範本加 group tones；FTF 八張缺 slot 用 blank 範本，rb 保持 Invalid。
`mux/onetone` 以副本加 group tones 調整 readout，並套用 ringup timing 初值。
使用者於 2026-10-20 裁定：「不只twotone，應該說除了onetone與readout optimize/mist等本身就在處理readout相關，其他實驗的readout預設值請優先使用ref。」
Framework 不認得 group/mux 或 resonators，不新增 group registry、RPC 或保存格式。
有序 alias 清單由通用 alias owner 解析；lab 在 Open／Reset 才把 `$participants` 名稱轉成原始索引，見 [[0079]]。
使用者於 2026-10-16 對只用 scalar mux alias 的方案裁定：「B」。

完整 readout 的 tones[i] 綁定 resonators[i]，cfg 不保存 writeback 目標或 group snapshot。
Lab cfg 的 participants 是非空、唯一且有效的原始 index list，依輸入順序投影 subset
給 program。底層 program／raw 的 active_tones 介面保持不變。Program 不知道 MG 或 resonators。RunRecord 保存完整 readout 與 selection，
資料軸保留原始序號，不重新編號。使用者於 2026-10-16 提議：「一個MG component類，內部屬性綁定到R1, R2，gui中預設指定MG，寫回目標也是從MG的綁定取」，並補充「為了能只讀特定幾個R，可以考慮MG外cfg另外提供mask之類的」。對 active_tones 方案裁定：「可以」。

[[0079]] 的 lab 表單契約使用完整具名 tones list 與獨立 participants。元素位置是原始 tone index，name 不可編輯；participants 的順序是執行順序。Open／Reset 在 lab 捕捉邊界一次把 alias 名稱轉成原始 index，非 participant drive／sweep 初始化為 None；一般編輯保值。Readout 身分全 tone 必填。Runtime 直接把 indices 交給既有 program／raw active_tones 接口。這不把 group snapshot 或 writeback 目的地塞進 cfg，也不讓 framework 解讀 group。具體實驗 cfg 與 recipe 接線由 lab 交付。

Program cfg 驗證錯誤沿既有宿主 result／Run 診斷定位 `active_tones` 並拒絕 Run，不保證欄位變紅。
FormState 的 preview／describe 不執行 lowering，也不新增 FormEditor 外部錯誤投影。
此分界沿用 [[0065]]。使用者於 2026-10-16 裁定：「報錯我選B」。

Writeback 由 adapter 經 Run／Load 既有 capture 讀取 `$mux.resonators`，以完整原始序號對應成功 active tones。
Onetone 沒有 qubits 時保留手填 readout。缺 capture 時只提 module，並顯示原因。
Capture 可解析時同時提出 readout slot 提案，將 `$mux.module.readout` 指向新模組的解析後 root。
Resonators 數不等於完整 N 時仍保留 module 與 slot 提案，省略逐 resonator scalar，並顯示原因。
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

`simulate` 是例外：它不依賴任何其他 `zcu_tools` package，包括同列的穩定工具，由 `simulate-closed` 檢查，見〈模擬器是封閉的子模塊，模型由使用方聲明〉。

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
- **依賴**：cfg_model 只依賴外部函式庫、utils、core.waveform_format 與 core.complex_codec，不讀寫檔案。Classifier 的複數中心共用既有 tag codec，不另定 library 格式。
- **准入**：型別只有在兩個以上有狀態模組都需要時才放進來。
- **改動**：核心 schema 會寫進檔案，例如每個工作點的 `library.yaml`。Library 的 format／format_version 只描述 header 與容器結構，不描述個別 module／waveform Spec 的欄位。新增、刪除或修改 Spec 欄位不自動升 library 版本。目前 library 維持 2.0，不遷移、不加相容讀取；entry schema 仍依現行定義驗證，不能以 forward minor 放寬 extra 策略。

使用者於 2026-10-15 裁決：「只是改Module而已真的需要升Library版本嗎？就像加一個Component原件定義，不應該讓舊版本不能讀對吧」「不要一次驗證，而是best-effort」「1可以，2A，3保留2.0」。Library 逐條驗證既有資料，合法條目不因另一筆 Spec 過時而失去可讀性。無效 Spec 的略過與原因查詢，以及讀取不改檔的規則由 [[0063]] 規定。

`program` 以 `type`／`style` 對應到執行用的 Module builder，取代 cfg 上的 `build()`。`resources` 的 module library 以核心 schema 作為 DocumentStore 的 model，並實作核心的 Protocol。驗證 context 注入固定 library 與 asset capability。Program 的 ModuleCfgFactory.from_core 集中轉換成 native cfg，透過 assets.path_for 取得絕對路徑。Native arb 只保留 data key 與必填 path，不保留 provider，不查 global DB。

### 分析 app 的 entry/component owner

2026-10-20 裁決：「新增 gui/app/analysis_project/ 共用過渡 owner」。
2026-10-20 使用者裁決：「1B可以，2要，3讓migrations裡支援遷移，工具本身不兼容」，分析工具改開 ResultEntry 並選目標 component，舊資料由 migrations 遷移，工具不加相容層。
2026-10-23 使用者裁決：「fluxdep／dispersive 維持 fluxonium 專用」。

`gui.app.analysis_project` 集中 fluxdep／dispersive 共用的 AnalysisEntry、AnalysisTarget、
EntryDialog、entry CLI 與 RPC 投影。AnalysisEntry 以 project_root 下的 result/ 與 Database/
開啟既有 ResultEntry，目標只接受 fluxonium 類 component。Component 省略時解析 global 的
qubit alias。Dialog 在 alias 缺失或錯誤時要求使用者明示選擇，不自選第一個 component。
分析 app 不使用舊 chip／qubit 兩級目錄與 params.json，也不呼叫 Database Save。
Predictor 與 dispersive 格式由 resources 擁有。Fluxdep 匯出 predictor、更新 global 已宣告欄位，
並以 record_analysis 留 ledger。Dispersive 讀 entry-local predictor，匯出獨立 dispersive artifact。
Lab kind 的註冊函式由 `apps/` 組合根透過 lab 描述交給 analysis behavior；behavior 先註冊，再開啟 target。Framework 不 import `zcu_lab`。
Notebook 的既有 params.json handoff 仍由 resources.qubit_params 擁有，不是分析 app 的 fallback。
共用 GUI、session、remote、measure 與 workflows 不依賴這個分析 owner。
`.importlinter` 的 `shared-no-analysis-project` contract 保護此方向。
中立的 `nearest_existing` 位於 `gui.widgets.file_dialog`，callers 直接匯入，不經分析 owner re-export。

Native 光譜的數值投影與選擇分屬兩個 owner。`analysis.fluxdep.project_native_spectrum` 擁有 Container 到光譜的投影：caller 明示 variable 與 device_axis，
它驗證 2-D 複數資料與另一軸的頻率單位並換成 MHz；fluxdep 與 dispersive 不各自猜資料布局。
`gui.app.analysis_project` 另擁有共用的 native source 讀取、variable／axis 預選與 dialog mechanics；它只接 caller 提供的 device choices，
不取得 component 或 predictor，也不含 fluxdep 的 Type、Inherit 或 overlay。Session 與共用 dialog 仍不依賴這個分析 owner。
這是 advisor 於 2026-10-27 依使用者同日授權「C也是，可以直接派」訂定的分工。

### Device snapshot 標量的解讀 owner

`device.snapshot_scalar_values` 是 acquisition snapshot 到具名 `DeviceScalar(value, unit)` 的唯一解讀入口。
名稱缺席時不回傳該鍵；在場但缺 value、為 bool 或非有限值時報錯並指名。缺 unit 視為 native。Apps 不解析 driver settings 或 cfg。
`analysis.flux` 擁有單位比對：只核對實際參與換算的其他儀器，單位精確相等，native 只等於 native，不換算。
被掃儀器取掃描軸，不重複使用其 snapshot 值；整份 snapshot 未記錄與已記錄但某儀器缺席是不同狀態，缺席儀器依 [[0082]] 貢獻 0。
Device API 的局部參數與錯誤細節留在 module 文件與 docstring。
這是 advisor 於 2026-10-27 依使用者同日授權「C也是，可以直接派」訂定的分工。

### 模組之間的合作

- 需要另一個模組負責的能力，而它沒有提供時，擴充那個模組或停下來問。不在呼叫端重做。
- 兩個有狀態模組需要合作時，由需要的一方定義最小 Protocol，由組合根注入實作；或把組合邏輯放到上一層。
- 注入留給真正由外部決定的實作。框架常用的幾種支援能力直接內建，以內部內聚的子模組實現；使用者腳本為了獨立性自行集成演算法也可以。使用者於 2026-10-20 裁定：「C太多注入也不是很好維護，有些可以是框架提供支援的幾種功能，已內部內聚子模塊實現，而不是用注入的方式。script有些是為了獨立性而集成算法。」
- 呼叫端驗證被呼叫者的輸出格式、組被呼叫者的內部版面、或修補被呼叫者回傳的資料，都表示責任放錯位置。

### 可執行投影

`.importlinter` 的 C8–C13、C15、C16、C19 與 C21 投影本篇規則：

| contract | 類型 | 內容 |
|---|---|---|
| C8 | `layers` | 前端框架 > `experiment` > 能力模組 |
| C9 | `acyclic_siblings` | `zcu_tools` 的頂層 package 之間沒有循環 |
| C10 | `independence` | `program`、`resources`、`device`、`qick_remote` 彼此獨立 |
| C11 | `forbidden` | `cfg_model` 不依賴有狀態模組、實驗與前端；使用 `utils`、`core.waveform_format` 與 `core.complex_codec` |
| C12 | `forbidden` | 穩定工具包含 core，不依賴有狀態模組 |
| C13 | `forbidden` | `mcp` 只使用 `gui.remote` 與各 app 的 wire spec |
| C15 | `forbidden` | 使用者實驗的 core 不依賴 GUI framework |
| C16 | `forbidden` | `zcu_tools` 不 import `zcu_lab` |
| C19 | `forbidden` | `core` 在 repo 內只依賴 `utils` |
| C21 | `forbidden` | `zcu_tools` 不反向依賴使用者入口 `scripts` |

`.importlinter` 的 `experiment-directories-independent` contract 檢查上述實驗與家族邊界，不新增 `ignore_imports` 例外。

`composition-roots-above-framework-and-lab` 的 `layers` contract 約束 `apps : mcp_servers` > `zcu_lab` > `zcu_tools`。Lab 與 framework 不反向 import 任一組合根，不新增 `ignore_imports`。

`lab-assembly-only` 的 `forbidden` contract 列出 assembly 以外的 lab 直接子模組，保護組合根與開發工具的直接 import 邊界；`migrations` 與使用者空間 `scripts` 不在 source_modules 中。

`simulate-closed` 的 `forbidden` contract 以 `zcu_tools.simulate` 為 source，禁止它 import 其他所有頂層 package 與 `format_version` module，保護〈模擬器是封閉的子模塊，模型由使用方聲明〉。寫法與 [[0080]] 的 `datafile-standalone` 相同：明列清單，不新增 `ignore_imports`。

`zcu_tools`、`zcu_lab`、離線 `migrations` 與 repo 的 `apps`、`mcp_servers`、`scripts`、`tools` 都是 import-linter 的 root packages。只為離線遷移存在的程式與對照表放在 `migrations/`，`zcu_lab` 與 `zcu_tools` 不 import 它，遷移完成後可整個刪除。使用者於 2026-10-14 裁定：「我自己會預期，我遷移完後刪除migrations就可以了」。三個 GUI 與 measure MCP 組合根各注入一個 lab 描述，兩個 analysis MCP 只注入 GUI module 名稱；framework 不反向載入使用者定義。Tests 的 lab 獨立性由 review 檢查 import，必要時以 grep `zcu_lab` 輔助，不新增 tests import contract。

債務以確切的 module 對逐條列在 `ignore_imports`，不用萬用字元，每組註解寫明「債務，由誰移除」。`ignore_imports` 只減不增。

## 債務

| 債務 | 移除方式 | 記帳位置 |
|---|---|---|
| `program → resources` | `program` 改依賴 `cfg_model` 的 Protocol | C10 `ignore_imports`：`program.v2.modules*` → `resources.context` |
| `notebook/analysis` | 通用分析移到 `analysis`；實驗專屬 helpers 已由 `zcu_lab` 附件擁有 | 尚未搬移 |
| `utils` 的 `shot_classification`、`tomography` | 移到 `analysis` | 尚未搬移 |
| `analysis/fluxdep/io.py` 直接用 h5py | 檔案讀寫移到 `datafile` | 尚未搬移 |

## 後果

- 增減功能時依賴圖的形狀不變：工具之間的邊很少變，有狀態模組之間沒有邊。兩個有狀態模組要合作，做法固定是 Protocol 或上移組合。
- 新增跨 package 依賴會被 contract 擋下或在 review 中出現，不會靜默成為先例。

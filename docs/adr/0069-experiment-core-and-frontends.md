# ADR-0069 — 實驗核心、前端包裝與具名圖形產物

**狀態：** accepted。
**關聯：** [[0062]] 的實驗 workflow、[[0063]] 的保存、[[0065]] 的 cfg、[[0066]] 的 operation、[[0067]] 的 GUI。

## 問題

本篇處理核心與前端責任混用的問題。原本的 v2 實驗 class 同時提供 Notebook 的 last_result 便利行為、量測、分析與保存，GUI adapter 再補表單、參數轉換與統一介面。Notebook 呼叫習慣牽動核心契約，GUI 因而必須傳遞 cfg 外參數。Figure、數值分析與前端生命週期也由同一層處理。

只改用函式、拆 package 或增加宣告式 definition，不能解決這些責任問題。目標是先分開核心與前端，保留使用者直接撰寫 Python 和原生 Matplotlib 的能力。

## 決策

### 核心與前端

核心保留無跨次可變狀態的實驗 class。NotebookAdapter 先綁定 soc、soccfg、DeviceManager 與 plot host 引用，再由 `nb_adapter(core)` 建立各自保存 records 的 NotebookExperiment。Caller 建立 experiment instance，不要求零參數 constructor。同步實驗不各寫扁平參數 wrapper，caller 直接提供 typed config／options。GUI adapter 直接使用核心，兩前端不互相包裝。

RunRecord 泛型組合 nullable typed cfg 與實驗專屬 Result。Typed cfg 的位置是 RunRecord.cfg；檔案的原始 CfgSnapshot envelope 由 StoredRun 保留，不把 typed record 當成 lossless rewrite envelope。AnalysisRecord 組合 explicit source RunRecord、實際 options、typed Analysis 與純具名 figures；cfg 與 result 從 source 取得，不另外保存一份可混用的來源。

核心操作形狀為：

```python
run(config, *, context) -> Result
save(source: RunRecord, destination) -> None
load(source: Path) -> RunRecord
from_stored(stored: StoredRun) -> RestoredRun[RunRecord]

# Only synchronous analysis cores provide this operation.
analyze(source: RunRecord, options, *, plots) -> Analysis
```

直接呼叫 core.run 不自動建立 record，也不保存 last state。共用 Notebook 包裝層建立 RunRecord，同步分析成功後建立 AnalysisRecord。Interactive core 不實作 analyze，GUI adapter／plugin 與 Notebook 獨立分析工具各自負責。最低共用 core 契約不能強制 analyze，不補空方法、不探測新舊簽名，也不建立互動 capability registry 或通用 callback framework。

RunContext 提供單次 run 的 soc、soccfg、具名 devices、plots 與 cancel_signal，不保留 QickContext alias。Caller 在 run 前固定名稱到 BaseDevice 的綁定，核心只借用 driver。核心仍決定每個 sweep point 何時 setup；setup helper 在整批 setup 前驗證所有必要名稱，不查全域 manager。Connect、disconnect、registry 與 ResourceManager 生命週期留在前端 owner。Hardware handles 不進 cfg 或 records。

每次 run 建立新的 context、Plots 與 StopSignal，硬體 handles 可以重用，停止與錯誤狀態不能沿用。Executor、子實驗、Schedule 與 experiment setup helper 共用這個 StopSignal。只有 helper 呼叫低層 driver 的 stop_event 介面時才取出 Event，device 模組不反向依賴 experiment。ProgramBuilder 保留 acquire-local cancel flag，SNR 等 data-driven early stop 不取消整次 run。既有 retry、partial result、raise_if_error 與 GUI operation 分類保持不變，不引入另一個 outcome 或 runner。Progress bar 暫留環境注入。

Load／analyze 不要求硬體；run 在缺少必要 handles 或 binding 時拒絕，不自行查找或建立資源。單次 buffer、tracker 與 cache 歸 run。NotebookExperiment 每次 run 從已綁定的 manager 取得一次 driver mapping，再建立 context；同次 run 不重查 registry。建構入口不讀裝置資訊，load／analyze／save 也不查裝置。同一 manager 更新 registry 後，下次 run 使用新綁定；重新賦值 caller 變數不重綁既有入口。GUI 的 RunService 與 workflow RunSession 仍各自在 operation 邊界建立 context。

影響 acquisition 的實驗選項全部進 typed config，包括 T1 uniform。核心擁有設定驗證及環境無關預設；GUI 擁有編輯表示、標籤、expression 與 md／module library seed。核心分析不讀 live GUI 狀態。Hardware handles 不進 cfg 或 record。

共用包裝層隔離 caller cfg／options 與 core 工作輸入，成功後保留該次來源與選項。Record 的欄位關聯固定，不深拷貝大型 Result，不凍結 Figure artists。Record-owned cfg／options 可被使用者刻意修改，不提供 deep-freeze、讀取時複製或完整不可變歷史保證。

同步核心 options 與 Analysis 是實驗專屬 typed 資料，options 欄位擁有預設值，不另設 Notebook defaults 或動態簽名。實驗類別以 `Options` 引用既有分析選項型別，有獨立 post-analysis 選項時使用 `PostOptions`。Caller 以 `T1Exp.Options(...)` 建立獨立物件，不另匯入 Options。沒有選項的實驗不補空型別，共用 Protocol 不增加這項要求，NotebookExperiment 不轉送這些類屬性。Analysis 不含 Figure 或 writeback。Notebook 同步入口回傳 AnalysisRecord，互動工具也保留成功成果的來源、實際 options 與純圖。Notebook 的 analyze(options, *, source=None) 與 save(destination, *, source=None) 預設最近成功 run／load 的 last_run，無來源就拒絕。分析歷史 source 不改 last_run 或 save 的預設來源。Core 與獨立互動工具仍要求明確來源。Notebook 不新增 writeback。

GUI 插件自行定義 typed 成功輸出的欄位。插件決定是否輸出選項、隨機 seed、時間或其他重現資訊。Framework 保存來源、插件輸出與圖，不追查插件的隱式依賴，也不保證完整 options 或可重現性。`params` 保持表單輸入，不在 Done 時替換成終態 options；不新增通用 committed-options owner。GUI 與 remote 讀同一份已提交輸出。

實驗跨入 zcu_tools 時使用公開絕對 import，預設置頂。只有具體重型依賴或初始化限制才延遲，不藉此外移模組或建立設定式 discovery。

### Notebook 中斷與部分結果

Schedule 在自己的 with 保護區域接住 KeyboardInterrupt，將同次 StopSignal 記為 interrupted，保留 buffer 內已完成的資料，讓實驗在離開 scope 後組成 partial result。Notebook 共用包裝層將正常返回的 interrupted 視為 stopped，發布可保存的 RunRecord，而非丟棄這次資料。Failed signal 與逃出 core.run 的例外仍失敗，不取代上一筆成功 record。

這個責任放在 Schedule 與 Notebook 包裝層，不要求每個實驗各包一層 try/catch，也不加 stop widget、背景 thread 或事件 pump。保護區域外的中斷、重複中斷及收尾例外不保證產生 partial record，細節由 Schedule／Notebook 公開契約維護。

原裁決日期未記錄，使用者於 2026-10-16 map 結案時確認沿用，原話「都採用建議」。原裁決原話為「但我不想要每個實驗都包一層try/catch，Schedule用with大概也很難攔截吧？」。

### Experiment tag 與 adapter

實驗核心宣告持久化 tag，GUI adapter 轉接或明確覆寫 experiment_tag。Framework 只讀 adapter 的公開 tag，不取得實驗類別，也不從泛型反推實驗類。這讓核心擁有持久化身分，同時保留 adapter 黑箱；代價是每個 adapter 必須明確接上 tag。

原裁決日期未記錄，使用者於 2026-10-16 map 結案時確認沿用，原話「都採用建議」。原裁決原話為「插隊問題我的想法是tag應該放實驗類讓adapter轉接或複寫才對」及「不太建議暴露實驗類給框架，最好都經由adapter轉接」。

### 圖形產物與呈現

`plots` 是本次操作的繪圖能力，不是全域服務查詢器。

```python
fig, ax = plots.subplots("fit", ...)
viewer = plots.liveplot_1d("measurement", ...)
plots.adopt("diagnostic", external_fig)
```

以上入口建立或接收 Figure 時即具名登記。所有具名圖都是本次操作的圖形產物，不做第二次 register_result。實驗直接操作原生 Figure／Axes，factory 不包裝整套 Matplotlib。

不呈現環境仍建立與更新圖並可保存，只是不開視窗或安排互動展示。Liveplot 在執行期間呈現。第三方自行建圖須經 adopt 才保證接入，不能撤回其已發生的顯示副作用。

新入口不依賴 ambient plotting scope，也不要求實驗作者寫相關 with。Factory 必須明確接合 host；不能只把舊 plt.subplots 包一層，仍暗中依賴 routing 或每次切換全域 backend。

Adapter／Notebook 分析工具掌握操作或 session 使用期，共用 plots 處理圖形機制。操作中的 Plots 提供建圖、liveplot、adopt 與 host 接合；完成成果使用另一個純具名 Figure 容器，不將 Plots 本身當作 AnalysisRecord.figures。純容器不提供建圖、接管、liveplot 或 host commands。

Presentation owner 保留明確的呈現使用期與 release 責任，不能複製 mapping 後丟失所有權。跨存活 owner 接管仍拒絕。停止 producer、最後 refresh、釋放呈現與保留可保存的原生 Figure 分開處理。替換目前引用不關閉使用者持有的舊圖，release 後原生 artist 操作與 savefig 仍可使用。

圖已登記不表示操作成功。Core、host finish 或必要收尾失敗時不提交新成功 record，失敗診斷圖不得混入上一筆成功分析。

GUI application 持有結果與具名圖集合，保存不反向依賴 Qt widget。前端持有 canvas 與選圖狀態。截圖取目前選圖，分析圖保存涵蓋整個集合；不新增多圖排版編輯器。

### 三種分析生命週期

- T1 提供同步 analyze，消費 explicit RunRecord 與 typed options。Notebook 成功後發布 AnalysisRecord；GUI 發布插件定義的 typed 分析輸出、來源與圖。
- Singleshot GE 核心另提供 post_analyze。Post 消費 adopted primary 的來源與 calibration，不重新 primary fit，不讀未重新分析的 initial_state 表單。Primary／post 各自持有图與 GUI writeback proposal。成功 primary 替換清空目前 post，primary 失敗保留前次成功組。清空引用不關閉使用者持有的舊圖。不要求其他實驗提供空 post 方法。
- OneTone FluxDep 核心只負責 acquisition 與資料保存／載入，不實作 analyze。GUI adapter／INTERACTIVE plugin 自行分析，Notebook 使用獨立分析工具。重用現有 TwoLinePicker 與 Qt-free 選線規則，不把 Notebook 工具接回 core.analyze 或共用 Adapter 分析。啟動返回不代表成功；Done 驗證及收尾後發布該次成果，Cancel／failure 不取代上一筆成功。Notebook 保留實際 options，GUI 插件仍自行決定成功輸出的欄位。

Notebook run／load 正常返回後更新目前 RunRecord，清空目前分析與圖引用；失敗保留舊紀錄。同步分析成功才成組替換成果，分析舊 RunRecord 不替換目前 run。互動工具綁定自己的來源，晚到完成仍按 G5 發布自身成果，不取最新 run，也不新增 generation gate。清空引用不銷毀使用者持有的舊圖或結果，不新增完整歷史管理。

### 保存、發現與範圍

實驗透過 AXES_SPEC／GroupedAxesSpec 擁有 RunRecord 到保存資料的映射，datafile 擁有檔案格式。保留 inner-first axes、Result-native shape、SI disk units、roles、labels、dtype、comment／tag 與 complex calibration。Caller 指定 exact path，save 不依目前 run。本機單檔 save 是明示 Labber 匯出，load 只讀 native data.h5，兩者不是對稱格式。只有 Labber 的舊資料須先用既有 migrations 離線轉換，不提供執行期 fallback，也不代跑真實資料遷移。Run 保存與重存的 native replace、Labber 新檔及失敗語意由 [[0063]] 規定。

PersistableExperiment.from_stored 使用公開 restore_run(stored, spec=...)，不做 I/O、不查 registry 或動態匯入實驗。Notebook 的 load 只收 Path，先以 datafile 讀成 StoredRun，再經同一轉換回 typed RunRecord。GUI LoadService 讀檔後交 adapter.restore；GUI 只持有 adapter 的 opaque result。RestoredRun 以 record 型別為泛型，同時保留核心 typed record 與 GUI opaque handle 的型別資訊，cfg_issue 為 None 或非空的不可用原因。

Restore 先驗證 native 格式、experiment tag、必要 variables、axes、unit、dtype、shape 與 builder result，再處理 typed cfg。缺 /cfg、非法 JSON 或非法 CfgSnapshot envelope 仍是格式錯誤。合法 envelope 的 cfg_type、schema major 或 concrete model values 不相容時，保留有效資料、令 RunRecord.cfg=None，並發出 CfgUnavailableWarning；cfg_issue 攜帶同一原因。同 major 的 minor 與未知 cfg keys 沿既有 known projection，StoredRun 保留 raw cfg JSON 與歷史 metadata，不重拍來源或裝置。

GroupedLoadData.cfg_snapshot 允許 None，資料 builder 讀 variables，不拿目前 cfg 補值。正常 run 仍有有效 cfg。需要 cfg 的分析或 data 保存入口明確拒絕 cfg=None，不需要 cfg 的分析照常，不由 NotebookAdapter 一律阻擋。GUI Load 不回填表單，也不從 Form、library、default 或 live device 補回歷史 cfg。原始 cfg 保存在 native 資料檔，透過公開 datafile.load_run_data 離線讀取為 StoredRun，GUI 不提供檢視入口。使用者裁定見 [[0063]]。

使用者於 2026-10-14 對框架讀檔後交 StoredRun 給 adapter 回答：「看起來沒問題」；對資料優先、cfg 可缺回答：「我認為可以」。2026-10-15 以「1可以，2採用忽略不支援欄位的方式，3同意，4一起做吧」確認 Load 不回填與 Labber-only 先離線遷移。

一般實驗使用預設 save／load，作者可用相同 RunRecord 介面明確 override，不要求額外 experiment_codec，也不聲稱支援任意 Python 物件。Workflow streaming／grouped artifacts 保留自己的契約，不改成 one-shot record 格式。PersistableExperiment、RecordExperiment 與 NotebookAdapter 的 save／load 只接本機路徑，不公開 server_ip／port；獨立 datafile／workflow transport 保留自己的契約。

Notebook save 吸收 unique path 的便利層並回傳實際 path，再呼叫 exact saver。Path helper 不建立檔案或鎖，不宣稱原子 reservation。AnalysisRecord 保留記憶體內來源與 options，分析圖另外保存；不新增完整 analysis session 磁碟格式，也不撤銷實驗既有保存義務。

保留 explicit catalog、v2／v2_gui 分離，以及單檔／package 並存。框架不要求固定內部檔名，不靠掃描或 import 副作用發現實驗。

範圍是所有實驗及其 Notebook／GUI caller，不以 GUI catalog 為上限。一般 T1、singleshot/ge、onetone/flux_dep 分別代表同步、post 與互動分析三種生命週期。

所有實驗與 caller 都使用這套核心與前端契約，不保留 pyplot 相容層或舊協議 fallback。Executor、動畫框架、acquisition、排程與硬體鎖機制不在本篇範圍。

## 顯式 cfg 與裝置環境

DeviceManager 的 registry、lock 與 close claims 屬於 instance，不提供預設 singleton。Notebook 與 GUI composition 持有或接收 manager；共享 driver 的 caller 共享同一 manager，不各自取得關閉權。既有 alias、close failure 與 driver lock 契約保留，不以 destructor 自動關閉，不把 disconnect 當作歸零或 RF off。

Notebook 使用 `CfgEnv(ml, device_manager)` 借用固定 library snapshot 與 caller 的 DeviceManager。顯式呼叫 `make_cfg(raw_cfg, CfgModel, env, overrides=...)` 以該 library 的 validation context 組裝 typed cfg，不讀取或設定 devices。參數值由 caller 在建立 raw cfg dict 時直接讀 point 或 roles，不交給 CfgEnv 解析 expression 或延遲求值。組裝失敗直接報錯，不 refresh 來源或退回舊 cfg。

CfgEnv 與 RunContext 分工不同。前者可跨次組裝重用；後者只有單次執行能力，不讓核心取得 point、library 或 registry 管理權。底層 `assemble_experiment_cfg` 接顯式 validation context，不讀硬體。Run 不重新組裝 cfg、不重連、不補裝置。需要新的參數或 library 資訊時，caller 明確讀取來源、重建 CfgEnv，再次呼叫 make_cfg。Driver mapping 不是資源 lease，也不凍結 driver 狀態。Notebook 初始化 cell 重跑就重建，不探查 namespace 或自動重用，舊連線仍由使用者明確關閉。GUI 的固定來源、CfgRef／revision 與 acceptance 不因 Notebook 入口而重新查詢硬體。

模擬環境由啟動時明確選用的 Python `build_environment()` 組裝，回傳已綁定 soc、匹配 soccfg、同一 DeviceManager 與公開描述。GUI 與 Notebook 共用此 bundle；一般 soc_changed 不建立 device，不保留動態切換入口。成功回傳時清理責任交給宿主，部分清理失敗保留資源與原錯誤，不假裝 ready。

FTF 的 YAML 由 resources 驗證；experiment.ftf 擁有接線、裝置與逐 round snapshot，simulate 只收數值並提供 joint model、evolution、population conversion 與 ADC。初值只在 startup 套用，每 round 讀當下 device 值；讀取失敗不降級。正常 metadata 只保留 I/Q 與公開 provenance，不含模型真值。舊 TLS 執行機制不留相容層。依賴由 .importlinter C8、C9、C10、C12 約束；完整物理限制與 readout 起點凍結近似見 [模擬環境契約](draft/simulation-environments.md)。

可信 Python 不是硬體 sandbox。使用者自行隔離主機資源，不新增模擬專用裝置黑名單或自動斷線政策，也不預建 plugin registry 或空 backend。

## 取捨

- 明確 factory 使建圖依賴可讀，但任意第三方 pyplot 操作不再自動路由；adopt 處理返回的 Figure，不提供任意 pyplot 全域狀態隔離。
- 圖獨立於 Analysis，避免數值型別依前端改變，代價是 caller 必須把資料、來源與圖一起發布。
- 共用 NotebookAdapter 減少同步實驗的重複 wrapper，caller 直接使用核心 typed config／options。互動分析維持前端工具，不將 widget／session 協調塞進共用 Adapter。
- 不呈現仍建圖，保留實作一致性與保存能力，接受建圖及 artist 更新成本。
- 不以單一同步方法統一互動 session；通用繪圖不保證自動提供互動輸入。
- 不保留 pyplot 相容層或第二套分析協議；代價是每個實驗與 caller 都必須直接使用 explicit factory、typed config／options 與 records。

## 七項政策

### G1：一般圖與 liveplot

一般 subplots 立即建立並具名登記，GUI 於函數完成後呈現。Liveplot 立即呈現，worker 透過 typed update 傳遞資料，GUI owner 更新 artists。Worker 不任意直接修改 active live Figure／Axes。實驗不需寫前端分支，也不能以 draw lock 取代這項執行緒契約。

### G2：圖形生命週期

停止 producer、最後有效 refresh、保留 Figure 與釋放呈現資源分開處理。取消請求不立即銷毀圖，既有 partial 正常返回仍可呈現最後資料。新操作不關閉使用者持有的舊圖。

Notebook backend adapter 處理顯示與 close，避免 cell 結束重複顯示，並提供明確釋放呈現資源的方式。close 對 widget 與保存的影響需依 backend 核對，不使用全域 close("all") 清理別人的圖。外部 Figure 的 adopt 不能撤回已發生的顯示副作用。

### G3：名稱與接管

操作內一名一圖。同名同 Figure 重複 adopt 等冪，同名異圖及同圖異名拒絕。已由其他操作持有的 Figure 拒絕接管，不偷移 canvas。第一版不提供 transfer 或 alias。舊圖由原集合保存，跨操作重畫使用資料。

### G4：多圖保存

逐圖記錄是否曾成功保存，以 stage 與圖名命名，單圖也遵守相同規則。保存前固定本次集合、產物 identity 與目的地，在 I/O 前檢查安全路徑與名稱碰撞。Identity 用來將完成紀錄對應回原產物，不是內容 dirty 版本。

同一份圖一旦保存成功，修改 artist、視圖或目的地不清除成功紀錄。不新增 dirty 通知或自動內容追蹤。新產物從未保存開始。

部分失敗保留成功項，不宣稱整組已保存，也不承諾跨檔原子保存。Save All 包含已保存的 artifact，重寫選定的 native、PNG 與 JSON；Labber 每次產生新檔，失敗保留上次成功連結。每次呼叫回報本次結果，不自動 retry，保存權威與失敗邊界見 [[0063]]。

### G5：晚到結果

Notebook 不新增 generation／request identity 的發布限制。較早啟動的互動或分析較晚完成時，允許覆蓋目前分析，操作順序由使用者負責。該次數值、來源 Result、實際 options 與圖集合仍須成組發布，不能混用不同操作的資料。

既有 GUI busy、operation token 與已取消 session 的生命週期規則不變。這項裁決不重新定義 run 中斷，也不刪除 GUI 的既有保護。

### G6：Notebook 互動前端

Notebook 互動分析直接使用 Notebook widget，不增加前端可用性 preflight 或 backend 偵測。建立或使用 widget 的實際錯誤正常傳遞，不自動降級到 inline、無頭分析或自動 Done。

同步分析的不呈現繪圖模式維持不變。它不是無頭互動工作流。

### G7：GUI adapters

GUI application 統一使用插件定義的 typed 分析輸出與 plots，不另要求通用終態 options。Interactive plugin／frontend 使用 session-owned 圖集合，不探測新舊簽名，不新增反射或雙協議 fallback。

### 不使用自訂 pyplot routing backend

實驗與 GUI caller（包括 measure 與 fluxdep search）都經 explicit factory／host 建圖，專案沒有自訂的 pyplot routing backend、routing scope 或其設定。這不是移除 Matplotlib rendering：原生 Figure／Axes、Qt canvas、Agg 與 ipympl 仍負責實際繪製。共享 runtime 保留 backend 選擇、host 初始化、shutdown handling、mathtext lock／prewarm、容器、owner scheduling、attach／detach 與 rendering 初始化。

## 公開入口

- [Experiment](../../lib/zcu_tools/experiment/README.md) 說明 RunRecord／AnalysisRecord、nullable cfg、RunContext 與 default／grouped persistence。核心不再保存前一次結果。
- [Notebook](../../lib/zcu_tools/notebook/README.md) 說明 NotebookAdapter、GEPostAnalyzer 與 FluxDepAnalyzer。同步分析可省略 source，獨立互動工具仍接明確來源；Done／Cancel 與晚到成果保留各自來源。
- [GUI plotting](../../lib/zcu_tools/gui/plotting/README.md) 說明原生 Figure 與 Qt presentation 的不同生命週期。Plots.finish 完成最後刷新，Plots.release 只釋放呈現。NamedFigures 保留完成後的原生圖。
- Measure 的 ArtifactKey 以 stage 與圖名識別成果。`measure.artifacts` 擁有 ArtifactTracker 與 SaveService，捕捉本次保存來源與目的地。Save All 包含已保存成果，個別失敗不撤銷先前成功項。共用 `experiment.artifacts` 服務 GUI 與 Notebook，固定項目集中隱藏格式選擇、編碼與失敗隔離。Workflow 的單檔保存仍用共用 prepared writer，不強制改走 entry Save All。
- [Session](../../lib/zcu_tools/gui/session/README.md) 說明 DeviceManager owner 與 Use Simulate Env coordinator。一般 SoC 連線不建立 FakeDevice，coordinator 先完成真實裝置斷線，再發布已綁定來源的 mock 環境。

Standalone [liveplot](../../lib/zcu_tools/plotting/liveplot/README.md) 保留自己的 backend 與 close 契約，不等同於 Plots host。

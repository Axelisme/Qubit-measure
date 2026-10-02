# 實驗核心、前端包裝與具名圖形產物

**狀態：** 69 個核心、相關 caller 與共用依賴已在 integration 完成實作，舊自訂 pyplot routing backend 已退場。整體雙軸審查發現的接縫缺口正在修正，正式接受尚未完成，landing 另需使用者授權。本文仍是 ADR 草案，不代表持久分支已採用，也不取代現行 [實驗 workflow](../0062-experiment-workflow.md)、[保存](../0063-persistence-ownership.md)、[cfg](../0065-cfg-editing.md)、[operation](../0066-operation-lifecycle.md) 與 [GUI](../0067-gui-application.md) 契約。

## 問題

本次重整處理核心與前端責任混用的問題。舊 v2 實驗 class 同時提供 Notebook 的 last_result 便利行為、量測、分析與保存，GUI adapter 再補表單、參數轉換與統一介面。Notebook 呼叫習慣牽動核心契約，GUI 因而必須傳遞 cfg 外參數。Figure、數值分析與前端生命週期也由同一層處理。

只改用函式、拆 package 或增加宣告式 definition，不能解決這些責任問題。目標是先分開核心與前端，保留使用者直接撰寫 Python 和原生 Matplotlib 的能力。

## 核准目標

### 核心與前端

核心保留無跨次可變狀態的實驗 class。共用 NotebookAdapter 接收 experiment instance，綁定可重用 hardware handles 與 plot host。同步實驗不再各寫扁平參數 wrapper，caller 直接提供 typed config／options。GUI adapter 直接使用核心，兩前端不互相包裝。

RunRecord 泛型組合 cfg 與實驗專屬 Result。Result 只表達資料，不攜帶 cfg_snapshot。AnalysisRecord 組合 explicit source RunRecord、實際 options、typed Analysis 與純具名 figures；cfg 與 result 從 source 取得，不另外保存一份可混用的來源。

核心操作形狀為：

```python
run(config, *, context) -> Result
save(source: RunRecord, destination) -> None
load(source: Path) -> RunRecord

# Only synchronous analysis cores provide this operation.
analyze(source: RunRecord, options, *, plots) -> Analysis
```

直接呼叫 core.run 不自動建立 record，也不保存 last state。共用 Notebook 包裝層建立 RunRecord，同步分析成功後建立 AnalysisRecord。Interactive core 不實作 analyze，GUI adapter／plugin 與 Notebook 獨立分析工具各自負責。最低共用 core 契約不能強制 analyze，不補空方法、不探測新舊簽名，也不建立互動 capability registry 或通用 callback framework。

RunContext 提供單次 run 的 soc、soccfg、具名 devices、plots 與 cancel_signal，不保留 QickContext alias。Caller 在 run 前固定名稱到 BaseDevice 的綁定，核心只借用 driver。核心仍決定每個 sweep point 何時 setup；setup helper 在整批 setup 前驗證所有必要名稱，不查全域 manager。Connect、disconnect、registry 與 ResourceManager 生命週期留在前端 owner。Hardware handles 不進 cfg 或 records。

每次 run 建立新的 context、Plots 與 StopSignal，硬體 handles 可以重用，停止與錯誤狀態不能沿用。Executor、子實驗與 Schedule 共用這個 StopSignal，device setup 使用其 Event。ProgramBuilder 保留 acquire-local cancel flag，SNR 等 data-driven early stop 不取消整次 run。既有 retry、partial result、raise_if_error 與 GUI operation 分類保持不變，不引入另一個 outcome 或 runner。Progress bar 暫留環境注入。

Load／analyze 不要求硬體；run 在缺少必要 handles 或 binding 時拒絕，不自行查找或建立資源。單次 buffer、tracker 與 cache 歸 run。NotebookAdapter 每次 run 由 caller 提供的硬體與 driver mapping 建立 context；GUI 的 RunService 與 workflow RunSession 各自在 operation 邊界建立 context。

影響 acquisition 的實驗選項全部進 typed config，包括 T1 uniform。核心擁有設定驗證及環境無關預設；GUI 擁有編輯表示、標籤、expression 與 md／module library seed。核心分析不讀 live GUI 狀態。Hardware handles 不進 cfg 或 record。

共用包裝層隔離 caller cfg／options 與 core 工作輸入，成功後保留該次來源與選項。Record 的欄位關聯固定，不深拷貝大型 Result，不凍結 Figure artists。Record-owned cfg／options 可被使用者刻意修改，不提供 deep-freeze、讀取時複製或完整不可變歷史保證。

同步核心 options 與 Analysis 是實驗專屬 typed 資料，options 欄位擁有預設值，不另設 Notebook defaults 或動態簽名。Analysis 不含 Figure 或 writeback。Notebook 同步入口回傳 AnalysisRecord，互動工具也保留成功成果的來源、實際 options 與純圖。Notebook 不新增 writeback。

GUI 插件自行定義 typed 成功輸出的欄位。插件決定是否輸出選項、隨機 seed、時間或其他重現資訊。Framework 保存來源、插件輸出與圖，不追查插件的隱式依賴，也不保證完整 options 或可重現性。`params` 保持表單輸入，不在 Done 時替換成終態 options；不新增通用 committed-options owner。GUI 與 remote 讀同一份已提交輸出。

實驗跨入 zcu_tools 時使用公開絕對 import，預設置頂。只有具體重型依賴或初始化限制才延遲，不藉此外移模組或建立設定式 discovery。

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

Presentation owner 保留明確的呈現使用期與 release 責任，不能複製 mapping 後丟失所有權。跨存活 owner 接管仍拒絕。停止 producer、最後 refresh、釋放呈現与保留可保存的原生 Figure 分開處理。替換目前引用不關閉使用者持有的舊圖，release 後原生 artist 操作與 savefig 仍可使用。

圖已登記不表示操作成功。Core、host finish 或必要收尾失敗時不提交新成功 record，失敗診斷圖不得混入上一筆成功分析。

GUI application 持有結果與具名圖集合，保存不反向依賴 Qt widget。前端持有 canvas 與選圖狀態。截圖取目前選圖，分析圖保存涵蓋整個集合；不新增多圖排版編輯器。

### 三種分析生命週期

- T1 提供同步 analyze，消費 explicit RunRecord 與 typed options。Notebook 成功後發布 AnalysisRecord；GUI 發布插件定義的 typed 分析輸出、來源與圖。
- Singleshot GE 核心另提供 post_analyze。Post 消費 adopted primary 的來源與 calibration，不重新 primary fit，不讀未重新分析的 initial_state 表單。Primary／post 各自持有图與 GUI writeback proposal。成功 primary 替換清空目前 post，primary 失敗保留前次成功組。清空引用不關閉使用者持有的舊圖。不要求其他實驗提供空 post 方法。
- OneTone FluxDep 核心只負責 acquisition 與資料保存／載入，不實作 analyze。GUI adapter／INTERACTIVE plugin 自行分析，Notebook 使用獨立分析工具。重用現有 TwoLinePicker 與 Qt-free 選線規則，不把 Notebook 工具接回 core.analyze 或共用 Adapter 分析。啟動返回不代表成功；Done 驗證及收尾後發布該次成果，Cancel／failure 不取代上一筆成功。Notebook 保留實際 options，GUI 插件仍自行決定成功輸出的欄位。

Notebook run／load 正常返回後更新目前 RunRecord，清空目前分析與圖引用；失敗保留舊紀錄。同步分析成功才成組替換成果，分析舊 RunRecord 不替換目前 run。互動工具綁定自己的來源，晚到完成仍按 G5 發布自身成果，不取最新 run，也不新增 generation gate。清空引用不銷毀使用者持有的舊圖或結果，不新增完整歷史管理。

### 保存、發現與範圍

實驗擁有 RunRecord 到檔案的映射，caller 指定 exact path；save 不依目前 run。預設編解碼沿用 AXES_SPEC／GroupedAxesSpec 與 canonical Labber HDF5，將 cfg envelope 與純 Result 資料分開。保留 inner-first axes、Result-native shape、SI disk units、roles、labels、dtype、comment／tag 與 complex calibration。既有目的地仍拒絕，不新增覆寫、跨程序鎖或原子保存保證。

RunRecord.cfg 允許 None。正常 run 保有有效快照，load 對缺少或無法驗證的 cfg 保留有效資料與既有診斷，不用目前設定補成來源。需要 cfg 的分析或操作在其 owner 入口拒絕；無需 cfg 的分析仍可使用，不由 NotebookAdapter 一律阻擋。預設 canonical saver 保留缺 cfg 時拒絕，GUI 回填／寫回保留 skip。Shape、units、roles、dtype 的資料完整性驗證不放寬。

一般實驗使用預設 save／load，作者可用相同 RunRecord 介面明確 override，不要求額外 experiment_codec，也不聲稱支援任意 Python 物件。Workflow streaming／grouped artifacts 保留自己的契約，不改成 one-shot record 格式。PersistableExperiment、RecordExperiment 與 NotebookAdapter 的 save／load 只接本機路徑，不公開 server_ip／port；獨立 datafile／workflow transport 保留自己的契約。

Notebook save 吸收 unique path 的便利層並回傳實際 path，再呼叫 exact saver。Path helper 不建立檔案或鎖，不宣稱原子 reservation。AnalysisRecord 保留記憶體內來源與 options，分析圖另外保存；不新增完整 analysis session 磁碟格式，也不撤銷實驗既有保存義務。

保留 explicit catalog、v2／v2_gui 分離，以及單檔／package 並存。框架不要求固定內部檔名，不靠掃描或 import 副作用發現實驗。

最終範圍是所有實驗及其 Notebook／GUI caller，不以現有 GUI catalog 為上限。先以一般 T1、singleshot/ge、onetone/flux_dep 驗證同步、post 與互動分析，再分批遷移其餘實驗及必要的共用 helper。

過渡期間允許尚未遷移的實驗因舊契約而報錯，不為了保持它們可用而加入 pyplot 相容層。報錯須對應到未遷移項目，不能忽略已遷移路徑的 regression，也不能把過渡狀態當成最終交付。這不授權重寫 executor 或動畫框架，既有 acquisition、排程與硬體鎖機制保持。

## 顯式 cfg 與裝置環境

DeviceManager 的 registry、lock 與 close claims 屬於 instance，不提供預設 singleton。Notebook 與 GUI composition 持有或接收 manager；共享 driver 的 caller 共享同一 manager，不各自取得關閉權。既有 alias、close failure 與 driver lock 契約保留，不以 destructor 自動關閉，不把 disconnect 當作歸零或 RF off。

Notebook 使用 `CfgEnv(md, ml, device_manager)` 集中借用資源。顯式呼叫 `make_cfg(raw_cfg, CfgModel, env, overrides=...)` 時取得當次 device snapshot，再組裝 typed cfg。讀取失敗直接報錯，不退回舊 snapshot，不執行 setup。md 目前只作資源引用，raw cfg 中的 `md.foo` 在建立 dict 時取值，不新增 expression 或延遲求值。

CfgEnv 與 RunContext 分工不同。前者可跨次組裝重用；後者只有單次執行能力，不讓核心取得 md、ml 或 registry 管理權。底層 `assemble_experiment_cfg` 仍接顯式 snapshot，不讀硬體。GUI 的固定來源、CfgRef／revision 與 acceptance 不因 Notebook 入口而重新查詢硬體。

GUI 的 Use Simulate Env 入口由獨立 coordinator 編排，不由一般 soc_changed 通知偷偷註冊 FakeDevice。通過既有互斥檢查後，先經 owner disconnect 已連線的真實 devices，FakeDevice 保留。全部成功才確認或建立 FakeDevice、註冊、將其即時數值 reader 綁定 MockSoc，再交給 SoC owner。既有有效 mock source 不重設；matching predictor 的配套亦由 coordinator 組裝。

任一 disconnect 或組裝失敗就回報未完成，不自動重連，不假裝 ready。清理只處理本次建立的資源。綁定時確認 source 存在且可用，不接受先綁名稱、acquire 才查 registry。MockSoc 與 SimEngine 不持有 manager；每次 acquire 讀取 reader 一次，同次 acquire 保持固定 operating point。Reader 失敗不降級固定 flux。低層未綁 source 的固定 flux 與白雜訊模式仍可明確使用。

此入口允許之後再連線真實 device，不保證持續隔離，也不在每次 run 前自動斷線。自動建立 context 是後續方向，不納入本次 coordinator，也不預建 plugin registry 或空 hook。

## 取捨

- 明確 factory 使建圖依賴可讀，但任意第三方 pyplot 操作不再自動路由；adopt 處理返回的 Figure，不提供任意 pyplot 全域狀態隔離。
- 圖獨立於 Analysis，避免數值型別依前端改變，代價是 caller 必須把資料、來源與圖一起發布。
- 共用 NotebookAdapter 減少同步實驗的重複 wrapper，caller 直接使用核心 typed config／options。互動分析維持前端工具，不將 widget／session 協調塞進共用 Adapter。
- 不呈現仍建圖，保留實作一致性與保存能力，接受建圖及 artist 更新成本。
- 不以單一同步方法統一互動 session；通用繪圖不保證自動提供互動輸入。
- 接受未遷移實驗在中間階段報錯，避免維護第二套繪圖或分析協議；代價是必須逐項追蹤遷移與驗證，不能只用三個標準實驗通過推定整批完成。

## 七項核准政策與待落實接縫

下列政策已核准，部分共用繪圖與保存能力已落實，但完整 caller 遷移尚未完成。末節列出接線與驗證義務，不能將政策核准或底層能力通過視為端到端保證。

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

部分失敗保留成功項，不宣稱整組已保存，也不承諾跨檔原子保存。Save All 只保存未保存項，Retry 只重試未完成項。明確再次匯出另行表達，retry 不構成覆寫原始資料的授權。

### G5：晚到結果

Notebook 不新增 generation／request identity 的發布限制。較早啟動的互動或分析較晚完成時，允許覆蓋目前分析，操作順序由使用者負責。該次數值、來源 Result、實際 options 與圖集合仍須成組發布，不能混用不同操作的資料。

既有 GUI busy、operation token 與已取消 session 的生命週期規則不變。這項裁決不重新定義 run 中斷，也不刪除 GUI 的既有保護。

### G6：Notebook 互動前端

Notebook 互動分析直接使用 Notebook widget，不增加前端可用性 preflight 或 backend 偵測。建立或使用 widget 的實際錯誤正常傳遞，不自動降級到 inline、無頭分析或自動 Done。

同步分析的不呈現繪圖模式維持不變。它不是無頭互動工作流。

### G7：未遷移 adapters

GUI application 統一使用插件定義的 typed 分析輸出與 plots，不另要求通用終態 options。三個標準實驗的 concrete adapters 先接入新核心，其他實驗與 adapters 隨後分期遷移，不要求每個中間階段都維持舊 caller 可用。Interactive plugin／frontend 使用 session-owned 圖集合，不探測新舊簽名，不新增反射或雙協議 fallback。

完整清單須包含未出現在 GUI catalog 的實驗、組合量測 caller 與必要 helper。共用基底、fake 和支援框架需另外分類，不以 class 數量代替公開實驗清單。每項都需對應遷移範圍與驗證證據。

### 舊自訂 backend 退場

三個標準實驗與其餘實驗／GUI caller 接入 explicit factory／host 後，移除專案自訂的 pyplot routing backend，以及只服務它的 routing／scope 和設定。不以離屏 pyplot scope 延長舊核心的使用期。

退場包含共享 backend 的相關 GUI caller，例如 measure 與 fluxdep search，不能只刪除 backend 檔案而留下必要路徑未接線。這不是移除 Matplotlib rendering：原生 Figure／Axes、Qt canvas、Agg 與 ipympl 仍負責實際繪製。

共享 runtime 的 backend 選擇、host 初始化、shutdown handling 與 mathtext lock／prewarm 要分別核對。移除舊路由職責，保留或整理仍必要的容器、owner scheduling、attach／detach 及 rendering 初始化。

## 已實作的公開入口與驗證邊界

- [Experiment](../../../lib/zcu_tools/experiment/README.md) 說明 RunRecord／AnalysisRecord、nullable cfg、RunContext 與 default／grouped persistence。核心不再保存前一次結果。
- [Notebook](../../../lib/zcu_tools/notebook/README.md) 說明 NotebookAdapter、GEPostAnalyzer 與 FluxDepAnalyzer。同步分析使用 explicit source；互動工具的 Done／Cancel 與晚到成果保留各自來源。
- [GUI plotting](../../../lib/zcu_tools/gui/plotting/README.md) 說明原生 Figure 與 Qt presentation 的不同生命週期。Plots.finish 完成最後刷新，Plots.release 只釋放呈現。NamedFigures 保留完成後的原生圖。
- Measure 的 ArtifactKey 以 stage 與圖名識別成果。ArtifactTracker 記錄每張圖是否曾成功保存，SaveService 捕捉本次保存來源與目的地。Save All 選尚未保存的成果，個別失敗不撤銷先前成功項。
- [Session](../../../lib/zcu_tools/gui/session/README.md) 說明 DeviceManager owner 與 Use Simulate Env coordinator。一般 SoC 連線不建立 FakeDevice，coordinator 先完成真實裝置斷線，再發布已綁定來源的 mock 環境。

三個參考實驗已有各自的軟體接受紀錄。其餘核心與 callers 已整合。全 task 的專門 Standards／Spec 審查指出 Notebook 失敗提交、grouped persistence 接線及 typed analysis 等缺口；修正候選仍須驗證與重新審查。報告不等於正式接受或 landing 授權。修正前集中行為測試兩次得到 7431 passed、7 skipped；7 項因缺少 fluxonium_1.h5 未執行。全 repo type／lint 仍有既有診斷，不能以行為測試通過宣稱所有檢查全綠。

本次未操作硬體，也未重跑所有 Notebook cells 或 FFmpeg。既有 VSCode 60-frame 人工觀察只覆蓋當時的探針與版本。Standalone [liveplot](../../../lib/zcu_tools/plotting/liveplot/README.md) 保留自己的 backend 與 close 契約，不等同於新的 Plots host。

## 轉正為現況的條件

先完成三個標準實驗的 Notebook／GUI 接線與 G1–G7 契約驗證，再以完整清單逐項確認其他實驗及必要 caller 已遷移。以公開 seam 驗證 cfg snapshot、typed analysis、primary／post 來源、互動 Done／Cancel、具名多圖與保存失敗；以直接審閱確認責任、依賴與匯出。

舊自訂 backend 退場須有適用 GUI caller 的行為證據。刪除路由與專用測試後，仍須驗證新 host 的呈現、刷新、失敗與釋放行為；不新增「舊檔案不存在」的測試，也不以刪檔結果作為替代成功的證明。

不得用一般 T1 通過推定 interactive 或 post-analysis 已驗收。Notebook inline／widget 顯示與 Qt thread 接合需有對應觀察，不以數值測試代替。驗證完成後才將已落實部分寫入現行 ADR 與 module README。

---
status: accepted
---

# Persistence：保存權威與資料契約

## 脈絡

App memento、Experiment Data File、workflow run artifact、`params.json`、SampleTable 和 waveform asset 都會落盤，但保存目的不同。以檔案格式統一它們，會讓通用讀寫層決定量測語意、完成條件或使用者路徑政策。本決策依保存目的分配權威，不建立統一的 persistence service。格式、欄位和各類資料的操作細節由各 owner 維護。

## 決策

### 保存目的與 owner

- App memento 保存偏好和工作狀態，不是量測備份。各 app 擁有 schema、version、capture／restore 和觸發時機；`gui.session.persistence.SingleFileCaretaker` 提供單檔 I/O、格式錯誤的 default degradation 與 atomic replace。Measure framework 只在 lifecycle close 保存 Qt 視窗尺寸與左面板寬度，不保存或還原 domain session。SoC、device、entry／point、tabs／input 的檔案與重開政策由啟動腳本擁有，見 [[0064]]；setup／point 的 aliases.yaml 仍由 resources.entry 擁有。shared caretaker 不規定 close-only，也不訂閱高頻事件代替 app 決策。
- Experiment Data File 是正規的 `data.h5`，保存一個 Experiment Result 的全部 Data Variables。Labber 是供 Log Browser 使用的衍生輸出，不是原始資料權威。`experiment.axes_spec` 定義 typed Result／cfg 與 persisted axes、units、variables 和 metadata 的 mapping；`datafile` 擁有 [[0080]] 的通用容器格式與讀寫，不反向依賴 experiment。Container 只認 vars、docs、groups；cfg、metadata 的具名型別、編碼與驗證由 experiment 擁有。正常載入在 experiment 邊界還原 typed Result，不把 generic variable mapping 洩漏給分析端。
- Workflow run 保存跨步驟的提交與 audit。Engine 擁有 manifest、journal 與 best-effort iteration record.json；workflow 擁有 native 量測檔及圖形，並在 point record 記錄用途與檔名。Engine 每輪提供唯讀 absolute iteration_dir，不保存、追蹤或保證 workflow 檔案。單次量測 Completed 與保存分開，責任與失敗規則見 [[0062]]。
- `params.json` 是跨 workflow 的 typed parameter handoff。`QubitParams` 擁有 section 更新與 project identity；不把 sample arrays 或 dense curves 塞進參數檔。generic table storage 不負責這類語意。
- 參數容器由 `resources.entry` 組合條目、setup 範本與獨立工作點。值與 opaque metadata 共用一份文件，setup／point 重用 `resources.parameters` 的 2.0 header、metadata 讀寫與 value staging；DocumentStore 擁有單檔衝突與原子提交，不換算數值。Entry 不產生、解釋或驗證參數 metadata，clone 原樣複製；library 來源記錄與本條目 ledger 引用驗證仍由 library owner 擁有。Accepted executor 對參數只提交 caller 提供的 metadata。GUI context owner 與 Notebook 經公開 capability 使用 entry。GUI proposal 留在 frontend，Apply 才轉成 resources concrete items。兩者共用 accepted executor，resources 不 import frontend 或 experiment，由 C8 分層與 C9 無循環 contract 約束。
- 獨立參數能力由 `resources.parameters` 擁有。Caller 只傳既有文件路徑與自己的 Pydantic value model 給 `open_parameters`，不組裝 DocumentStore，不要求 entry、kind registry、general、roots 或 ledger。`components` 保存直接值，`metadata` 是 opaque YAML mapping 旁表。一般欄位回傳數字等原值，metadata 從獨立入口讀寫。DocumentStore 仍是唯一單檔交易 owner。Resources 不查 metadata 的來源，不解釋 stderr 或 comment，也不因改值自動產生或清除 metadata。Caller 在同筆 edit 中明確替換 metadata mapping，提交沿用既有 mapping-leaf 合併與衝突規則。FieldView 與完整 value-model staging 由 resources 共用，沒有第二套交易或 validator framework。依賴仍由 C8 分層、C9 無循環與 C10 有狀態能力模組獨立 contract 約束。
- 使用者於 2026-10-13 裁決：「可以改metadata」。新參數容器寫 `format_version: "2.0"`，只讀 major 2，保留同 major 的 forward-minor 契約。入口直接拒絕舊 major 或任何含 `provenance` 旁表的文件，錯誤含路徑、讀到版本與期望版本 2.0。不雙讀、不附轉換工具，由使用者自行修改既有文件。這項參數格式變更不改 generic DocumentStore、ModuleLibrary 或其他成果的版本契約。
- Setup 與每個 point 各自的 authored alias 表由 `resources.entry` 保存為獨立的 `aliases.yaml`，不是 memento 或 ledger。ContextService 擁有有序表的語法與 Apply 時機，resources 擁有檔案表示、建立時複製及單檔原子替換。MCP 的單次 aliases／target 覆寫不保存它。
- SampleTable v2 是跨 producer／consumer 的座標交換契約。`resources.sample_table.schema` 擁有座標、單位及解析語意；條目樣本使用 append-only JSONL，讀取取每個 run_id 最後一整行，CSV 只作匯出。舊 CSV reader 留在 root `migrations/` 供離線轉換，runtime 不作 fallback。
- Waveform asset 擁有播放 arrays、reference time axis 和 duration。Core waveform_format 驗證 samples 與編解碼 NPZ，只驗 recipe JSON 語法。Resources 管理 formula 語意、命名與實體 I/O。ModuleLibrary Spec 只保存 asset key。Native arb cfg 保存 key 與必填本機絕對 path，不另存播放長度或 provider。Program/sim 讀 path 的 samples，在完整 asset duration 取樣，不解讀 formula，也不查 global DB。

### 設定快照與 acquired 事實

設定快照的正本只在 `data.h5` 的 `/cfg`，以 CfgSnapshot 保存 values、cfg_type 與 schema_version。Labber 從同一份 prepared payload／snapshot 產生，供 Log Browser 使用，不是另一份讀回權威。Notebook 與 GUI 都讀 native `data.h5`。

Ledger 的 acquired 只記 run_id、tab／tab_id、experiment、point、roles 與事件時間，不保存 cfg_summary，也不從 Form 或 RunRecord.cfg 序列化第二份設定。只量測未保存仍可追加 acquired，不建立 run 目錄。

RecordsLedger 讀取 envelope 與依 kind 選出的 payload 時忽略未知欄位，包括舊 1.0 行的 cfg_summary。讀取不改寫原行或附檔，get 的 raw event_json 保留未知內容。已知欄位型別、必填欄位、格式 major、kind、entry identity 與 JSON 損壞仍按契約拒絕；append 仍驗證已知欄位。不升格式 major，也不做 strip／rewrite migration。

使用者於 2026-10-14 對保存位置選擇「A」，正本只放資料檔。對前端讀回裁定：「Notebook與GUI應該都讀data.h5，只有log browser讀Labber檔。」2026-10-15 確認：「1可以，2採用忽略不支援欄位的方式，3同意，4一起做吧」。其中 ledger 採忽略未知欄位，不改寫歷史行；Labber-only 資料先離線遷移，Load 不回填表單。

### MCP call log 與 ledger

MCP call log 是 best-effort 除錯紀錄，可以截斷、清除或停用，不併入 ledger，也不能代替 ledger 的事實紀錄。Ledger 的 origin 只標示 gui、notebook 或 mcp，不帶 MCP call id。跨 MCP、GUI 與 ledger 傳遞 call id 會增加耦合，卻無法讓可丟棄的 call log 成為可靠證據。使用者於 2026-10-08 以「採用建議」確認刪除這條跨層 call id 傳遞，只保留 origin。

這個分工不增加跨檔原子性、fsync 或掉電保證。Ledger append 與輸出檔案的失敗邊界仍依本篇，不因 call log 存在就視為已提交。

### Run 輸出位置與 writer 權責

新格式根目錄統一為 `result/`，與 GUI runtime 一致，不另設 `results/`。`.gitignore` 保留 `result/`，移除 `/results/`。
使用者於 2026-10-20 裁決：「好，統一result，那gitignore也做處理」。

Entry 只由 `result/<name>/` 構成。`resources.entry.ResultEntry` 提供條目的不可變身分、exact Result path、Database 輸出位置與 scoped ledger。`create`／`open` 只檢查 Result，不檢查或建立 `Database/<name>/`。第一次 Save 才由 writer 建立所需的 Database 目錄，已有的同名 Database 照用，不另加混雜偵測。

`rename_entry` 只移動 Result，不改內容或 UUID，也不移動 Database 舊輸出。改名後重新 open，新 Save 使用 `Database/<新名>/`，舊輸出留在原處。Database 不是 entry 的存在或 clone 身分邊界，因此沒有雙目錄改名與復原協定。

使用者於 2026-10-20 裁決：「甲，簡單的機制加一致的規則比非預期的兼容邏輯好維護與使用，同樣名字的Database會混雜屬於有意不處理的設計。」對改名裁決：「改名時Database不動，Database是輸出位置，不是什麼要綁定的目錄」。

`SaveLayout` 根據這些 paths、前端啟動實驗的名稱、run_id、保存時間與 artifact key 計算各格式的 exact outputs。它不建立目錄或寫檔，也不從舊的日期目錄猜 entry。C8 分層與 C9 無循環 contract 涵蓋 experiment 使用 datafile／resources 的方向。

新 run 保存契約把資料夾綁定一次 acquisition。第一次 Save 才建立 `Database/<name>/runs/<experiment>_<run_id>/`，名稱保留完整 run_id。再次 Save 使用同一 run_id，更新其中的 `data.h5`、具名圖片與 `analysis/<pane>.json`。圖片另有 Result 的 point／全域 figures 副本，檔名為 `<experiment>_<run_id>_<圖名>.png`。對等 paths 使用同一份 bytes。

Labber 每次 Save 都在 `Database/<name>/Labber/YYYY/MM/Data_MMDD/` 取 `<experiment>_<id6>_<序號>.hdf5` 新檔名。實驗名中的 `/` 轉 `_`，id6 是 run_id 的後六碼。序號從 1 起避讓既有檔案，一個 run 可以對應多個 Labber 檔。只量測不存檔時，acquired 事件不建立 run 資料夾。保存不改 acquisition 的開始、結束或完成狀態。

使用者於 2026-10-19 表示：「Labber的檔名是run_id對人太不友善了，應該要有實驗名比較好，id可能取後6位用於grep」，並裁決「採用建議」。實驗名使用各前端啟動名稱，GUI 用 adapter 名稱，workflow 用 manifest.workflow，Notebook 用 spec.tag。Migration 傳入 legacy.experiment 計算 native 位置，不重命名舊 Labber 檔。

Workflow 的兩側目錄都使用 `workflows/<workflow名>_<run_id>/`，保留完整 run_id。Workflow 名中的 `/` 同樣轉 `_`。Result 保存 manifest／journal，Database 保存 acquisition 與圖，iteration 位於其 `iter/` 下。可讀前綴不取代 metadata 中的 run 身分，也不引入額外的身分側檔或新的序號避讓規則。

使用者於 2026-10-20 提出：「不然名字包含完整run_id，但前綴也有實驗名讓用戶看？」並以「2可以 4可以」確認完整 run_id 前綴命名與後續遷移範圍。對 Result 圖片與 workflow 套用相同規則回覆：「3」。Labber 維持原有 id6 加序號的檔名，遷移工具與消費端改用新版面，不新增舊路徑相容邏輯。

`experiment.native_persistence.prepare_run_data` 從 typed RunRecord 與 axes spec 建立同一份 SI／離散 payload 和 cfg snapshot。Native 與 Labber writers 分別消費它，不先寫 native 再讀回轉成 Labber，也不互相呼叫。Datafile writer 只處理自己的格式與 caller 指定的目的地，不選路徑、不擷取 live context、不決定 batch。`experiment.artifacts` 的共用保存編排負責 artifact 選集、輸出順序與逐輸出失敗隔離，一個 writer 失敗不阻擋其他輸出。

RunMetadata 的 run_id、時間、completion、RunSnapshot 與 SoftwareProvenance 是 caller 提供的歷史事實。

舊 Native `zcu.experiment-data` 2.1 的歷史欄位規則如下；新檔由 [[0080]] 的 zcu.container 3.0 保存，欄位進入 docs/metadata。`started_at` 接受 UTC ISO 字串或 None，`completion` 接受 complete、partial、stopped 或 unknown。None 與 unknown 表示缺少歷史證據，不從檔案時間、檔名或 run_id 補造，也不加遷移來源標記。正常量測路徑仍必填開始時間，只產生已知完成狀態。2.1 的 started_at attr 必填，以空字串編碼 None；completion attr 以 unknown 編碼未知。缺失 attr、非法型別與非法時間仍拒絕。

同一個容器 reader 長期接受舊 major 2，轉成 Container 後寫回一律使用 3.0，不再提供 major-2 writer。來源完整子樹以 opaque groups/legacy 保留。同 major-3 的 forward minor 與未知內容在 load、修改、save 往返時保留，不降版。格式解碼由 [[0080]] 擁有，實驗欄位是否合法由 experiment 的文件 schema 驗證。

GUI Load 與共用 typed loader 保留未知歷史值，不把 unknown 改成 complete。Workflow `load_trace` 是 complete-only 分析入口：None＋complete 可載入，unknown 與 partial／stopped 同樣拒絕，因為未知完成度不能證明 acquisition 完整。Notebook 仍只載入 RunRecord，不把載入資料變成 acquisition evidence。現有前端不顯示或排序這兩個 metadata 欄位，本格式變更不新增 UI。

使用者於 2026-10-19 對 started_at 可為 None、completion 加 unknown、不加來源標記的建議裁決：「可以，採用建議」。

`RunMetadata.device_snapshot` 保存 Run 開始前讀取的全部已連線 device 設定。GUI 在取得 operation hardware lease 後、adapter.run 前讀取。Notebook 在使用者的程序內同步執行，由 caller 擁有資源，不新增 lease；snapshot 在建立 RunContext 後、core.run 前讀取。

兩路每台只呼叫一次 get_info。任一台失敗就拒絕 Run，不寫設定、不進 adapter/core，也不用 cache 補值；無 device 記 `{}`。Snapshot 表示實驗前狀態，不是 setup 後或每個 core 的 checkpoint。後續 sweep 不改寫它，實驗使用的值由 cfg 的 dev、flux_unit、sweep 範圍與 persisted axes 記錄。

Experiment 將 snapshot 寫入 Container 的 `docs/metadata.device_snapshot`。Key 是 device 名稱，value 是 get_info 的設定。Snapshot 為 None 時寫 JSON null，空 mapping 則寫 `{}`。舊 native 2.x 的 `/device` 由 legacy reader 映入相同欄位。單位沿各 DeviceInfo 的儲存欄位。YOKO 的 `value_unit` 是 A 或 V，與 current／voltage mode 一致；Fake 是 native。Value 與 rampstep 沿該單位保存。RF 頻率為 Hz，功率為 dBm，不加 `value_unit`。Snapshot 不另造單位欄位或換算值。GUI 和 Notebook Save 都使用當次 Run 的固定 snapshot，不重讀硬體或取下一輪值。

Labber 成功後，保存編排將 labber_path 放入 SaveReport 與 saved ledger 事件，不寫入 RunMetadata。輸出結果與 saved ledger 事件由保存編排產生，RecordsLedger 只驗證與追加事件。Ledger 追加失敗不能把已寫好的檔案改標為量測失敗，也不代表那些檔案已回滾。

`experiment.artifacts.AnalysisDocument` 與 encoder 擁有窗格 JSON 格式，信封也可由 to_doc 表達為 [[0080]] 的 docs/analysis。現有 artifact 仍寫 analysis/<pane>.json，SaveLayout 與 ledger 不改。`experiment.artifacts` 定義 ArtifactItem 與固定 DataItem／FigureItem／AnalysisItem。一次 SaveArtifact 包含多個 items，`save_artifacts` 依種類選 native／Labber／PNG／JSON，逐項編碼一次，對等目的地共用 bytes。

DataItem 保留 typed run 準備成功或失敗的結果；cfg=None 或 producer 準備失敗只使對應 data outputs 失敗，同選集的圖與分析仍嘗試保存。每次呼叫回傳最新 SaveReport 並追加 saved；ledger 失敗向外傳遞，不 retry 或 rollback。共用核心不依賴 GUI 或 Qt，由 `.importlinter` 的 `artifacts-core-no-ui` contract 檢查。

`NotebookExperiment.save_run` 綁定 PointView，使用 entry owner 的 capture_point_storage，保存 adapter 驗證過的 acquisition source。SavedRun 固定該 run 的資料、context 與 evidence，PartialSaveError 保留同一 handle 供明確再存，不建立恢復收據或自動 retry。

GUI Save／Save All 與 MCP 保存使用這條 run 保存流程。Save All 包含已保存的 artifact，每次重寫全部選定成員。GUI 在 owner thread 同步完成並回傳完整 SaveReport，不建立 Save handle、worker 或 save-busy phase。Data pane 唯讀顯示每個 output 的 path、status 和 error，不讓使用者改目的地。Notebook 的本機單檔 save 保留原入口，不猜格式；`save_run` 使用 bound PointView 的公開 capability。C8 與 C9 涵蓋 notebook → experiment/datafile/resources 的依賴，C12 保持 datafile 不依賴 stateful resources。

GUI Native Load 由 LoadService 呼叫 datafile.load 讀成 Container，使用 experiment.metadata_from_container 驗證 metadata 並比對 experiment tag，再交 adapter.restore(container)。BaseAdapter 使用核心 from_stored，共用資料優先轉換見 [[0069]]。核心與 Notebook 保持 typed RunRecord，GUI 只持有 opaque record 與明示 cfg_issue，不探測 cfg／result。完整 RunMetadata 沿 RunRecord.metadata 保留原 run_id、時間、completion 與 Labber 子欄位，不包含輸出路徑，也不產生 acquired。Load 不回填表單，不以目前 Form、library、default 或 live device 補回歷史 cfg。

使用者於 2026-10-16 選擇「A」，採用修改文件、不新增 GUI 入口的方案。原始 cfg 保存在 native 資料檔，透過公開 datafile.load 離線讀取 Container 的 docs/cfg。GUI 不提供原始 cfg 檢視入口，不擴大本次範圍。

相同 opened entry 且可定位歷史 point 時，context owner 一次讀取檔案的 point 和已解析 alias names，固定 Load 時的持久值供 Analyze／Post／Writeback。檔案不保存原始 point／library 版本，不能把這份來源當成原始量測參數。跨 entry、EMPTY 或無歷史 point 時沒有 analysis source，不回退目前 alias。Save storage 在 Load request 時獨立捕捉；同 entry 沿歷史 point／run_id 保存，跨 entry 在 mapping 和 I/O 前拒絕。

Load Data 同時提供 File 選檔與 opened entry 的 native 檔案清單。清單掃描 `Database/<entry>/runs/*/data.h5`，完全不讀 ledger。Run 身分、experiment、point、開始／結束時間與 completion 來自各檔的 RunMetadata／RunSnapshot；未知歷史時間保持空白，completion 保留 unknown。Experiment 的 `read_run_metadata` 透過 datafile.read_doc 只讀並驗證 metadata，不讀 cfg 或量測 arrays，也不保證整份資料可載入。Metadata 讀取失敗時，run_id 為空字串，experiment、point、時間與 completion 欄位留空。清單保留 exact Path 與既有錯誤資訊，選取仍回傳 exact path，由既有 Load 路徑回報失敗。清單不解析新舊目錄名來補欄位，避免讓路徑命名成為第二份歷史身分來源。查詢不建立目錄或事件，遷移也不手造 ledger。

使用者於 2026-10-20 確認檔案清單與成功 Load 分開：「我以為只要是data.h5就可以看到，能不能Load另說」。對清單完全不讀 ledger、顯示欄位只從檔案 metadata 讀取的方案回覆：「可以」。

### Setup 與 point alias 表

Setup 的 alias 位於 `<entry>/aliases.yaml`。每個 point 的 alias 位於 `<entry>/points/<p>/aliases.yaml`，與 `point.yaml`、`module_cfg.yaml` 並列。從 setup 新建 point 時有檔才複製，clone 則只複製來源 point 的檔。各表之後獨立修改，不繼承或覆寫 setup。既有 point 缺檔視為空表，不自動複製、不回退 setup。

使用者於 2026-10-20 裁定：「採用乙，alias. yaml的獨立檔案，找不到就視內容為空，與其他兩個資料容器並列」。

`EntryAliases` 保留名稱到原字串的 mapping 順序、空列與 `$` softlinks，不保存解析後名稱。檔案使用 `format: zcu.aliases`，寫入 `format_version: "1.0"`，讀取接受 major 1 的 forward minor。空 mapping 與缺檔都讀成空表。格式、結構或 alias 語法錯誤直接失敗，不回退 preset。

Dialog Apply 先驗整表，再由 entry owner 寫 sibling temporary 並 atomic replace，成功後才安裝 memory 和發布變更 fact。失敗保舊表、舊檔與 dialog draft，可修正後再 Apply。未選 point 時 Apply 以 `FailedPreconditionError(no_context)` 拒絕，不寫 setup。普通 `set_alias_rows` 只改 memory，不要求 point。Apply 不關閉 dialog，Cancel 不撤回先前已 Apply 的內容。

切換 point 載入該 point 的表，Open／Reset 捕捉目前 point 的 alias。已保存的空表或空列不自填。Run 保留原 capture，後續改表或切 entry 不改歷史 Analyze／Writeback／Save 的來源。這個單檔替換不承諾跨檔交易、多 writer merge 或掉電恢復。

### 資料表示與完整性

參數容器的 YAML、快照與視圖使用同一個工作單位數字，metadata 內容也不換算。元件 model 擁有型別、extra 策略與驗證。這條規則不改 Experiment Data File 的單位契約；datafile 的格式責任仍由 datafile 擁有。

參數容器框架只消費已註冊的 model 與角色宣告。具體 kinds、欄位與內建角色由 `zcu_lab.components` 擁有。Lib 外的組合根顯式傳入共用 ComponentRegistry，在 startup 註冊，不在 import 或 reload 時註冊。`.importlinter` 的 `entry-definitions-composition-only` 與 `framework-no-user` 禁止框架反向 import 定義。使用者 model 可用 validator 處理舊欄位；框架不猜兼容規則。

DocumentStore 分開保存 raw YAML 節點、驗證後的 normalized tree 與 public typed snapshot。一般讀取及空 edit 不寫檔；非空 commit 以最新 normalized tree 合併與驗證，再把差異套回 raw 節點。Model 的讀檔轉換在下一次 commit 寫回，未改節點保留原樣。

ModuleLibrary 是讀取可寫檔的明確例外。Library owner 先驗 header、容器與可定址名稱，再逐條驗證 Spec 及所屬 provenance。壞 Spec、entry-local provenance 或不存在的本地事件使該 root 及附屬來源被移除，不影響合法兄弟。孤立或不可定址的 provenance、header 或容器錯誤整份拒絕，不備份、不清理。Ledger I/O、schema、附檔與非預期錯誤傳遞，不降級為壞條目。

DocumentStore 的同一把 sidecar lock 涵蓋初始讀取、refresh、edit 進入及 commit 重讀的清理。清理候選通過完整驗證後，store 先將完整原檔逐 byte 複製成同目錄、不覆寫既有檔案的時間戳 .bak，再以 sibling temporary 與 atomic replace 清理主檔。成功後只寫一行 warning，列出備份路徑與移除 roots。合法檔與 header-only 不重寫；snapshot 仍只讀合法的已發布記憶體，不新增 GUI／MCP 顯示。

備份失敗不替換主檔；replace 失敗保留原主檔及完成的備份，兩者都不發布候選。讀取清理已完成後，後續 draft、body 或 conflict 失敗不回滾清理。新 draft 仍嚴格驗證，不能以清理吞掉 caller 的壞值。ResultEntry clone 先用來源條目的相同政策清理，再複製文件及匯入剩餘合法來源。此例外只適用明確啟用政策的 library，不擴張到 setup、point 或其他 store，也不承諾跨檔交易或掉電恢復。Library 版本只看容器，見 [[0072]]。

使用者於 2026-10-15 裁決：「其實我覺得很簡單，讀取時發現有問題，就重命名當前檔案或複製成.bak檔，本體留合法的即可，也不用處理顯示的問題。」對先複製完整原檔、備份名加時間戳的建議回覆：「照建議做」。

Format 與 format_version 只描述容器結構。檔頭不符或 major 不同時拒絕載入；forward-minor 保留容器層剩餘未知欄位，不覆寫元件或 nested model 的 extra 策略。

任何 minor 都由元件定義者的 model 決定 defaults、validators 與舊寫法演進。Forbid 拒絕未知欄位，ignore 丟棄它們，allow 保留並公開它們；ignore 丟棄的欄位在非空 commit 時移除。

框架不提供版本遷移鏈或相容 loader。Entry 不為值寫入或格式轉換產生 metadata。

Experiment persistence 使用 inner-first axes；disk payload 由 Result-native shape 對應，save／load 不要求 caller 補 transpose。物理單位由 experiment mapping 明示；不能為未定義物理量的 scalar 捏造 A／V。Data Variable 是結果語意，state／phase 等離散座標仍是 axis。正規 data.h5 的各 Data Variable 可以使用不同 grid，共用一份結果身分與 metadata。Labber 預設的 canonical one-shot grouped 輸出要求共同 grid 與明確 variable-to-channel mapping。必需 variables 由 experiment 決定，不由 generic writer 猜測。異質 streaming workflow 有獨立 layout 與 completeness 規則，不能因同為 HDF5 就視為同類檔案。

Grid 不同的 grouped 實驗可由 spec 明確宣告 independent Labber layout，各 variable 保留自己的 1D／2D axes。Writer 不因 grid 不同自動切換，native 格式與載入契約不變。使用者於 2026-10-19 要求「Labber存檔問題請獨立修」。採用明確宣告，讓異質 grid 的實驗能匯出 Labber，同時保留未宣告實驗的 common-grid 驗證。

結構完整、量測完成與持久提交不同。Complete Experiment Result 指所需 variables 齊全，不保證所有預定測點完成。Operation stopped／failed 不單獨決定已提交資料能否分析。Autofluxdep 在 node 正常返回、patch 驗證後提交 row；mid-node exception 不把半列視為 committed。已取得 measurement 而 fit／provide 失敗時，保留 raw measurement 並分別標示提供結果的狀態；skip 以 audit event 表示，不以 NaN 推斷。Node-row 與 flux-level commit 是不同進度；partial／stopped／failed run 仍可保有先前已提交的 evidence。Terminal exports／reports 取用 committed journal evidence，不能反向覆寫 canonical row。

2026-10-21 註：autofluxdep 已移除，上一段的 node-row、flux-level commit 與 terminal export 規則是歷史記錄。現行 workflow 的提交與保存邊界見 [ADR-0062](0062-experiment-workflow.md)，不把舊 node 契約套到新 engine。

### 失敗、路徑與遷移邊界

Run accept 先追加一筆 accepted，再逐項提交參數、library 或樣本。項目失敗以 `WritebackApplyError` 公開 accepted event_id、成功前綴 applied 與原始 cause，保留前項與事件，停止後項。Notebook 直接看到此錯誤；GUI context owner 發布實際成功前綴，GUI service 標記前綴後原樣拋 cause，wire 不改原 producer 分類。GUI writeback 只接受 SavedPoint，未選點時在 append 前拒絕，提示先選或建立點，不自動建點或寫 point=null 樣本。一般 sample append 的 point 選填契約不受影響。

Memento JSON 損壞或不符 codec 格式時，共用 caretaker 回 fresh defaults，restore outcome 帶明確錯誤供 host 顯示。載入不移檔、不備份，也不新增保存保護狀態。Host 之後照原有時機保存。FormInput 的固定 data／selections 格式與遞迴還原見 [[0079]]，所有 caretaker hosts 共用格式失敗的處理。

Provider、其他 restore／apply 故障仍傳播。一般 read I/O 失敗沿用錯誤 outcome 與 defaults，不冒充 decode 失敗。Caretaker 不懂 cfg，也不自行決定何時保存。量測資料、參數和未知 artifact 格式版本不能沿用 memento 的 whole-file default degradation，必要 variable、shape 或 unit 不符時仍須回報可定位的錯誤。

Autofluxdep artifact 建立失敗不啟動 run；canonical row write／flush／journal 失敗會使 run 失敗，並保留此前已 flush 的證據。Terminal export／report 失敗要顯示，但不把成功量測 row 改標為 measurement failure。Stop 仍嘗試 finalize；取消請求不保證 finalize 成功，也不表示 rollback。單檔 replace、HDF5 flush 和 journal append 不是跨檔交易或掉電持久保證；不可從 manifest 的 accepted 標記推導任意 crash 後可安全 resume。跨檔失敗窗口的讀取、修復和 resume 規則尚未驗證。

2026-10-21 註：上一段描述已移除的 Autofluxdep artifact，不是現行 runtime 的失敗契約。現行 Experiment writer 依下段處理 exact path；Workflow 保存見 [ADR-0062](0062-experiment-workflow.md)。

Experiment writer 寫 caller 指定的 exact path。Labber 拒絕既有目的地；native 預設也拒絕，只有明確 replace=True 才用同目錄 temp、關檔後 atomic replace，失敗清 temp 並保留舊目的地。SaveLayout 的 Labber suffix 只避讓當時已存在的檔案，不是跨程序 filename reservation；collision 由上層回報該輸出失敗，不隱藏 retry。單檔 replace 與 ledger append 不是跨檔交易，也不保證掉電恢復。Run identity 由 run-scoped metadata 記錄，不由可讀 slug 或檔名猜測。Workflow manifest 連結 metadata root 和 heavy-data root，不把 exports 當另一份原始權威。

Normal runtime 只接受各自契約指定的格式，不猜測 legacy shape／unit；版本必須與 artifact kind 一起判斷，合法 streaming v1 不因數字小於 one-shot v2 而視為同類 legacy。Experiment 的舊 converter 和 GUI fallback 已退休。SampleTable legacy 資料只能由 caller 明確呼叫 pure DataFrame conversion；此函式不自動改寫 CSV。QubitParams 的 project identity migration 由其 owner 管理，不因本 ADR 授權寫入使用者資料，也不建立跨資料類型的共用 migration layer。

### 共享資料與引用

`params.json` 的 QubitParams 與工作點的 `Q1.params` 是獨立資料，不自動同步。QubitParams 的 caller 以 typed 語意方法讀寫；更新一個 section 保留其它已知和未知 section。Timestamp 只表示 section 修改時間，不是 dependency provenance 或自動失效。單一 typed module 不等於跨程序互斥，也不代表 mtime 同步可解決多 writer 衝突。

SampleTable v2 明示 `dev_value` 的 A／V base unit，flux 與 row frame 由獨立欄位表示。Flux 解析依 explicit、row-frame、fallback-frame 順序，並標明每列來源；fallback frame 必須與該列 unit 一致。自描述 unit 不保證缺少 flux／frame 的列能不靠外部輸入算出 flux。分析用 correction／alignment 不默默寫回原始樣品資料，也不自動修正 explicit flux。Autofluxdep export 用 run 建立時的 unit snapshot，不以 export 當下的 live device state 重讀舊結果。

2026-10-21 註：autofluxdep 已移除，上一段最後一句只保留舊 export 的歷史契約。現行 SampleTable 座標與單位契約見 [sample_table owner](../../lib/zcu_tools/resources/sample_table/README.md)。

Asset repository 的 rename／delete 只處理資產，不掃描或改寫 ModuleLibrary 的 nested references。舊 key 可能因此成為 dangling reference；使用端解析缺失資產時明確失敗，由 caller／使用者修正引用。這避免 repository 反向理解 experiment cfg，也避免跨 owner 批次寫入。Capture-bound assets.path_for 驗 key 與檔存在，不載入 samples。Native cfg 的 path 以本機絕對路徑與 POSIX separators 序列化。Data.h5 不附帶 NPZ。移動 asset 或跨 OS 後，caller 須明確重定位 path；讀量測 arrays 不需要重播。覆寫同 path 會改變之後播放的 bytes。本決策不引入 asset versioning、copy-on-run、packaging 或自動引用遷移。

## 後果

維護者先找保存對象的 owner，再查其就地格式契約：[`datafile`](../../lib/zcu_tools/datafile/README.md)、[`experiment`](../../lib/zcu_tools/experiment/README.md)、[`QubitParams`](../../lib/zcu_tools/resources/qubit_params.md)、[`parameter entry`](../../lib/zcu_tools/resources/entry/README.md)、[`SampleTable`](../../lib/zcu_tools/resources/sample_table/README.md)、[`waveform assets`](../../lib/zcu_tools/resources/waveform_assets.md)、[`program waveform`](../../lib/zcu_tools/program/v2/README.md) 與 [main GUI](../../lib/zcu_tools/gui/app/measure/README.md)。本 ADR 不保證多檔 all-or-nothing、crash safety、多 writer 協調、autosave、resume、browser 或任何使用者資料遷移。

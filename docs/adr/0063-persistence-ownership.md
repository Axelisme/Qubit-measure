---
status: accepted
---

# Persistence：保存權威與資料契約

**Last updated:** 2026-11-01

## 脈絡

App memento、Experiment Data File、workflow run artifact、`params.json`、RecordTable 和 waveform asset 都會落盤，但保存目的不同。以檔案格式統一它們，會讓通用讀寫層決定量測語意、完成條件或使用者路徑政策。本決策依保存目的分配權威，不建立統一的 persistence service。格式、欄位和各類資料的操作細節由各 owner 維護。

## 決策

### 保存目的與 owner

- App memento 保存偏好和工作狀態，不是量測備份。各 app 擁有 schema、version、capture／restore 和觸發時機；`gui.session.persistence.SingleFileCaretaker` 提供單檔 I/O、格式錯誤的 default degradation 與 atomic replace。Measure framework 只在 lifecycle close 保存 Qt 視窗尺寸與左面板寬度，不保存或還原 domain session。SoC、device、entry／point、tabs／input 的檔案與重開政策由啟動腳本擁有，見 [[0064]]；setup／point 的 aliases.yaml 仍由 resources.entry 擁有。shared caretaker 不規定 close-only，也不訂閱高頻事件代替 app 決策。
- Experiment Data File 是正規的 `data.h5`，保存一個 Experiment Result 的全部 Data Variables。Labber 是供 Log Browser 使用的衍生輸出，不是原始資料權威。`experiment.axes_spec` 定義 typed Result／cfg 與 persisted axes、units、variables 和 metadata 的 mapping；`datafile` 擁有 [[0080]] 的通用容器格式與讀寫，不反向依賴 experiment。Container 只認 vars、docs、groups；cfg、metadata 的具名型別、編碼與驗證由 experiment 擁有。正常載入在 experiment 邊界還原 typed Result，不把 generic variable mapping 洩漏給分析端。
- Workflow run 保存跨步驟的提交與 audit。Engine 擁有 manifest、journal 與 best-effort iteration record.json；workflow 擁有 native 量測檔及圖形，並在 point record 記錄用途與檔名。Engine 每輪提供唯讀 absolute iteration_dir，不保存、追蹤或保證 workflow 檔案。單次量測 Completed 與保存分開，責任與失敗規則見 [[0062]]。
- `params.json` 是跨 workflow 的 typed parameter handoff。`QubitParams` 擁有 section 更新與 project identity；不把 sample arrays 或 dense curves 塞進參數檔。generic table storage 不負責這類語意。
- 參數容器由 `resources.entry` 組合條目、setup 範本與獨立工作點。值與 opaque metadata 共用一份文件，setup／point 重用 `resources.parameter_metadata` 的 2.0 header 與 metadata 讀寫，以及 `resources.parameter_fields` 的 value staging；DocumentStore 擁有單檔衝突與原子提交，不換算數值。Entry 不產生、解釋或驗證參數 metadata，clone 原樣複製；library 只保存 Spec，不建立模組來源或查 ledger。Accepted executor 對參數只提交 caller 提供的 metadata。GUI context owner 與 Notebook 經公開 capability 使用 entry。GUI proposal 留在 frontend，Apply 才轉成 resources concrete items。兩者共用 accepted executor，resources 不 import frontend 或 experiment，由 C8 分層與 C9 無循環 contract 約束。
- 使用者於 2026-10-13 裁決：「可以改metadata」。新參數容器寫 `format_version: "2.0"`，只讀 major 2，保留同 major 的 forward-minor 契約。入口直接拒絕舊 major 或任何含 `provenance` 旁表的文件，錯誤含路徑、讀到版本與期望版本 2.0。不雙讀、不附轉換工具，由使用者自行修改既有文件。這項參數格式變更不改 generic DocumentStore、ModuleLibrary 或其他成果的版本契約。
- Setup 與每個 point 各自的 authored alias 表由 `resources.entry` 保存為獨立的 `aliases.yaml`，不是 memento 或 ledger。ContextService 擁有有序表的語法與 Apply 時機，resources 擁有檔案表示、建立時複製及單檔原子替換。MCP 的單次 aliases／target 覆寫不保存它。
- `RecordWriteback` 只 append adapter 組裝的任意 JSON object，不選座標裝置、不注入 `dev_value/dev_unit`，也不因裝置資料缺失而省略 proposal。框架只補 `component`、`run_id`、`point`、`at` 與 `source` provenance 信封。`component` 來自明示的 table binding，producer 同名欄位不能覆寫它；舊列缺此欄仍可讀，不補寫。格式由 adapter 間的 policy 決定，見 [lab authoring 的 Record policy](../../zcu_lab/v2/measure-authoring.md#record-policy)。元件記錄放在 `records/components/<target>.jsonl`，事件 ledger 維持 `records/ledger.jsonl`，附件維持 `records/<event UUID>.json`，兩個命名空間不共用行格式。
- `resources.record_table` 擁有 append-only JSONL，每個 run_id 只讀最後一整行，CSV 只作匯出。`project_device_coordinates(..., device_name=...)` 將 caller 指定的裝置投影為分析 frame，再套用 opt-in flat v2 座標與 flux 解析。不猜裝置，不讀舊 flat 欄位作投影 fallback。Runtime 不讀舊名稱或 `samples/` 路徑。

使用者 2026-10-25：「我認為框架應該對SampleWriteback保持中立無知，可能可以叫RecordWriteback也說不定，行為就是append一個格式不拘的json到紀錄的jsonl中，格式由adapter間的隱式policy決定，資訊也尤其組裝，當然文檔與註解要寫這個policy好讓人維護」。
同日確認完整改名：「我認為sample可以遷移到新名字，稱現在還沒有真實測量的時間遷是最安全的」。
Wire kind、written key 與 session prefix 使用 `record`、`records` 與 `record-<n>`。舊 API 名稱不保留。
- Waveform asset 擁有播放 arrays、reference time axis 和 duration。Core waveform_format 驗證 samples 與編解碼 NPZ，只驗 recipe JSON 語法。Resources 管理 formula 語意、命名與實體 I/O。ModuleLibrary Spec 只保存 asset key。Native arb cfg 保存 key 與必填本機絕對 path，不另存播放長度或 provider。Program/sim 讀 path 的 samples，在完整 asset duration 取樣，不解讀 formula，也不查 global DB。

### 設定快照與 acquired 事實

設定快照的正本只在 `data.h5` 的 `/cfg`，以 CfgSnapshot 保存 values、cfg_type 與 schema_version。Labber 從同一份 prepared payload／snapshot 產生，供 Log Browser 使用，不是另一份讀回權威。Notebook 與 GUI 都讀 native `data.h5`。

Ledger 的 acquired 只記 run_id、experiment、point、roles 與事件時間，不保存 cfg_summary，也不從 Form 或 RunRecord.cfg 序列化第二份設定。只量測未保存仍可追加 acquired，不建立 run 目錄。

RecordsLedger 忽略未知欄位，包括舊 1.0 行的 cfg_summary 與 entry_id。讀取不改寫原行或附檔，get 的 raw event_json 保留未知內容。部分內容投影只驗請求欄位與必要祖先；明確整筆投影 events()／get().event 仍驗完整 payload model。歷史 import 的 source_entry_id 只留在 raw，不讀取或驗證，source_event_id 與 source_event_json 的引用契約保持。

Append 完整驗證本次新事件。既有歷史只驗追加需要的 JSON、版本、event／run ID、duplicate、完整尾行與既有安全條件，未使用的壞 payload 不阻擋新事件。無效新事件仍拒絕且不改原 bytes。Index 不預驗未使用的 attachment reference；真正使用附件時，先驗 canonical reference 與 root containment，再解析路徑或開檔。不升格式 major，也不做 strip／rewrite migration。

使用者於 2026-10-14 對保存位置選擇「A」，正本只放資料檔。對前端讀回裁定：「Notebook與GUI應該都讀data.h5，只有log browser讀Labber檔。」2026-10-15 確認：「1可以，2採用忽略不支援欄位的方式，3同意，4一起做吧」。其中 ledger 採忽略未知欄位，不改寫歷史行；Labber-only 資料先離線遷移，Load 不回填表單。

### MCP call log 與 ledger

MCP call log 是 best-effort 除錯紀錄，可以截斷、清除或停用，不併入 ledger，也不能代替 ledger 的事實紀錄。Ledger 的 origin 只標示 gui、notebook 或 mcp，不帶 MCP call id。跨 MCP、GUI 與 ledger 傳遞 call id 會增加耦合，卻無法讓可丟棄的 call log 成為可靠證據。使用者於 2026-10-08 以「採用建議」確認刪除這條跨層 call id 傳遞，只保留 origin。

這個分工不增加跨檔原子性、fsync 或掉電保證。Ledger append 與輸出檔案的失敗邊界仍依本篇，不因 call log 存在就視為已提交。

### Run 輸出位置與 writer 權責

新格式根目錄統一為 `result/`，與 GUI runtime 一致，不另設 `results/`。`.gitignore` 保留 `result/`，移除 `/results/`。
使用者於 2026-10-20 裁決：「好，統一result，那gitignore也做處理」。

Entry 只由 `result/<name>/` 構成。`resources.entry.ResultEntry` 提供條目名稱、UTC 建立時間、exact Result path、Database 輸出位置與 scoped ledger。`create`／`open` 只檢查 Result，不檢查或建立 `Database/<name>/`。第一次 Save 才由 writer 建立所需的 Database 目錄，已有的同名 Database 照用，不另加混雜偵測。

同一 Result root 下，entry 歸屬由資料夾名稱決定，不產生或使用 entry UUID。Setup／point 的舊 entry_id 即使格式不合法也忽略，讀取不改 bytes，明確寫入時移除。新 ledger 事件與 RunSnapshot 不含 entry_id，舊事件的該欄位不參與讀取或追加。Run ID 與 event ID 不變。Role capture 仍核對 point source 與 owning roots。Clone 使用來源位置當下的內容。缺 records 時 ledger 讀取回空且不建目錄，第一次 append 才建立。

2026-10-28 使用者詢問「這裡的UUID能做什麼？什麼是目錄名不能而必須要UUID?」，先以「選B」採用位置歸屬，UUID 不再當跨檔身分。隨後在 A「完全移除 entry UUID」與 B「只保留開啟中 setup 被換成不同 UUID 時拒絕操作」之間選擇「A」。

A 包含新建 entry 不產生 UUID。Setup／point 接受有或沒有 entry_id 的舊檔並忽略其值，下次明確寫入去掉。新 ledger／量測 metadata 與公開 wire 不再包含 entry_id，setup 換檔也不因 UUID 拒絕。Run ID 與 ledger 事件 ID 不受影響。

Setup 同名替換後，可經公開 SetupView.refresh、ResultEntry.new_point 或重新開啟 entry 取得新內容，不以 UUID 拒絕。GUI 的 entry 與 point 已安裝文件由同一個批次 watcher 監看，嚴格驗證後才安裝；holder pending、revision 與失敗保留舊內容的規則見 [[0081]]。

2026-10-29 使用者要求：「setup與point要有一致的檢測行為」。這取代先前不新增 setup 自動監看的安排，不改 UUID 移除規則。

Point 是實驗語意下具有一致環境參數的單位，不限於 flux。2026-10-28 使用者定義：「一個point不是一個flux點的意思，而是在其中的實驗語義下，具有一致的環境參數的意思，不只包含flux。而對於實驗本身就在掃某個環境參數，其一致性要求本來就不包含那個參數，所以哪個參數不會影響一致性」。

`rename_entry` 只移動 Result，不改文件內容，也不移動 Database 舊輸出。重新 open 後的新 Run 記錄新名稱，新 Save 使用 `Database/<新名>/`。已有 Run 重存仍沿它記錄的名稱，舊輸出留在原處。Database 不是 entry 的存在或 clone 身分邊界，因此沒有雙目錄改名與復原協定。

使用者於 2026-10-20 裁決：「甲，簡單的機制加一致的規則比非預期的兼容邏輯好維護與使用，同樣名字的Database會混雜屬於有意不處理的設計。」對改名裁決：「改名時Database不動，Database是輸出位置，不是什麼要綁定的目錄」。

`bind_run_outputs` 在 resources.entry 內依目前 Result／Database roots 與 Run 記錄的 entry 名稱、point 綁定輸出位置，不開 ResultEntry、不要求 entry 或 point 存在。缺名稱使用 unnamed_entry，缺 point 保持 None，Result 圖副本放 entry 層 `figures/`。GUI Save 不使用 live selection、捕捉時的 storage 或原檔位置決定目的地。

`SaveLayout` 接收 exact entry paths、caller 產生的 stem、point 與保存時間，再按 artifact key 計算 exact outputs。它不建立目錄或寫檔，不補前綴或 run_id，也不正規化 stem。Stem 必須是安全的單一路徑段，非法值直接拒絕。C8 分層與 C9 無循環 contract 涵蓋 experiment 使用 datafile／resources 的方向。

GUI 的 `Adapter.save_stem(context)` 委派自己的 `AxesSpec.save_stem` 或 `GroupedAxesSpec.save_stem`。Adapter 可用自己的 spec 覆寫，不改核心的原 spec。Notebook 沿 core spec 的同一 callback；workflow 的 `write_outputs(..., save_stem=...)` 由 caller 提供 callback。這些入口共用 `SaveContext`，包含完整 adapter path／tag／workflow name、完整 run_id、歷史 point、解析後 roles、可用 cfg values 與保存時間。缺 cfg 時傳 None，不讀 live context 補值。

預設 callback 將名稱中的 `/` 轉成 `.`，接上依 mapping 順序以 `-` 連接的 role values，再接完整 run_id，各段以 `_` 連接。Role list 按原順序攤平，未綁名稱不加入。沒有 roles 時省略該段。例如 `twotone.time_domain.t1_Q1-R1_20261027T120000Z-a1b2c3`。Mux adapters 依 cfg participants 原索引順序以 `+` 列參與 qubits，例如 `mux.t1_Q1+Q2+Q3_20261027T120000Z-a1b2c3`。Readout-only onetone 列 resonator labels，缺 cfg 名稱時回預設 role 命名，不猜 qubit 身分。

Database 的 runs 平放與按 point 分層都可行，目前選平放，日後有移植需要再改。2026-10-28 使用者裁決：「Database平放與按point分感覺都可以，我認為目前平放就先保持好了」。

Native run 目錄為 `Database/<name>/runs/<stem>/`。再次 Save 的 callback 回傳相同 stem 時，更新其中的 `data.h5`、具名圖片與 `analysis/<pane>.json`。圖片另有 Result point／全域 figures 副本，檔名為 `<stem>_<圖名>.png`。Native 同名目的仍由 saver 明確 replace=True 原子覆寫，不加避讓規則。自訂 callback 若依保存時間改名，後續保存會寫到另一個目錄。

Labber 初始路徑為 `Database/<name>/Labber/YYYY/MM/Data_MMDD/<stem>.hdf5`。只有初始名字已存在才沿用既有 numeric suffix helper。Helper 不作跨程序原子保留。每次 Save 可以產生另一個 Labber 檔，不改 acquisition 時間或完成狀態。只量測不存檔時，acquired 事件不建立 run 目錄。

Labber comment 在 cfg 旁保存 `snapshot: {point, roles}`，保留原 comment 與 timestamp。這讓檔案離開 entry 後仍帶目標身分。Reader 忽略不支援的 envelope keys，不要求歷史檔補 snapshot。不升 wire 或 Container 版本，不搬動舊輸出。格式演進與辨識方式見 [存檔歷程](../storage-format-history/2026-10-27-labber-identity.md)。

使用者於 2026-10-27 裁決：「Q-L1-a+b，名字也許要有adapter」；「Q-L3我的想法是adapter決定怎麼生成名字，可以有預設實現。」同日要求：「run_id也是由adapter行數補上，請你遵守框架不做多餘行為的設計理念，將可能需要的資訊作為參數傳給save_stem。」框架只使用 callback 的結果，不在後方再補 run_id。

2026-10-27 合併前檢查將預設與 mux callback 改為保留完整 run_id。只取後六碼會讓不同 acquisition 共用 native stem，而 replace=True 原本只用於同一 run 重存。保留完整 run_id 避免這項資料覆寫退化，SaveLayout 與 writer 不補救或改寫 callback 的結果。自訂 callback 仍自行決定名稱，但須避免不同 run 共用 stem。

這項裁決取代單次 run 與 Labber 的固定 experiment／run_id 命名。2026-10-19 的實驗名加 id6 與 2026-10-20 的完整 run_id 前綴規則不再適用這兩種輸出。離線 migration 仍明確提供原有 native stem，不改名舊 Labber 檔。

Workflow 本身的兩側目錄維持 `workflows/<workflow名>_<run_id>/`，保留完整 run_id，workflow 名中的 `/` 轉 `_`。Result 保存 manifest／journal，Database 保存 acquisition 與圖，iteration 位於其 `iter/` 下。此處不是 workflow 的 Labber export，沒有套用 save_stem。

`experiment.native_persistence.prepare_run_data` 從 typed RunRecord 與 axes spec 建立同一份 SI／離散 payload 和 cfg snapshot。Native 與 Labber writers 分別消費它，不先寫 native 再讀回轉成 Labber，也不互相呼叫。Datafile writer 只處理自己的格式與 caller 指定的目的地，不選路徑、不擷取 live context、不決定 batch。`experiment.artifacts` 的共用保存編排負責 artifact 選集、輸出順序與逐輸出失敗隔離，一個 writer 失敗不阻擋其他輸出。

`RunRecord.metadata` 是中性的 `JsonObject`，預設為空 dict，建構時 deepcopy。「未填」表示缺 key，不使用 None 狀態。Record 不解讀 key，也不持有 native schema 或 reader；這份 JSON 物件與 datafile 的 `docs["metadata"]` 一對一。

`RunMetadata` 是 experiment 擁有的 native metadata schema。產生方以它組好內容，再用 `model_dump(mode="json")` 填入 record。`RunMetadataContent` 是讀取方按需驗證欄位的工具，不存進 record。

Native 與 artifact Save 在任何 I/O 前驗證完整 schema 與原 JSON 的有限值。GUI 在修改 comment 與 prepare_ledger 前完成驗證。`save_run` 與含 native DataItem 的 `save_artifacts` 另比對 experiment tag；analysis-only Save 只驗完整 schema 與有限 JSON，不比對 tag，也不新增 expected identity。

寫入使用原 JSON 的全部 key，包括作者額外 key 與已廢棄 key，不以 schema 投影覆蓋原內容。Record 不持有廢棄 key 清單，也不新增相容層。

2026-11-01，使用者要求：「但RunRecord是相當中性且底層的，不能帶有上層資訊的認知，且要夠通用靈活，可以放想要的格式，與datafile對接」。對 schema／reader 分工與 Save 驗證選擇「3aA 3bA」，對保留已廢棄 key 選擇「廢棄key選A」。

RunMetadata 的 run_id、時間、completion、RunSnapshot 與 SoftwareProvenance 是 caller 提供的歷史事實。RunSnapshot 保存 entry_name、point 與 roles，不帶 entry UUID。Experiment 的 ParameterSnapshot 保存 resources 擁有的作用中參數內容、library Specs、authored aliases 與 fixed names。Session 的 capture／restore adapter 連接這兩個 owner，不讓 resources 或通用 ParameterContext 依賴 experiment。C8 分層、C9 無循環與 parameter-context-no-experiment contract 保護這個方向。

GUI 在 Run admission 固定作用中文件組的參數環境，global 與 point 都可 Run、Analyze 與 Save。Global run 的 snapshot.point 為 None。Save 使用 metadata 內的快照，不重新讀 live 文件。Load 只還原檔內快照；缺快照回沒有 general、created_at、target 或 storage 的空內容，不補造歷史資訊。Analyze／Post 使用這份內容。寫回提案從歷史內容求值及解析 alias／slot，Apply 則重新取得當下作用中的文件組。Global 寫 entry 層，POINT 寫選定 point；未開 entry 才以 no_context 拒絕非空寫回，空歷史環境不是拒絕理由。

2026-10-28 使用者說明 Load 的來源：「我認為run R確實可以記錄是哪個entry與point，但Load為什麼要看entry/point資訊？他又不需要，他只需要當時capture的環境就好？」對沒有快照的舊量測補充：「沒有快照的舊量測，可以用空環境來生成分析參數預設值，而不是不能分析。」當時的「1A 2A」裁決中，寫回選擇 GUI 目前選的 point。

Save 目的地隨後改採記錄的 entry／point。2026-10-28 使用者要求：「我希望load只需要給data.h5的路徑，不要求其在對應的儲存位置，但再儲存的目標就是內部記錄的entry與point。」「我認為Save的部分，應該不用找？」「直接生成路徑，然後覆寫就好」。缺名稱與 point 的處理分別是「我認為沒有記錄entry_name，那就用unnamed_entry之類的」「point應該也可以比照辦理，unnamed_point」。這項 Save 裁決取代存到目前開啟 entry 的方案，不改寫檔內歷史 metadata。下述 2026-10-29 規則只取代其中缺 point 使用 unnamed_point 的部分。

2026-10-29，使用者對無 point 量測與圖副本位置回答：「1我認為可以改成可以，Database本來就不看point，全域figures的位置也比照point內的 2應該也可以 3可以，用inspect改 4預設從全域繼承，或是指定empty或是clone其他point 5我有考慮統一名字」。對舊 run 缺 point 也放 entry 層 figures、移除 unnamed_point 的方案回覆：「A1 B採用建議」。因此 global 與舊 run 的空 point 都使用 entry 層 `figures/`，unnamed_entry 保留。

新檔由 [[0080]] 的 zcu.container 3.0 保存，實驗欄位進入 docs/metadata。`started_at` 接受 UTC ISO 字串或 None，`completion` 接受 complete、partial、stopped 或 unknown。None 與 unknown 表示缺少歷史證據，不從檔案時間、檔名或 run_id 補造。正常量測路徑仍必填開始時間，只產生已知完成狀態。

Reader 只接受 zcu.container major 3。同 major 的 forward minor 與未知內容在 load、修改、save 往返時保留，不降版。舊 native 2.x 不再支援。使用者於 2026-10-24 裁定不支援的舊資料 Load 失敗即可，因此移除舊 reader 與批次升級工具。實驗欄位是否合法仍由 experiment 的文件 schema 驗證。

GUI Load 與共用 typed loader 保留未知歷史值，不把 unknown 改成 complete。Workflow `load_trace` 是 complete-only 分析入口：None＋complete 可載入，unknown 與 partial／stopped 同樣拒絕，因為未知完成度不能證明 acquisition 完整。Notebook 仍只載入 RunRecord，不把載入資料變成 acquisition evidence。現有前端不顯示或排序這兩個 metadata 欄位，本格式變更不新增 UI。

使用者於 2026-10-19 對 started_at 可為 None、completion 加 unknown、不加來源標記的建議裁決：「可以，採用建議」。

`RunMetadata.device_snapshot` 保存 Run 開始前讀取的全部已連線 device 設定。GUI 在取得 operation hardware lease 後、adapter.run 前讀取。Notebook 在使用者的程序內同步執行，由 caller 擁有資源，不新增 lease；snapshot 在建立 RunContext 後、core.run 前讀取。

兩路每台只呼叫一次 get_info。任一台失敗就拒絕 Run，不寫設定、不進 adapter/core，也不用 cache 補值；無 device 記 `{}`。Snapshot 表示實驗前狀態，不是 setup 後或每個 core 的 checkpoint。後續 sweep 不改寫它，實驗使用的值由 cfg 的 dev、flux_unit、sweep 範圍與 persisted axes 記錄。

Experiment 將 snapshot 寫入 Container 的 `docs/metadata.device_snapshot`。Key 是 device 名稱，value 是 get_info 的設定。Snapshot 為 None 時寫 JSON null，空 mapping 則寫 `{}`。單位沿各 DeviceInfo 的儲存欄位。YOKO 的 `value_unit` 是 A 或 V，與 current／voltage mode 一致；Fake 是 native。Value 與 rampstep 沿該單位保存。RF 頻率為 Hz，功率為 dBm，不加 `value_unit`。Snapshot 不另造單位欄位或換算值。GUI 和 Notebook Save 都使用當次 Run 的固定 snapshot，不重讀硬體或取下一輪值。

Labber 成功後，保存編排將 labber_path 放入 SaveReport 與 saved ledger 事件，不寫入 RunMetadata。輸出結果與 saved ledger 事件由保存編排產生，RecordsLedger 只驗證與追加事件。Ledger 追加失敗不能把已寫好的檔案改標為量測失敗，也不代表那些檔案已回滾。

`experiment.artifacts.AnalysisDocument` 與 encoder 擁有窗格 JSON 格式，信封也可由 to_doc 表達為 [[0080]] 的 docs/analysis。現有 artifact 仍寫 analysis/<pane>.json，SaveLayout 與 ledger 不改。`experiment.artifacts` 定義 ArtifactItem 與固定 DataItem／FigureItem／AnalysisItem。一次 SaveArtifact 包含多個 items，`save_artifacts` 依種類選 native／Labber／PNG／JSON，逐項編碼一次，對等目的地共用 bytes。

DataItem 保留 typed run 準備成功或失敗的結果；cfg=None 或 producer 準備失敗只使對應 data outputs 失敗，同選集的圖與分析仍嘗試保存。每次呼叫回傳最新 SaveReport 並追加 saved；ledger 失敗向外傳遞，不 retry 或 rollback。共用核心不依賴 GUI 或 Qt，由 `.importlinter` 的 `artifacts-core-no-ui` contract 檢查。

`NotebookExperiment.save_run` 綁定 PointView，使用 entry owner 的 capture_point_storage，保存 adapter 驗證過的 acquisition source。SavedRun 固定該 run 的資料、context 與 evidence，PartialSaveError 保留同一 handle 供明確再存，不建立恢復收據或自動 retry。

GUI Save／Save All 與 MCP 保存使用這條 run 保存流程。Save All 包含已保存的 artifact，每次重寫全部選定成員。GUI 在 owner thread 同步完成並回傳完整 SaveReport，不建立 Save handle、worker 或 save-busy phase。Data pane 唯讀顯示每個 output 的 path、status 和 error，不讓使用者改目的地。Notebook 的本機單檔 save 保留原入口，不猜格式；`save_run` 使用 bound PointView 的公開 capability。C8 與 C9 涵蓋 notebook → experiment/datafile/resources 的依賴，C12 保持 datafile 不依賴 stateful resources。

GUI Native Load 由 LoadService 呼叫 datafile.load 讀成 Container，透過 experiment 的 metadata content 只讀驗 experiment tag，再交 adapter.restore(container)。Adapter 依 spec 呼叫 restore_run，共用資料優先轉換見 [[0069]]。核心與 Notebook 保持 typed RunRecord，GUI 只持有 opaque record 與明示 cfg_issue，不探測 cfg／result。

Load 將 `docs["metadata"]` 的全部 key 原樣保留在 `RunRecord.metadata`，包括原 run_id、時間、completion、snapshot、provenance 及額外或已廢棄 key。Load 不預驗 admission 以外的 metadata 欄位。Reader 按需讀取，`MetadataFieldError` 保留欄位、`/docs/metadata/<field>` 文件路徑與原始原因。Load 不加入輸出路徑，也不產生 acquired。Load 不回填表單，不以目前 Form、library、default 或 live device 補回歷史 cfg。

原始 cfg 正本保存在 native 資料檔的 docs/cfg，可透過公開 datafile.load 離線讀取。使用者於 2026-10-16 選擇「A」時不新增 GUI 入口；2026-10-23 批准的 Results browser 是唯讀檢視例外，呈現保存的 cfg_type、schema_version 與以 values 為根的相對路徑。瀏覽器不修改檔案、不回填表單，也不重新分析。視窗與前端邊界見 [[0067]]；Load 與 adapter 的 opaque record 契約不變。

Load 接受任意位置的 data.h5，不讀目前 entry 或歷史 point 的外部文件。未使用的 metadata、參數與 library 區塊延後驗證，使用壞值時報來源與欄位，不降成空內容。Save、完整 metadata／component／document 投影與 edit 仍完整驗證。Leaf read 只執行請求欄位的 annotation、field validators，包括 before，與必要祖先檢查，不執行 model 層級 before／after validators。缺值沿該欄位的 default／default_factory，同文件重讀保持穩定。Whole-component projection、edit、seed 與 clone 執行完整 model，包括轉換與跨欄位檢查。需要跨欄位不變式的 consumer 使用 whole-component projection；只靠 model validator 的 legacy 轉換需明確 edit 或 parameter upgrade，不在 leaf read 補做。Header／major、schema_version、JSON／YAML 表示、實際使用的 attachment path 與 ledger 追加安全規則保持。

Run 與 Load 的預覽及實際 Save 共用 SaveService 的公開 layout planner。
Planner 使用目前 roots、metadata 記錄的 entry 名稱與 point，以及 adapter.save_stem 的結果，只呼叫一次 bind_run_outputs，回傳具名的 binding 與 layout。
預覽只使用 layout，Save 以同一 layout 寫檔、以 binding.prepare_ledger 準備 ledger。
正常 Load 完成後立即有預覽，Analyze／PostAnalyze 後也有預覽，不依賴 run.storage。
Native eligibility 只檢查 schema 的 required keys 是否齊全，不驗值。空 dict 或缺 native 必填 key 時仍保留 artifact／member declarations，但沒有輸出預覽，也不寫 ledger acquired，GUI Save 以 storage_unavailable 拒絕。
這項判斷共用 experiment 的 reader helper，不另列必填 key，也不以完整驗證代替 eligibility。Eligibility 通過不代表可保存，實際 Save 仍在任何 I/O 前完整驗證。
預覽不建立目錄，也不標為 Saved。
預覽只隔離 experiment.metadata 的 MetadataFieldError，保留含壞命名欄位的歷史資料供 Load；錯誤帶欄位、文件路徑與原始原因。
其他規劃錯誤照常傳出，Save 仍完整驗證 metadata，不將壞內容改成空快照。
Save report 發布實際路徑、成功或逐項失敗。
預覽與 Save 使用各自計畫時的保存時間；自訂 stem 若依時間改名，實際輸出仍由該次 Save 計畫決定。
實際 Save 建立目錄並覆寫同名 data.h5、分析與圖。Result 圖副本放到記錄的 `points/<point>/figures/`，缺 point 放 entry 層 `figures/`。不新增 source.json 或另一份來源權威。

Tab／Analyze 逐欄讀取自身需要的 run_id、experiment 或 device_snapshot，不受其他缺欄影響。缺自身 key 時回 None，experiment 保留 adapter_name fallback；有 key 但值壞時仍拋 MetadataFieldError。

2026-10-29，先前方案保留已綁定 Run 的路徑預覽，讓無 storage 的 Run 首次 Save 前只有 declarations、沒有 outputs。
使用者回答是否保留這項安排時說「1A, 但第一次save前沒預覽很怪」。
隨後，使用者對預覽與 Save 共用路徑計算、讓 Load 首次 Save 前也有預覽的修正方案回覆「1做」。
這項修正取代無 storage 時預覽為空的安排，已綁定 Run 的路徑保持不變。

Load Data 同時提供 File 選檔與 opened entry 的 native 檔案清單。清單掃描 `Database/<entry>/runs/*/data.h5`，完全不讀 ledger。Run 身分、experiment、point、開始／結束時間與 completion 來自各檔的 RunMetadata／RunSnapshot；未知歷史時間保持空白，completion 保留 unknown。清單透過 `datafile.read_doc` 讀 metadata，只解碼與驗證上述顯示欄位，不要求完整 RunMetadata 或 provenance，也不讀 cfg 或量測 arrays。清單可顯示不代表整份資料可載入。Metadata 讀取失敗時，run_id 為空字串，experiment、point、時間與 completion 欄位留空。清單保留 exact Path 與既有錯誤資訊，選取仍回傳 exact path，由既有 Load 路徑回報失敗。清單不解析新舊目錄名來補欄位，避免讓路徑命名成為第二份歷史身分來源。查詢不建立目錄或事件，遷移也不手造 ledger。

使用者於 2026-10-20 確認檔案清單與成功 Load 分開：「我以為只要是data.h5就可以看到，能不能Load另說」。對清單完全不讀 ledger、顯示欄位只從檔案 metadata 讀取的方案回覆：「可以」。

### Setup 與 point alias 表

Entry 的 global alias 位於 `<entry>/aliases.yaml`。每個 point 的 alias 位於 `<entry>/points/<p>/aliases.yaml`，兩層都與 `parameters.yaml`、`library.yaml` 並列。新 point 預設 inherit global 三檔，library 經 copy_library；也可明確選 empty 或 clone。Empty 只建最小有效參數文件，不繼承使用者值，也不建 library 或 aliases。Clone 必須提供來源，只複製來源 point 的文件；其他模式禁止帶來源。來源缺 library 或 aliases 時目的也保持缺席。各組之後獨立修改，選 point 後整組替換 global，不逐名稱回退。

兩層缺 library 或 aliases 都當空、revision 為 None，讀取不建檔，首次明確編輯才建檔。外部先建檔時 writer 報 revision 衝突，不覆寫候選。缺參數文件仍是錯誤。

2026-10-29 使用者說明：「行為須一致，沒有就當空」，並確認：「A，我目前先設計成整個作用域換成point，沒有個別的fallback邏輯」。

使用者於 2026-10-20 裁定：「採用乙，alias. yaml的獨立檔案，找不到就視內容為空，與其他兩個資料容器並列」。

使用者於 2026-10-23 對 alias 表加 participants 列的選項裁定：「甲」。Alias 保存目前工作焦點，parameters 保存元件事實。慣例列 `participants` 保存本次要量的有序名稱，例如 `[Q3, Q1]`，不在元件 schema 加入本次工作選取。

`AliasValue = str | list[str]` 由 resources 的 alias 資料 owner 定義。Session rows、事件、保存與 remote 邊界沿用這個型別。`EntryAliases` 保留 mapping 順序、空字串、空清單與 `$` softlinks。清單以 YAML list 保存，元素可為非空識別名或 backward `$alias.field` link。Session AliasTable 逐元素驗證與解析，保持順序；錯誤元素拒絕整次讀取，不略過或取首項。空清單與空字串皆未綁。缺列、未綁與 invalid 的公開讀取失敗由 session 指名 alias。元件及 device 的實際可讀性仍由原 owner 判斷，表的保存不檢查裝置可用性，也不將 alias 的名稱轉成 tone index。

`EntryAliases` 保存 authored 值，不保存解析後名稱。Run snapshot、storage 及 acquired metadata 則固定當次解析後的單一名稱或有序名稱清單。歷史載入不重展全域表；單一名稱消費者遇清單明確拒絕。Alias 資料型別由 resources 擁有，experiment／frontends 向下引用，不反向依賴 session。這條依賴由 `layers` 的 C8 及 `top-level-acyclic` 的 C9 contract 約束。

檔案使用 `format: zcu.aliases`，寫入 `format_version: "1.0"`，讀取接受 major 1 的 forward minor。空 mapping 與缺檔都讀成空表。嚴格讀取與寫入仍直接回報格式、結構或 alias 語法錯誤，不回退 preset。

供 session 使用的 resources 讀取結果在一次 I/O 中固定解碼內容、原始內容錯誤與 observed bytes revision。Session 保存這份結果及自己的 alias 語法錯誤，建立固定 capture。壞 aliases 不阻擋 point 的參數與 library；實際 alias 讀取或投影才報該 capture 的原錯。錯誤狀態不是有效空表，修復後的新 capture 也不改舊 capture。Revision 不保證內容有效，修復與衝突語意見 [[0081]]。Session 透過 resources 的讀取契約取得資料，依賴方向由 C8／C9 約束。Caller 不重讀或補驗檔案。

2026-10-29，使用者對 alias 錯誤隔離、owner 分工與 observed revision 的文件補充回覆「同意」。這項隔離實作下方的按需讀取原則，不放寬 alias writer 的驗證。

`apply_alias_rows` 先驗整表，再由 entry owner 寫 sibling temporary 並 atomic replace，成功後才安裝 memory 和發布變更 fact。GLOBAL 寫 entry 表，POINT 寫選定 point 表；`setup=True` 固定選 entry 表。未開 entry 時以 `FailedPreconditionError(no_context)` 拒絕，不暫存、不自動開 entry。Startup 先開 entry，再呼叫此入口；不提供 memory-only alias 初值或 presets。Inspect 完成有效編輯後保存所選 Destination，失敗保留輸入、舊表與舊檔，關閉不撤回已保存內容，見 [[0081]]。

開 entry 載入 global 表，切換 point 載入該 point 的表，回 Global 則重新使用 entry 表。Open／Refresh／Reset 捕捉作用中組的 alias，已保存的空表或空列不自填。選 point 時，Inspect 仍可編輯 entry 表，但不改作用中 point 或舊 capture。Run 保留原 capture，後續改表或切 entry 不改歷史 Analyze／Writeback／Save 的來源。這個單檔替換不承諾跨檔交易、多 writer merge 或掉電恢復。

### 資料表示與完整性

參數容器的 YAML、快照與視圖使用同一個工作單位數字，metadata 內容也不換算。元件 model 擁有型別、extra 策略與驗證。這條規則不改 Experiment Data File 的單位契約；datafile 的格式責任仍由 datafile 擁有。

參數容器框架只消費已註冊的 model 與角色宣告。具體 kinds、欄位與內建角色由 `zcu_lab.components` 擁有。Lib 外的組合根顯式傳入共用 ComponentRegistry，在 startup 註冊，不在 import 或 reload 時註冊。`.importlinter` 的 `entry-definitions-composition-only` 與 `framework-no-user` 禁止框架反向 import 定義。使用者 model 可用 validator 處理舊欄位；框架不猜兼容規則。

2026-10-23 使用者對以下寫入慣例回覆「認同」：一次使用者動作、每份文件一筆交易，不承諾跨文件原子提交。Owner 寫入 API 以「一份文件的一組變更」為原形，單項寫入是長度 1 的批次，避免逐項重開同份文件、驗證與提交。

resources.entry、DocumentStore 與 ledger_models 在 owner 內部分開讀取投影與寫入驗證。參數讀取保留未使用的 raw 內容，同一 component 的壞 leaf 不擋其他 leaf，不先建立整份嚴格 model。Edit／append 不因此放寬寫入契約，壞值仍由 owner 指出路徑，不略過或降成空內容。DocumentStore 保存 raw YAML 節點，完整驗證時產生 normalized tree 與 typed snapshot。一般讀取及空 edit 不寫檔；非空 commit 以最新 normalized tree 合併與驗證，再把差異套回 raw 節點。Model 轉換在下一次 commit 寫回，未改節點保留原樣。

ModuleLibrary 讀取不改檔。Library owner 先驗 header、容器與可定址名稱，再逐 root 驗證 Spec。無效 Spec 從讀取投影略過，公開 `invalid_entries()` 回報 root 與原因，合法兄弟仍可用。Header、容器及 I/O 錯誤傳遞。Strict refresh 不略過無效 Spec，失敗保留已發布快照。讀取與空 edit 保留原 bytes，不建立 .bak。

Library 不再保存 per-leaf provenance 或 stderr。舊 `provenance:` 容忍並忽略，非空寫入時自然移除。讀取與寫入都不需要 ledger 或 entry_id。DocumentStore 對未變 bytes 沿用驗證快取，不做來源依賴檢查。新 draft 仍嚴格驗證，不略過 caller 的壞值。

ResultEntry clone 只複製 point 參數、metadata、合法 library Specs 與來源 point 的 alias，不改來源檔。跨 entry 不匯入事件、不複製附檔，也不寫 source 或 cloned_from。`import` kind 保留為歷史讀取類型，不再有 producer。Clone 失敗只清理本次目的 point，不承諾跨檔 snapshot 或交易。

2026-10-28 使用者裁決：「我其實沒有要知道來源的需求，這功能純屬於錦上添花」「來源我一直以為只是metadata，沒想要解析」。這次移除模組來源機制與讀取清理例外，取代 2026-10-15 的 library 先備份再清理政策。Library 容器仍是 2.0，變更記錄見 [移除模組來源](../storage-format-history/2026-10-28-remove-module-provenance.md)。

Format 與 format_version 只描述容器結構。檔頭不符或 major 不同時拒絕載入；forward-minor 保留容器層剩餘未知欄位，不覆寫元件或 nested model 的 extra 策略。

任何 minor 都由元件定義者的 model 決定 defaults、validators 與舊寫法演進。Forbid 拒絕未知欄位，ignore 丟棄它們，allow 保留並公開它們；ignore 丟棄的欄位在非空 commit 時移除。

框架不提供 runtime 相容 loader。Component 文件有獨立的版本與明確執行的升級鏈，見下節。Entry 不為值寫入或格式轉換產生 metadata。

Experiment persistence 使用 inner-first axes；disk payload 由 Result-native shape 對應，save／load 不要求 caller 補 transpose。物理單位由 experiment mapping 明示；不能為未定義物理量的 scalar 捏造 A／V。Data Variable 是結果語意，state／phase 等離散座標仍是 axis。正規 data.h5 的各 Data Variable 可以使用不同 grid，共用一份結果身分與 metadata。Labber 預設的 canonical one-shot grouped 輸出要求共同 grid 與明確 variable-to-channel mapping。必需 variables 由 experiment 決定，不由 generic writer 猜測。異質 streaming workflow 有獨立 layout 與 completeness 規則，不能因同為 HDF5 就視為同類檔案。

Grid 不同的 grouped 實驗可由 spec 明確宣告 independent Labber layout，各 variable 保留自己的 1D／2D axes。Writer 不因 grid 不同自動切換，native 格式與載入契約不變。使用者於 2026-10-19 要求「Labber存檔問題請獨立修」。採用明確宣告，讓異質 grid 的實驗能匯出 Labber，同時保留未宣告實驗的 common-grid 驗證。

結構完整、量測完成與持久提交不同。Complete Experiment Result 指所需 variables 齊全，不保證所有預定測點完成。Operation stopped／failed 不單獨決定已提交資料能否分析。Autofluxdep 在 node 正常返回、patch 驗證後提交 row；mid-node exception 不把半列視為 committed。已取得 measurement 而 fit／provide 失敗時，保留 raw measurement 並分別標示提供結果的狀態；skip 以 audit event 表示，不以 NaN 推斷。Node-row 與 flux-level commit 是不同進度；partial／stopped／failed run 仍可保有先前已提交的 evidence。Terminal exports／reports 取用 committed journal evidence，不能反向覆寫 canonical row。

2026-10-21 註：autofluxdep 已移除，上一段的 node-row、flux-level commit 與 terminal export 規則是歷史記錄。現行 workflow 的提交與保存邊界見 [ADR-0062](0062-experiment-workflow.md)，不把舊 node 契約套到新 engine。

### 資料容器的讀取契約

資料容器採 Tolerant Reader。每個 app 只讀、只驗當次操作需要的檔案與欄位。用不到的欄位缺少、多出或格式不同都不報錯；實際使用時才檢查它是否存在、數值是否可用。有問題時在使用處報錯並指名欄位，或在該功能可選時關閉並註明原因。容器可以不再讀取舊欄位，不必轉換或提供相容別名。寫入端仍寫出完整、正確的格式。

這項規則也適用於按 point 或個別分析成果移植資料。App 不因其他檔案或區塊缺席、格式不同而拒絕使用已具備的資料。2026-10-28，使用者說明：「我會預期一個app只會需要給他需要的資料，而不是檢查用不到的格式與欄位」。2026-10-29 再次確認：「基本上可以，但我要提醒，我一致的設計規則是：只檢查需要用到的資訊」。

Point availability 只判參數文件可讀且能解析成 owner 的文件結構，保留 header 與目前 schema version 檢查，不讀 library 或 aliases。Leaf／component 格式問題在使用或寫入時報錯，不讓整個 point 不可用。讀不到或解析不了的 point 仍標不可用並附原因，其他 point 正常；UI 停用、不可當 clone 來源、tooltip 與 remote error 的呈現不變。Availability 不作為 use_point、Load、Save 或其他讀取的額外前置條件，檔案同步見 [[0081]]。

2026-10-26，使用者裁決：「我對這種資料容器的行為預期都是：可以不讀舊格式欄位，但不要因為少了欄位或多了欄位就報錯。」這是一般資料讀取的欄位容忍原則。既有容器 header／major、component `schema_version` 的拒讀規則與元件 model 驗證契約仍由各 owner 決定，不以此新增相容層。

量測與擬合分階段。量測端只把資訊記錄成 JSONL，不預先決定分析需要的格式。分析時才按需處理、組合所需資訊。讀取工具與分析工具只檢查實際使用的欄位，不驗整份 schema、版本或與當下分析無關的欄位；上述 owner 的既有格式契約不因此放寬。

2026-10-26，使用者說明原因：「所以我才希望將擬合與測量分開階段，因為你不會知道擬合需要什麼資訊，也不知道想要什麼格式，只要都記錄成jsonl，透過要分析時按需處理來組合需要的資訊。所以請你不要讓分析工具與讀取工具有非必要的格式檢查」

### Component 文件版本與明確升級

Entry 根與每個 point 都使用 `parameters.yaml`、`library.yaml`、`aliases.yaml`。
檔名由 `resources.document_names` 擁有，entry 以根層 `parameters.yaml` 識別。
Runtime 不讀舊檔名；只有舊 setup 的 entry 開啟時列出舊路徑與明確改名指令。
原升級入口的 `--scope filenames` 只改名、不改 bytes，提供 dry-run、外部備份與 apply。
新舊並存時拒絕、不覆寫。需要內容升級時先 filenames，再 components。
Components 只枚舉新名，schema_version 與容器版本不因檔名改變。
操作與失敗限制見 [升級指南](../parameter-upgrades.md#先改檔名再升級內容)。

2026-10-29，使用者對統一文件檔名的選項回覆「A1 B採用建議」。

每份存 component 參數的 setup、point 與 entry template 都有 `schema_version: N`。它描述整份 component schema，不取代容器的 format／format_version。Lab 與 component models 一起在既有 ComponentRegistry 註冊具名 ParameterSchemaPolicy，提供目前版本與 upgrade 命令。未配置就失敗，lib 不預設 lab 版本，也不把 policy 參數散到 ResultEntry／open／create／template。

Runtime 只接受目前版本。缺版或版本不同時，讀取處列出文件版本、目前版本與可執行的 upgrade 命令，不推測、不改檔。優先新增有預設值的欄位，結構變更才增加一個 vN→vN+1 純函式步驟。

`resources.parameter_upgrade` 擁有通用鏈與 YAML I/O。Caller 提供完整舊格式分類、每版完整驗證及純函式步驟。未標版文件只有在 operator 明確 upgrade，且完整分類確認後才視為 v1。Runner 驗證每步輸出正好是下一版，current 文件也先驗證再跳過。預設 dry-run 不建檔或鎖，逐文件、逐欄位列出差異。Apply 使用 resources 的 per-user runtime file lock，重讀、完整驗證、exclusive byte backup、sibling temporary 與 atomic replace。失敗保留原檔與已完成的備份，其餘文件照常處理。這不是多檔或掉電交易。

`resources.upgrade_backup` 擁有 parameter、predictor、record table 與 processed spectrum 升級共用的備份定位。它依來源最近的 `result`／`Database` 所屬資料專案，選擇旁側 `<專案根>-backup`，不是工具 checkout 旁，也不把備份留在資料項目內。Helper 在任何寫入前檢查目的地；caller 才建立目錄並保存原 bytes。它不依賴 migrations，C20 禁止 runtime 反向依賴離線遷移，C21 禁止 library 反向依賴 scripts。Override、相對鏡像、resolved `-backup` 名稱與 symlink 規則由[備份位置](../parameter-upgrades.md#備份位置)擁有，不在各升級工具重做判斷。

2026-10-28，使用者對升級備份移到資料項目外、與遷移備份同處的選項回覆「2A」。2026-10-29 對上述共用 owner 與外部備份分工的文件補充回覆「同意」。

Lab 的 component flux 純轉換由長期升級步驟與一次性 `storage_mapping` 共用。Migrations 透過 caller 的完整文件 batch callback 在 current schema 驗證前轉換 seed／value，不先發布舊格式再升級。框架不解讀 lab 欄位。離線 YAML 沒有 device catalog，只驗共用名稱語法，device 存在性留在使用處檢查。`migrations → zcu_lab.components` 是 caller 指向定義者的邊，C14、C16、C20 保護反向隔離，不限制 caller 載入。

Lab 擁有目前版本、步驟與真實舊文件形狀。`scripts/upgrade_parameters.py` 只組合 lab plan 與通用 main。長期鏈不依賴可刪除的 `migrations/`，後者仍只做舊 repo 匯入。C14、C16 禁止 lib 反向 import lab；C20、C21 禁止 lib 依賴 migrations 或 scripts。Caller 注入取代反向載入。

2026-10-27 記錄使用者採用升級鏈的原因：「每次改component都會讓舊資料不支援很頭痛」，並對整份文件版本與通用 runner／lab steps 的建議回覆「12採用建議」。框架測試在公開 runner、讀寫與 main 接縫注入合成步驟，不 import lab；真實 lab 步驟以離線樣本驗收。

### 失敗、路徑與遷移邊界

2026-10-23 使用者對同一份文件一次 edit 的方案裁定：「同意」。Run accept 先追加一筆 accepted，再依目的地文件分組提交，組內順序保留。提交順序為 parameters 一組、library module／waveform 一組、樣本逐筆。Parameters 與 library 各使用一次 edit，任一項無效整組不寫。Executor 回 `WritebackResult`，含 accepted event_id 與逐項 committed values。組失敗以 `WritebackApplyError.result` 保留已提交組的全部項目，並公開 event_id、applied 與原始 cause，保留前組與事件，停止後組。applied 仍計項目數，失敗組計零。Notebook 直接看到此錯誤；GUI context owner 發布實際成功前綴，GUI service 標記前綴後原樣拋 cause，wire 不改原 producer 分類。

GUI writeback 接受當下作用中的 SavedEntry 或 SavedPoint，不自動建 point。Global 的 acquired ledger 與 record row 都可保存 point=null，沿既有 nullable 欄位契約，不新增格式版本。這取代原本 GUI 禁止 point=null 樣本的限制。

Preview 的 destination_context 與已提交 receipt 的 destination 只含 entry_name、point，不含 kind。GLOBAL 的 point 為 null，POINT 為 label；已提交 receipt 的 entry_name 必為字串。未開 entry 仍可預覽合法 draft，destination_context 兩欄與 current 都為 null，但預覽不授權 Apply。MCP 保留 opaque actual_path 與固定來源證據，Apply 不以 preview 的位置鎖定目的地。

2026-10-23 裁定：「resources／context 的 writeback executor 回具名結果：accepted event_id 加逐項 confirmed committed values」「值由實際寫入的 owner 當筆形成，GUI 只關聯 item ID／pane／目的地，不再整批後 readback、不用 proposal 冒充 actual」。Point／library 批次 setter 在 draft 逐項驗證後捕捉隔離值，整組 commit 成功才回傳；sample append 回傳當筆實際序列化的完整 row。Executor 依實際提交順序保留這些值，accepted payload 的 writes 仍保留供應順序。CommittedWrite.index 是該項在供應 tuple 的位置，GUI 用它關聯 draft item ID，不以 path 或前綴長度配對。父／子路徑後續覆寫不改首筆收據。GUI 不重開目的地讀回。每個 batch 仍只追加一筆 accepted event，不增加 per-item callback、worker、retry、rollback 或第二次 readback。

Memento JSON 損壞或不符 codec 格式時，共用 caretaker 回 fresh defaults，restore outcome 帶明確錯誤供 host 顯示。載入不移檔、不備份，也不新增保存保護狀態。Host 之後照原有時機保存。FormInput 的固定 data／selections 格式與遞迴還原見 [[0079]]，所有 caretaker hosts 共用格式失敗的處理。

Provider、其他 restore／apply 故障仍傳播。一般 read I/O 失敗沿用錯誤 outcome 與 defaults，不冒充 decode 失敗。Caretaker 不懂 cfg，也不自行決定何時保存。量測資料、參數和未知 artifact 格式版本不能沿用 memento 的 whole-file default degradation，必要 variable、shape 或 unit 不符時仍須回報可定位的錯誤。

Autofluxdep artifact 建立失敗不啟動 run；canonical row write／flush／journal 失敗會使 run 失敗，並保留此前已 flush 的證據。Terminal export／report 失敗要顯示，但不把成功量測 row 改標為 measurement failure。Stop 仍嘗試 finalize；取消請求不保證 finalize 成功，也不表示 rollback。單檔 replace、HDF5 flush 和 journal append 不是跨檔交易或掉電持久保證；不可從 manifest 的 accepted 標記推導任意 crash 後可安全 resume。跨檔失敗窗口的讀取、修復和 resume 規則尚未驗證。

2026-10-21 註：上一段描述已移除的 Autofluxdep artifact，不是現行 runtime 的失敗契約。現行 Experiment writer 依下段處理 exact path；Workflow 保存見 [ADR-0062](0062-experiment-workflow.md)。

Experiment writer 寫 caller 指定的 exact path。Labber 拒絕既有目的地；native 預設也拒絕，只有明確 replace=True 才用同目錄 temp、關檔後 atomic replace，失敗清 temp 並保留舊目的地。SaveLayout 的 Labber suffix 只避讓當時已存在的檔案，不是跨程序 filename reservation；collision 由上層回報該輸出失敗，不隱藏 retry。單檔 replace 與 ledger append 不是跨檔交易，也不保證掉電恢復。Run identity 由 run-scoped metadata 記錄，不由可讀 slug 或檔名猜測。Workflow manifest 連結 metadata root 和 heavy-data root，不把 exports 當另一份原始權威。

Storage migration 的 migration_id 是單次操作 ownership token，不是 entry 身分。Manifest／report 與初始化暫存名稱都使用 migration 專屬名稱，setup／point／run／ledger 產出不依賴 entry UUID。需要重跑或 cleanup 的舊 ownership 欄位 entry_id 會使操作拒絕並指名檔案。工具不讀其值、不自動轉換或刪除來源、備份與目的地。Operator 使用舊版完成，或確認後自行處理。重跑、備份、cleanup 的 ownership 與越界路徑保護保持。

Normal runtime 只接受各自契約指定的格式，不猜測 legacy shape／unit；版本必須與 artifact kind 一起判斷，合法 streaming v1 不因數字小於 one-shot v2 而視為同類 legacy。Experiment 的舊 converter 和 GUI fallback 已退休。既有 pure DataFrame conversion 只產生分析 frame，不是 raw-record loader，不補造 `devices`，也不自動改寫 CSV。QubitParams 的 project identity migration 由其 owner 管理，不因本 ADR 授權寫入使用者資料，也不建立跨資料類型的共用 migration layer。

### 共享資料與引用

`params.json` 的 QubitParams 與工作點的 `Q1.params` 是獨立資料，不自動同步。QubitParams 的 caller 以 typed 語意方法讀寫；更新一個 section 保留其它已知和未知 section。Timestamp 只表示 section 修改時間，不是 dependency provenance 或自動失效。單一 typed module 不等於跨程序互斥，也不代表 mtime 同步可解決多 writer 衝突。

Opt-in flat v2 分析 frame 明示 `dev_value` 與 `dev_unit`，單位為 A、V 或 native，flux 與 row frame 由獨立欄位表示。原始 record 不要求這些欄位。Flux 解析依 explicit、row-frame、fallback-frame 順序，並標明每列來源；fallback frame 必須與該列 unit 一致。自描述 unit 不保證缺少 flux／frame 的列能不靠外部輸入算出 flux。分析用 correction／alignment 不默默寫回原始樣品資料，也不自動修正 explicit flux。Autofluxdep export 用 run 建立時的 unit snapshot，不以 export 當下的 live device state 重讀舊結果。

2026-10-21 註：autofluxdep 已移除，上一段最後一句只保留舊 export 的歷史契約。現行 RecordTable 座標與單位契約見 [record_table owner](../../lib/zcu_tools/resources/record_table/README.md)。

Asset repository 的 rename／delete 只處理資產，不掃描或改寫 ModuleLibrary 的 nested references。舊 key 可能因此成為 dangling reference；使用端解析缺失資產時明確失敗，由 caller／使用者修正引用。這避免 repository 反向理解 experiment cfg，也避免跨 owner 批次寫入。Capture-bound assets.path_for 驗 key 與檔存在，不載入 samples。Native cfg 的 path 以本機絕對路徑與 POSIX separators 序列化。Data.h5 不附帶 NPZ。移動 asset 或跨 OS 後，caller 須明確重定位 path；讀量測 arrays 不需要重播。覆寫同 path 會改變之後播放的 bytes。本決策不引入 asset versioning、copy-on-run、packaging 或自動引用遷移。

## 後果

維護者先找保存對象的 owner，再查其就地格式契約：[`datafile`](../../lib/zcu_tools/datafile/README.md)、[`experiment`](../../lib/zcu_tools/experiment/README.md)、[`QubitParams`](../../lib/zcu_tools/resources/qubit_params.md)、[`parameter entry`](../../lib/zcu_tools/resources/entry/README.md)、[`RecordTable`](../../lib/zcu_tools/resources/record_table/README.md)、[`waveform assets`](../../lib/zcu_tools/resources/waveform_assets.md)、[`program waveform`](../../lib/zcu_tools/program/v2/README.md) 與 [main GUI](../../lib/zcu_tools/gui/app/measure/README.md)。本 ADR 不保證多檔 all-or-nothing、crash safety、多 writer 協調、autosave、resume、browser 或任何使用者資料遷移。

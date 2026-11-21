---
status: accepted
---

# 0063 — Persistence：保存權威與資料契約

**Last updated:** 2026-11-17，global／point 共用一套參數型別與單檔開檔入口

## 脈絡

App memento、Experiment Data File、workflow run artifact、`params.json`、RecordTable 和 waveform asset 都會落盤，但保存目的不同。以檔案格式統一它們，會讓通用讀寫層決定量測語意、完成條件或使用者路徑政策。本決策依保存目的分配權威，不建立統一的 persistence service。格式、欄位和各類資料的操作細節由各 owner 維護。

## 決策

### 保存目的與 owner

**規則。** 各 app 擁有 memento 的 schema、版本、擷取、還原與保存時機。共用的 `SingleFileCaretaker` 提供單檔 I/O、格式錯誤時的 defaults 與 atomic replace。Measure framework 只在 lifecycle close 保存 Qt 視窗尺寸與左面板寬度。SoC、device、entry／point、tabs／input 的檔案與重開政策由啟動腳本擁有，見 [[0064]]。

Experiment 擁有 typed Result、cfg 與 metadata 的編解碼、驗證，以及 axes、units 和 Data Variables 的映射。正常載入在 experiment 邊界還原 typed Result，不把 generic variable mapping 交給分析端。`datafile` 只擁有 [[0080]] 的 vars／docs／groups 容器，不反向依賴 experiment。Workflow engine 保存跨步驟的 manifest、journal 與 best-effort iteration `record.json`。它每輪供應唯讀、絕對的 `iteration_dir`。Workflow 保存 native 量測檔及圖形，並在 point record 記各檔用途與名稱。Engine 不追蹤、不保證這些檔案；單次量測 Completed 與保存的分工見 [[0062]]。

`QubitParams` 擁有 `params.json` 的 typed handoff、section 更新與 project identity。`resources.entry` 組合 entry、global 參數與獨立 point。值和 opaque metadata 在同一文件提交。DocumentStore 處理單檔衝突與原子提交，不換算數值。Entry 不產生、解釋或驗證 metadata，clone 原樣複製。Library 只保存 Spec，不建立模組來源或查 ledger。GUI context owner 和 Notebook 都經公開 capability 使用 entry，並共用 accepted executor。Executor 只提交 caller 提供的參數 metadata；GUI proposal 留在 frontend，Apply 才轉為 resources concrete items。Resources 不 import frontend 或 experiment，依 C8、C9。

`resources.entry` 另存各 scope 的 authored `aliases.yaml`。ContextService 擁有有序表語法與 Apply 時機。`RecordWriteback` 只 append adapter 組裝的 JSON object；框架只補 `component`、`run_id`、`point`、`at`、`source`。明示的 table binding 決定 `component`，同名 producer 欄位不能覆蓋它。`resources.record_table` 擁有 append-only JSONL，每個 run_id 讀最後一整行，CSV 只作匯出。`ArbWaveformDatabase` 擁有資產命名、formula 語意與 I/O。`core.waveform_format` 驗樣本、編解碼 NPZ，只驗 recipe 的 JSON 語法。

**例子。** 一次量測的全部 Data Variables 保存到 `data.h5`。Log Browser 使用衍生 Labber 檔。同一批 adapter 提案若附元件記錄，該記錄另寫 `records/components/<target>.jsonl`，不與 `records/ledger.jsonl` 或 `records/<event UUID>.json` 附件共用行格式。欄位範例見 [Record policy](../../zcu_lab/v2/measure-authoring.md#record-policy)及 [RecordTable 投影](../../lib/zcu_tools/resources/record_table/README.md#裝置投影projectionpy)。

**邊界。** Memento 不備份量測，Measure framework 不還原 domain session；共用 caretaker 不規定 close-only，也不訂閱高頻事件。跨 workflow 參數不存 sample arrays 或 dense curves，不由 generic table storage 解讀。參數文件只讀 major 2，接受同 major 的 forward minor，寫 `format_version: "2.0"`。入口拒絕舊 major 或 `provenance` 旁表，錯誤列出路徑、讀到的版本與期望的 2.0。使用者自行修改舊文件，沒有雙讀或轉換工具。共用 header、metadata I/O 與 value staging 見 [entry owner](../../lib/zcu_tools/resources/entry/README.md#視圖與-metadata)。其他容器不因此升版。

MCP 的單次 aliases／target override 不寫 alias 表。RecordWriteback 不選裝置、不注入 `dev_value/dev_unit`，也不因缺裝置資料而省略提案。舊列缺 `component` 仍能讀，不補寫。投影只用 caller 指定的裝置，不猜裝置、不以舊 flat 欄位回退。Runtime 不讀舊 API 名稱或 `samples/`；wire kind、written key、session prefix 分別為 `record`、`records`、`record-<n>`。

永久波形的播放陣列、reference time axis 與 duration 屬於資產。Writeback 使用 entry 下 global／point 共用的 `<database_path>/arb_waveforms`。Library 的 `arb` Spec 只存 key。Native `arb` cfg 帶 key 與本機絕對 path；`arb_memory` 帶樣本而無 key 或 path，不能序列化或進持久 Spec，見 [[0072]]。Program 對兩種來源都用完整樣本時長，不解讀 formula、不查 global DB；模擬器讀編譯後的 envelope。

**依據。**
- 2026-10-13，使用者：「可以改metadata」。原裁決未附入庫討論，參數容器變更見 Git `e85d40caf`。
- 2026-10-25，使用者：「我認為框架應該對SampleWriteback保持中立無知，可能可以叫RecordWriteback也說不定，行為就是append一個格式不拘的json到紀錄的jsonl中，格式由adapter間的隱式policy決定，資訊也尤其組裝，當然文檔與註解要寫這個policy好讓人維護」。同日確認改名：「我認為sample可以遷移到新名字，稱現在還沒有真實測量的時間遷是最安全的」。原裁決未附入庫討論，改名見 Git `1a684aa75`。
- 2026-11-19，使用者：「波形我觉得存 arbitrary 就好了，arbitrary 就是拿来存这种特别的东西的。其他采用建议。」見[記憶體 arb 討論](../discussions/2026-11-20-in-memory-arb-waveforms.md)。

### 設定快照與 acquired 事實

**規則。** 設定快照的正本只在 `data.h5` 的 cfg 文件，即現行 Container 的 `docs/cfg`。`CfgSnapshot` 保存 values、cfg_type 與 schema_version。Notebook 與 GUI 讀 native 檔；Labber 從同一份 prepared payload／snapshot 匯出，只供 Log Browser 使用。Ledger 的 acquired 只記 run_id、experiment、point、roles 與事件時間，不從 Form 或 RunRecord.cfg 產生第二份設定，也不寫 cfg_summary。

**例子。** Run 完成後未按 Save，仍可追加 acquired，但不建立 run 目錄。舊 ledger 的 `cfg_summary` 或 `entry_id` 不參與投影，`get` 的 raw `event_json` 仍保留未知欄位。

**邊界。** 部分投影只驗請求欄位與必要祖先；`events()`／`get().event` 的整筆投影仍驗完整 payload。歷史 import 的 `source_entry_id` 只留在 raw，`source_event_id` 與 `source_event_json` 的引用契約不變。Append 完整驗證新事件，對歷史只驗追加所需的 JSON、版本、event／run ID、duplicate、完整尾行與既有安全條件。未使用的壞 payload 不擋追加，無效新事件不改原 bytes。Index 不預驗未使用的附件；開檔或解析路徑前驗 canonical reference 與 root containment。不升 ledger major、不重寫歷史行或附件。Labber-only 舊資料先離線遷移；Load 不回填表單。

**依據。** 2026-10-14，使用者對快照位置選「A」，並說：「Notebook與GUI應該都讀data.h5，只有log browser讀Labber檔。」2026-10-15 確認：「1可以，2採用忽略不支援欄位的方式，3同意，4一起做吧」。見[量測成果與設定快照](../discussions/2026-10-14-artifact-persistence.md)及 [RecordsLedger](../../lib/zcu_tools/resources/entry/README.md#records-ledger)。

### MCP call log 與 ledger

**規則。** MCP call log 是 best-effort 除錯紀錄，不併入 ledger。Ledger 記持久的事實，其 origin 只標示 gui、notebook 或 mcp，不帶 MCP call id。

**例子。** 清除、截斷或停用 MCP call log 後，仍以 ledger 的事件為準，不依賴可丟棄的 call id 對照。

**邊界。** Call log 不證明 ledger 或輸出檔已提交，也不增加跨檔原子性、fsync 或掉電保證。

**依據。** 2026-10-08，使用者以「採用建議」確認移除跨層 call id，只保留 origin。傳遞 call id 會耦合 MCP、GUI 與 ledger，卻不能讓 call log 成為可靠證據。原裁決未附入庫討論。

### Run 輸出位置與 writer 權責

**規則。** `result/` 是 Result root，`.gitignore` 只保留這個新格式名稱，不用 `/results/`。`ResultEntry` 的存在與歸屬由 `result/<name>/` 決定。Entry 不產生 UUID；create／open 只檢查 Result。Save 和資產發布各按需建立自己的 Database 目錄，沿用同名 Database、不偵測混雜。`ResultEntry` 提供名稱、UTC 建立時間、exact Result path、Database 輸出位置與 scoped ledger。Point 表示實驗語意下環境參數一致的單位，不限於 flux；一致性不包含當次掃描的參數。

`bind_run_outputs` 依目前 Result／Database roots、Run 記錄的 entry 名稱與 point 綁定，不開 entry、不查 entry 或 point 存在性。GUI Save 不從 live selection、捕捉時的 storage 或原檔位置選目的地。`SaveLayout` 用 exact paths、caller 的安全單段 stem、point 與保存時間計算輸出位置，不寫檔、不改 stem。GUI 委派 adapter 自己的 `AxesSpec`／`GroupedAxesSpec` callback，Notebook 沿 core spec，workflow 的 Labber export 由 caller 提供 callback。共用 `SaveContext` 包含完整前端名稱、完整 run_id、歷史 point、解析後的 roles、可用 cfg values 與保存時間；缺 cfg 傳 None，不補 live 值。預設與 mux callback 使用完整 run_id。框架不加前綴或補 run_id，自訂 callback 負責避免不同 run 共用 stem。

`prepare_run_data` 依 typed RunRecord 與宣告的 spec 準備同一份 SI／離散 payload 與 cfg snapshot，native 與 Labber writer 各自消費，不互讀或互呼。Writer 只管自己的格式與 caller 指定的 exact path，不選路徑、不擷取 live context、不決定 batch。`experiment.artifacts` 擁有選集、輸出順序與逐輸出失敗隔離。`RunRecord.metadata` 是中性的 `JsonObject`，預設 `{}`，建構時 deepcopy，缺 key 表示未填。它與 `docs/metadata` 一對一，不帶 native schema 或 reader。Experiment 擁有 producer 用的 `RunMetadata` 與讀者按需驗證用的 `RunMetadataContent`。Native 與 artifact Save 在 I/O 前完整驗證 schema 及有限 JSON，GUI 也在修改 comment 與 prepare_ledger 前驗證。Native data Save 另比對 experiment tag，analysis-only Save 不比對、不新增 expected identity。寫入保留原 JSON 的全部 key，包括作者額外 key 與已廢棄 key。

Run metadata 的身分、時間、completion、RunSnapshot 與 SoftwareProvenance 都是 caller 提供的歷史事實。RunSnapshot 記 entry_name、point、roles；ParameterSnapshot 記作用中參數、library Specs、authored aliases 與 fixed names。GUI 在 Run admission 固定文件組，global 與 point 都可 Run、Analyze、Save。Save 不重新擷取 live 文件。Session 的 capture／restore adapter 連接兩位 owner；resources 與通用 `DocumentSetCapture` 不依賴 experiment。Device snapshot 是 Run 前所有已連線 device 的設定。GUI 在 lease 後、adapter.run 前讀取；Notebook 由 caller 持有資源，在建立 RunContext 後、core.run 前讀取，不加 lease。每台讀一次 `get_info`；沒有已連線的 device 時記 `{}`，任一讀取失敗就拒絕 Run，不進 adapter／core、不補快取、不寫設定。掃描與 Save 不改寫或重讀這份 snapshot。

**例子。** 將 `data.h5` 移到別處再 Load，Analyze／Post 只用檔內 capture。缺快照時用空環境，不造 general、created_at、target 或 storage。再次 Save 用目前 roots 和檔內身分。缺 entry 名稱用 `unnamed_entry`；缺 point 保持 None，圖副本放 entry 層 `figures/`，有 point 則放 `points/<point>/figures/`。相同 stem 的 native `data.h5`、具名 PNG 與 `analysis/<pane>.json` 重存覆寫。自訂 stem 隨保存時間變動時，另寫新目錄。預設與 mux stem、Labber 日期樹、圖副本命名和 workflow 雙側目錄見 [entry 輸出位置](../../lib/zcu_tools/resources/entry/README.md#輸出位置)與 [Labber 命名與身分](../storage-format-history/2026-10-27-labber-identity.md)。

**邊界。** `rename_entry` 只移動 Result、不改文件、不移動 Database。新 Run 使用重新開啟後的名稱，已有 Run 重存沿原記錄名稱。Database 不決定 entry 存在或 clone 的身分，不另做雙目錄改名或復原。舊 global／point 的 `entry_id` 即使壞值也忽略，讀不改 bytes、明確寫入時移除。新 ledger、Run metadata 與公開 wire 不帶它，舊 ledger 不用它投影或追加。Run／event ID、point source 與 roots 的核對不變。Clone 用來源當下內容。無 records 時讀取回空，首次 append 才建目錄。同名替換可用 `ParameterView.refresh`、`ResultEntry.new_point` 或重開讀取，不以 UUID 拒絕。GUI 用同一批次 watcher 監看已安裝的 global 與 point；嚴格驗證後才安裝，pending／revision 與失敗保留舊內容見 [[0081]]。

Native runs 平放於 `Database/<name>/runs/`。Workflow 本身仍以獨立的 Result／Database roots 保存 manifest／journal 與 acquisition／圖，不套用單次 Save 的 stem。Labber 用日期路徑，只有初始名字已存在才找 numeric suffix，不作跨程序保留。重存不改 acquisition 時間或完成狀態。Labber comment 帶 `snapshot: {point, roles}`，保留原 comment、timestamp。Labber reader 忽略不支援的 envelope key，不要求歷史檔補 snapshot，不升 wire 或 Container 版、不移舊輸出。Labber 成功路徑只記在 SaveReport 和 saved 事件；保存編排產生事件，RecordsLedger 驗證、追加，失敗不把已寫檔案改標為量測失敗或視為已回滾。

GUI Save／Save All 與 MCP 走同一保存流程。`SaveArtifact` 可含多個固定 DataItem／FigureItem／AnalysisItem。每個 item 編碼一次，相同內容的對等輸出共用 bytes。DataItem 的 cfg 或準備失敗只影響 data outputs。AnalysisDocument 擁有窗格 JSON 信封，也可用 `to_doc` 放進 `docs/analysis`，不改 SaveLayout 或 ledger。每次呼叫回最新 SaveReport 並追加 saved；ledger 失敗向外傳遞，不 retry 或 rollback。Save All 也重寫已保存的選定成員。GUI 在 owner thread 同步回 report，不開 handle、worker 或 save-busy phase。Data pane 只顯示 path、status、error。Notebook 的本機單檔 save 不猜格式；`save_run` 用 bound `ParameterView` 的公開 `capture_storage` 與 adapter 驗證的 acquisition source。`SavedRun` 固定資料、context 與 evidence；`PartialSaveError` 保留同一 handle 供明確重存，不自動 retry 或建立恢復收據。依賴由 C8、C9、C12、`parameter-context-no-experiment` 及 `artifacts-core-no-ui` 約束。

Native reader 只接受 [[0080]] 的 zcu.container major 3，往返保留同 major forward minor 及未知內容，不降版；不支援舊 native 2.x。Experiment 驗文件 schema。`started_at` 可為 UTC ISO 字串或 None，`completion` 可為 complete、partial、stopped、unknown。缺歷史證據用 None 或 unknown，不由檔名、mtime 或 run_id 補造。正常量測仍填開始時間與已知完成狀態。Workflow `load_trace` 只接受 complete，允許 None＋complete；Notebook 只載入 RunRecord，不製造 acquisition evidence。這項格式不另增前端的開始時間或完成度顯示與排序。

GUI LoadService 讀 Container、只先驗 experiment tag，再交 adapter.restore；experiment 和 Notebook 維持 typed RunRecord，GUI 只持有 opaque record 與明示 cfg_issue，見 [[0069]]。Load 保留全部原 metadata，不加輸出路徑、不追加 acquired，不讀外部 entry 或 point，不用 Form、library、default 或 live device 補 cfg。逐欄讀取只看該欄，缺欄回 None，experiment 可 fallback 至 adapter_name；壞值仍以 `MetadataFieldError` 帶欄位、`/docs/metadata/<field>` 與原始原因報錯。Results browser 唯讀顯示原 cfg_type、schema_version 及以 values 為根的相對路徑，不回填、不改檔、不重新分析。也可經 `datafile.load` 離線查看 `docs/cfg`。見 [[0067]] 與 [experiment owner](../../lib/zcu_tools/experiment/README.md#實驗介面與資料)。

Load Data 的 File 選檔與 opened entry 清單分開。清單掃描 `Database/<entry>/runs/*/data.h5`，以 `datafile.read_doc` 只讀 run_id、experiment、point、開始／結束時間及 completion；不讀 ledger、cfg、arrays 或驗 provenance。未知時間留空、unknown 不改。Metadata 讀取失敗時 run_id 用空字串，其他欄留空，保留 exact path 與錯誤，選取仍由 Load 報告結果。清單不從目錄名猜身分、不建立目錄或事件，遷移也不為清單手造 ledger。

Run／Load／Analyze／PostAnalyze 及 Save 共用 [SaveService planner](../../lib/zcu_tools/gui/app/measure/artifacts/README.md)，每次只綁一次。預覽只拿 layout，不依賴 `run.storage`、不建目錄、不標為 Saved；Save 使用同次 binding 準備 ledger。預覽與 Save 各用當次保存時間。Native eligibility 共用 experiment 的 required-key 檢查，只驗齊全、不驗值；缺 key 保留 declarations，但無預覽、不寫 acquired，Save 回 storage_unavailable。通過仍須在寫入前完整驗證。預覽只隔離 `MetadataFieldError`，其他規劃錯誤照常傳出。Save report 給實際路徑與逐項結果，不另建 `source.json`。

歷史內容供 Analyze／Post 與提案求值、解析 alias／slot；Apply 重取當下作用中文件組，不寫歷史來源。Global 寫 entry 層，POINT 寫所選 point；只在未開 entry 時以 no_context 拒絕非空寫回。Leaf read 與完整驗證的分工見[資料容器的讀取契約](#資料容器的讀取契約)。`docs/metadata.device_snapshot` 的 key 是裝置名，value 是設定。None 寫 JSON null，空 mapping 寫 `{}`。設定保留 get_info 的單位，YOKO 值與 rampstep 用 mode 對應的 A 或 V，Fake 用 native，RF 頻率為 Hz、功率為 dBm、不加 `value_unit`。不換算或另造單位欄位，當次使用值仍由 cfg 的 dev、flux_unit、sweep 範圍與 persisted axes 記錄。

**依據。**
- 2026-10-16，使用者選「A」，原 cfg 可離線讀，不加 GUI 入口。2026-10-23 核准唯讀 Results browser 例外。原裁決未附入庫討論，見 Git `b6fbfa57c`、`2bc9107ea`。
- 2026-10-19，使用者對可缺的開始時間、unknown 完成度與不加來源標記答「可以，採用建議」。2026-10-24 裁定不支援的舊資料 Load 失敗即可。原裁決未附入庫討論，見 Git `b1956e3db`、`0751411df`。
- 2026-10-20，使用者：「好，統一result，那gitignore也做處理」；「甲，簡單的機制加一致的規則比非預期的兼容邏輯好維護與使用，同樣名字的Database會混雜屬於有意不處理的設計。」對改名：「改名時Database不動，Database是輸出位置，不是什麼要綁定的目錄」。對 Load Data 清單說：「我以為只要是data.h5就可以看到，能不能Load另說」，並答「可以」，同意只從檔案 metadata 讀取。原裁決未附入庫討論，見 Git `d3cd468c6`、`5d35a569b`、`22f999f32`。
- 2026-10-27，使用者：「Q-L1-a+b，名字也許要有adapter」；「Q-L3我的想法是adapter決定怎麼生成名字，可以有預設實現。」又說：「run_id也是由adapter行數補上，請你遵守框架不做多餘行為的設計理念，將可能需要的資訊作為參數傳給save_stem。」同日合併前檢查發現只取後六碼可能覆寫其他 acquisition，預設及 mux 改用完整 run_id。2026-10-19 與 2026-10-20 的固定命名由[存檔歷程](../storage-format-history/2026-10-27-labber-identity.md)保存；離線 migration 仍明確供應舊 native stem，不改名舊 Labber 檔。
- 2026-10-28，使用者問「這裡的UUID能做什麼？什麼是目錄名不能而必須要UUID?」，先答「選B」採位置歸屬，再答「A」完全移除 entry UUID。同日：「一個point不是一個flux點的意思，而是在其中的實驗語義下，具有一致的環境參數的意思，不只包含flux。而對於實驗本身就在掃某個環境參數，其一致性要求本來就不包含那個參數，所以哪個參數不會影響一致性」。對 runs 位置說：「Database平放與按point分感覺都可以，我認為目前平放就先保持好了」。見[entry 與量測快照的格式歷程](../storage-format-history/2026-10-28-entry-names-and-run-parameters.md)；原裁決未附入庫討論。
- 2026-10-28，使用者：「我認為run R確實可以記錄是哪個entry與point，但Load為什麼要看entry/point資訊？他又不需要，他只需要當時capture的環境就好？」「沒有快照的舊量測，可以用空環境來生成分析參數預設值，而不是不能分析。」寫回裁決「1A 2A」使用目前選定的 point，現行目的地隨作用中文件組。對重存說：「我希望load只需要給data.h5的路徑，不要求其在對應的儲存位置，但再儲存的目標就是內部記錄的entry與point。」「我認為Save的部分，應該不用找？」「直接生成路徑，然後覆寫就好」「我認為沒有記錄entry_name，那就用unnamed_entry之類的」。原裁決未附入庫討論，見 Git `69a634620` 與上述格式歷程。
- 2026-10-29，使用者要求：「setup與point要有一致的檢測行為」。對 global Run 與圖副本等回答：「1我認為可以改成可以，Database本來就不看point，全域figures的位置也比照point內的 2應該也可以 3可以，用inspect改 4預設從全域繼承，或是指定empty或是clone其他point 5我有考慮統一名字」。對缺 point 放 entry 層 figures 答「A1 B採用建議」，取代格式歷程中的舊安排。對首次保存前預覽先指出「1A, 但第一次save前沒預覽很怪」，再對預覽與 Save 共用計畫答「1做」。原裁決未附入庫討論，見 Git `d0fe4fde9`、`d26c34008`、`d9caffe91`。
- 2026-11-01，使用者：「但RunRecord是相當中性且底層的，不能帶有上層資訊的認知，且要夠通用靈活，可以放想要的格式，與datafile對接」。對 schema／reader 分工與 Save 驗證答「3aA 3bA」，保留廢棄 key 答「廢棄key選A」。原裁決未附入庫討論，見 Git `36f6b1e42`。
- 2026-11-18，使用者過目 native 同名重存的 agent 選擇：「我大概看了一遍，基本上是没问题」，並說「并不算是裁决，而算是建议」。這是看過的建議，僅供參考，不是使用者裁決，也不是未覆核。原討論未入庫，見 Git `1f4b16aa0`。

### 同一種資料不因 global 或 point 分兩套

**規則。** 同一種資料不因層級分成兩套型別、函式或方法。`point: str | None` 選檔，None 是預設的 global；ContextService 或 Notebook caller 持有 focus。Entry 根與各 point 都有 `parameters.yaml`、`library.yaml`、`aliases.yaml` 這組文件。ResultEntry 每種檔有一個同形入口，各只讀自己的檔。名稱要指出行為與定位，不用 setup 稱呼 global。

**例子。** `SetupView`／`PointView` 收成 `ParameterView`，兩層也共用 `ParameterDocument`、`ParameterGeneral`、`capture_storage(view, roles=None)` 與 `BoundDocumentWritebackTarget(view, storage)`。用 `entry.parameters(point=None)`、`entry.library(point=None, *, assets=None)`、`entry.aliases(point=None)` 開 global，傳 label 則開該 point。切換 global 用 `use_point(None)`。

**邊界。** 只有 point 有的欄位用同一型別的選填欄位；格式不同的檔，如 `EntryTemplate`，才獨立成型別。資料物件不帶 global／point 身分，來源與兩個 roots 留在私有 binding，不占用 `ParameterView` 的公開名稱。Roles 比對來源與兩個 roots；capture 由 view 來源推出 entry／point，writeback target 建構時比對 view 與 storage 路徑。Clone 比 result_root，拒絕以 global view 當來源；從 global 建 point 用 `mode="inherit"`。磁碟格式、header、檔名與路徑不變。

**依據。** 2026-11-16，使用者：「使用一個中性的View來取代SetupView/PointView，類似的物件也比照辦理，讓setup/point的概念只是管理層類似focus的指標，底層資料結構是一樣的」「沒指定point就是操作在global，預設就是global，也就是沒有focus的point，對應到這個entry下的全域資料容器檔，不要再有setup這個名詞與屬性了，然後精簡這堆類似的類與方法，我無法從名字看出其行為與定位」。三個開檔入口的依據是同日的「這三個檔是一起切換的，捕捉也是一起捕捉的，值得給這三個的合併一個名字，而這三個的相關方法也要一致」。`use_point` 簽名由 designer 代裁，使用者 2026-11-17 追認，回答中的「四筆追認」涵蓋這一項。見[文件組與型別討論](../discussions/2026-11-17-global-point-parameter-types-and-document-set.md)。

### Global 與 point alias 表

**規則。** Global 與各 point 保存各自的 authored alias 表，和參數、library 並列。選 point 時整組換入，不逐名稱回退 global。Alias 表保存工作焦點，parameters 保存元件事實。Resources 擁有 `AliasValue = str | list[str]` 與檔案表示；session 驗證及解析有序的名稱與 backward `$alias.field` link。Run snapshot、storage 與 acquired 固定解析後的名稱或有序清單，歷史讀取不重展 live 表。Experiment 與前端向下引用資源型別，不讓 resources 依賴 session，見 C8、C9。

**例子。** `participants: [Q3, Q1]` 記本次參與順序，無須改元件 schema。新 point 預設 inherit global 三檔，library 走 `copy_library`。Empty 只建最小有效參數文件；clone 只複製指定來源 point 的文件，來源缺 library 或 aliases，目的也缺。此後各組獨立。

**邊界。** 缺 library 或 aliases 當空，revision 為 None，讀不建檔，首次明確編輯才建；session 編輯若遇外部先建檔，報 revision 衝突不覆寫。缺參數文件仍報錯。Clone 必須給來源，其他模式禁止給。空清單與空字串都未綁，缺列、未綁或壞資料由 session 指名 alias。解析不得略過壞元素或擅取首項；只接受單名的 caller 遇清單拒絕。存表不檢查設備可用性、不換成 tone index。檔名、YAML 清單形狀、`zcu.aliases` 1.0 與 forward-minor 契約見 [entry alias owner](../../lib/zcu_tools/resources/entry/README.md#global-與-point-alias-表)。嚴格讀寫失敗不退 preset。

一次 I/O 固定解碼、原始內容錯誤與 observed bytes revision，session 也保存語法錯誤。壞 aliases 不擋參數或 library；實際使用或投影才報原錯，無效不是空表。修復後的新 capture 不改舊 capture；caller 不重讀或補驗。`apply_alias_rows` 先驗整表，entry owner 成功原子替換後才安裝 memory 與發布 fact。None 寫 global；label 只可寫目前 focus 的 point，否則拒絕 `InvalidInputError(point_not_focused)`。未開 entry 報 `FailedPreconditionError(no_context)`，不暫存或自動打開；startup 先開 entry，再經同一入口，不提供 memory-only alias 初值或 presets。Inspect 可在 point focus 時改 global，不改作用中文件組或舊 capture。編輯失敗保留輸入、舊表與舊檔，關閉不撤回已保存內容。Open／Refresh／Reset 捕捉當時使用的表，不自填空列；切回 Global 使用 global 表。舊 Run 的 Analyze／Writeback／Save 保留原 capture 作來源。單檔保存不承諾跨檔、多 writer merge 或掉電恢復，revision 與衝突見 [[0081]]。

**依據。**
- 2026-10-20，使用者：「採用乙，alias. yaml的獨立檔案，找不到就視內容為空，與其他兩個資料容器並列」。2026-10-23 對加入 participants 答「甲」。原裁決未附入庫討論，見 Git `1a25d7a05`、`09113733b`。
- 2026-10-29，使用者：「行為須一致，沒有就當空」，並確認：「A，我目前先設計成整個作用域換成point，沒有個別的fallback邏輯」。對隔離 alias 錯誤、owner 分工與 observed revision 的補充答「同意」，不放寬 writer 驗證。原裁決未附入庫討論。後續開檔結果的設計見[同一次 alias 檔案觀察](../discussions/2026-11-17-alias-file-cached-observation.md)。

### 資料表示與完整性

**規則。** 參數的 YAML、快照與 view 用相同工作單位，metadata 不換算。元件 model 擁有型別、驗證、預設與 extra 策略。`zcu_lab.components` 擁有 kinds、欄位、內建角色；lib 外的組合根在 startup 顯式註冊 ComponentRegistry，不在 import 或 reload 註冊。框架只消費註冊宣告，不反向 import 定義，不猜相容規則。

一次使用者動作對每份文件提交一次。Owner API 接受該檔的一組變更，單項是長度 1 的批次。Resources、DocumentStore 與 ledger_models 分開讀取投影和寫入驗證。DocumentStore 保留 raw YAML，完整驗證產生 normalized tree 與 typed snapshot；未變 bytes 可沿用驗證快取，不查來源依賴。空 edit 與讀取不寫檔。非空 commit 在最新 normalized tree 合併並驗證，再將差異套回 raw，寫回 model 轉換、保留未改節點。Library 不解析模組來源；clone 複製 point 參數、metadata、合法 Specs 及 aliases，不改來源。跨 entry clone 不複製事件或附檔、不寫 source 或 cloned_from。`import` kind 只讀歷史，不再有 producer。Clone 失敗只清本次目的 point，不承諾跨檔 snapshot 或交易。

**例子。** Library 先驗 header、容器及可定址名稱，再逐 root 驗 Spec。一個 root 的 Spec 壞了，合法兄弟仍可讀，`invalid_entries()` 回失敗 root 與原因。Header、容器與 I/O 錯誤仍傳出；strict refresh 不略過壞 Spec，失敗保留上次快照。讀取與空 edit 不改原 bytes、不做 .bak；新 draft 仍嚴格驗證。Library 2.0 不再保存 per-leaf provenance 或 stderr，容忍舊 `provenance:`，非空寫入移除它。兩種操作都不查 ledger 或 entry_id。

**邊界。** 容器 header、major 不符直接拒絕。Forward minor 保留容器層未知欄位，不取代 nested model 的 extra 策略。Model 的 forbid 拒絕未知欄位，ignore 丟棄並在非空 commit 移除，allow 保留並公開。框架沒有 runtime 相容 loader；component 另有顯式升級鏈，Entry 不替寫值或格式轉換產生 metadata。

Experiment Data File 的映射仍由 experiment 擁有，不沿參數單位。Spec 用 inner-first axes 對應 Result-native shape，不要 caller transpose；未定義物理量的 scalar 不捏造 A／V。State／phase 是離散 axis；Data Variable 表達結果語意。Native 的多個變數可各用自己的 grid，共用身分與 metadata。Labber 預設 common grid，須明示 variable-to-channel mapping；異質 grid 要由 spec 宣告 independent，各自保存 1D／2D axes，不自動切換、不改 native 契約。異質 streaming workflow 有自己的 layout 與 completeness，不因同是 HDF5 就視為同類。Result 所需 variables 齊全不代表測完所有點；停止、失敗、已提交資料能否分析是不同問題。現行 workflow 提交與保存見 [[0062]]，詳見 [experiment owner](../../lib/zcu_tools/experiment/README.md#實驗介面與資料)與 [[0080]]。

**依據。**
- 2026-10-19，使用者：「Labber存檔問題請獨立修」。明示 independent layout，不放寬未宣告實驗的 common-grid 檢查。原裁決未附入庫討論，見 Git `ca3f0f242`。
- 2026-10-21，原 ADR 註記 autofluxdep 已移除；舊 node-row、flux-level commit 與 export 規則不適用現行 workflow，留 Git 歷史。
- 2026-10-23，使用者對每份文件一次交易答「認同」。見[寫入慣例](../discussions/2026-10-23-store-read-efficiency.md)，該討論的舊 library 來源驗證已不適用。
- 2026-10-28，使用者：「我其實沒有要知道來源的需求，這功能純屬於錦上添花」「來源我一直以為只是metadata，沒想要解析」。這項決定接替 2026-10-15 的 library 讀取清理政策，見[移除模組來源](../storage-format-history/2026-10-28-remove-module-provenance.md)。

### 資料容器的讀取契約

**規則。** 每次操作只讀、只驗所需的檔案、欄位與必要祖先。沒用到的欄位缺少、多出或格式不同，不構成讀取錯誤；也不要求舊欄位有相容別名或轉換。實際使用時發現壞值，要指出來源和欄位，或在功能可選時關閉並註明原因，不把壞值降成空內容。量測與擬合分階段，量測只記原始 JSONL，分析再組合所需資訊，不為分析預定行格式。

**例子。** Point availability 只看參數文件是否可讀、能否解析結構，以及 header／schema_version；不讀 library 或 aliases，不檢查未使用的 leaf。壞檔標記該 point 不可用並附原因，UI 停用、不能作 clone 來源，沿用 tooltip 與 remote error。其他 point 正常。Availability 不成為 use_point、Load 或 Save 的另一項准入條件。同步見 [[0081]]。

**邊界。** 同一 component 的壞 leaf 不擋其他 leaf。Leaf read 執行該欄的 annotation、field validators（含 before）與祖先檢查，不跑 model 層 before／after。缺欄用該欄的 default／default_factory，同文件重讀保持穩定。跨欄位不變式要用完整 component 投影；完整 metadata／component／document 投影及 Save、edit、seed、clone 仍完整驗證。只靠 model validator 的 legacy 轉換，須明確 edit 或 upgrade。分析工具不預驗整份 schema、版本或無關欄位，但各 owner 對容器 header／major、component schema_version、JSON／YAML 表示、實際用到的附件路徑及 ledger 追加安全規則不變。這也適用只移植 point 或個別分析成果。見[entry owner](../../lib/zcu_tools/resources/entry/README.md)。

**依據。**
- 2026-10-26，使用者：「我對這種資料容器的行為預期都是：可以不讀舊格式欄位，但不要因為少了欄位或多了欄位就報錯。」並說：「所以我才希望將擬合與測量分開階段，因為你不會知道擬合需要什麼資訊，也不知道想要什麼格式，只要都記錄成jsonl，透過要分析時按需處理來組合需要的資訊。所以請你不要讓分析工具與讀取工具有非必要的格式檢查」。
- 2026-10-28，使用者：「我會預期一個app只會需要給他需要的資料，而不是檢查用不到的格式與欄位」。2026-10-29 確認：「基本上可以，但我要提醒，我一致的設計規則是：只檢查需要用到的資訊」。原裁決未附入庫討論，見 Git `c30bf2bbd`、`d0fe4fde9` 與[歷史快照的按需讀取](../storage-format-history/2026-10-28-entry-names-and-run-parameters.md#native-run)。

### Component 文件版本與明確升級

**規則。** 存 component 的 global、point 與 entry template 各自用整份文件的 `schema_version`，不取代容器的 format／format_version。Lab 在 ComponentRegistry 註冊具名 `ParameterSchemaPolicy`，提供目前版本與命令。未配置即失敗，lib 不預設 lab 版本、不把 policy 散傳到 entry／template API。Runtime 只讀目前版本；缺版或版本不符時，回報文件版本、目前版本與可執行的升級命令，不猜、不寫。優先加有預設的欄位；結構變更才新增 vN→vN+1 純函式。

**例子。** 只有 `setup.yaml` 等舊檔名的 entry 不可開啟，錯誤指名舊路徑與改名指令。`resources.document_names` 擁有檔名，以根層 `parameters.yaml` 識別 entry。先以 `--scope filenames` 只改檔名、不改 bytes，再按需升級 components。新舊並存拒絕、不覆寫。Components 只枚舉新名，改名不升容器或 component 版本。操作見[先改檔名再升級內容](../parameter-upgrades.md#先改檔名再升級內容)。

**邊界。** `resources.parameter_upgrade` 擁有通用鏈與 YAML I/O，caller 提供完整舊格式分類、各版驗證與純函式步驟。未標版文件只在明確 upgrade 且完整分類後視為 v1。Runner 驗每步恰為下一版、current 也先驗再跳過。Dry-run 不建檔、不鎖，逐檔逐欄列差異。Apply 以 per-user runtime file lock 重讀並完整驗證，做 exclusive byte backup，再以 sibling temporary 做單檔 atomic replace。失敗保留原檔及已完成備份，其他檔繼續。這不是多檔或掉電交易。

`resources.upgrade_backup` 統一 parameter、predictor、record table 與 processed spectrum 的備份定位。以最近的 `result`／`Database` 所屬資料專案選旁側 `<專案根>-backup`，不放 checkout 或資料項目裡。Helper 寫入前驗目的地；caller 才建目錄、寫原 bytes。Override、相對鏡像、resolved `-backup` 與 symlink 規則見[備份位置](../parameter-upgrades.md#備份位置)，各工具不重做。

Lab 擁有目前版本、步驟與真實舊文件形狀；`scripts/upgrade_parameters.py` 只組合 lab plan 與通用 main。長期鏈不依賴可刪除的 `migrations/`。一次性 `storage_mapping` 與長期鏈共用 lab 的 flux 純轉換；migration 由 caller 的完整文件 batch callback 先轉 seed／value 再驗 current schema，不先發布舊格式。框架不解讀 lab 欄位。離線 YAML 只驗裝置名稱語法，不驗 catalog；使用時才驗裝置存在。C14、C16、C20、C21 隔離 lib 與 lab、migrations、scripts，不禁 caller 注入。框架在公開 runner／讀寫／main 接縫驗合成步驟，真實 lab 步驟以離線樣本驗收。

**依據。**
- 2026-10-27，使用者：「每次改component都會讓舊資料不支援很頭痛」，對整份文件版本及通用 runner／lab steps 答「12採用建議」。產品落點見[版本歷程](../storage-format-history/parameter-schema-version.md)，原裁決未附入庫討論。
- 2026-10-28，使用者對備份移出資料項目且和遷移備份同處答「2A」。2026-10-29，對共用 owner 與外部備份分工答「同意」；對統一檔名答「A1 B採用建議」。見[升級指南](../parameter-upgrades.md)及[檔名歷程](../storage-format-history/document-file-names.md)，原裁決未附入庫討論。

### 失敗、路徑與遷移邊界

**規則。** Run accept 先追加一筆 accepted，再按目的地提交：arb 資產一組、parameters 一組、library module／waveform 一組、records 逐筆。組內保留順序。Parameters 與 library 各一次 edit，任一項無效整組不寫。`WritebackResult` 回 accepted event_id 與逐項 committed values。Ledger append 失敗時原樣拋出錯誤，不開始後續寫入。Append 成功後若有一組失敗，executor 停止後續組，保留事件與已提交項目。`WritebackApplyError.result` 包含 accepted `event_id` 與已確認的 `committed` 前綴，也保留資產 I/O 失敗前已確認的項目。例外本身公開 `event_id`，以 `applied` 計已確認的項目數，並以 `__cause__` 保留原始錯誤。Notebook 直接收到例外。GUI context owner 發布實際成功前綴，service 標記後原樣拋出 cause；wire 保留 producer 分類。

Owner 在實際寫入時形成逐項值。批次 setter 驗 draft 並捕捉隔離值，整組 commit 後才回傳；record append 回傳當筆完整序列化的 row。Executor 回實際提交順序，accepted 的 writes 保留供應順序。`CommittedWrite.index` 是供應 tuple 的位置，GUI 用它關聯 item ID／pane／目的地，不用 path 或前綴長度推算，也不重開讀回、不拿 proposal 當 actual。後項覆寫父／子路徑不改先前收據。不作 per-item callback、worker、retry 或 rollback。

**例子。** 同批 library 引用新 arb key 時，先發布資產。Owner 先查整組，同 key 同 time、I、Q 回 `existing`，保留原檔與 recipe；全零 Q 等同缺 Q。撞 key 且樣本不同，整組不寫。新 key 回 `created` 並寫不帶 recipe 的 NPZ，不覆寫舊檔。Ledger 只記路徑、樣本數與時長。見 [waveform assets](../../lib/zcu_tools/resources/waveform_assets.md)及 [[0008]]。

**邊界。** Apply 取當下 `DocumentSetRef`，不自動建 point。Global acquired 與 record row 都可用 point=null，不升格式。Preview 的 destination_context 與 receipt 的 destination 只有 entry_name、point，無 kind。GLOBAL 的 point 為 null，POINT 為 label；receipt 的 entry_name 必為字串。未開 entry 仍可預覽合法 draft，兩欄與 current 為 null，但不授權 Apply。MCP 保留 opaque actual_path 與固定來源證據，Apply 不被 preview 位置鎖定。

Memento JSON 損壞或 codec 不符時，caretaker 回 fresh defaults 與供 host 顯示的錯誤。不移檔、不備份、不加保存保護，host 仍按原時機保存。一般讀 I/O 失敗也回錯誤 outcome 和 defaults，不冒充 decode 失敗。Provider 與其他 restore／apply 故障仍傳出。Caretaker 不懂 cfg；FormInput 格式與還原見 [[0079]]。量測資料、參數及未知 artifact 版本不能照 memento 降成 defaults。必要 variable、shape 或 unit 不符須回報可定位的錯誤。

Experiment writer 用 caller 的 exact path。Labber 拒絕已有目的地，native 預設也拒絕。明示 `replace=True` 才在同目錄寫 temp、關檔後 atomic replace，失敗清 temp 並保留舊目的地。Labber suffix 只避讓當時存在的檔案，collision 由上層回報、不隱藏 retry。單檔 replace 與 ledger append 不構成跨檔交易或掉電保證。Run 身分來自 run-scoped metadata，不從 slug 或檔名推論；Workflow manifest 連結 metadata 和 heavy-data roots，exports 不成為原始權威，現行 workflow 保存見 [[0062]]。

Runtime 按 artifact kind 和各自的版本契約讀取，不猜 legacy shape／unit；streaming v1 不因版號較小就當成 one-shot legacy。舊 experiment converter 與 GUI fallback 退役；pure DataFrame conversion 只產分析 frame，不是 raw loader，不補 `devices` 或改 CSV。QubitParams 的 project identity 遷移由其 owner 管，不因本 ADR 寫入使用者資料，不建通用 migration layer。離線 storage migration 使用 `migration_id` 辨識單次操作，manifest、report 與初始化暫存名皆屬該操作。需要重跑或 cleanup 時遇舊 ownership `entry_id` 便指名檔案拒絕，不讀值、不轉換、不自動刪改來源、備份或目的。用舊版完成，或確認後自行處理。Ownership 與越界路徑保護見[遷移指南](../storage-migration.md)和 [migration 格式歷程](../storage-format-history/2026-10-28-entry-names-and-run-parameters.md#離線-migration)。

**依據。**
- 2026-10-21，原 ADR 註記 Autofluxdep artifact 已移除；舊失敗與 finalize 規則留 Git 歷史，不視為現行契約。
- 2026-10-23，使用者對每份文件一次 edit 答「同意」。同日裁定：「resources／context 的 writeback executor 回具名結果：accepted event_id 加逐項 confirmed committed values」「值由實際寫入的 owner 當筆形成，GUI 只關聯 item ID／pane／目的地，不再整批後 readback、不用 proposal 冒充 actual」。分組見[寫入慣例](../discussions/2026-10-23-store-read-efficiency.md)；具名結果見 Git `927b14e7a`、`9a8bdd7e1`，原裁決未附入庫討論。
- 2026-11-19，使用者對 Apply 發布新 key、不覆寫舊資產答「可以」。與 recipe 寫檔的比較見[arb 資產寫回討論](../discussions/2026-11-20-arb-asset-writeback.md)。

### 共享資料與引用

**規則。** `params.json` 的 QubitParams 與工作點的 `Q1.params` 獨立，不自動同步。Typed 更新一個 section 時保留其他已知與未知 section。分析的 correction／alignment 不寫回原始 record、不改 explicit flux。Asset repository 的 rename／delete 只動資產，不掃描或改寫 ModuleLibrary 的 nested references。

**例子。** 把 `data.h5` 搬到另一個系統後，可讀量測陣列，不必重播。Native cfg 的資產 path 是本機絕對路徑，以 POSIX separators 序列化。檔案不內嵌 NPZ，要重播時由 caller 明確重定位。Rename 後引用舊 key，使用端解析失敗，由 caller 或使用者修正。Capture-bound `assets.path_for` 只驗 key 與檔案存在，不載 samples。

**邊界。** QubitParams 的 timestamp 只表示 section 修改時間，不是 provenance 或失效訊號；typed API 或 mtime 不提供跨程序互斥、多 writer 協調。Opt-in flat v2 frame 明示 `dev_value`、`dev_unit`，單位為 A、V 或 native；原始 record 不要求這些欄位。Flux 依 explicit、row-frame、fallback-frame 順序解析，記錄每列來源。Fallback frame 須符合該列 unit，單有 unit 不能補缺失的 flux 或 frame。覆寫同一 asset path 會改之後播放的 bytes；不提供資產 versioning、copy-on-run、packaging 或自動遷移引用，也不讓 repository 反向理解 experiment cfg 或跨 owner 批次寫入。

**依據。** 現行讀寫見 [QubitParams](../../lib/zcu_tools/resources/qubit_params.md)、[RecordTable](../../lib/zcu_tools/resources/record_table/README.md)及 [waveform assets](../../lib/zcu_tools/resources/waveform_assets.md)。2026-10-21，原 ADR 註記舊 autofluxdep export 已移除；其使用 Run 當時 snapshot 的契約留 Git 歷史。原段未記逐字裁決。

## 後果

維護者先找保存目的對應的 owner，再查格式契約：[`datafile`](../../lib/zcu_tools/datafile/README.md)、[`experiment`](../../lib/zcu_tools/experiment/README.md)、[`QubitParams`](../../lib/zcu_tools/resources/qubit_params.md)、[`parameter entry`](../../lib/zcu_tools/resources/entry/README.md)、[`RecordTable`](../../lib/zcu_tools/resources/record_table/README.md)、[`waveform assets`](../../lib/zcu_tools/resources/waveform_assets.md)、[`program waveform`](../../lib/zcu_tools/program/v2/README.md)與 [main GUI](../../lib/zcu_tools/gui/app/measure/README.md)。本 ADR 不承諾多檔 all-or-nothing、crash safety、多 writer 協調、autosave、resume、browser 或任何使用者資料遷移。

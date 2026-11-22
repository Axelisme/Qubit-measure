---
status: accepted
---

# 0069 — 實驗核心、前端包裝與具名圖形產物

**關聯：** [[0062]] 的實驗 workflow、[[0063]] 的保存、[[0065]] 的 cfg、[[0066]] 的 operation、[[0067]] 的 GUI。

## 問題

本篇處理核心與前端責任混用的問題。原本的 v2 實驗 class 同時提供 Notebook 的 last_result 便利行為、量測、分析與保存，GUI adapter 再補表單、參數轉換與統一介面。Notebook 呼叫習慣牽動核心契約，GUI 因而必須傳遞 cfg 外參數。Figure、數值分析與前端生命週期也由同一層處理。

只改用函式、拆 package 或增加宣告式 definition，不能解決這些責任問題。目標是先分開核心與前端，保留使用者直接撰寫 Python 和原生 Matplotlib 的能力。

## 決策

### GUI simulation workers

使用者正常使用的 GUI 模擬 startup 明確選擇 `max(1, (os.cpu_count() or 1) - 1)` 個 worker，保留一核且至少使用一個。這項規則適用於 single_fluxonium、integer_fluxonium、six_fluxonium 與 FTF。共用策略由 `gui.app.measure.startup_context.simulation_workers(cpu_count)` 擁有，startup 注入 CPU 數；工廠不查詢 CPU 數。FTF GUI startup 傳 `warmup=True`，在背景做 CPU 準備與 GPU 預編譯，見[〈顯式 cfg 與裝置環境〉](#顯式-cfg-與裝置環境)。例如八核主機的 GUI 使用七個 worker，CPU 數未知時使用一個。

工廠與 Engine 的預設保持 `workers=1`。測試、benchmark、研究與直接呼叫不套用 GUI 策略，可明確選擇一核或兩核。FTF Notebook 仍保留兩核並啟用 warmup。Engine 只在多條 timeline 間平行，單條 timeline 或單 worker 串行。演化期間 native BLAS 限為一個執行緒。本規則不承諾加速倍數，實際 acquisition 的加速量另行量測。

six_fluxonium 的六個 source 各有自己的 Engine，但來源準備仍逐一執行，不同時啟動六個 pool。Engine 僅在多 timeline 的 evolve 期間持有 worker pool。

2026-11-16，使用者裁決：「我要讓我正常使用的模擬速度儘量快，可以使用cpu數-1核」，並選擇涵蓋全部 GUI 模擬環境的「C」。使用者澄清：「你測試或研究用1或2核可以，但我正常使用請用多核」。使用者再次確認：「six_fluxonium也要，模擬器的都要。」因此全部 GUI 模擬 startup 使用多核，直接呼叫預設不變；此裁決取代先前 GUI startup 固定一個 worker 的規則。

### 核心與前端

核心保留無跨次可變狀態的實驗 class。NotebookAdapter 先綁定 soc、soccfg、DeviceManager 與 plot host 引用，再由 `nb_adapter(core)` 建立各自保存 records 的 NotebookExperiment。Caller 建立 experiment instance，不要求零參數 constructor。同步實驗不各寫扁平參數 wrapper，caller 直接提供 typed config／options。GUI adapter 直接使用核心，兩前端不互相包裝。

RunRecord 泛型組合 nullable typed cfg 與實驗專屬 Result。Typed cfg 的位置是 RunRecord.cfg；檔案的原始 CfgSnapshot envelope 由 StoredRun 保留，不把 typed record 當成 lossless rewrite envelope。AnalysisRecord 組合 explicit source RunRecord、實際 options、typed Analysis 與純具名 figures；cfg 與 result 從 source 取得，不另外保存一份可混用的來源。

核心操作形狀為：

```python
run(config, *, context) -> Result

# Optional analysis signatures; unimplemented operations raise NotImplementedError.
analyze(source: RunRecord, options, *, plots) -> Analysis
post_analyze(source: RunRecord, primary, options, *, plots) -> PostAnalysis

# External persistence functions, not core methods.
save_labber(source: RunRecord, destination: Path, *, spec) -> None
save_run(source: RunRecord, destination: Path, *, spec, replace=False) -> None
load_run_with_metadata(source: Path, *, spec) -> RunRecord
restore_run(stored: Container, *, spec) -> RestoredRun[RunRecord]
```

直接呼叫 core.run 不自動建立 record，也不保存 last state。共用 Notebook 包裝層建立 RunRecord，同步分析成功後建立 AnalysisRecord。前端只根據 core.run 的返回或拋錯決定是否發布：正常返回就發布，拋錯就交付原錯誤、將 operation 標 failed，不發布新 record，也不取代上一筆成功 record。

前端不尋找 Schedule、不讀它的 root buffer，也不從任意 core 失敗中合成 failed record。作者若要攔截錯誤、保存累積資料或定義 partial 結果，必須在 core 內明確處理。這使前端不必知道作者用了哪種 scheduler、buffer 或 Result 形狀。2026-10-31，使用者裁定：「對框架來說core run就是黑盒子，只是接收context然後回傳result的函數，內部要怎麼註冊buffer與怎麼攔截報錯都是作者的自由，scheduler只是一個給作者用的工具，框架不知道scheduler，stop_signal只是一個框架表達盡快結束並回傳result的訊號通道，在這個模型下，框架沒有partial的認知，更別說buffer要求與result格式對齊了，所有錯誤攔截與partial語意都由作者維護與負責，可以有工具與慣例」。

Interactive core 不實作 analyze，GUI adapter／plugin 與 Notebook 獨立分析工具各自負責。`Experiment` 的分析方法只提供簽名並 raise `NotImplementedError`，不強制實作 analyze，也不探測新舊簽名，也不建立互動 capability registry 或通用 callback framework。

RunContext 提供單次 run 的 soc、soccfg、具名 devices、plots 與顯式 scope，不保留 QickContext alias。Caller 在 run 前固定名稱到 BaseDevice 的綁定，核心只借用 driver。核心仍決定每個 sweep point 何時 setup；setup helper 在整批 setup 前驗證所有必要名稱，不查全域 manager。Connect、disconnect、registry 與 ResourceManager 生命週期留在前端 owner。Hardware handles 不進 cfg 或 records。

每次 run 建立新的 context、Plots 與 Scope，硬體 handles 可以重用，停止請求不能沿用。GUI 與 Notebook 建 run 根，workflow 使用作者明填 parent 下的 operation child。Schedule(scope=..., name=None) 在入口建立 schedule child，每次 acquire 建 acquire child，離開時 release。Driver 與 clock 只接 scope.stop，不反向依賴 Scope。TimelineBuilder 只組裝 program，不接停止訊號。

StopSignal 只承載停止請求與不解析的自由文字 reason，release 只解除父子關係，不上交 partial 或 error。Schedule 的 outcome 與資料累積屬於作者工具，不寫回框架。Schedule 的 AcquisitionRunner 保留 acquire-local cancel flag，SNR 等 data-driven early stop 不取消整次 run。Runtime 可向作者交付本次 acquire 的合作停止狀態，但不把它當成前端的完成事實，也不污染下次 acquire。Retry 留在取數 owner，partial 結果由作者定義，不引入另一個 runner。Progress bar 暫留環境注入。

Scope 是 experiment 的共用控制與生命週期契約。作者看 scope.cancelled，底層只借 stop bit。Kind/name 路徑、fork status、selector 歧義與根 ScriptCancelled 的出口規則見 [[0075]]。2026-10-22 使用者要求「signal scope應該搭配有作用域保證的with以及for來規範scope範圍」。Host 與 caller 都不得以 latest child 或 callable 反射補 parent/name。Workflow 根及所有 child 具有相同 conveniences；scope.dir 鏡射 scope 路徑，scope.commit 每個 scope 只能一次，commit 當下保存 record 快照。Reader 走目錄樹並按 commit 序號排序。

Load／analyze 不要求硬體；run 在缺少必要 handles 或 binding 時拒絕，不自行查找或建立資源。單次 buffer 與 cache 歸 run。NotebookExperiment 每次 run 從已綁定的 manager 取得一次 driver mapping，再建立 context；同次 run 不重查 registry。建構入口不讀裝置資訊，load／analyze／save 也不查裝置。同一 manager 更新 registry 後，下次 run 使用新綁定；重新賦值 caller 變數不重綁既有入口。GUI 的 RunService 與 workflow RunSession 仍各自在 operation 邊界建立 context。

影響 acquisition 的實驗選項全部進 typed config，包括 T1 uniform。核心擁有設定驗證及環境無關預設；GUI 擁有編輯表示、標籤、expression 與 md／module library seed。核心分析不讀 live GUI 狀態。Hardware handles 不進 cfg 或 record。

共用包裝層隔離 caller cfg／options 與 core 工作輸入，成功後保留該次來源與選項。Record 的欄位關聯固定，不深拷貝大型 Result，不凍結 Figure artists。Record-owned cfg／options 可被使用者刻意修改，不提供 deep-freeze、讀取時複製或完整不可變歷史保證。

同步核心 options 與 Analysis 是實驗專屬 typed 資料，options 欄位擁有預設值，不另設 Notebook defaults 或動態簽名。實驗類別以 `Options` 引用既有分析選項型別，有獨立 post-analysis 選項時使用 `PostOptions`。Caller 以 `T1Exp.Options(...)` 建立獨立物件，不另匯入 Options。沒有選項的實驗不補空型別，Experiment 基底不增加這項要求，NotebookExperiment 不轉送這些類屬性。Analysis 不含 Figure 或 writeback。Notebook 同步入口回傳 AnalysisRecord，互動工具也保留成功成果的來源、實際 options 與純圖。Notebook 的 analyze(options, *, source=None) 與 save(destination, *, source=None) 預設最近成功 run／load 的 last_run，無來源就拒絕。分析歷史 source 不改 last_run 或 save 的預設來源。Core 與獨立互動工具仍要求明確來源。Notebook 不新增 writeback。

GUI 插件自行定義 typed 成功輸出的欄位。插件決定是否輸出選項、隨機 seed、時間或其他重現資訊。Framework 保存來源、插件輸出與圖，不追查插件的隱式依賴，也不保證完整 options 或可重現性。`params` 保持表單輸入，不在 Done 時替換成終態 options；不新增通用 committed-options owner。GUI 與 remote 讀同一份已提交輸出。

實驗跨入 zcu_tools 時使用公開絕對 import，預設置頂。只有具體重型依賴或初始化限制才延遲，不藉此外移模組或建立設定式 discovery。

### 身分、定位與顯示

#### 綁定身分留在 adapter

**規則。** GUI adapter 從固定 capture 解析角色綁定的元件名。Run metadata 保存該次來源，寫回時由 adapter 核對元件名及順序。量測 cfg 與表單欄位不保存角色身分。Adapter 把位置推導成普通數值填進 cfg。需要顯示元件名的實驗以自己的 analyze `options` 接收標籤，GUI adapter 從同次 capture 提供顯示預設。Core 與分析不解析角色身分。

**例子。** `$qubit1` 綁 QA，QA 在完整 mux readout 的第二個位置，adapter 把 `qubit1_tone=1` 交給 core。Local Z 分析只印出 options 裡的 QA，寫回則由 adapter 用來源身分比對而不用圖例。

**邊界。** Notebook 使用者以自己的上下文綁定元件，手動提供同樣的 tone index、校準值與標籤。不透過 adapter 也能呼叫 core。標籤不是身分，不用來決定寫回目標。

**依據。** 2026-11-20，使用者答「A」，並說「我目前姑且的想法是，可能 analyze 开放一个参数接口，可以让 adapter 填之类的。」「其實gui顯示的options與core要的options是解耦的」。見[身分與中性位置的討論](../discussions/2026-11-22-two-qubit-identity-and-neutral-positions.md)。

#### 只驗實驗所需的能力

**規則。** 實驗不因範例樣品而限制元件的物理子種類。需要驗證物理種類時，先指出序列或分析依賴的能力，再由該實驗的表單推導拒絕不相容的種類。

**例子。** Two-qubit 不要求 fluxonium 或 transmon。兩個位置須是 `$mux.qubits` 中不同的成員，並遵守 [ADR-0072 的有序 mux 綁定](0072-module-boundaries.md#mux-group-是實驗室的-component-定義)。Open／Defaults 檢查 owner 大類、對應順序與已綁 readout；提交時再用同一次 capture 核對實際 readout 的完整 tone 數。需要 CZ 校準時才要求 coupler 綁定，不要求特定 coupler 種類。

**邊界。** Owner 大類、連結存在與順序是 mux 拓撲契約，不是物理子種類限制。相同 tone 數不能證明 readout 的物理順序，仍由操作員核對。沒有固定 group 的 core 不猜測這項關係。

**依據。** 2026-11-21，使用者答「1a，沒有理由限制就不要限制 2a」。同日追認種類中性化的共用規則：「啊，其余大部分都没有问题啊，整批追认。」Two-qubit 的兩次驗證由 designer 於 2026-11-22 代裁，使用者未覆核。見[身分與中性位置](../discussions/2026-11-22-two-qubit-identity-and-neutral-positions.md)及[mux 綁定檢查](../discussions/2026-11-22-two-qubit-mux-binding-checks.md)。

#### 同一位置不跨層翻譯

**規則。** 同一位置從 alias、recipe、cfg、表單到預設顯示只用一個名字。衍生欄位沿用該位置名，不為了在兩套名稱間對照而加轉換。

**例子。** Two-qubit 使用 repo 已有的 `qubit1`、`qubit2`。第一個位置的 tone index 叫 `qubit1_tone`，配對的 CZ slot 叫 `cz_qubit1`，預設圖例叫 Qubit1。

**邊界。** GUI point aliases 與 Notebook 內建角色名稱相同，綁定仍獨立。位置名不是實際元件名，顯示時可用 options 換成元件名。變長的 mux 選取沿用既有 index 契約。舊 point 按[操作指南](../../zcu_lab/v2/two_qubit/guide.md#重綁舊-point)一次重綁，不設舊名 alias。

**依據。** 2026-11-21，使用者追認 designer 代裁：「策零零七一追认代采」。沿 [ADR-0079](0079-form-field-transitions.md#remote-與-recipe) 2026-10-18 的原話：「既然核心是用index，那就用適合核心的方案，不要加一堆難維護的轉換邏輯」。見[身分與中性位置的討論](../discussions/2026-11-22-two-qubit-identity-and-neutral-positions.md)。

### Notebook 中斷與部分結果

NotebookAdapter 在 main thread 執行 core.run 期間替換 SIGINT handler。第一次中斷設定 stop 與 reason，並啟動計時器。Core 在期限內正常返回就取消計時器並照常發布；到期或真的收到第二次中斷時，handler 拋 KeyboardInterrupt，不發布 record。Schedule 與 AcquisitionRunner 不吞 KeyboardInterrupt，作者自行決定如何組成可返回的資料。

Adapter 的 interrupt_timeout 是正有限秒數，預設10秒，非法值在入口拒絕。POSIX 由 timer 經 signal.pthread_kill 向 main thread 送 SIGINT，可打斷其阻塞中的 C 呼叫；沒有 pthread_kill 時使用 _thread.interrupt_main，可能要等阻塞呼叫返回。Finally 先取消並 join timer，再還原 handler，避免遲到的 signal 中斷下一段工作。非 main thread 不替換 handler。

2026-10-31，使用者說明：「notebook只會發出一次ctrl-c，後續按中斷不會重發，而且太久沒結束會彈窗警告，我的想法是adapter在第一次按就設定stop以及一個計時器，超時就raise」，並以「10秒」確認預設期限。這取代 Schedule 攔截 KeyboardInterrupt 再讓前端判定 partial 的舊契約。

前端的 stop_requested 只表示這次 run 期間送過停止請求，不判斷 Result 完整性。RunMetadata.completion 的 complete／stopped／unknown 及開跑暫時值 partial 保持不變；stopped 改表示上述停止事實。GUI、workflow 與 recipe 的歸屬及終局規則見 [[0075]]、[[0062]]、[[0066]]。

### Experiment tag 與 adapter

實驗核心以 `AXES_SPEC.tag` 宣告持久化身分。GUI Adapter 值接收 spec，`experiment_tag` 由 `spec.tag` 取得，不另外宣告實驗類或覆寫 tag。Framework 只讀 adapter 的公開 tag，不取得實驗類別，也不從泛型反推實驗類。

原裁決日期未記錄，使用者於 2026-10-16 map 結案時確認沿用，原話「都採用建議」。原裁決原話為「插隊問題我的想法是tag應該放實驗類讓adapter轉接或複寫才對」及「不太建議暴露實驗類給框架，最好都經由adapter轉接」。

### 圖形產物與呈現

`plots` 是本次操作的繪圖能力，不是全域服務查詢器。

```python
fig, ax = plots.subplots(num="measurement")
line = live_line(ax)


@plots.on_plot_thread
def update(data):
    xs, ys = data
    return line.set(xs, ys)


fig.show()
plots.adopt("diagnostic", external_fig)
```

2026-11-01，全部實驗已遷移至以上具名建圖與 owner 更新入口。

`live_line` 從 lab 支援套件取得，工具與 lib 機制的分工見 [[0072]]。以上入口建立或接收 Figure 時即具名登記。所有具名圖都是本次操作的圖形產物，不做第二次 register_result。實驗直接操作原生 Figure／Axes，factory 不包裝整套 Matplotlib。

不呈現環境仍建立與更新圖並可保存，只是不開視窗或安排互動展示。Liveplot 在執行期間呈現。第三方自行建圖須經 adopt 才保證接入，不能撤回其已發生的顯示副作用。

新入口不依賴 ambient plotting scope，也不要求實驗作者寫相關 with。Factory 必須明確接合 host；不能只把舊 plt.subplots 包一層，仍暗中依賴 routing 或每次切換全域 backend。

Adapter／Notebook 分析工具掌握操作或 session 使用期，共用 plots 處理圖形機制。操作中的 Plots 提供建圖、`on_plot_thread`、adopt 與 host 接合；完成成果使用另一個純具名 Figure 容器，不將 Plots 本身當作 AnalysisRecord.figures。純容器不提供建圖、接管、liveplot 或 host commands。2026-11-01，操作中的圖更新已完成此入口的遷移。

Presentation owner 保留明確的呈現使用期與 release 責任，不能複製 mapping 後丟失所有權。跨存活 owner 接管仍拒絕。停止 producer、最後 refresh、釋放呈現與保留可保存的原生 Figure 分開處理。替換目前引用不關閉使用者持有的舊圖，release 後原生 artist 操作與 savefig 仍可使用。

依 2026-10-22 operation 繪圖邊界裁定，workflow 的每次 `env.run` attempt 使用獨立 operation Plots，與 workflow 層 `env.plots` 共用 presentation host。返回、丟棄與失敗先 finish，下一 attempt release 舊 operation。`workflows.WorkflowPlots` 擁有 workflow 名稱限制與兩組圖的生命週期；通用 `Plots.fork()` 只建立同 host 的新集合。Pane 與 MCP 投影目前 operation 為 `operation/<name>`，workflow 圖名保持不變並在登記時拒絕該前綴。Terminal 保留最新 operation，tab replacement 與 close release 兩組。保存圖片由作者從 `Completed.value.result` 重畫，不增加 result wrapper 或 Completed 圖形欄位。這不改 G3 的名稱與接管檢查。

圖已登記不表示操作成功。Core、host finish 或必要收尾失敗時不提交新成功 record，失敗診斷圖不得混入上一筆成功分析。

GUI application 持有結果與具名圖集合，保存不反向依賴 Qt widget。前端持有 canvas 與選圖狀態。截圖取目前選圖，分析圖保存涵蓋整個集合；不新增多圖排版編輯器。

### 呈現規範

**規則。** [實驗呈現指南](../../zcu_lab/v2/figure-style.md)定義圖、表單文字、guide 與 recipe 說明的寫法。R0 是共同的驗收原則。懂物理但不熟悉實作的操作員，只看表單、guide、liveplot 與 Analyze 圖，就能說出實驗量什麼、每個參數做什麼、每個圖形元素代表什麼，以及如何判讀結果。

圖只放物理量、模型、擬合結果與判讀所需的標註。Run 身分放在可見的 Analyze 摘要，不印在圖上。Adapter 用該次 cfg 的 Reps×Rounds 與實驗已回傳的實際施加值提供摘要行。GUI 與 MCP 的 [`AnalysisPresentation`](../../lib/zcu_tools/gui/app/measure/remote/README.md#analysis-presentation) 顯示同一份行，不各自組裝。摘要行不經 [[0062]]〈一個 run 的 operation〉的 workflow operation wire。圖的標題與圖例由各實驗 core 編寫，不建立共用 formatter。Run 頁的按鈕名為 `Defaults`，表單的物理欄位仍用 `Reset`。

[R7](../../zcu_lab/v2/figure-style.md#r7在固定存檔幾何檢查版面) 只在 12×9 in、150 dpi 的 PNG 檢查版面。Measure 的 Run、Analysis、Post-analysis、Workflow 和 Data 以 12×9 in 排版原生 Figure，再將整張圖等比放入圖磚，其餘空間留白。磚形不改變 Figure 排版。Data 保留選圖縮圖；[`View figure` 閱讀面](../../lib/zcu_tools/gui/app/measure/ui/README.md)以同樣的幾何等比顯示全圖。GUI 只驗圖完整嵌入、等比和無裁切，互動畫布另驗縮放後的資料座標。

一維系列若能用現有 shot 資料估計 SEM，就畫逐點 shot SEM，spectrum 也一樣。IQ 散點及二維圖不畫逐點 error bar。Shot SEM 是每發樣本標準差除以保留 shot 數的平方根，表示該點平均值的預期誤差。Run 與 Analyze 的一維連續掃描，沒有擬合曲線時連接量測點，有擬合曲線時只留量測點及可用的誤差棒。類別軸不連線。顯示用時間寫 `µs`，gain、amplitude 與倍率寫 `a.u.`，機率寫出機率名稱，計數寫 `count`。存檔單位照[保存、發現與範圍](#保存發現與範圍)，不受顯示詞彙影響。

**例子。** T2 Ramsey 的 Analyze 圖按規範畫量測點、密網格 fit 與「T2* = 值 ± 誤差 µs」。cfg 的 Reps×Rounds 和實際施加的 detune 放在摘要。Rabi 和 T1 的 shot SEM 表達各點平均值的預期誤差。Two-tone spectrum 有 shot SEM 時也畫誤差棒，沒有擬合曲線時連接量測點。

**邊界。** 採集摘要不列完成輪數、是否提前停或 shot 數，不擴充 `RunMetadata`，也不改 [[0062]] 的 Schedule facts 規則。各實驗決定是否保存實際施加值。重開檔案缺某個值就不顯示，不能用現行表單補值。Shot SEM 估計器歸 `zcu_tools.analysis`；fit 參數誤差與 RB 的跨 seed 標準差留在原 owner。只存平均的實驗在家族修正時加二階 moment，不移除舊欄位；舊檔仍可 Load，缺資料時不畫 error bar。除此之外，呈現修正沿用現有資料，不改取數。指南的 fit 品質、population 軸、分支標籤，以及表單、guide 與 recipe 文字細則是 designer 代裁，使用者未覆核。Ding 博論只作版式與標註參考，不把相近的圖視為同一實驗。

**依據。** 見[呈現規範討論](../discussions/2026-11-22-presentation-rules.md)與[摘要及誤差棒的責任分工](../discussions/2026-11-22-shared-presentation-facts.md)。原始問答未入庫。

- 2026-11-20，使用者：「實驗不是能用就行，至少要一個對實作無知的使用者角度，從cfg與guide以及liveplot與分析圖，搭配物理專業名詞就能完整理解這個實驗，包含gui/recipe。」使用者以「1A 2A，統一度量衡 3採用」採用單位方案 A 和整份 R0–R9 清單，並說「gain的單位名應該寫a.u.而不是冗長的dimensionless」。
- 2026-11-22，使用者：「1我以為圖的名字已經足夠定位」。重整選項後回答：「1可以，圖應該放純理論實作者關注的重點，而不是實驗作者關注的資訊。2我有考慮給部分實驗的點加上errorbar，通常是比如rabi/t1之類的一維數據，對數據值的大小比較敏感的。spectrum之類的或是二維圖就不用」。
- 2026-11-22，使用者：「關於實驗liveplot與分析圖，我認為可以參考ding博倫的格式，這才是人看的圖。」
- 2026-11-22，使用者對摘要和按鈕答「1a, 存檔看有沒有需求 2a」，分別採用現有值與 `Defaults`。
- 2026-11-22，Data 頁保留縮圖選圖、改以閱讀面驗 R7，是 designer 依夜間授權代裁，使用者未覆核。使用者同日要求：「那注意排版啊，那个 GUI 的版面配置可能不是正规的四乘三或者什么的，然后字体大小之类的要注意缩放，比如说在 Save 页面可能会变得比较窄」。見[在 Measure Data 頁讀原圖](../discussions/2026-11-22-data-figure-reading.md)。
- 2026-11-22，使用者在 shot SEM 與輪間散布之間選「A」，補充「像我其实更想要设计的，其实应该是平均标准差，也就是这个点的预期误差。」數值估計器放 analysis、標題與圖例留在 core 是 designer 代裁，使用者未覆核。
- 2026-11-22，使用者：「如果拟合曲线本身就是连线的，那数据点就不用。但如果没有拟合曲线，那就要连线。」使用者選「全面套用」，並要求 spectrum 也畫可用的 shot SEM。見[量測點何時連線與 spectrum 的 error bar](../discussions/2026-11-22-data-point-lines.md)。

### 三種分析生命週期

- T1 提供同步 analyze，消費 explicit RunRecord 與 typed options。Notebook 成功後發布 AnalysisRecord；GUI 發布插件定義的 typed 分析輸出、來源與圖。
- Singleshot GE 核心另提供 post_analyze。Post 消費 adopted primary 的來源與 calibration，不重新 primary fit，不讀未重新分析的 initial_state 表單。Primary／post 各自持有图與 GUI writeback proposal。成功 primary 替換清空目前 post，primary 失敗保留前次成功組。清空引用不關閉使用者持有的舊圖。不要求其他實驗提供空 post 方法。
- OneTone FluxDep 核心只負責 acquisition 與資料保存／載入，不實作 analyze。GUI adapter／INTERACTIVE plugin 自行分析，Notebook 使用獨立分析工具。重用現有 TwoLinePicker 與 Qt-free 選線規則，不把 Notebook 工具接回 core.analyze 或共用 Adapter 分析。啟動返回不代表成功；Done 驗證及收尾後發布該次成果，Cancel／failure 不取代上一筆成功。Notebook 保留實際 options，GUI 插件仍自行決定成功輸出的欄位。

Notebook run／load 正常返回後更新目前 RunRecord，清空目前分析與圖引用；失敗保留舊紀錄。同步分析成功才成組替換成果，分析舊 RunRecord 不替換目前 run。互動工具綁定自己的來源，晚到完成仍按 G5 發布自身成果，不取最新 run，也不新增 generation gate。清空引用不銷毀使用者持有的舊圖或結果，不新增完整歷史管理。

### 保存、發現與範圍

實驗透過 AXES_SPEC／GroupedAxesSpec 擁有 RunRecord 到保存資料的映射，datafile 擁有檔案格式。保留 inner-first axes、Result-native shape、SI disk units、roles、labels、dtype、comment／tag 與 complex calibration。Caller 指定 exact path，save 不依目前 run。本機單檔 save 是明示 Labber 匯出，load 只讀 native data.h5，兩者不是對稱格式。只有 Labber 的舊資料須先用既有 migrations 離線轉換，不提供執行期 fallback，也不代跑真實資料遷移。Run 保存與重存的 native replace、Labber 新檔及失敗語意由 [[0063]] 規定。

2026-10-24 使用者要求「實驗基底只剩宣告」。`Experiment[Result, Cfg]` 只保留 `AXES_SPEC` 與 run／analyze／post_analyze 簽名；`from_stored` 契約改為外部函式 `restore_run(stored, spec=...)`，不做 I/O、不查 registry 或動態匯入實驗。Notebook 的 load 只收 Path，先以 datafile 讀成 StoredRun，再經同一轉換回 typed RunRecord。GUI LoadService 讀檔後交 adapter.restore；GUI 只持有 adapter 的 opaque result。RestoredRun 以 record 型別為泛型，同時保留核心 typed record 與 GUI opaque handle 的型別資訊，cfg_issue 為 None 或非空的不可用原因。

Restore 先驗證 native 格式、experiment tag、必要 variables、axes、unit、dtype、shape 與 builder result，再處理 typed cfg。缺 /cfg、非法 JSON 或非法 CfgSnapshot envelope 仍是格式錯誤。合法 envelope 的 cfg_type、schema major 或 concrete model values 不相容時，保留有效資料、令 RunRecord.cfg=None，並發出 CfgUnavailableWarning；cfg_issue 攜帶同一原因。同 major 的 minor 與未知 cfg keys 沿既有 known projection，StoredRun 保留 raw cfg JSON 與歷史 metadata，不重拍來源或裝置。

GroupedLoadData.cfg_snapshot 允許 None，資料 builder 讀 variables，不拿目前 cfg 補值。正常 run 仍有有效 cfg。需要 cfg 的分析或 data 保存入口明確拒絕 cfg=None，不需要 cfg 的分析照常，不由 NotebookAdapter 一律阻擋。GUI Load 不回填表單，也不從 Form、library、default 或 live device 補回歷史 cfg。原始 cfg 保存在 native 資料檔，透過公開 datafile.load_run_data 離線讀取為 StoredRun，GUI 不提供檢視入口。使用者裁定見 [[0063]]。

使用者於 2026-10-14 對框架讀檔後交 StoredRun 給 adapter 回答：「看起來沒問題」；對資料優先、cfg 可缺回答：「我認為可以」。2026-10-15 以「1可以，2採用忽略不支援欄位的方式，3同意，4一起做吧」確認 Load 不回填與 Labber-only 先離線遷移。

一般實驗將明確 RunRecord 與 spec 交給外部存讀函式，不要求額外 experiment_codec，也不聲稱支援任意 Python 物件。Workflow streaming／grouped artifacts 保留自己的契約，不改成 one-shot record 格式。外部 persistence 函式與 NotebookExperiment 的 save／load 只接本機路徑，不公開 server_ip／port；獨立 datafile／workflow transport 保留自己的契約。

Notebook save 吸收 unique path 的便利層並回傳實際 path，再呼叫 exact saver。Path helper 不建立檔案或鎖，不宣稱原子 reservation。AnalysisRecord 保留記憶體內來源與 options，分析圖另外保存；不新增完整 analysis session 磁碟格式，也不撤銷實驗既有保存義務。

保留 explicit catalog、v2／v2_gui 分離，以及單檔／package 並存。框架不要求固定內部檔名，不靠掃描或 import 副作用發現實驗。

範圍是所有實驗及其 Notebook／GUI caller，不以 GUI catalog 為上限。一般 T1、singleshot/ge、onetone/flux_dep 分別代表同步、post 與互動分析三種生命週期。

所有實驗與 caller 都使用這套核心與前端契約，不保留 pyplot 相容層或舊協議 fallback。Executor、動畫框架、acquisition、排程與硬體鎖機制不在本篇範圍。

#### Native 可選變數與 Labber 投影

**規則。** `GroupedAxesSpec` 至少宣告一個必填變數。`VariableSpec(required=False)` 只在整個 Result z 欄位為 `None` 時略寫；讀檔缺可選變數時不交給 builder，也不補資料。缺必填、缺 Result 欄位或已有變數不合宣告仍拒絕。實驗層把現有變數映射到 SI，先用 [[0080]] 的單一 Variable 驗證器檢查，再按 spec 投影 Labber 選集。單檔匯出與 artifacts 共用這個準備過程。

**例子。** 傳統 Amp Rabi 保留原 `data` 的 Gain 軸和平均 complex IQ，有六矩才另存 `iq_moments`。Labber 明選原 `data`，普通 reader 仍讀單一 log。只有平均 IQ 的舊檔與校準產物用同一 spec 重開，不畫逐點誤差棒。

**邊界。** `required=False` 只管 native 變數可否缺席，不決定哪些變數匯出。未明定 Labber 選集時匯出所有已存在的變數；不因選集自動切換 grid layout。不放寬既有 tag、cfg schema、軸或 dtype 驗證，也不新增歷史 reader。Datafile 與 Labber writer 不判讀實驗語意。

**依據。** 2026-11-22，designer 依「这个点的预期误差」所定的 shot SEM 方向代裁單一 spec 與獨立選集，使用者未覆核這兩項保存細節。比較與驗證見 [Amp Rabi 的 shot moments](../discussions/2026-11-22-amp-rabi-shot-moments.md)。

### Cfg 欄位改名與 schema 版本

為了讓同一角色用同一名稱，可以改既有實驗的持久 cfg 欄位。改名時提升 `cfg_schema_version` 的 major，不遷移舊 cfg，也不提供 alias。只要舊 run 的 tag 與結果資料仍相容，Load 就保留 Result 與 metadata，令 `RunRecord.cfg=None`。需要 cfg 的分析不會改用新預設值繼續跑。

例如 `two_qubit/local_z` 將掃描欄位 `gate_counts` 改為 `gate_count` 並提升 cfg schema major，Result 欄位與存檔軸維持原樣。若只改 guide、表單標籤或圖上的文字就能達到目的，保留 cfg 欄位與版本。若必須保留舊 run 依原 cfg 重新分析的能力，先不改名；相容方案要另外裁定。

使用者在 2026-11-19 對以單數改名為先例、不提供 alias 的提案回覆「全采用建议」。選項與首批範圍見[實驗命名與顯示討論](../discussions/2026-11-20-experiment-naming-and-display.md)。

#### 只有模擬測試資料時的舊結果格式

**規則。** 確認只有模擬器測試資料的家族更改 native 結果格式時，不為不相容的舊 layout 新增 reader。保留原檔，不改寫或遷移。相容 layout 仍沿用既有 cfg fallback。

**例子。** Two-qubit joint readout 的新檔使用 `terminal_qubit1`、`terminal_qubit2`。舊軸與現行宣告不符時 Load 拒絕，要讀原檔使用對應的舊版。

**邊界。** 有實機資料的家族沿用本 ADR 的載入相容規則。不確定某家族有沒有實機資料，先問使用者。這項例外不授權刪除舊檔，也不要求其他家族改格式。

**依據。** 2026-11-22，使用者答「我沒有舊資料，可以直接換到新格式」，並補充「像是chain/herald/two_qubit等實驗，我都沒在真實硬體上跑過，只有測試用的模擬器數據，後者不重要，所以這些都不用在意舊格式數據的載入問題」。見[two-qubit native 軸名討論](../discussions/2026-11-22-two-qubit-native-axis-cutover.md)。

## 顯式 cfg 與裝置環境

DeviceManager 的 registry、lock 與 close claims 屬於 instance，不提供預設 singleton。Notebook 與 GUI composition 持有或接收 manager；共享 driver 的 caller 共享同一 manager，不各自取得關閉權。既有 alias、close failure 與 driver lock 契約保留，不以 destructor 自動關閉，不把 disconnect 當作歸零或 RF off。

Notebook 使用 `CfgEnv(ml, device_manager)` 借用固定 library snapshot 與 caller 的 DeviceManager。顯式呼叫 `make_cfg(raw_cfg, CfgModel, env, overrides=...)` 以該 library 的 validation context 組裝 typed cfg，不讀取或設定 devices。參數值由 caller 在建立 raw cfg dict 時直接讀 point 或 roles，不交給 CfgEnv 解析 expression 或延遲求值。組裝失敗直接報錯，不 refresh 來源或退回舊 cfg。

CfgEnv 與 RunContext 分工不同。前者可跨次組裝重用；後者只有單次執行能力，不讓核心取得 point、library 或 registry 管理權。底層 `assemble_experiment_cfg` 接顯式 validation context，不讀硬體。Run 不重新組裝 cfg、不重連、不補裝置。需要新的參數或 library 資訊時，caller 明確讀取來源、重建 CfgEnv，再次呼叫 make_cfg。Driver mapping 不是資源 lease，也不凍結 driver 狀態。Notebook 初始化 cell 重跑就重建，不探查 namespace 或自動重用，舊連線仍由使用者明確關閉。GUI 的固定來源、CfgRef／revision 與 acceptance 不因 Notebook 入口而重新查詢硬體。

模擬 builder 回傳具名 `SimulationEnvironment`，包含已綁定 soc、匹配 soccfg、同一 DeviceManager 與公開描述。一般可信 Python loader 仍支援零參數 `build_environment()`；FTF startup 與 Notebook 直接呼叫公開 `experiment.ftf.ftf(...)`。

GUI 不載入 environment，由選定的 startup 腳本經一般 facets 安裝 SoC 並逐一 register drivers。一般 soc_changed 不建立 device，不保留動態切換入口。成功回傳後由 caller 安排清理；只有 register 成功的 driver 才交給 GUI 關閉，失敗與未交付資源由腳本清理。資源交接與 `on_close` 時序見 [[0064]]。部分清理失敗保留資源與原錯誤，不假裝 ready。

FTF Notebook 範例在組裝時明確傳 `warmup=True`，worker 數保留兩顆 host CPUs，至少使用一個。FTF GUI startup 明確傳共用策略選出的 CPU 數減一個 worker，至少一個，並傳 `warmup=True`。Factory 不等 CPU 數值準備與 GPU 預編譯就返回，兩者在背景執行。`experiment.ftf` 擁有代表性數值工作與 process-wide single-flight；`SimulationEnvironment` 暴露進度、future 與 ready 等待。直接呼叫 `ftf()` 預設 `workers=1, warmup=False`。背景工作不接觸正式 devices、RNG stream 或 resident Engine。Program binding 只知道通用 ready callback。Acquire 在 admission 後、消耗 sequence 前等準備完成。失敗由 environment 回報並退回 on-demand 編譯。外部計時 owner 先等 ready 再開始計時；GUI 狀態顯示由前端接線，不由數值層更新。依賴沿用 C8、C9、C10、C12，沒有新增反向依賴。

2026-11-18，使用者說：「这个 ADR 0069不预编译的裁决，我怎么没印象呢？我好像没有。…我预期 GUI 启动时会预编译啊。」對兩案「A：GUI 傳 `warmup=True`，CPU 準備與 GPU 預編譯都在背景做；B：只預編譯 GPU」答「A」。原先「GUI 不預編譯」由 2026-10-20 的 `4302e30d8` 改入，沒有使用者裁決紀錄。使用者希望「使用者体感到的执行模拟速度才不会不一致」。代價是新機器或 Numba／OpenCL 快取失效後，factory 一返回就 acquire，第一次就得等全部準備。Intel Arc、d=24、7 workers 的冷快取量測中，立即 acquire 花 76.9 s，不預編譯花 22.3 s，等準備完成再 acquire 花 3.3 s。量測條件見[討論紀錄](../discussions/2026-11-18-ftf-gpu-dop853-backend.md#留下的問題)。

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

一般 subplots 立即建立並具名登記，GUI 於函數完成後呈現。Liveplot 立即呈現，worker 呼叫以 `@plots.on_plot_thread` 包裝的更新函式，本體在 host owner 執行緒修改 artists。Worker 不任意直接修改 active live Figure／Axes。實驗不需寫前端分支，也不能以 draw lock 取代這項執行緒契約。2026-11-01，全部實驗的更新已遷移至此 owner 執行路徑。

### G2：圖形生命週期

停止 producer、最後有效 refresh、保留 Figure 與釋放呈現資源分開處理。取消請求不立即銷毀圖。作者組成的結果正常返回後仍可呈現最後資料，前端不判斷它是否為 partial。新操作不關閉使用者持有的舊圖。

Notebook backend adapter 處理顯示與 close，避免 cell 結束重複顯示，並提供明確釋放呈現資源的方式。close 對 widget 與保存的影響需依 backend 核對，不使用全域 close("all") 清理別人的圖。外部 Figure 的 adopt 不能撤回已發生的顯示副作用。

### G3：名稱與接管

操作內一名一圖。同名同 Figure 重複 adopt 等冪，同名異圖及同圖異名拒絕。已由其他操作持有的 Figure 拒絕接管，不偷移 canvas。第一版不提供 transfer 或 alias。舊圖由原集合保存，跨操作重畫使用資料。

### G4：多圖保存

逐圖記錄是否曾成功保存，以 stage 與圖名命名，單圖也遵守相同規則。保存前固定本次集合、產物 identity 與目的地，在 I/O 前檢查安全路徑與名稱碰撞。Identity 用來將完成紀錄對應回原產物，不是內容 dirty 版本。

同一份圖一旦保存成功，修改 artist、視圖或目的地不清除成功紀錄。不新增 dirty 通知或自動內容追蹤。新產物從未保存開始。
使用者 2026-11-18 過目此條：「我大概看了一遍，基本上是没问题」，並說明這類規則「并不算是裁决，而算是建议」；它是 agent 的選擇，引用時視為僅供參考的先例。

部分失敗保留成功項，不宣稱整組已保存，也不承諾跨檔原子保存。Save All 包含已保存的 artifact，重寫選定的 native、PNG 與 JSON；Labber 每次產生新檔，失敗保留上次成功連結。每次呼叫回報本次結果，不自動 retry，保存權威與失敗邊界見 [[0063]]。

### G5：晚到結果

Notebook 不新增 generation／request identity 的發布限制。較早啟動的互動或分析較晚完成時，允許覆蓋目前分析，操作順序由使用者負責。該次數值、來源 Result、實際 options 與圖集合仍須成組發布，不能混用不同操作的資料。
使用者 2026-11-18 過目此條：「我大概看了一遍，基本上是没问题」，並說明這類規則「并不算是裁决，而算是建议」；它是 agent 的選擇，引用時視為僅供參考的先例。

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
- [Session](../../lib/zcu_tools/gui/session/README.md) 說明 DeviceManager 與一般 SoC／device 操作。模擬資源由可信 startup 腳本經一般 facets 交付，不由 session coordinator 載入。一般 SoC 連線不建立 FakeDevice。選定腳本、`on_close` 與 driver 關閉責任見 [[0064]]。

獨立 liveplot plotter／backend 已退休。即時更新使用 [Plots 的 `on_plot_thread`](../../lib/zcu_tools/plotting/README.md)，格式由 lab 工具提供。2026-11-01，即時更新遷移已完成。

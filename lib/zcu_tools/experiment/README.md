# `zcu_tools.experiment` — 實驗層

**Last updated:** 2026-10-30（固定值存檔軸）

**讀者：** framework-dev。實驗介面、Run 資料、持久化映射與裝置操作邊界。

`experiment/` 定義實驗介面、Result 與保存資料之間的映射，以及實驗 cfg 的支援機制。這個位置不表示其現有程式碼均與 QICK 無關。`cfg_assembler.py` 提供 Run mapping 的共用值型別，`utils/sweep.py` 使用 `program.v2` 的 `SweepCfg`。程式指令的建構與 acquire 由 [program/v2](../program/v2/README.md) 及 [experiment/v2/runtime](v2/runtime/README.md) 接合。

## Shared async script host

`script_host.ScriptHost(scope)` 接 caller 以 catalog 名建立的 workflow／recipe 根，建構時不啟動 thread 或 loop。`run(coroutine)` 在呼叫它的 worker 執行一次作者 coroutine，完成 finally 與 executor drain 後返回或 raise，不能重用。`await call(fn, scope=None, name=None, on_completed=None)` 在 executor 執行 `fn(CallContext)`，回 CallResult。未填 scope 時 operation 掛根，不預建 iteration。Context.scope 是該 operation child，epoch 記起跑值。子層取消由 owner 合作回 partial。Root 與 live_paths 是公開讀取介面。正常 executor completion 回到 worker loop 時先呼一次 on_completed，再檢查根取消；fn 或 hook 的真錯誤原樣傳播。作者 Task 不會被 cancel，finally 留在同一 worker。

`await ask(question)` 以唯一 request id 等待具名 Answer，`answer(request_id, answer)` 驗證當前問題；錯誤、重複或遲到答案不消耗新問題。根取消會釋放 Ask 並 raise 普通 ScriptCancelled，作者 ask 不回 Optional。根取消前已開始的 call 先 drain，發布 completion 後再 raise。ScriptCancelled 不存進 Host.error。

`control(fn)` 把數微秒級的 memory-only callback 排入與 cancel／pause／resume／answer 相同的 loop 序列，呼叫端等待回傳值或原例外。Fn 不 await、不阻塞、不 I/O，也不重入作者程式。Worker 上 inline 執行，foreign thread 經 call_soon_threadsafe 與 acknowledgement；run 前或 loop 關閉後 raise RuntimeError。`cancel(scope=None, expected=None)` 的 child 選擇器在 loop 序列匹配。根取消則在 Host lock 內設 token 並記 typed ack，第一次呼叫也不等 loop notification。根 ack 的 active_path 固定為根路徑，重複請求回同一 ack，不再排 notification。Loop 稍後釋放 Ask 與 pause gate。Ack 不表示作者 finally 或 executor 已 drain。Scope 也可填 label#N。省略 scope 選根，明填時使用 Scope resolver 與 `CancelResult(cancelled, reason, active_path, candidates)`，acknowledgement 不表示 operation 已 drain。Acquire 只停一次取得，Schedule 停該段後續取得，iteration 停本輪，下一個顯式 iteration 使用新 token，run 是當前 operation 的別名。Recipe runtime 另用 control 原子檢查當前 tab.run token，analysis／Ask／閒置時 run cancel 是 no-op。`close()` 合作 drain，off-worker 等到 coroutine／finally／executor 完成，worker 自己呼叫時只請求停止，不自等。

Pause 只停止目前 operation、增加 epoch 並關 gate，不停止 iteration。Resume 只開 gate。CallResult 固定起跑 epoch，並原子擷取 fn 完成時的 partial 與 stopped 事實。Partial 包含本 operation 被停止及已完成 child 的取消。Stopped 只表示 operation 自己的 token 已 set。Workflow env 只丟棄 Pause 造成的 stale stopped attempt，再重跑相同 request；單純 scope cancel 的 partial 直接交付。Host 不重跑外部效果。Scope.commit 同步寫 snapshot，不等 pause gate，也不更換 iteration。根取消打開 async gate，使作者 finally 可同步保存已有成果。

`Scope(kind, name=None, stop=None)` 建立 workflow、recipe 或 run 根。`child(label)` 建作者 child，`branch(kind, name=...)` 建 iteration、operation、schedule 或 acquire。兩者都以 with 或 release 收尾。Name 若非 None 必須非空，且不得含 #、/、:，作者 label 也不得使用保留 kind。路徑使用 `workflow:<name>/iteration#1/operation:T1#1`，同 parent 的各 kind 從 1 編號。`active_path` 在第一個 fork 停下，`live_paths` 列出所有存活葉的完整路徑。

WorkflowEnv 繼承同一個 Scope，host.root is env。Child 透過注入的 ScopeRuntime 執行 run／call／setup_device／wait_until，沒有目前 scope 或 experiment 到 workflows 的反向 import。共用 outcomes、facts、identity 與 RunFunction 由 operations.py 擁有。Scope 的 identity 與 get_device_cfg 由根提供並給 child 繼承，缺能力在使用時 raise MissingCapability。需要 workflow 圖的 helper 明收 plots。

Scope.dir 第一次使用才建目錄，按 kind／作者 label 與六位 sequence 鏡射，name 不進目錄名。每個 scope 同步 commit 一次，record.json 保存 scope_path、name 與全 run commit_seq。後續 record mutation 不改快照。Root commit 寫 data root。作者用 env.iterations() 決定 iteration 壽命，generator 的 next／close 釋放前一趟／最後一趟。

`scope.cancel(selector)` 搜尋整棵存活子樹。Selector 可用 kind、kind#N、operation:name 或完整葉路徑。多筆命中回 reason="ambiguous" 與 candidates，不任選最新 child。`scope.cancelled` 表示自身停止，`scope.stop` 供低層 bit API 使用。StopSignal 負責出生繼承、向下停止、第一個 error 與 release 時的 partial 上交。Operation child 上交到此截止，後續兄弟不繼承已交付的失敗。GUI run 根接收 Schedule 完成事實，但不因此設自己的 stop bit。

`Schedule(cfg, buffer, scope=..., name=None)` 在 context 入口建立 schedule child，每次 acquire 建立 acquire child，退出時 release。`RunContext.scope` 將當次 run 或 operation 的顯式 scope 交給 core；底層 driver、clock 與 program builder 仍只接 stop bit。

Recipe 與 workflow 都使用此 host，作者以原生 async／await 等待，不使用 generator protocol 或 ScriptDriver。Workflow 在作者 finally 與 executor drain 後嘗試關閉所有 progress bars，再發布終態。Cleanup 失敗保留原始執行錯誤並附 note；沒有執行錯誤時，cleanup 失敗使 run failed。

## Explicit simulation environments

`simulation.py` provides `SimulationEnvironment`, `white_noise` and
`load_environment`. A trusted Python file defines zero-argument `build_environment`.
It returns a bound SoC, its matching compile config, the exact populated
`DeviceManager` and a public description. GUI and Notebook share this bundle;
there is no controller argument, registry discovery or runtime mode switch.
`create_device(type_name, address)` retrieves explicitly addressed startup
FakeDevices from their process-wide class container, preserving model bindings
and settings after disconnect/name changes. The environment keeps only modeled
address metadata, not another instance registry. Unknown types/addresses fail
explicitly. Anonymous startup devices have no reconnect identity.
Successful builder return transfers cleanup to the host. Read the
[acquisition environment contract](../program/v2/simulation/README.md) before
writing a builder or numeric source. Public Run metadata contains only the
optional opaque `simulation_id`; private replay inputs remain on the binding.
WhiteNoise numeric code lives in `simulate.white_noise`. This composition root
wraps its statistics and complete raw source in program's `IntegratedRound`.
Decimated acquisition keeps per-execution `RoundSignals`. ADC and white_noise
structurally satisfy program's `RawShots` without importing program. Processing
or a getter requests shots from a separate retained identity. Direct means keep
their existing moment calculation and RNG stream.

[`ftf`](ftf/README.md) composes a named YAML sample, fixed V/A memory devices,
fixed-coordinate joint circuits, explicit evolution/population conversion and the
single-receiver ADC. It uses the same bundle and acquisition contracts, without
adding GUI physics or exposing state labels to ordinary Run metadata.

[`single_fluxonium`](single_fluxonium/README.md) plans complete microwave/fast-flux
timelines and assembles a one-source mock environment. Its source composes DC-basis
population conversion, one-cavity ADC statistics, deferred raw shots and decimated
traces. Hosts can call the optional `SimulationEnvironment.close_backend` after
acquisition drains; device cleanup remains separate.

[`six_fluxonium`](six_fluxonium/README.md) composes six independent single-cavity
sources with six local DC drivers and one shared coil. Each circuit uses the sum
of its local and coil reduced flux. Mux gen0/ADC0..5 and pulse gen13/ADC6 use
per-point nearest-bare-cavity routing. Sparse integrated histograms preserve
marginal counts without building a joint model; raw shots retain physical ADC
sources independently of backend lifetime.

`readout_plan.py` owns shared QICK drive/readout waveform conversion without
settings or circuit physics. `drive_track` detaches microwave slots with the
same support, carrier-phase and envelope semantics across the three environments.
`classify_readout` separates terminal blocks, triggered nonterminal projective
reads and untriggered cavity drives at each point. Projective reads group the
coherent tones and bound receiver triggers at one projection time. A block owns
triggers from its start until the next block start, including ringdown after
pulse end. Nonterminal sampling must finish strictly before the next block.
`ReadoutPartition.terminal_pulses` retains terminal tones even when no ADC observes
them. Projective block counts and each ADC's per-event grid counts must match
across points; classification rejects differences with the point and counts.
`read_event_contents` detaches and deduplicates each trigger's own pulse/grid
contents. `readout_contents` retains the terminal-only planning contract.
`logical_event_contents` retains chronological physical events and each grid's
original channel-local trigger column. `map_readout_events` localizes phases,
expands point/event contents and supplies the per-content output permutation.
Single and FTF share this mapping; six uses single's preparation interface.
The shared planner converts common-origin carrier phases to pulse-local phases. Environment planners supply
`ReadoutDrive` calibration and retain responsibility for receiver binding,
physical models, evolution and sampling.

Before evolution or sampling, each environment also requires equal projective
counts and per-ADC/event grid counts across templates in one numeric preparation.
Conflicts name the templates and layouts. Six applies this rule per routed
physical source, not across independent cavities. Single's public `preflight`
validates a cavity's complete ADC observation set; `prepare_readouts` builds one
shared projection history. Six preflights all routed sources before preparing any.
All ADCs retain the same logical-event axis, using empty grids for events they
do not observe. No ADC must observe terminal, but each must observe some event.

Environment runtimes project the density at each triggered nonterminal block's
start and continue each rank-1 DC energy outcome through evolution. ADC samples
one complete outcome history per shot and shares it between herald and terminal
reads. Integrated counts flatten the outcome and terminal-state axes. ADC CTMC
jumps during herald affect IQ only, not later density evolution. This excludes
MIST, IQ-clustered projection and data-dependent feedback for active reset.
Pruning thresholds and diagnostics belong to the
[single](single_fluxonium/README.md#complete-timelines),
[six](six_fluxonium/README.md#routing-and-acquisition) and
[FTF](ftf/README.md#cavity-drives) settings/runtime contracts.
Templates without projective reads retain their existing raw, integrated and
decimated outputs.

## Offline experiment probe

`probe.probe(experiment, cfg, environment, output_dir, *, options=None)` 用 caller
建立的 `SimulationEnvironment` 執行公開 run、native save、restore 與 analyze。
每次建立新的 Scope 與 headless Plots，以 Agg canvas 保存 live／analysis PNG，不建立 Qt application。
Caller 仍負責環境的 backend 與 devices 清理，probe 不更改 seed 或接線。
`ProbeReport` 保留 Result 欄位的 shape／dtype、NaN 視為相等的逐值讀回比較、分析摘要、PNG 路徑與階段秒數。
階段失敗時回傳原始 exception 與 failed_stage，並記錄 traceback；成功前已產生的檔案保留。

`probe.main(argv)` 載入 caller 指定的可信任 `module:class`，依 AXES_SPEC 的 cfg 型別驗證 YAML／JSON，
建立並清理 white_noise 或 single_fluxonium 環境。它印出 JSON 並回傳同型報告。
`python -m zcu_tools.experiment.probe --help` 列出參數；module shell 以 exit status 區分成功與階段失敗。

## 實驗介面與資料

`artifacts.AnalysisDocument` 擁有 analysis JSON 信封。`to_doc()` 回傳可放進通用容器 `docs/analysis` 的有限 JSON object，容器不解釋其鍵。既有 artifact encoder 仍產 `analysis/<pane>.json`，不改 SaveLayout 或 ledger。選用 AnalysisDocument 時不載入 artifact saver。

- `records.py` 的 `RunRecord` 將 nullable cfg、純 Result 與 optional `metadata` 配對。建立時 deepcopy cfg 與 metadata，Result 保留 caller 的陣列引用。宿主完成量測後以 `replace(record, metadata=...)` 裝入歷史事實。`metadata.py` 擁有 frozen Pydantic `RunMetadata`、按欄位讀取的 `RunMetadataContent`、frozen dataclass `RunSnapshot`／`SoftwareProvenance` 與 `CfgSnapshot`。新量測使用 RunMetadata，載入的 record 使用 RunMetadataContent。兩者的 `project()` 都完整驗證歷史文件，Save 使用這個入口。Content 的 property 只驗請求欄位，讀取完整 snapshot 或 provenance 時驗該區塊。欄位解析失敗時拋出公開的 `MetadataFieldError`，繼承 `ValueError`，保留 `field`、`document_path` 與原始 cause。預覽可只隔離這類錯誤，不隱藏其他規劃失敗；整份 `project()` 的錯誤契約不變。輸出 Labber path 只在 SaveReport 與 ledger 中。宿主以 `model_copy` 更新完整 metadata，公開保存入口會重新驗證。
- `RunSnapshot` 記錄 entry_name、point 與 roles，不含 entry UUID。舊 metadata 的 entry_id 被忽略，不從它推導名稱。
- `metadata.ParameterSnapshot` 保存 resource-owned 參數內容、library Specs、authored aliases 與 fixed names。它不含持久 target 或 storage。還原時不預驗未使用的區塊，讀取 owner 在使用值、Spec 或 alias 時報錯。GUI Run 在提交 worker 前固定這份 snapshot，Save 使用 RunMetadata 中的內容，不重新捕捉 live point。
- `base.py` 的 `Experiment[Result, Cfg]` 只宣告 `AXES_SPEC` 與 `run`、`analyze`、`post_analyze` 簽名。未實作的操作 raise `NotImplementedError`，無持久化宣告時 `AXES_SPEC=None`。核心不保存上次結果，也不提供存讀方法。NotebookExperiment 管理 records，NotebookAdapter 只綁定可重用環境。`native_persistence` 的 `save_run(record, path, spec=...)`、`load_run(path, spec=...)`、`load_run_with_metadata(path, spec=...)` 與 `restore_run(container, spec=...)` 接明確宣告。`save_labber(record, path, spec=..., comment=..., tag=...)` 另作單檔 Labber 匯出，cfg 缺失時拒絕。路徑都是本機 `Path`，caller 選路徑與覆寫政策，不提供 server_ip／port 或格式猜測。只有 Labber 的舊檔須先用 `migrations/` 離線轉換。
- `native_persistence.py` 的 `prepare_run_data` 將明確 RunRecord/spec 映射為 `PreparedRunData`，持有 SI `Container`、`CfgSnapshot`、原始 `LabberMetadata` 與 Labber layout。準備不做 I/O，也不要求 metadata 或 Labber common grid。`to_container(record.metadata)` 驗證 metadata，再加入 `docs/cfg={values,cfg_type,schema_version}` 與 `docs/metadata`。Metadata 以 `model_dump(mode="json")` 序列化。Load 保存 detached metadata content，不預驗未用欄位。非法文件的錯誤指出來源檔。`save_run` 只收 record，不另收 metadata；缺 metadata 即拒絕。`load_run_with_metadata` 回 record，`load_run` 保留 record／snapshot tuple。
- `native_persistence.read_run_metadata(path)` 用 datafile 的 `read_doc` 只讀歷史 metadata，不載 vars 或 cfg，也不回退到全檔 load。需要完整歷史證據的 caller 可取得 typed `RunMetadata`，不受壞 vars 阻礙。Saved-runs 清單另從 datafile.read_doc 投影顯示欄位，不要求 provenance 或完整 snapshot。缺文件 raise 指名來源的 `KeyError`，非法文件或 metadata raise 指名來源的 `ValueError`。已載 Container 的完整歷史讀取使用 `metadata_from_container(container)`，不重讀檔案。Fluxdep 用 `device_snapshot_from_container(container)` 只投影 device settings，不驗 cfg、provenance 或 entry snapshot；缺值回 None，已記錄空 mapping 保留為空。`metadata_from_container` 與 `read_run_metadata` 是明確完整投影，仍使用完整 RunMetadata 驗證。`metadata_content_from_container` 只固定文件，property 在使用欄位時驗證，錯誤指名 `/docs/metadata/<field>`。Container 缺 metadata 時 raise 指名 `/docs/metadata` 的 `ValueError`，caller 在轉譯錯誤時附來源路徑。
- 公開 `restore_run(container, spec=...)` 不做 I/O。它核對 metadata 的 experiment tag 與宣告的 variables、axes、units、dtype、shape，再重建 Result 與 typed cfg。未使用的 completion、timestamps、entry snapshot 與 provenance 不擋還原，實際讀取或完整投影才報錯。`VariableSpec.variable_from_result`／`loaded_from_variable` 負責 Result 與 `datafile.Variable` 的映射。Container axes 按 NumPy 維度順序，spec 與 loaded axes 為 inner-first，數值陣列不 transpose。GUI、Notebook 與 native load 共用 `restore_run`。合法 cfg envelope 的 type、schema major 或 concrete validation 不相容時，record.cfg=None，`RestoredRun.cfg_issue` 與 `CfgUnavailableWarning` 說明同一原因。缺 cfg 或非法 envelope 仍拒絕。Record 保留 metadata；unknown vars/docs/groups 與 raw cfg 由原 Container 保真，不塞進 typed record。
- `artifacts` 的 `save_artifacts(artifacts, *, record, ...)` 編排一次保存。它只取 `record.metadata`，缺失或非法時在寫檔前拒絕。`SaveLayout` 原樣使用 caller 的 stem，native 為 `runs/<stem>/data.h5`，Run 圖片與窗格 JSON 共用該目錄。Result 的 point／全域圖片副本為 `<stem>_<圖名>.png`。Writer 在首次 Save 建立 Database 目錄，entry create／open 不建立它。Labber 日期樹內初始名為 `<stem>.hdf5`，只有碰撞時才取 numeric suffix。Comment 在 cfg 旁保存歷史 snapshot.point 與 snapshot.roles。同 stem 重存覆寫 native、PNG 與窗格 JSON，不覆寫 Labber。
- `AxesSpec.save_stem` 與 `GroupedAxesSpec.save_stem` 接收 `SaveContext`，預設用 dotted frontend name、hyphen-joined roles 與完整 run_id 組名。GUI adapter 可在自己的 spec 覆寫；Notebook 沿 core spec 使用同一 hook。框架傳完整 name／run_id、point、roles、可用 cfg values 與 saved_at，不在 hook 的結果後方補字。

  固定 DataItem 共用同一 PreparedRunData，FigureItem 交原生 Figure，AnalysisItem 交 AnalysisDocument。框架每項編碼一次，對等路徑共用 bytes。DataItem.from_run 將 typed record 的準備錯誤保留為該項目的失敗，不假造 cfg；GUI producer hook 的準備錯誤也交給同一保存流程。輸出與編碼失敗各自隔離，每次呼叫回傳最新 SaveReport 並追加 saved 事件。Ledger 失敗直接傳播，已寫檔案保留，不重跑整筆保存或 rollback。原子替換只對 Windows 短暫 sharing／lock violation 重試約 0.5 秒。共用核心不依賴 GUI 或 Qt，由 artifacts-core-no-ui import contract 約束。Notebook 以 `Experiment[Result, Cfg]` 保留 cfg／Result 配對，並將 `AXES_SPEC` 交給外部存讀函式。
- `acquisition.py` 集中 adapter 的 `AcquisitionEvidence`、UTC run identity 與兩種 capture。
  `capture_software_provenance` 只讀 package／Git／host 與已有的 SoC cached config，不查儀器。
  `capture_device_snapshot(devices)` 由 Run owner 在下述 checkpoint 呼叫，讀取全部 connected drivers 的設定並轉成隔離 JSON。
  量測時間和 completion 由前端 adapter 在 acquire 邊界記錄，不由 core 或 datafile writer 擷取。Evidence 複製版本 mapping，獨立於可修改的 RunRecord cfg／Result。
- GUI experiment 與 Notebook 的 Run 建立路徑各捕捉一次 device snapshot。GUI 在 hardware lease 後、adapter.run 前讀取。Notebook 同步執行，資源由 caller 擁有，不新增 lease；它在建立 RunContext 後、core.run 前讀取。兩路對全部已連線 device 各讀一次 get_info，只讀不寫。任一讀取失敗拒絕 Run，不進 adapter/core，不以 cached info 補值；無 device 記 `{}`。Snapshot 隨當次 Run 保存，後續 sweep 或 Save 不重讀、不覆寫。
- Workflow 的 `await env.run(core.run, cfg)` 每次 attempt 建立新的 `RunContext` 與獨立的 operation Plots，呼叫 core 前捕捉 connected device snapshot，回 `Completed[RunRecord] | Failed`。`Completed.facts: OperationFacts` 與 journal／commit 信封共用事實結構；`env.identity` 回 detached startup RunIdentity。Lab saver 將 Completed、identity、spec.tag 與 snapshot 交給 `record_with_metadata`，由 experiment 組裝 RunMetadata 並放入 RunRecord.metadata。保存只傳 record，native 使用 datafile 的通用 Container。作者與保存介面見 [workflows](workflows/README.md)。
- `context.py` 的 `RunContext` 提供單次 run 的 soc、soccfg、具名 devices、plots 與 scope。名稱到 driver 的對應在建立時固定，driver 仍由 caller 管理。RunContext 不提供儀器 setters；core 以 `context.devices[cfg.dev[role]]` 取得 driver，再使用實驗層 helper。每次 run 建立新 context、plots 與 Scope。GUI／Notebook 使用 run 根，workflow 使用 operation child；硬體 handle 可重用，不把上一輪停止或錯誤狀態帶入下一輪。核心不管理前端 last_result 或呈現資源生命週期。Workflow 的每次 env.run attempt 也建立新的 RunContext 與 operation Plots，使用和 workflow 趨勢圖相同的 presentation host；返回或失敗先 finish，Pause retry 使用新集合。Workflow 層 env.plots 跨 commit 保留，生命週期見 [workflows README](workflows/README.md#ask-與顯示)。
- `axes_spec.py` 用 `AxesSpec` 描述軸與資料 channel。zcu_lab 以 `AxesSpec.Axis`、`AxesSpec.Z` 宣告子規格；獨立的 `Axis`、`ZSpec` 名稱仍可使用。這些規格描述 Result 欄位如何對應檔案中的軸與資料 channel。

  `AxesSpec.Axis("frequency", label="Frequency", unit="Hz", scale=1e6)` 與 `AxesSpec.Z("signal", label="Signal", unit="V")` 接受關鍵字。既有位置參數仍可使用。
  `AxesSpec.Axis(values=(0, 1, 2), label="Population", unit="None", dtype=np.int64)` 宣告固定座標，不需要 Result 欄位。
  `field_name` 與 `values` 必須恰好給一個，否則宣告時 raise `ValueError`。
  固定軸與欄位軸使用相同保存格式與比例。z 的 shape 須符合所有軸長度，不符在保存時 raise `ValueError`。
  讀回固定軸不會把座標傳入 Result constructor。
  Grouped 使用 `VariableAxisSpec(..., values=...)`，與 `field_name`、`generated_arange` 互斥。
  `GroupedLoadData` 仍提供固定座標，由 Result builder 決定是否保留。

  `AxesSpec` 在宣告時核對 Result 的資料欄位，不要求 Result 包含 cfg。軸以 inner-first 排列；保存時套用宣告的單位比例，載入時還原成 Result 欄位。Native cfg snapshot 載入時由宣告的 cfg 型別驗證；Labber 匯出另以 comment envelope 保存 cfg。`save_labber` 使用 caller 指定的路徑，不替 caller 選檔名。對包含多個 Data Variable 的單一 Result，`GroupedAxesSpec`／`VariableSpec` 描述 variable 與欄位的映射，由 typed builder 重建純 Result；Grouped 與 single 都透過 `save_labber(record, path, spec=...)` 匯出，cfg 缺失時拒絕。`labber_layout` 預設 common，要求共同 grid；明確宣告 independent 時，各 variable 保留自己的 1D／2D axes。單檔 save 與 artifact 保存都遵守這項宣告，不自動切換，也不改 native 格式。Grouped native load 由 `load_run_with_metadata` 與 single 共用 restore，不從 Labber reconstruction 讀回。通用檔案格式與讀寫由 [datafile](../datafile/README.md) 負責；保存權責另見 [ADR-0063](../../../docs/adr/0063-persistence-ownership.md)。

`RunMetadata` 與 workflow operation 只記各自邊界的事實，不要求作者使用特定 acquisition 寫法。
舊文件的 `acquisitions` key 在載入時忽略，讀取不轉換或寫回檔案。
ProgramBuilder 的 retry 與 data-driven early stop 改由日誌提供診斷。

## Cfg 與父層支援

- 實驗 cfg 從 `zcu_tools.cfg_model` import `ConfigBase`，直接繼承它宣告輸入。
  Cfg 自行宣告用到的 reps、rounds、initial_delay、relax_delay 與驗證，不繼承 program 設定。
  Core 將執行控制參數明確交給 program builder 與 acquire。
- `ConfigBase` 不宣告 `dev`。
  需要 device 的實驗 cfg 自行宣告角色到 device 名稱的 mapping；只有確實可選的實驗才宣告 optional dev。Dev 不保存 DeviceInfo、設定 patch 或 snapshot。
  使用 flux 的實驗 cfg 必填 `flux_unit`，值為 A、V 或 native，core 不提供隱含 default。
  Run 不預查 cfg.dev 名稱是否存在。Core 實際取 `context.devices[cfg.dev[role]]` 時，缺名即報錯並指名 device，不設定未使用的 device。Cfg 格式與必填 unit 的驗證仍保留。
  `ConfigBase` 提供欄位驗證、`with_updates` 和 `to_dict`；native restore 集中處理歷史 cfg 不相容的 warning 與 cfg=None。
- `cfg_assembler.py` 提供 `ExperimentScalar` 與 `ExperimentCfgValue`，描述 Run 接縫的 mapping 值型別。GUI Form 完成 module materialization 與 sweep 轉換並驗證 adapter 輸入。Adapter.run 將輸入轉成實際實驗 cfg 並回傳 RunRecord，host 保留該 record。Notebook caller 直接提供 typed cfg。
- `config.py` 提供繪圖尺寸 `figsize`；cfg model 與組裝由各自 owner 負責。
- `utils/` 提供 cfg 到 datafile comment codec 的 mapping 與規則 sweep 建構。Comment envelope 的 JSON 編解碼與 Labber reader 的 labels／units／shape／numeric container 驗證由 datafile 擁有；experiment 保留 typed cfg validation、memory units 與 generated coordinate 語意。儀器角色與單位政策由使用者套件的 helpers 擁有。Sweep helper 使用 program/v2 的 `SweepCfg`。`experiment.utils.make_sweep` 是 core 與 Notebook 共用的 sweep 建構入口。它與 [v2/utils](v2/README.md#v2-工具與實驗輔助) 的硬體量化、SNR 和 tracker 工具分開。

## Device 操作邊界

DeviceInfo 描述設定與讀回結果。Device driver 提供儀器本身的操作，flux 或 RF 角色由使用者套件定義。使用者套件選 driver、明示單位與操作用途；具體 helpers 位於 `zcu_lab/v2/support/device.py`，見 [lab v2 說明](../../../zcu_lab/v2/README.md#實驗角色與-device-操作)。

Workflow 的同步 `get_device_cfg(name)` 與完整 cfg `setup_device(name, cfg)` 由宿主提供。使用者工具組合讀取、單位檢查與欄位修改。Framework 轉交完整 cfg；部分欄位的合併與 value／unit 處理由使用者工具負責。Measure MCP 的 partial `device_setup` 是另一個有界工具，見 ADR-0068。

## 子模組入口

Package root 保留公開 exports，只有存取對應名稱時才載入其 owner module；直接使用 persistence exports 仍會載入其所需依賴。

GUI 的 Library、Inspect 與 module writeback 共用 `gui.form` 與 `gui.form_adapter`，直接提交持久 Spec，不經 experiment 的第二套編輯樹。ModuleWriteback 的 proposal 是 ModuleSpec；experiment 仍只接收執行用 typed cfg。跨宿主契約見 ADR-0065、ADR-0008。

- [v2](v2/README.md)：program/v2 的通用 runtime 與量化、SNR、tracker 工具。[runtime](v2/runtime/README.md) 說明 Schedule、buffer、ProgramBuilder 與 executor 的執行機制。具體實驗與前端附件由使用者套件提供，組合根注入 catalog，不由框架反向載入。

# ADR-0060：measure-gui 的 agent 介面——共用 GUI 狀態的第二個 view

**狀態：** accepted（未實作）
**關聯：** [[0059]]（RPC channel）、[[0061]]（interactive plugin session）；[[0002]]（version guard / operation handle）、[[0008]]（CfgEditor session）、[[0013]]（remote adapter 為第二個 View）、[[0025]]（Stop feedback）、[[0033]]（刪改名不掃描參照）、[[0047]]（expected-error taxonomy）、[[0050]]（canonical cfg binding paths）。

## Context

量測 agent 經 MCP 操作 measure-gui，使用者同時在 GUI 上觀看，並可能在任何時刻自行操作或把工作交給 agent。agent 的量測流程由一連串判斷點組成：讀實驗說明、編輯 cfg、開始 run、看 live plot 決定是否中斷重來、帶參數分析並反覆調整、挑選要寫回的結果、存檔。

## 前提

- agent 操作的是 GUI 正在顯示的同一份狀態（[[0013]]）。agent 的改動即時出現在 GUI；使用者的改動 agent 以讀取得知。介面不推送變更通知，也不區分改動者。
- GUI 的既有畫面與非互動分析行為不變；measure flux picker 的拖曳改為本地 preview、有效 release 才 commit（[[0061]]）。
- 人機對話在 agent 所在的 session（例如 Claude Code）進行，不經 MCP。
- 介面只組合 GUI 既有能力，所需補充列於「實作依據」。

## 設計原則

| 原則 | 內容 |
| --- | --- |
| 同一份狀態 | agent 操作 GUI 的 tab、cfg 草稿、分析結果、寫回草稿、context、儀器與 predictor。 |
| 名詞沿用 GUI | tool 以 GUI 物件命名，agent 與使用者指稱同一個 tab、同一份寫回清單。 |
| 草稿先於提交 | cfg 與寫回的修改先落在 GUI 草稿，run、寫入才提交（[[0008]]）。 |
| 一個判斷點一個 tool | 每個判斷點是獨立 tool，不合併成批次；每個 tool 的回傳足以做該步的判斷。 |
| 一件事一條路 | 常用操作由特化 tool 提供；其餘 wire method 經 RPC channel（[[0059]]），兩者不重疊。 |
| 索引與內容分離 | 索引類 tool 只回答「有什麼」；內容由各自的 tool 讀取。 |
| 能機械推導的就提供 | 可由既有資料算出的值（例如 `eta_s`、正規化後的 sweep）由介面回傳。 |
| 省 context | 圖一律回傳檔案路徑；大型值只回摘要，指名讀取時才回完整值。 |

## 共通規則

### 錯誤

會失敗的操作以錯誤回報，附 stable `reason`、訊息與修正提示（[[0047]]）；成功時不回傳空結果。常用 `reason`：`busy`（有衝突的操作進行中）、`unsaved`（有未存檔結果）、`missing`（前置條件不足，附缺少的項目）、`conflict`（參數互相矛盾）、`use_tool`、`not_cancellable`。version guard 衝突時重讀狀態再重試。

### 非同步與短暫等待

長時間操作回傳 operation id（`op`），以 `wait` 等待、`cancel` 取消。標示「短暫等待」的 tool 會在內部等待數秒：期間完成就直接回傳結果，否則回傳 `{status: "running", op}`。

### GUI 跟隨

寫入類 tool 會把 GUI 切到被操作的 tab 與對應子 tab，讀取類 tool 不切換。這是固定行為，沒有參數控制。

| Tool | GUI 切到 |
| --- | --- |
| `tab_open` | 新 tab 的 run 子 tab |
| `tab_edit`、`tab_run` | run |
| `tab_analyze`、帶 `payload` 的 `tab_interact` | analysis 或 post |
| 帶 `write` 的 `writeback` | analysis 或 post |
| `tab_save` | data |

### cfg 編輯語法

`tab_edit`、`writeback` 的 module／waveform 項目與 `ml_edit` 使用同一套編輯語法：依序套用的 `[{path, value}]`，path 為 canonical path（[[0050]]）。

```text
{path: "relax_delay", value: 30.5}                                  常數
{path: "qub_pulse.freq", value: {__kind: "eval", expr: "q_f"}}      md 表達式
{path: "readout.ref", value: "readout_rf"}                          切換參照的 library 項目
{path: "sweep.freq", value: {start: 815, stop: 875, expts: 301}}    整個 sweep
```

- 依序套用，遇錯即停、不回滾；錯誤指出失敗的 path 與已套用筆數。
- 切換 ref 會移除原本的子路徑，須先切 ref 再改子欄位。
- `{__kind: "value_ref", key}` 在套用時解析一次並寫入常數。

**sweep 一律整體修改。** sweep 的欄位彼此連動，只能在 sweep 路徑上給整個物件，不接受端點路徑（例如 `sweep.freq.start`）：

| sweep 種類 | 接受的形式 |
| --- | --- |
| 一般 | `{start, stop, expts}` 或 `{start, stop, step}` |
| 置中 | `{center, span, expts}` 或 `{center, span, step}` |

- `expts` 與 `step` 恰好給一個：都給回 `conflict`，都沒給回 `missing`；其餘欄位必須給齊。
- 形式必須符合該 sweep 的種類；中心被鎖定時不可給 `center`。
- `start`、`stop`、`center` 可用 md 表達式；`span`、`expts`、`step` 只接受數值。
- 正規化由 GUI 的 `SweepEditor` 負責；回傳附上實際值 `{start, stop, expts, step}`，表達式端點另附求值結果。

## Tool 規格（40 個特化 tool，另有 [[0059]] 的 3 個 RPC tool）

### A. 連線與狀態

**`connect(port?, launch = "never" | "if_missing" | "new", clean = false)`**

| `launch` | 已有 GUI | 沒有 GUI |
| --- | --- | --- |
| `"never"`（預設） | 接上 | `reason="no_gui"` |
| `"if_missing"` | 接上 | 啟動並接上 |
| `"new"` | `reason="port_in_use"` | 啟動並接上 |

未給 `port` 時自動尋找。`clean=true` 在啟動時不還原上次的 GUI session。wire 版本不相容時報錯。已連上時重複呼叫回傳目前狀態。回傳 `{launched, port, versions: {wire, gui, mcp}, status}`。連線隨 MCP 結束而關閉，不另提供 disconnect。

**`shutdown(discard_unsaved = false)`**
以 GUI 的正常關閉流程（保存 session、斷開儀器、清理）關閉目前連上的 GUI。有 run 進行中回 `busy`；有未存檔結果回 `unsaved`，確認後以 `discard_unsaved=true` 關閉。關閉逾時回 `{stopped: false}`，由使用者處理。

**`status()`** — 索引：

```text
{
  project: {chip, qubit, resonator} | null,
  soc: {connected, mock},
  context: {active},
  devices: [{name, connected}],
  predictor: {loaded},
  ready: {can_run, missing: [...]},
  tabs: [{tab, experiment, running}],
  running: [{op, tab, kind: "run" | "analyze" | "device"}]
}
```

`running` 包含所有進行中的操作，不論由誰啟動，其 `op` 可用於 `wait`／`cancel`。

### B. 專案與 SoC

**`project(chip?, qubit?, resonator?, scope?)`**
不帶參數時讀取 `{chip, qubit, resonator, result_dir, database_path}`；帶參數時設定 project 並回傳同樣內容。`scope` 為既有 result scope id，用於沿用既有結果目錄。

**`soc_connect(address, port)`**
同步連線實體 SoC，回傳 `soc_info()` 的內容。再次呼叫即改連另一塊板子。

**`soc_info(include_cfg = false)`**
回傳是否連線、位址，以及各通道的 generator／readout 類型、converter port、sample rate、最大 pulse／buffer 長度。`include_cfg=true` 附完整 QICK cfg。

### C. Context 與知識庫

**`contexts()`** — 索引：`{active, labels}`。

**`context_use(label)`**
切換 context；未知 label 報錯並列出可用 label。

**`context_create(label?, bind_device?, clone_from = "current")`**
建立並切換到新 context，回傳 label。未給 `label` 時由 `bind_device` 的目前值與單位產生預設 label。`clone_from` 預設複製目前 context 的 ml／md，`null` 建立空白 context。

**`md_get(keys?)`**
回傳 `{values: {key: value}}`。未指定 `keys` 時，非純量值只回摘要（例如 `"3 × 3 matrix"`）；指定時回完整值。

**`md_set(values: {key: value})`**
依序寫入，遇錯即停、不回滾；回傳 `{key: {before, after}}`。

**`ml_get(name?)`**
未給 `name` 時列出 modules 與 waveforms（名稱、種類、描述）；給 `name` 時回傳該項 cfg。

**`ml_roles()`**
列出可建立的 role 模板 `[{role_id, label, kind, default_name}]`。

**`ml_create(role_id, name?)`**
由 role 模板建立 module／waveform，預設值由 md 帶入；未給 `name` 時用 `default_name`。回傳 `{name, kind, cfg}`。

**`ml_edit(name, edits, save_as?)`**
以 cfg 編輯語法修改 library 項目並存檔；任何一步失敗則 library 不變。`save_as` 存為新項目、原項目不動。存檔時 md 表達式求值為數值，library 不保存與 md 的連動。回傳存入的 `{name, cfg}`。

**`ml_rename(name, new_name, kind?)`**、**`ml_delete(name, kind?)`**
種類由名稱判斷，module 與 waveform 同名時須給 `kind`；名稱衝突時報錯。參照該項目的 cfg 會改為 inline 值（值保留，不再連結 library，[[0033]]），回傳中提示此影響。

### D. 實驗與 tab

**資訊來源**

| 需要知道 | 取得方式 |
| --- | --- |
| 如何做這個實驗 | `guide(experiment)` |
| cfg 格式與目前值 | `tab_get(tab, include=["cfg"])` |
| 分析參數與目前值 | `tab_get(tab, include=["analyze_params"])` |
| 分析結果 | `tab_analyze` 的回傳，或 `tab_get(tab, include=["analysis", "post"])` |
| 寫回內容 | `writeback(tab, stage)` |
| 可存項目與存檔狀態 | `tab_get(tab, include=["artifacts"])` |
| 原始數值 | `tab_save` 後以 `load_labber_data`／`load_grouped_labber_data` 讀取資料檔 |

**`experiments(prefix?)`** — 索引：`[{name, summary}]`，`summary` 為 guide behavior 的第一句。

**`guide(experiment)`**
回傳該實驗的 guide `{behavior, expects_md, expects_ml, typical_writeback, recommended}`。guide 是實驗的操作說明，不是格式契約。

**`tab_open(experiment, from_file?)`**
開新 tab，回傳 `{tab, experiment}`。`from_file` 載入既有資料檔（不需 SoC）；資料檔與實驗不相容時報錯，且不留下 tab。

**`tab_close(tab, discard_unsaved = false)`**
關閉 tab。執行中回 `busy`；有 artifact 為 `not_saved` 或 `unsaved_changes` 時回 `unsaved` 並列出，確認後以 `discard_unsaved=true` 關閉。

**`tab_get(tab, include = ["summary"])`**

| include | 內容 |
| --- | --- |
| `summary` | `{experiment, state: {running, analyzing, has_result, has_analysis, has_post}, source_file}` |
| `cfg` | 每個可設 path 的種類（scalar／sweep／ref key）、型別、目前值、ref 可選項、是否鎖定 |
| `analyze_params` | primary 與 post 的參數定義與目前值 |
| `analysis`、`post` | 分析 summary 與圖檔路徑 |
| `artifacts` | `[{key, kind: "data" \| "image", default_path, status}]`；`status` 為 `no_result`、`not_saved`、`unsaved_changes`、`saved` 之一 |

**`tab_edit(tab, edits)`**
以 cfg 編輯語法修改 tab 的 cfg 草稿。執行中回 `busy`。回傳 `{applied, valid, errors?}` 與被修改 sweep 的實際值；`valid=false` 時 `errors` 列出不合法的欄位與原因。

**`tab_run(tab)`**
以目前 cfg 草稿開始 run，回傳 `{op}`。前置條件不足回 `missing`（`soc`、`active_context`、`valid_cfg`）；已有其他 run 回 `busy` 並附其 tab 與 op。取消的 run 保留已取得的結果，可照常分析與存檔。

**`tab_live(tab)`**
回傳 `{running, progress: [{label, percent}], elapsed_s, eta_s, figure}`；`figure` 為 run pane 圖的路徑。run 結束後 `running=false`，`figure` 為最終圖。沒有 run 也沒有結果時回 `reason="no_run"`。

**`tab_analyze(tab, stage = "primary" | "post", params?)`**（短暫等待）
對目前資料分析並寫入 analysis／post pane。

- `params` 為部分覆寫，寫入 GUI 的分析參數並沿用；未知參數或型別不符時報錯並列出合法參數。
- 完成時回傳 `{status: "finished", summary, figure, params, invalidated}`；`params` 為實際使用的完整參數，`invalidated` 列出被取代的內容（新的 primary 分析取代寫回草稿並清除 post 結果）。
- 互動式分析立即回傳 `{status: "interactive", op}`，以 `tab_interact` 操作。
- post 需要先有 primary 結果；分析失敗時報錯。

**`tab_interact(tab, payload?)`**
操作互動式分析。介面不解讀子命令，只轉送給互動分析外掛註冊的方法。

- 不帶 `payload`：回傳 `{plugin, info, state, commands, figure}`。`commands` 為外掛註冊的子命令與參數定義（`ParamSpec`）；`state` 為外掛目前的結構化選取狀態。
- `payload = {command, args}`：參數依外掛宣告驗證後執行一個子命令，回傳 `{info, state, figure}`。
- `done` 為所有外掛共有的子命令，完成分析，結果經原本的 `op` 送出；取消用 `cancel(op)`。

**`writeback(tab, stage = "primary" | "post", write?)`**

- 不帶 `write`：回傳 `{destination, items: [{id, kind: "md" | "module" | "waveform", target, description, current, proposed}]}`。md 項目為值；module／waveform 項目為 cfg，目標不存在時 `current` 為 `null`。
- `write = [{id, target?, value?, edits?}]`：只寫入列出的項目。`target` 改寫入名稱，`value` 改 md 值，`edits` 以 cfg 編輯語法修改 module／waveform。依序處理、遇錯即停，寫入目前的 active context（`destination`），回傳 `{written: {target: {before, after}}}`。

**`tab_save(tab, artifacts = "all" | [key, ...], paths?, comment?)`**
以 artifact 為單位存檔。`"all"` 依 GUI Save All 的順序存下所有可存項目。`paths` 覆寫個別路徑，其餘用預設路徑；`comment` 寫入 data。回傳 `{saved: {key: path}}`，為實際寫入的路徑（資料檔重名時自動加後綴）。

### E. 非同步

**`wait(op, timeout = 60)`**
回傳 `{status: "running" | "finished" | "cancelled" | "failed", elapsed_s, progress?, eta_s?, error?, feedback?}`。

- `timeout` 上限 300 秒；逾時回 `running` 與進度，不是錯誤。`wait` 期間 session 無法對話，長操作以較短的 timeout 分段等待，段與段之間可對話並用 `tab_live` 看圖。
- op 失敗時回 `failed` 與 `error: {reason, message}`，不丟錯誤；只有 `wait` 本身的錯誤（未知 op、連線中斷）才丟錯誤。
- 使用者在 GUI 以 Stop 附言中止時回 `cancelled` 並附 `feedback`（[[0025]]）。
- 結果由對應的 tool 讀取（`tab_live`、`tab_get`、`devices`）。

**`cancel(op)`**
取消 run、互動式分析或儀器設定，短暫等待後回傳 `{status: "cancelled" | "finished" | "cancelling"}`；`cancelling` 時以 `wait` 確認。post 分析不可取消，回 `not_cancellable`。

### F. 儀器

**`devices(name?)`**
未給 `name` 時為索引 `[{name, type, connected}]`；給 `name` 時回傳 `{name, type, address, connected, error, fields: [{name, type, current, settable, choices?}]}`。

**`device_connect(name, type?, address?)`**
同步連線，回傳該儀器的 `devices(name)` 內容。給 `type` 與 `address` 為首次連線；只給 `name` 時重連已記住的儀器。儀器預設被記住並跨 session 保存。

**`device_disconnect(name, forget = false)`**
同步斷線；`forget=true` 同時遺忘該儀器。

**`device_set(name, values)`**（短暫等待）
先依可設定欄位驗證，欄位不存在、不可設定或不在 `choices` 內時報錯並列出合法欄位。數值使用儀器原生單位（例如 YOKO 電流以 A）。完成時回傳設定後的 `fields`；ramp 較久時回傳 `op`，可 `cancel`。ramp 步長等保護由儀器驅動與 GUI 既有機制負責。

### G. Predictor

**`predictor_info()`** — `{loaded, source, EJ, EC, EL, flux_half, flux_period, flux_bias}`。

**`predictor_load(path? | model?, flux_bias?)`**
`path` 由 `params.json` 的 fluxdep_fit 區段載入，`model = {EJ, EC, EL, flux_half, flux_period}` 直接建立，兩者恰好給一個。取代目前的 predictor，回傳 `predictor_info()` 內容。

**`predict(value, transitions = [[0, 1]])`**
回傳 `[{transition, freq_mhz}]`，`value` 為儀器原生單位的設定值。預測值作為掃描起點，不作為量測結果寫回。

**`predictor_calibrate(value, freq_mhz, transition = [0, 1])`**
以一個量測點校正 `flux_bias` 並重新安裝 predictor，回傳 `{flux_bias_before, flux_bias_after}`。

### H. 畫面

**`screenshot(target = "window" | "setup" | "device" | "predictor" | "inspect" | "arb_waveform")`**
回傳 PNG 路徑，不切換 GUI。`setup` 截取目前開啟的 setup 對話框（啟動時或工具列開啟皆同）。

### 經 RPC 的操作

未列於上方的 wire method 經 [[0059]] 的 RPC channel 呼叫，例如 MetaDict 刪除、result scope 清單、predictor 卸載、arbitrary waveform、value source 與 GUI prompt 對話框。mock SoC 與 fake device 屬開發用途，由開發模式下的專用 tool 提供，不屬於本介面。

## 典型流程

```text
connect()
project(chip="Q5_2D", qubit="Q1", resonator="R1")
soc_connect("192.168.10.179", 8887)
device_connect("flux_yoko", type="YOKOGS200", address="USB0::…")
context_create(bind_device="flux_yoko")
device_set("flux_yoko", {value: 2e-3})                → {status: "running", op: 1}
wait(1)
predictor_load(path="result/Q5_2D/Q1/params.json")
predict(2e-3)                                         → [{transition: [0, 1], freq_mhz: 842.7}]
md_set({q_f: 842.7, qf_w: 15})

guide("twotone/freq")
tab_open("twotone/freq")                              → {tab: "t1"}
tab_get("t1", include=["cfg"])
tab_edit("t1", [{path: "sweep.freq", value: {center: {__kind: "eval", expr: "q_f"}, span: 40, expts: 201}}])
tab_run("t1")                                         → {op: 2}
wait(2, timeout=20)                                   → running
tab_live("t1")                                        → 看 live plot 決定繼續、或 cancel 後 tab_edit 重來
wait(2)                                               → finished
tab_analyze("t1")                                     → q_f = 845.1 MHz
tab_analyze("t1", params={model_type: "sinc"})        → 比較後決定採用哪次結果
writeback("t1")                                       → 檢視 current 與 proposed
writeback("t1", write=[{id: "md-1"}])
tab_save("t1", comment="q_f = 845.1 MHz")
tab_close("t1")
```

接手使用者的工作時，以 `status` 找到 tab，以 `tab_get` 讀取其 cfg 與分析結果，再從適當的步驟繼續。

## 實作依據

| Tool | 依據的既有能力 |
| --- | --- |
| `connect`、`shutdown` | MCP bridge 的 launch／connect；`app.shutdown` |
| `status` | GUI overview（readiness、project、context、SoC、tabs、running） |
| `project`、`soc_connect`、`soc_info` | `project.info`、`startup.apply`；`soc.connect`；`soc.info` |
| `contexts`、`context_use`、`context_create` | `context.labels`、`context.active`、`context.use`、`context.new` |
| `md_get`、`md_set` | `context.md_get`、`context.md_get_attr`、`context.md_set_attr` |
| `ml_get`、`ml_roles`、`ml_create` | `context.ml_get` 與 editor 讀取、`context.ml_list_roles`、`context.ml_create_from_role` |
| `ml_edit` | `editor.new`、`editor.set_field`、`editor.commit`，失敗時 `editor.discard` |
| `ml_rename`、`ml_delete` | `context.ml_rename_*`、`context.ml_del_*` |
| `experiments`、`guide` | `adapter.list`、`adapter.guide` |
| `tab_open`、`tab_close` | `tab.new`、`tab.load_data`、`tab.set_active`、`tab.close` |
| `tab_get` | `tab.snapshot`、`tab.get_cfg`、`tab.get_analyze_params`、`tab.get_post_analyze_params`、分析結果讀取 |
| `tab_edit` | `tab.set_cfg` |
| `tab_run`、`tab_live` | `tab.run_start`；`operation.progress` 與 run pane 截圖 |
| `tab_analyze` | `tab.analyze`、`tab.post_analyze` |
| `writeback` | `tab.writeback_preview`、`tab.writeback_set`、`tab.writeback_apply` |
| `tab_save` | `tab.save_data`、`tab.save_image` |
| `wait`、`cancel` | `operation.await`；`tab.run_cancel`、`analyze.cancel`、`device.cancel_operation` |
| `devices`、`device_*` | `device.list`、`device.snapshot`、`device.setup_spec`、`device.connect`、`device.reconnect`、`device.disconnect`、`device.forget`、`device.setup` |
| `predictor_*`、`predict` | `predictor.info`、`predictor.load`、`predictor.set_model_params`、`predictor.predict`；`PredictorService.calibrate_flux_bias` |
| `screenshot` | `view.screenshot`、`dialog.screenshot` |

需新增或調整，均不改變 GUI 畫面：

- 子 tab 切換：只影響顯示的 wire method，供 GUI 跟隨使用。
- `tab.get_cfg` 補上型別、ref 可選項與鎖定狀態。
- cfg 編輯語法：sweep 整體修改與衝突檢查、sweep 端點接受 md 表達式、回傳錯誤清單。
- `context.new` 接受 `label`。
- `tab.snapshot` 補上 artifact 存檔狀態。
- `tab.writeback_preview` 補上 md 項目的 current 與 module／waveform 項目的 current／proposed cfg。
- `predictor_calibrate` 的 wire method。
- `tab_interact` 的 GUI-side wire method：service-owned session 保存 committed `state` 與 operation（[[0061]]）；plugin 宣告子命令與 `ParamSpec`，wire 驗證後執行共用 action，GUI frontend 直接呼叫相同 typed action，不經 JSON。GUI-local preview 只以 `preview_active` presentation metadata 回報，不取代 committed state。

## 範圍外

本介面不提供：工作點（flux 點）的組合操作與跨工作點表格、predictor 的曲線與 matrix element、由 run cfg 直接建立 library 項目、衍生值寫入、run 歷史與比較、樣品表（SampleTable）寫入、狀態變更通知、冪等重試鍵。

## Consequences

- agent 的每個判斷點對應一個 tool，使用者在 GUI 上看到與 agent 相同的狀態與畫面。
- 同一個操作只有一個入口；低頻操作經 RPC channel，不增加特化 tool。
- `tab_interact` 依賴互動分析外掛的子命令重構，須在該重構完成後實作；其餘 tool 可先行實作。

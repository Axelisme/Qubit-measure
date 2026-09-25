# ADR-0060：measure-gui 的 agent 介面——共用 GUI 狀態的第二個 view（基礎版）

**狀態：** proposed（設計草稿；決定後取代 [[0059]] 的 workflow tool 清單，[[0059]] 的 RPC channel 保留並對量測 agent 開放）
**關聯：** [[0002]]（version guard / operation handle）、[[0008]]（CfgEditor session）、[[0013]]（remote adapter 為第二個 View）、[[0047]]（expected-error taxonomy）、[[0050]]（canonical cfg binding paths）。

## Context

使用者需要即時看到 agent 在做什麼，也可能自己動手後再交給 agent。因此 agent 的每個操作都落在 GUI 正在顯示的同一份狀態上（[[0013]]：remote adapter 與 MainWindow 平級，操作同一個 Controller／State）。

現有 measure-gui MCP 的問題在介面形狀：81 個 tool 多數是 wire method 的 1:1 投影，agent 要自己串起 tab、subtab、editor_id、handle，同一件事又有 bundle 與細粒度兩條路。本 ADR 先定義一組穩定的**基礎介面**；進階能力等基礎介面穩定後再逐項加入（見「後續」）。

## 範圍與前提

- **不改 GUI 的 UIUX。** GUI 畫面與既有行為不變。
- **agent 介面是另一個 view。** agent 的操作改變 GUI 的同一物件，使用者即時看到；使用者的改動 agent 透過讀取狀態得知。不推送變更通知、不區分改動者。
- **對話在 agent session 裡。** agent 跑在使用者面前的前端 session（例如 Claude Code），使用者改了什麼、要 agent 做什麼、agent 要確認什麼，都直接在 session 對話；MCP 介面不承擔人機對話。
- **基礎介面只包裝現有能力。** 每個 tool 對應 GUI 已有的 service／wire 行為，只做組合與整理回傳，不引入新的領域功能。

## 設計原則

| # | 原則 | 具體做法 |
| --- | --- | --- |
| P1 | **同一份狀態，兩個 view** | agent 操作 GUI 的 tab、cfg 草稿、pane 結果、writeback 草稿、context、device、predictor；需要知道現況就讀狀態。 |
| P2 | **名詞沿用 GUI 物件** | agent 與使用者談論的是同一個 tab、同一份 writeback 清單。 |
| P3 | **草稿先於提交** | cfg 與 writeback 的編輯先落在 GUI 草稿上，run／apply 才提交（[[0008]]）。 |
| P4 | **一個判斷點一個 tool** | 讀 guide → 編輯 cfg → 開始 run → 看 live plot 決定是否中斷重來 → 帶參數分析並反覆調整 → 審核寫回 → 存檔，每一步都是自然判斷點，各自是獨立的 tool，不合併成 batch。每個 tool 的回覆足以做該步的判斷。 |
| P5 | **一件事一條路** | 常用操作特化成 tool；其餘 wire method 只經 RPC channel；已特化的 method 不能經 RPC 重複呼叫。 |
| P6 | **非同步只有一種** | 長操作回傳 operation id；一個 `wait`、一個 `cancel`。 |
| P7 | **錯誤可行動** | 錯誤帶 stable `reason` 與 `hint`（[[0047]]）；guard 衝突時重讀狀態再重試。 |
| P8 | **省 context** | 預設精簡，細節用 `include=`；圖回檔案路徑；陣列降採樣或匯出。 |

## Decision：Tool 集合（31 特化 + 3 RPC）

### A. 連線與狀態（3）

**`connect(port?, launch = "never" | "if_missing" | "new", clean = false)`**
接上 GUI。

| `launch` | 已有 GUI | 沒有 GUI |
| --- | --- | --- |
| `"never"`（預設） | 接上 | `reason="no_gui"` |
| `"if_missing"` | 接上 | 啟動並接上 |
| `"new"` | `reason="port_in_use"` | 啟動並接上 |

- 未給 `port` 時沿用現有的自動尋找，找不到才用預設 port。
- `clean=true` 只在實際啟動時有效：不還原上次的 GUI session。
- wire 版本不相容時直接報錯。
- 已連上時重複呼叫直接回傳目前狀態。
- 回傳 `{launched, port, versions: {wire, gui, mcp}, status}`，`status` 同 `status()`。
- 不提供 disconnect；MCP 結束時連線自動關閉。

**`shutdown()`**
關閉目前連上的 GUI，不論由誰啟動。走 GUI 正常的關閉流程（保存 session、斷開儀器、清理）。有 run 在跑時回 `reason="busy"`；優雅關閉逾時回 `{stopped: false}`，由使用者處理，不提供強制結束。

**`status()`** — 索引，只回答「有什麼、在哪裡」，具體內容由各自的 tool 讀取：

```text
{
  project: {chip, qubit, resonator} | null,
  soc: {connected, mock},
  context: {active: "051115_2.000mA" | null},
  devices: [{name, connected}],
  predictor: {loaded},
  ready: {can_run, missing: ["soc", "active_context"]},
  tabs: [{tab: "t3", experiment: "twotone/freq", running: false}],
  running: [{op: 17, tab: "t3", kind: "run" | "analyze" | "device"}]
}
```

- 純讀取，不切換 GUI 畫面。
- `ready.missing` 由現有 readiness 四旗標翻譯而來。
- `running` 列出所有進行中的操作，不論由誰啟動；`op` 可直接用於 `wait`／`cancel`。
- 儀器欄位與值、context 清單與 md、tab 的 cfg／結果、進度、SoC 硬體資訊、project 路徑都不在此，分別由 `devices`、context 類 tool、`tab_get`／`tab_live`、`wait`、RPC 讀取。

### B. 環境（3）

**`project(chip?, qubit?, resonator?, scope?)`**
不帶參數時讀取目前 project：`{chip, qubit, resonator, result_dir, database_path}`；帶參數時設定 project（`startup.apply`）並回傳同樣內容。`scope` 對應既有的 result scope id，用於沿用既有結果目錄；scope 清單走 RPC。重新設定時沿用 GUI 現有行為，不另加限制。

**`soc_connect(address, port)`**
同步連線實體 SoC，回傳 `soc_info()` 的內容；連不上時快速失敗。重新呼叫即改連另一塊板子，不另提供 disconnect。

**`soc_info(include_cfg = false)`**
讀取 SoC 硬體資訊：是否連線、位址、各通道的 generator／readout 類型、converter port、sample rate、最大 pulse／buffer 長度。`include_cfg=true` 附完整 QICK cfg。

mock 模式（mock SoC 與 fake device）屬於開發用途，不在量測介面中，經 RPC 或開發工具啟動。context 的建立與切換見 C 類；儀器連線見 F 類。

### C. Context 與知識庫（6）

**`contexts()`**
context 索引：`{active, labels}`。

**`context_use(label)`**
切換 context；未知 label 報錯並列出可用的 label。

**`context_create(label?, bind_device?, clone_from = "current")`**
建立並切換到新 context，回傳 label。`label` 可自由指定；未指定時由 `bind_device` 的目前值與單位產生預設 label（沿用現有規則），兩者皆無時使用預設命名。`clone_from` 預設從目前 context 複製 ml／md，`null` 表示空白 context。

**`md_get(keys?)`**
讀 MetaDict，回傳 `{values: {key: value}}`。不帶 `keys` 時回傳全部，非純量值（矩陣、長陣列）只回摘要（例如 `"3 × 3 matrix"`）；以 `keys` 指定時回完整值。

**`md_set(values: {key: value})`**
依序寫入，遇錯即停、不回滾（沿用現有語意）；回傳 `{key: {before, after}}`。刪除走 RPC。

**`ml_get(name?)`**
不帶 `name` 列出 modules 與 waveforms（名稱、種類、描述）；帶 `name` 回傳該項 cfg。

ModuleLibrary 的寫入（由 role 建立、改名、刪除、修改欄位）屬常用操作，應有特化 tool；支援方式另行討論，定案前暫經 RPC。

### D. 實驗與 tab（12）

#### 資訊從哪裡取得

guide 是這個實驗的 skill：說明它量什麼、假設 context 已有哪些值、建議的操作順序、常見問題。它是散文，不是格式契約。具體格式各有獨立的讀取方式：

| 需要知道 | 取得方式 | 內容 |
| --- | --- | --- |
| 怎麼做這個實驗 | `guide(experiment)` | adapter guide：behavior、expects_md、expects_ml、typical_writeback、recommended |
| cfg 格式與目前值 | `tab_get(tab, include=["cfg"])` | 每個可設路徑的種類（scalar／sweep edge／ref key）、型別、目前值、ref 可選的 library 項目、是否鎖定 |
| 分析參數格式 | `tab_get(tab, include=["analyze_params"])` | primary 與 post 各自的參數名、型別、可選值、目前值；該實驗有哪些分析階段、是否互動式 |
| 分析結果 | `tab_analyze` 的回傳，或 `tab_get(tab, include=["analysis", "post"])` | summary 欄位與值、figure 路徑 |
| 寫回內容 | `writeback(tab, stage)`（不帶 `items`） | 每個項目的 id、種類（md／module／waveform）、target、current、proposed、是否勾選、目的地 context |
| 存檔路徑 | `tab_get(tab, include=["save_paths"])` | data、analysis image、post image 的預設路徑（依 GUI 檔名規則） |

#### Tools

**`experiments()`**
列出可用實驗：名稱、一行說明、分析階段（primary／post）與模式（fit／interactive）。

**`guide(experiment)`**
回傳該實驗的 guide 全文。開 tab 前後都可讀。

**`tab_open(experiment, from_file?)`**
在 GUI 開新 tab；`from_file` 載入既有資料檔（用於分析，不需 SoC）。回傳 tab id 與 cfg 摘要。

**`tab_close(tab)`**
關閉 tab；執行中的 tab 回 `reason="busy"`。

**`tab_get(tab, include = ["summary"])`**
`include` 可選 `cfg`、`analyze_params`、`analysis`、`post`、`writeback`、`save_paths`、`figures`；`summary` 含實驗名、分析階段、run／analysis 狀態。這也是 agent 讀取使用者在某個 tab 做了什麼的方式。

**`tab_edit(tab, edits)`**
依序套用 cfg 編輯到該 tab 的 cfg 草稿，GUI 表單即時更新。沿用現有編輯語法（[[0050]]）：

```text
{path: "relax_delay", value: 30.5}                          常數
{path: "qub_pulse.freq", value: {__kind: "eval", expr: "q_f"}}   連結 md 表達式
{path: "sweep.freq.start", value: 815.0}                    sweep 端點
{path: "readout.ref", value: "readout_rf"}                  切換參照的 library 模組
```

回傳套用數量與草稿是否有效。

**`tab_run(tab)`**
以該 tab 目前的 cfg 草稿開始 run，立即回傳 `{op}`；不附帶編輯、不自動分析。

**`tab_live(tab)`**
讀取 run 進行中的狀態：進度、run pane 的 live plot（PNG 路徑）。agent 據此決定繼續等、`cancel` 後 `tab_edit` 重來，或讓它跑完。run 結束後回傳最終的 run pane 圖。

**`tab_analyze(tab, stage = "primary" | "post", params?)`**
以 `params` 對目前資料分析，結果寫入 GUI 的 analysis／post pane；可用不同 `params` 反覆呼叫直到滿意。回傳 summary 與 figure 路徑；分析較久時回傳 `{op}`。互動式分析回 `status: "awaiting_user"`，由使用者在 GUI 完成。

**`writeback(tab, stage = "primary" | "post", items?, apply = false)`**
不帶 `items` 時只讀取草稿。`items = [{id, selected?, target?, value?, edits?}]` 勾選、改名、改 md 值，或以 cfg 編輯語法修改模組／波形項目的欄位；`apply=true` 套用到目前 context。回傳每個項目的 `{id, kind, target, current, proposed, selected}` 與目的地 context。

**`tab_save(tab, data = true, images = ["analysis"], data_path?, image_paths?, comment?)`**
不給路徑時用 `save_paths` 的預設路徑；`comment` 寫入資料檔註解。回傳實際寫入的檔案路徑。

**`data(tab, role?, max_points = 200, export = false)`**
回傳目前結果的降採樣軸與數值；`export=true` 回 `.npz` 路徑供 agent 自行分析。

#### GUI 跟隨 agent 的操作

agent 操作某個 tab 的某個階段時，GUI 一律切到該 tab 與對應的子 tab，讓使用者看到 agent 正在處理的畫面；這是各 tool 的固定行為，沒有開關參數：

| Tool | GUI 切到 |
| --- | --- |
| `tab_open` | 新 tab 的 run 子 tab |
| `tab_edit`、`tab_run`、`tab_live` | run（cfg 表單與 live plot 所在） |
| `tab_analyze` | analysis 或 post |
| `writeback` | analysis 或 post（writeback 清單所在） |
| `tab_save` | data |

`tab_get` 與 `data` 可一次讀多個面向，不對應單一子 tab，因此不切換。

### E. 非同步（2）

**`wait(op, timeout = 60)`**
等待 operation 結束；timeout 不是錯誤，回傳目前進度。`wait` 期間 session 無法對話，長操作應以較短的 timeout 分段等待，其間用 `tab_live` 看 live plot，也讓使用者能插話。結束後回傳狀態（finished／cancelled／failed）；產物由對應階段的 tool 讀取（`tab_live`、`tab_analyze` 的結果、`devices`）。

**`cancel(op)`**
取消任何 operation（run、互動式分析、device 操作）。

### F. 儀器（2）

**`devices(name?)`**
清單或單一儀器的欄位與狀態。

**`device_set(name, connect? , values?)`**
`connect = {type, address}` 連線（已連線則略過），`connect = false` 斷線；`values` 例如 `{value: 0.5e-3}`、`{output: true}`。ramp 類回傳 operation id。

### G. Predictor（2）

**`predictor(action = "info" | "load" | "set_params" | "clear", ...)`**
對應現有 predictor wire method。

**`predict(value, transition = [0, 1])`**
回傳預測頻率。只當掃描種子，不當結果寫回。

### H. 畫面（1）

**`screenshot(target = "window" | tab | dialog)`**
回 PNG 路徑；用於 agent 自己確認 GUI 呈現，或在 session 中給使用者看。

### I. RPC channel（3）

`rpc_list(domain?)`、`rpc_describe(method)`、`rpc_call(method, params)`，規格見 [[0059]]。承接低頻但必要的操作：context 列表、MetaDict 刪除、ModuleLibrary 建立／改名／刪除、cfg editor session、arb waveform、value source、device forget／取消、analyze cancel、GUI prompt 對話框（agent 不在使用者身邊時）等。已特化的 method 由 RPC 呼叫時回 `reason="use_tool"`。

## 情境演練

### 情境一：給環境參數，量到 T1

```text
connect()
project(chip="Q5_2D", qubit="Q1", resonator="R1")
soc_connect("192.168.10.179", 8887)
context_create(bind_device="flux_yoko")
device_set("flux_yoko", connect={type: "YOKOGS200", address: "USB0::…"}, values={value: 2e-3})  → op o1
wait("o1")
predictor("load", path="result/Q5_2D/Q1/params.json")
predict(2e-3)                                   → 842.7 MHz
md_set({q_f: 842.7, qf_w: 15})             → 作為掃描種子

# twotone：guide → cfg → run → live plot → 分析 → 寫回 → 存檔
guide("twotone/freq")                           → 先寬掃找真實的峰，再窄掃擬合；predictor 只當種子
tab_open("twotone/freq")                        → t1（GUI 切到 t1 的 run 子 tab）
tab_get("t1", include=["cfg"])                  → sweep.freq 以 q_f ± 1.5*qf_w 連結；readout.ref 可選 readout_rf／readout_dpm
tab_edit("t1", [{path: "rounds", value: 50}])
tab_run("t1")                                   → op o2
wait("o2", timeout=20)                          → running
tab_live("t1")                                  → 40%，live plot 看得到峰在 845 MHz 附近
wait("o2", timeout=60)                          → finished
tab_get("t1", include=["analyze_params"])       → model_type: lor | sinc
tab_analyze("t1")                               → q_f=845.1 MHz，fit 貼合（GUI 切到 analysis）
writeback("t1")                                 → md.q_f 842.7 → 845.1、md.qf_w → 0.8
writeback("t1", apply=true)
tab_get("t1", include=["save_paths"])           → Database/Q5_2D/Q1/…/Q1_qubit_freq_0925@051115_2.000mA.hdf5
tab_save("t1", comment="q_f = 845.1 MHz")       → GUI 切到 data 子 tab
tab_close("t1")

# amp rabi：live plot 顯示掃描範圍不足，中斷重來
tab_open("rabi/amp_rabi")                       → t2
tab_run("t2")                                   → op o3
wait("o3", timeout=15); tab_live("t2")          → 振盪只有半個週期
cancel("o3")
tab_edit("t2", [{path: "sweep.gain.stop", value: 0.4}, {path: "sweep.gain.expts", value: 101}])
tab_run("t2") → wait → tab_live                 → 兩個完整週期
tab_analyze("t2")                               → pi_gain=0.213，但第一點離群
tab_analyze("t2", params={skip: 1})             → pi_gain=0.211，殘差較小
writeback("t2")                                 → md.pi_gain、ml.pi_amp、ml.pi2_amp
writeback("t2", items=[{id: "md-2", selected: false}], apply=true)
tab_save("t2")

# T1：長量測，分段等待
tab_open("time_domain/t1") → tab_run("t3")      → op o4
wait("o4", timeout=60); tab_live("t3")          → 衰減曲線合理，繼續
wait("o4", timeout=240)                         → finished
tab_analyze("t3") → writeback("t3", apply=true) → tab_save("t3")
```

### 情境二 a：接手使用者的 tab

```text
connect(launch="never")
status()                                   → t4 twotone/freq, active, analysis: failed
tab_get("t4", include=["cfg", "analysis", "figures"])
data("t4", max_points=150)                 → 峰貼在掃描邊緣
tab_edit("t4", [{path: "sweep.freq.start", value: 838.0}, {path: "sweep.freq.stop", value: 858.0}])
tab_run("t4") → wait → tab_live("t4")
tab_analyze("t4")                          → fit ok
（在 session 問使用者：找到 q_f=848.3 MHz，要寫回並繼續做 rabi 嗎？）
writeback("t4", apply=true)
```

### 情境二 b：分析舊資料

```text
tab_open("time_domain/t1", from_file="Database/Q5_2D/Q1/…/Q1_t1_0918.hdf5")   → t9
tab_analyze("t9")
tab_analyze("t9", params={dual_exp: true})
data("t9", export=true)                    → .npz，agent 自行擬合比較
```

### 情境二 c：排查問題

```text
tab_get("t4", include=["cfg", "figures"])
devices()                                  → jpa_sgs output=false
（在 session 問使用者：JPA pump 目前關閉，要打開嗎？）
device_set("jpa_sgs", values={output: true})
tab_run("t4") → wait → tab_live("t4")      → 峰值恢復
```

## 對現有架構的需求

基礎介面只組合現有能力：

| Tool | 組合的現有能力 |
| --- | --- |
| `status` | `gui_overview` + tab 清單 |
| `contexts` | `context.labels`、`context.active` |
| `context_use` | `context.use` |
| `context_create` | `context.new`，**新增** `label` 參數 |
| `md_get`／`md_set` | `context.md_get`／`md_get_attr`／`md_set_attr` |
| `ml_get` | `context.ml_get` + editor 讀取 |
| `project` | `project.info`、`startup.apply` |
| `soc_connect` | `soc.connect(kind=remote)` |
| `soc_info` | `soc.info` |
| `experiments` | `adapter.list` + adapter capabilities |
| `guide` | `adapter.guide` |
| `tab_open` | `tab.new`（+ `tab.load_data`）+ `tab.set_active` |
| `tab_close` | `tab.close` |
| `tab_get` | `tab.snapshot`（含 `save_paths`）、`tab.get_cfg`、`tab.get_analyze_params`／`get_post_analyze_params`、analyze／post result、writeback preview、figure |
| `tab_edit` | `tab.set_cfg`（既有編輯語法） |
| `tab_run` | `tab.run_start` |
| `tab_live` | `operation.progress` + run pane 截圖（`tab.get_figure(run)`） |
| `tab_analyze` | `tab.analyze`／`tab.post_analyze` + short-wait |
| `writeback` | `tab.writeback_preview` + `tab.writeback_set` + `tab.writeback_apply` |
| `tab_save` | `tab.save_data`（`data_path`、`comment`）+ `tab.save_image` |
| `wait`／`cancel` | `operation.await`；各類 cancel 合一 |
| `device_set` | `device.connect`／`disconnect`／`setup` |
| `predictor`／`predict` | 既有 predictor wire method |

需要補的只有三項，都不改 GUI 畫面：

- `data`：run result 的降採樣與匯出。
- 子 tab 切換：view-only wire method（與 `tab.set_active` 同性質），供各階段 tool 讓 GUI 跟隨到對應子 tab。
- cfg 格式投影：`tab.get_cfg` 目前回傳值與路徑種類，需補上型別、ref 可選項與鎖定狀態。

## 後續（基礎介面穩定後再評估）

附錄 A 的分析指出以下能力有需要，但都屬進階介面，暫不納入：

- 讀 cfg 時每個葉子標出來源（常數、表達式、參照、關閉），以及開關選用模組的編輯操作
- writeback 以更新模式寫入模組部分欄位；由 run cfg 建立 library 模組
- 衍生值寫入（`reset_f = r_f - q_f`）
- 工作點操作與跨工作點表格
- predictor 校正 flux bias、曲線與 matrix element
- 互動式分析的數值入口
- tab 的 run 歷史、排查用的 run 比較
- 狀態變更摘要、activity 紀錄
- 存檔註解、樣品表（SampleTable）
- `request_id` 冪等重試

## 與 [[0059]] 的關係

- [[0059]] 的七類 workflow tool 清單由本 ADR 的 31 個特化 tool 取代。
- [[0059]] 的 RPC channel 保留並對量測 agent 開放；開發 agent 也用它做 GUI 端改動的 e2e 驗證。

## Alternatives considered

- **以量測為名詞、原子化的 `measure()`**：中間步驟不落在 GUI 草稿，使用者無法即時跟隨或中途介入。
- **在 GUI 加 agent 專用 UI 或推送變更通知**：需要改 GUI，且 agent 讀狀態即可得知改動。
- **維持現有 81 個 tool**：名詞正確但粒度太細、雙路徑多。
- **Code execution**：繞過 permit、guard 與硬體互斥，不可接受。

## 待決問題

1. `tab_live` 的部分資料摘要內容（只給進度與圖，或附降採樣數值）。
2. 實作順序：建議先做 `status`、`tab_*`、`writeback`、`wait`，再做 `setup`、`experiments`、`data`、`devices`、`predictor`。

## 附錄 A：使用者流程分析（`notebook_md/single_qubit.md` 與 measure-gui）

本附錄是 Decision 的依據，記錄從實際使用流程檢查初版設計時的發現。初版把「參數」當成扁平的 `{path: value}`、把「寫回」當成單一的 commit，但實際流程中參數大多是**相對於知識庫的表達式與模組參照**，寫回則包含**挑選、改名、改值、由 run cfg 升級成模組、衍生值**；另外**工作點（flux point）是整個流程的外層迴圈**，predictor 則是貫穿其中的**種子來源與需要校正的模型**。

### A.1 流程階段

| 階段 | notebook 做的事 | 產出寫到哪裡 |
| --- | --- | --- |
| 0. 專案與硬體 | 設 chip/qubit/resonator 名稱、result_dir、database；連 SoC；登錄儀器（flux yoko、JPA yoko、JPA pump SGS）並設定模式與 rampstep | 專案設定、儀器 registry |
| 0'. 接線 | `md.res_ch`、`md.ro_ch`、`md.qub_4_5_ch`（依躍遷分的驅動通道） | MetaDict |
| 0''. 工作點 | `em.new_flux(value, clone_from, unit)` 或 `em.use_flux(label)`：每個 flux 值一個 context，從前一點複製 ml/md | context |
| 1. 讀取 bring-up | lookback → `timeFly`；註冊 `ro_waveform`；onetone freq → `r_f`、`rf_w`；power dep（只看圖）；onetone flux dep（互動選線）→ `flx_half`、`flx_int`、`flx_period`，再把 flux 設到由它們算出的點；註冊 `readout_rf` 模組 | md、ml、儀器 |
| 1'. JPA（選用） | JPA flux／freq／power／auto／check，開關 pump | 儀器、md |
| 2. 找 qubit | 註冊 qubit 波形；載入 predictor（`params.json`）並對齊 `flx_half`／`flx_period`；移 flux、開新 context；`q_f` 由 predictor 預測當掃描中心；twotone → `q_f`、`qf_w`；由量到的 `q_f` 校正 predictor `flux_bias` | md、predictor |
| 3. 脈衝校準 | length rabi → `pi_len`、`pi2_len`、`rabi_f`，並把本次 run 的 `qub_pulse` 改長度後註冊成 `pi_len`／`pi2_len` 模組；amp rabi 的掃描上限由 `pi_len` 模組的 gain 推得，結果 → `pi_gain` 並註冊 `pi_amp`／`pi2_amp` | md、ml |
| 4. Reset（選用分支） | single／dual／bath 三種，各有 freq（中心由 `r_f - q_f` 或 predictor 預測）→ length → gain → check；註冊 `reset_10`、`reset_bath` 等模組；之後的實驗可選擇是否加上 reset 模組 | md、ml |
| 5. 讀取最佳化 | 需要 pi pulse；freq／power／length 或 auto → `best_ro_*`；**更新**既有 `readout_dpm` 模組的欄位（含 `+0.1` 之類的衍生值） | md、ml（update） |
| 6. 同調時間 | T2Ramsey（主動 detune；擬合出的 detune 反過來**修正 `q_f`**）、T1（掃描長度與 relax_delay 取 `5*t1`）、T1 with tone、T2Echo、CPMG | md |
| 7. 記錄 | 把這個工作點的 `q_f`、T1、T2 等加進 `samples.csv`（SampleTable）；dump 儀器資訊 | 樣品表、檔案 |
| 8. 進階 | single shot、MIST、fast flux | md、ml |
| 外層迴圈 | 換下一個 flux 點，重複 2～7 | 每點一個 context |

### A.2 參數控制的樣態

| # | 樣態 | notebook 例子 | GUI 現況 |
| --- | --- | --- | --- |
| C1 | 相對知識庫的表達式 | `r_f ± 1.5*rf_w`、`relax_delay = 5*t1`、`trig_offset = timeFly + 0.05` | `EvalValue` 由 md 即時求值（`Md` seed） |
| C2 | 參照 library 模組／波形，可帶覆寫 | `"readout": "readout_dpm"`、`get_waveform("qub_flat", {length: 0.1})` | `ReferenceValue` |
| C3 | 選用模組開關 | `reset`、`init_pulse` 以註解切換 | `.reset(optional=True)` |
| C4 | 在多個 library 模組間擇一 | `readout_rf` 或 `readout_dpm`、`pi_amp` 或 `pi_len` | reference 選擇 |
| C5 | 執行期選項（不在 cfg 裡） | `earlystop_snr`、`fail_retry`、`uniform=False`、`detune`、`num_points` | 視 adapter |
| C6 | 分析參數 | `model_type`、`fit_bg_amp_slope`、`decay`、`skip`、`dual_exp`、`fit_fringe` | `AnalyzeParams` dataclass |
| C7 | 由知識庫推算儀器設定 | flux 設到 `(1-0.5)*(flx_int-flx_half)/0.5 + flx_half` | 無（手動） |
| C8 | predictor 當種子 | `q_f = predict_freq(cur, (0,1))`；reset 頻率由 predictor 推 | value source `predictor.*`；guide 明言「predictor 不是答案」 |
| C9 | 先寬掃再窄掃 | twotone 4000–6000 → `q_f ± 20` | 手動改 sweep |

### A.3 寫回的樣態

| # | 樣態 | 例子 | GUI 現況 |
| --- | --- | --- | --- |
| W1 | 擬合值寫 md，且可部分採用 | 寫 `q_f` 不寫 `qf_w`；`# md.best_ro_freq = md.r_f` 手動覆蓋 | `MetaDictWriteback`，可取消勾選、可改值 |
| W2 | 衍生值 | `reset_f = r_f - q_f`；T2Ramsey 修正 `q_f` | 無 |
| W3 | 由本次 run 的 cfg 升級成 library 模組（再改欄位） | `pi_len = qub_pulse.with_updates(length=pi_len)` | `ModuleWriteback`（`edit_schema` 可編輯） |
| W4 | 更新既有模組的部分欄位 | `update_module("readout_dpm", {...best_ro_*})` | 以 writeback 覆寫整個模組，無 partial update |
| W5 | 使用者判斷覆蓋擬合 | `pi_len = 1.0`、`res_probe_len = 5.0` | writeback 可改值；或手動 md 寫入 |
| W6 | 改名（retarget） | 同一結果寫到不同名稱的模組 | `target_name` 可改 |
| W7 | 寫到哪個 context | 永遠寫目前 flux 點 | `destination_context` = active context |
| W8 | 更新 predictor | `update_bias(calculate_bias(cur, q_f))` | `PredictorService.calibrate_flux_bias` 存在，**wire 上沒有** |
| W9 | 只看圖不寫回 | power dep、reset check、JPA check | 無 writeback items |
| W10 | 存檔附註解 | `save(comment=f"t1 = {t1}us")`、圖存 `flux_dir/image` | save 有檔名規則，無使用者註解 |
| W11 | 工作點彙總 | `sample_table.add_sample(dev_value, q_f, T1, T2…)` | main GUI 沒有 |

### A.4 初版設計漏掉的部分

1. **cfg 編輯的語意不只是設值（C1–C4）。** agent 需要能表達「設常數」、「連結到 md 表達式」、「改參照哪個 library 模組（可覆寫欄位）」、「開關選用模組」。讀 cfg 時每個葉子也要標出來源（常數、`md:r_f`、`predictor`、`ref:pi_amp`），agent 才知道改了 md 之後哪些值會跟著變。
2. **寫回是審核動作，不是一鍵 commit（W1–W7）。** 需要：挑選項目、改名、改值、編輯模組 cfg、指定目的地 context；模組類寫回要區分「新增」與「更新既有模組的部分欄位」。
3. **衍生寫入（W2）** 沒有位置：應允許以表達式寫入 md（`reset_f = r_f - q_f`），並把來源 measurement 記入 provenance。
4. **工作點是外層迴圈（0''、C7、外層迴圈）。** 應有一個操作同時完成：移動 flux（可用表達式）、建立或切換 context、指定從哪一點複製、回傳 predictor 在此點的預測。也需要跨工作點的表格讀取（每點的 `q_f`、`t1`…）。
5. **predictor 有完整生命週期（C8、W8）：** 載入（`params.json` 或 fluxdep 結果）、與 `flx_half`／`flx_period` 對齊、以量測點校正 `flux_bias`、預測單點／曲線／matrix element、多種躍遷（含 sideband 如 `r_f - q_f`）。預測值只能當掃描種子，不能當結果寫回。
6. **接線設定（0'）** 屬於環境：驅動通道依躍遷區分，應在 `setup` 或 context 中明確宣告，而不是散在 md。
7. **library 管理（1、2、3）：** 由 role 模板建立、由 measurement 的 run cfg 升級、更新欄位、檢視；目前只能經 `context_set` 的 `ml.` 路徑，不足以表達「從 m12 的 `qub_pulse` 建立 `pi_len` 並改長度」。
8. **互動式分析（flux dep 選線）** 目前只能由人在 GUI 拖線。agent 需要數值入口（直接給 `flx_half`／`flx_int` 候選，或自動對齊後回傳候選讓使用者確認）。
9. **執行期選項（C5）與分析參數（C6）** 應在 `experiments(name)` 的 schema 中與 cfg 分開列出，`measure` 分別傳入。
10. **只看圖的實驗（W9）** 是決策點：回傳要強調圖與資料摘要，不產生 proposed。
11. **工作點彙總（W11）** 與存檔註解（W10）缺席：需要 `record_sample` 與 measurement 的 `note`。
12. **輔助儀器狀態（JPA pump 開關）** 直接影響讀取品質；每個 measurement 的環境快照需包含所有儀器，`diff` 才抓得到。

### A.5 修正方向（基礎部分已併入 Decision，其餘列於「後續」）

| 修正 | 內容 |
| --- | --- |
| `measure` 的 `cfg` 改為編輯操作清單 | `{path, value}`、`{path, expr: "r_f + 0.5*rf_w"}`、`{path, ref: "readout_dpm", overrides: {...}}`、`{path, enabled: false}`；另加 `run_options` 與 `analyze_params` 兩個獨立參數 |
| `experiments(name)` 與 `get(m, include=["cfg"])` | 每個 cfg 葉子回傳 `{value, source}`；列出 run options 與 analyze params schema |
| `commit` 改為 `writeback(measurement, analysis?, select, edits, retarget, destination?)` | 項目帶型別（md／module／waveform）與 id；module 項目可傳欄位編輯；`destination` 預設目前 context |
| `context_set` 支援表達式 | `{"md.reset_f": {expr: "r_f - q_f"}}`，`reason` 可引用 measurement id |
| 新增 `library_define(name, from, edits?)` | `from` 為 role 模板、`m12.qub_pulse`（run cfg 中的模組）或既有模組；`mode = create | update` |
| 新增 `work_point(value? | expr? | label?, clone_from = "current", device = "flux")` | 移動 flux、建立或切換 context，回傳此點的 predictor 預測；`work_points(fields=[...])` 回傳跨工作點表格 |
| predictor 擴充 | `predictor(action = load | set_params | align | calibrate | info)`；`predict(at, transitions, kind = freq | curve | matrix)`；`calibrate` 以 measurement 的擬合值為輸入 |
| `setup` 加 `wiring` | `{res_ch, ro_ch, qub_ch: {"0-1": 11, "4-5": 1}}` |
| 互動式分析的數值入口 | `analyze(m, params={flx_half, flx_int})` 或 `analyze(m, mode="auto_align")` 回傳候選 |
| 新增 `record_sample(fields | from_context=true, note?)` 與 measurement `note` | 對應 SampleTable 與存檔註解 |

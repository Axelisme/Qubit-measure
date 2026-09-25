# ADR-0060：measure-gui 的 agent 介面——共用 GUI 狀態的第二個 view

**狀態：** proposed（設計草稿；決定後取代 [[0059]] 的 workflow tool 清單，[[0059]] 的 RPC channel 保留並對量測 agent 開放）
**關聯：** [[0002]]（version guard / operation handle）、[[0008]]（CfgEditor session）、[[0013]]（remote adapter 為第二個 View）、[[0025]]（user feedback wakeup）、[[0047]]（expected-error taxonomy）、[[0050]]（canonical cfg binding paths）。

## Context

使用者需要即時看到 agent 在做什麼，也可能隨時自己動手、再交還給 agent。因此 agent 不能在 GUI 之外另開一條「看不見的」執行路徑：它的每個操作都必須落在 GUI 正在顯示的同一份狀態上。

現有 measure-gui MCP 已經是這個架構（[[0013]]：remote adapter 與 MainWindow 平級，操作同一個 Controller／State），問題在介面本身：81 個 tool 多數是 wire method 的 1:1 投影，agent 要自己把 tab、subtab、editor_id、handle 串起來；cfg 只能設常數；寫回只能整批套用；工作點、predictor 校正、跨工作點彙總都沒有入口（見附錄 A）。

## 範圍與前提

- **不改 GUI 的 UIUX。** 不加 agent 專用面板、不標示改動者、不改變存檔等既有行為。
- **不區分改動者。** agent 與使用者的改動對彼此而言都只是「狀態變了」。
- **agent 介面只是另一個 view。** 它的「UIUX」是結構化的讀取與操作：用 agent 讀得快的形式呈現同一份狀態，用 agent 好表達的形式送出操作。
- 需要的後端補充（例如 tab 的 run 歷史、狀態變更摘要）只存在於 GUI 的 service／State 層與 agent 介面，不出現在 GUI 畫面上。

## 設計原則

| # | 原則 | 具體做法 |
| --- | --- | --- |
| P1 | **同一份狀態，兩個 view** | agent 的每個操作都改 GUI 的同一物件（tab、cfg 草稿、pane 結果、writeback 草稿、context、device、predictor），GUI 自然即時反映；使用者的操作改變同一狀態，agent 讀得到。 |
| P2 | **名詞沿用 GUI 物件** | tab、cfg、run、analysis／post、writeback、context、library、device、predictor、work point。agent 與使用者談論的是同一個「Twotone 2」。 |
| P3 | **草稿先於提交** | cfg 編輯與 writeback 編輯先落在 GUI 的草稿上（使用者看得到），run／apply 才提交。這是 GUI 既有模型（[[0008]]），agent 介面不繞過它。 |
| P4 | **看得見別人的改動** | 每個讀取回傳目前狀態；`status` 與 `wait` 附上自上次讀取以來的狀態變更摘要；guard 衝突的錯誤直接附上最新值。 |
| P5 | **一次回覆足以做下一個決定** | run 完成回傳擬合、品質、圖檔路徑、寫回草稿；讀 cfg 時每個值標出來源。 |
| P6 | **常用操作特化，低頻操作走 RPC** | 約 30 個特化 tool 涵蓋 single_qubit 流程；其餘 wire method 經 RPC channel（[[0059]]）。 |
| P7 | **非同步只有一種** | 所有長操作回傳 operation id；一個 `wait`、一個 `cancel`。 |
| P8 | **錯誤可行動、變更可重試** | 錯誤帶 stable `reason` 與 `hint`（[[0047]]）；會改狀態的 tool 接受 `request_id`，重送不重複執行。 |
| P9 | **省 context** | 預設精簡，細節用 `include=`；圖一律回檔案路徑；陣列預設降採樣或匯出。 |

## Decision：agent 看到的狀態

### cfg 葉子帶來源

讀取 cfg（`tab_get`、`experiments(name)`、`library(name)`）時，每個葉子回傳值與來源，對應 GUI 既有的 value 類型：

```text
"qub_pulse.freq":        {value: 845.1, source: "expr", expr: "q_f"}
"sweep.freq":            {start: 815.1, stop: 875.1, expts: 301, source: "expr", expr: "q_f ± 1.5*qf_w"}
"readout":               {source: "ref", ref: "readout_dpm", overrides: {}}
"reset":                 {source: "disabled", choices: ["reset_10", "reset_bath"]}
"relax_delay":           {value: 10.5, source: "const"}
```

agent 因此知道改了 md 的 `q_f` 之後哪些值會跟著變，也知道哪些模組可以切換。

### cfg 編輯操作

所有編輯 cfg 的 tool 接受同一種操作清單，路徑沿用 [[0050]] 的 canonical path：

```text
{path: "relax_delay", value: 30.5}                                  設常數
{path: "sweep.freq", expr: {center: "q_f", span: "40"}}             連結 md 表達式
{path: "readout", ref: "readout_rf", overrides: {"pulse_cfg.gain": 0.02}}   參照 library 並覆寫
{path: "reset", enabled: false}                                     關閉選用模組
{path: "reset", ref: "reset_bath"}                                  開啟並指定模組
```

### 狀態變更摘要

GUI 端把 EventBus 上的領域事件整理成精簡的變更摘要，`status` 與 `wait` 附上自 agent 上次讀取以來的項目：

```text
changes: [
  {what: "tab", tab: "t3", detail: "cfg changed: sweep.freq, relax_delay"},
  {what: "context", detail: "md.q_f 845.1 → 848.3"},
  {what: "device", name: "flux", detail: "value 0.5e-3 → 0.52e-3"},
  {what: "tab", tab: "t4", detail: "opened (twotone)"}
]
```

不記錄改動者；摘要只描述狀態變了什麼。

### tab 的 run 歷史

每個 tab 保留自己的 run 歷史（`t3#1`、`t3#2`…），每筆含當次 cfg 快照、環境快照（所有儀器狀態、context label、predictor 狀態）、分析結果與資料檔路徑。GUI 畫面仍只顯示最新結果；歷史只供 agent 介面讀取、比較與重跑。

## Decision：Tool 集合

### A. 連線與全局狀態（4）

**`connect(port?, launch = "if_missing" | "never" | "new", clean = false)`**
接上 GUI。`"never"` 用於接手使用者正在用的 GUI；`"new"` 用於開發驗證。回傳 `status()` 內容與程式碼版本。

**`shutdown()`**
只關閉由本 MCP 啟動的 GUI；接上的 GUI 回 `reason="not_owner"`。

**`status()`** — 唯一的定位讀取：

```text
{
  environment: {project, soc, context, work_point, devices: [{name, value, unit, output}], predictor},
  ready: {can_run: bool, missing: [...]},
  tabs: [{tab: "t3", experiment: "twotone/freq", active: true, running: false,
          last_run: "t3#2", analysis: "ok" | "failed" | null, writeback_pending: 2}],
  running: [{op, tab, kind, progress, eta_s}],
  changes: [...自上次讀取以來],
  prompts: [尚未回覆的 ask_user]
}
```

`tabs[].active` 是使用者目前在 GUI 上選中的 tab，接手時由此得知使用者正在看什麼。

**`activity(limit = 30, since?)`**
GUI 端保存的領域事件紀錄，涵蓋 agent 連線之前的操作（開 tab、改 cfg、run、分析失敗、寫回、儀器變動）。接手時用來了解之前發生了什麼。

### B. 環境與工作點（3）

**`setup(project?, soc?, wiring?, devices?, request_id?)`** — 宣告期望狀態，冪等：

```text
setup(
  project = {chip: "Q5_2D", qubit: "Q1", resonator: "R1"},
  soc = {kind: "remote", address: "192.168.10.179", port: 8887},
  wiring = {res_ch: 0, ro_ch: 0, qub_ch: {"4-5": 1, "1-4": 2}},
  devices = {
    flux_yoko: {type: "YOKOGS200", address: "USB0::…::INSTR", mode: "current", rampstep: 1e-6},
    jpa_sgs:   {type: "SGS100A", address: "TCPIP0::192.168.10.89::inst0::INSTR"},
  },
)
→ {changed: ["soc", "devices.flux_yoko"], unchanged: ["project"], problems: []}
```

已相同的部分不動作；會影響進行中操作的改變回 `reason="busy"`。`wiring` 寫入 md 的通道鍵。`devices.<name> = null` 表示斷線。

**`work_point(value? | expr? | label?, device = "flux_yoko", clone_from = "current", request_id?)`**
移到一個工作點：依 `value` 或 md 表達式（例如 `"flx_half + 0.5*(flx_int - flx_half)"`）移動 flux 源，並建立綁定該值的 context（從 `clone_from` 複製 ml／md），或以 `label` 切回既有 context。回傳 context label、實際儀器值與 predictor 在此點的預測（若已載入）。ramp 較久時回傳 operation id。

**`work_points(fields = ["q_f", "t1", "t2r"], labels?)`**
跨工作點表格：每個 context 一列，欄位取自各自的 md。

### C. 知識庫（4）

**`context_get(paths?, history = false)`**
路徑語法 `md.<key>`、`ml.<module>.<path>`，支援萬用字元。`history=true` 附上該值過去的變動（時間、舊值、新值、若由寫回產生則附 run id）。

**`context_set(values: {path: value | {expr}}, request_id?)`**
直接寫 md 或 ml 欄位；`{expr: "r_f - q_f"}` 由目前 md 求值後寫入常數。回傳 `{path: {before, after}}`。對應 notebook 中 `md.reset_f = md.r_f - md.q_f` 這類衍生寫入。

**`library(name?)`**
不帶 `name` 列出 modules 與 waveforms（名稱、種類、描述）；帶 `name` 回傳該項 cfg（葉子帶來源）。

**`library_define(name, kind = "module" | "waveform", from, edits?, mode = "create" | "update", request_id?)`**
`from` 可為 role 模板（`{role: "qub_probe"}`）、某次 run 的 cfg 內模組（`{run: "t3#2", module: "qub_pulse"}`）或既有 library 項目（`{library: "readout_rf"}`）；`edits` 用 cfg 編輯操作。`mode="update"` 只改列出的欄位。對應 notebook 的 `register_module(pi_len=qub_pulse.with_updates(...))` 與 `update_module("readout_dpm", {...})`。

### D. 實驗與 tab（9）

**`experiments(name?)`**
不帶 `name`：列出實驗 `{name, summary, reads_md, writes, ready}`。帶 `name`：guide、cfg（以目前 context 解析、葉子帶來源）、`run_options` schema、`analyze_params` schema（primary 與 post）、分析模式（fit／interactive）、典型耗時。

**`tab_open(experiment, from_tab? | from_run? | from_file?, edits?, request_id?)`**
在 GUI 開一個新 tab（使用者立即看到）。`from_run="t3#2"` 複製該 run 的 cfg；`from_file` 載入舊資料檔（用於分析，不需 SoC）。回傳 tab id 與 cfg 摘要。

**`tab_get(tab, include = ["summary"])`**
`include` 可選 `cfg`、`result`、`analysis`、`post`、`writeback`、`runs`、`figures`。這是 agent 讀取使用者在某個 tab 做了什麼的方式。

**`tab_edit(tab, edits, request_id?)`**
套用 cfg 編輯操作到該 tab 的 cfg 草稿，GUI 表單即時更新。回傳改變的葉子與其新來源。使用者同時修改同一欄位時回 `reason="conflict"` 並附最新 cfg。

**`tab_run(tab, edits?, run_options?, analyze = "auto" | "none", wait_s = 10, request_id?)`**
先套 `edits`，再執行；`analyze="auto"` 在 run 完成後做 primary 分析。`wait_s` 內完成回傳 `{run: "t3#3", status, analysis: {fit, quality, figure}, writeback: [...]}`；否則回傳 `{op, run, status: "running", eta_s}`。

**`tab_analyze(tab, stage = "primary" | "post", params?, run?)`**
對目前或指定 run 的資料（重新）分析，結果寫入 GUI 的 analysis／post pane。互動式分析（flux dep 選線）可直接給數值（`params={flx_half, flx_int}`）或 `params={mode: "auto_align"}` 取得候選。回傳 fit、quality、figure 與 writeback 草稿。

**`writeback(tab, stage = "primary" | "post", select?, retarget?, edits?, apply = false, destination?, request_id?)`**
編輯 GUI 上的 writeback 草稿：`select` 勾選項目、`retarget={"md-2": "qf_w_old"}` 改名、`edits` 改 md 值或模組欄位（cfg 編輯操作）；`apply=true` 寫入 `destination` context（預設目前 context）。回傳每個項目的 `{target, kind, current, proposed, selected, applied}`。

**`tab_save(tab, data = true, images = ["analysis"], note?)`**
沿用 GUI 的存檔路徑與檔名規則；`note` 寫入資料檔註解（對應 notebook 的 `save(comment=...)`）。

**`data(tab | run, role?, max_points = 200, export = false)`**
回傳降採樣的軸與數值；`export=true` 回 `.npz` 路徑供 agent 用 Python 自行分析。

### E. 非同步（2）

**`wait(ops?, timeout = 60)`**
以下任一發生即返回：operation 結束、`ask_user` 得到回覆、出現 error 診斷、相關狀態變更（例如 agent 正在用的 tab 或 context 被改）、timeout。回傳 `{settled: [...], running: [...], changes: [...], prompts: [...]}`；timeout 不是錯誤。

**`cancel(op)`**
取消任何 operation（run、互動式分析、device ramp）。

### F. 儀器（2）

**`devices(name?)`**
清單或單一儀器的完整欄位與最近錯誤。

**`device_set(name, values, request_id?)`**
例如 `{value: 0.5e-3}`、`{output: true}`、`{power: -20}`；ramp 類回傳 operation id。

### G. Predictor（2）

**`predictor(action = "info" | "load" | "set_params" | "align" | "calibrate" | "clear", ...)`**
- `load(path)`：`params.json` 或 fluxdep 結果；
- `align()`：以 md 的 `flx_half`／`flx_period` 對齊；
- `calibrate(run | {value, freq_mhz}, transition = [0,1])`：以量測值校正 `flux_bias`（GUI 端 `calibrate_flux_bias` 已存在，補上 wire method）；
- `info()`：EJ／EC／EL、對齊與 bias 狀態。

**`predict(at: {value} | {work_point}, transitions = [[0,1]], kind = "freq" | "curve" | "matrix")`**
`curve` 回傳一段 flux 範圍的頻率（檔案或降採樣）；可給 sideband 組合（例如 `r_f - f01`）。預測值只當掃描種子，說明中標示 `extrapolated` 等可信度。

### H. 使用者與排查（4）

**`ask_user(question, choices?, blocking = true, timeout = 600)`**
沿用 GUI 既有的 prompt 對話框；`blocking=false` 時答案之後出現在 `status`／`wait` 的 `prompts`。

**`screenshot(target = "window" | tab | dialog)`**
回 PNG 路徑。

**`diagnose(target? = tab | run | device | "soc", compare_to? = run)`**
彙整最近錯誤、GUI log 尾段路徑、SoC 與儀器健康狀態；給 `compare_to` 時比較兩個 run 的 cfg、環境快照與分析參數。

**`record_sample(fields? , from_context = true, note?)`**
把目前工作點的結果（`q_f`、`t1`、`t2r`…與儀器值）加進 result dir 的 `samples.csv`（SampleTable v2，[[0057]]），並保存儀器資訊。

### I. RPC channel（3）

`rpc_list(domain?)`、`rpc_describe(method)`、`rpc_call(method, params)`，規格見 [[0059]]。對量測 agent 開放，承接低頻但必要的操作：tab 關閉與切換、arb waveform、ModuleLibrary 刪除與改名、MetaDict 刪除、predictor 以外的 value source 讀取、device forget 等。已特化成 tool 的 wire method 由 RPC 呼叫時回 `reason="use_tool"`。

合計 33 個 tool（30 特化 + 3 RPC）。

## 情境演練

### 情境一：給環境參數，多輪量測到取得 T1

使用者：「Q1、SoC 192.168.10.179、flux 用那台 Yoko，偏到 2 mA，量 T1。」

```text
connect()
setup(project={...}, soc={...}, wiring={...}, devices={flux_yoko: {...}})
work_point(value=2e-3, clone_from="current")          → context "051115_2.000mA", predictor 0-1: 842.7 MHz
predictor("align")
experiments()                                         → twotone/freq ready；time_domain/t1 缺 ml.pi_amp
tab_open("twotone/freq", edits=[{path: "sweep.freq", expr: {center: "q_f", span: "40"}}])   → t1
tab_run("t1")                                         → t1#1 fit q_f=845.1 MHz, quality ok
writeback("t1", apply=true)                           → md.q_f 842.7 → 845.1, md.qf_w → 0.8
predictor("calibrate", run="t1#1")
tab_open("rabi/amp_rabi")                             → t2（cfg: qub_pulse.freq expr "q_f"）
tab_run("t2")                                         → t2#1 quality flags: ["under_sampled"]
tab_run("t2", edits=[{path: "sweep.gain", value: {start: 0, stop: 0.4, expts: 101}}])
                                                      → t2#2 pi_gain=0.213
writeback("t2", select=["md-1", "ml-1", "ml-2"], apply=true)   → md.pi_gain、ml.pi_amp、ml.pi2_amp
tab_open("time_domain/t1")                            → t3（cfg: pi_pulse ref "pi_amp", relax_delay expr "5*t1"）
tab_run("t3", wait_s=5)                               → op o7 running, eta 240 s
wait(["o7"], timeout=300)                             → t3#1 t1=35.4±1.2 us
writeback("t3", apply=true)
tab_save("t3", note="t1 = 35.4 us")
record_sample()
```

使用者在 GUI 上依序看到三個 tab 開啟、表單欄位被改、圖出現、writeback 清單被勾選與套用。

### 情境一'：沿 flux 掃多個工作點

```text
for v in [1.8e-3, 1.9e-3, 2.0e-3]:
    work_point(value=v)                              → predictor 預測 q_f
    tab_run("t1", edits=[{path: "sweep.freq", expr: {center: "q_f", span: "20"}}]) → writeback → predictor("calibrate", ...)
    tab_run("t2") → writeback → tab_run("t3") → writeback → record_sample()
work_points(fields=["q_f", "pi_gain", "t1"])
```

同一組 tab 在每個工作點重跑；cfg 以 md 表達式連結，切換 context 後自動取用該點的值。

### 情境二 a：使用者做到一半，請 agent 接手

使用者：「我在調 twotone，你接著做完到 rabi。」

```text
connect(launch="never")
status()          → tabs: [{tab: "t4", experiment: "twotone/freq", active: true, analysis: "failed"}]
activity(limit=10) → t4 run 兩次，第二次分析失敗 fit_diverged
tab_get("t4", include=["cfg", "runs", "figures"])
data("t4#2", max_points=150)                  → 峰貼在掃描邊緣
tab_edit("t4", [{path: "sweep.freq", value: {center: 848.0, span: 20, expts: 201}}])
tab_run("t4")                                 → t4#3 fit ok
ask_user("t4#3 找到 q_f=848.3 MHz（見 GUI 圖），要寫回並繼續做 rabi 嗎？", choices=["是", "否"])
writeback("t4", apply=true) → tab_open("rabi/amp_rabi") → …
```

agent 接手的是使用者自己的 tab；接手期間使用者若改了 t4 的 cfg，下一次 `wait`／`status` 的 `changes` 會列出。

### 情境二 b：分析舊資料

```text
tab_open("time_domain/t1", from_file="Database/Q5_2D/Q1/…/Q1_t1_0918.hdf5")   → t9
tab_analyze("t9", params={dual_exp: true})
data("t9", export=true)                       → .npz，agent 自行擬合比較
diagnose("t9", compare_to="t3#1")             → readout gain 與 flux 值不同
```

### 情境二 c：排查問題

```text
diagnose("t4")
  → soc ok；jpa_sgs output=false；與上次成功的 t1#1 相比：jpa_sgs.output true → false
ask_user("JPA pump 目前關閉，要打開嗎？")
device_set("jpa_sgs", {output: true})
tab_run("t4")                                 → 峰值恢復
```

## 對現有架構的需求

全部位於 GUI service／State 層與 MCP，GUI 畫面不變。

| 需求 | 現況 | 需要做的事 |
| --- | --- | --- |
| cfg 葉子來源投影 | value 類型已有（`EvalValue`、`ReferenceValue`、optional module） | 讀取時投影成 `{value, source}` |
| cfg 編輯操作 | `editor.set_field` 以 canonical path 設值 | 支援 expr、ref + overrides、enable／disable |
| 狀態變更摘要 | EventBus 事件已由 MCP piggyback | GUI 端整理成精簡摘要，依 agent 上次讀取位置提供 |
| activity 紀錄 | 無持久紀錄 | GUI 端保存近期領域事件 |
| tab run 歷史 | 每個 tab 只有最新結果 | 保留歷史 run 的 cfg 與環境快照、分析結果、資料檔路徑 |
| writeback 部分更新模組 | `ModuleWriteback` 覆寫整個模組 | 支援 update 模式 |
| 衍生寫入 | 無 | `context_set` 的 expr |
| library 由 run cfg 建立 | writeback 內部已有 `module_cfg_to_value` | 開放為獨立操作 |
| work point 組合操作 | context 綁定儀器值已有（`context.new(bind_device)`） | flux 移動 + context 建立的組合；跨 context 表格 |
| predictor 校正與曲線 | `calibrate_flux_bias`、`predict_freq_curve` 在 service 層 | 補 wire method |
| 互動式分析數值入口 | 只能在 GUI 拖線 | adapter 接受數值參數或 auto-align 候選 |
| 存檔註解 | 無 | 資料檔 comment |
| 樣品表 | main GUI 無 | `record_sample` 寫 SampleTable v2 |
| `request_id` 冪等 | 無 | MCP session 記錄近期 request_id 與結果 |

## 與 [[0059]] 的關係

- [[0059]] 的七類 workflow tool 清單由本 ADR 的 30 個特化 tool 取代。
- [[0059]] 的 RPC channel（`rpc.catalog`、guard policy 宣告、`rpc_*`）保留並對量測 agent 開放，承接低頻操作；開發 agent 也用它做 GUI 端改動的 e2e 驗證。

## Alternatives considered

- **以量測為名詞、原子化的 `measure()`（本 ADR 初版）**：中間步驟（改參數、挑寫回）不落在 GUI 草稿上，使用者只看到結果突然出現，無法即時跟隨或中途介入。
- **在 GUI 加 agent 專用 UI（標示改動者、agent 面板、自主程度開關）**：需要改 GUI UIUX；共用狀態加上變更摘要已足以讓雙方看見彼此的改動。
- **維持現有 81 個 tool**：名詞正確但粒度太細，缺 cfg 表達式、寫回審核、工作點、predictor 校正等能力。
- **Code execution**：繞過 permit、guard 與硬體互斥，不可接受；`data(export=true)` 提供離線分析的出口。

## 待決問題

1. tab run 歷史的保存範圍與上限（只在記憶體，或隨 session 持久化）。
2. 狀態變更摘要的粒度：每個 cfg 欄位一筆，或每個 tab 合併成一筆。
3. `work_point` 是否在切換前自動關閉或保留目前的 tab（目前設計為保留，cfg 以表達式跟著新 context）。
4. 實作分期：建議先做 cfg 來源投影與編輯操作、`tab_run`／`tab_analyze`／`writeback`、狀態變更摘要，再做 work point、predictor、run 歷史、activity、diagnose。

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

### A.5 修正（已併入上方 Decision）

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

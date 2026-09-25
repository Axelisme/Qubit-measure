# ADR-0060：以量測 agent 為中心的 measure-gui MCP 介面（第一性原理設計）

**狀態：** proposed（設計草稿，不遷就現有 tool；決定後取代 [[0059]] 的 workflow tool 清單，[[0059]] 的 `rpc.catalog` 與 guard policy 宣告保留為開發者通道）
**關聯：** [[0002]]（version guard / operation handle）、[[0013]]（remote adapter 為第二個 View）、[[0025]]（user feedback wakeup）、[[0047]]（expected-error taxonomy）、[[0052]]（EventMeta origin attribution）、[[0053]]（gate presence）。

## Context

現有 measure-gui MCP 是 GUI 的遙控器：名詞是 tab、subtab、pane、editor、handle，agent 必須在腦中把「我要量 T1」翻譯成「開 tab → 改 cfg path → run_start → poll handle → analyze_start → writeback_list → writeback_apply(subtab_id) → save_data」。這些名詞服務的是人的視覺版面，不是 agent 的決策。

從量測 agent 的角度，實際在做的事只有：

1. 確認環境（project、SoC、儀器、目前工作點）。
2. 決定下一個實驗與參數。
3. 執行、看圖與擬合品質、判斷可不可信。
4. 不可信就調參重跑或重新分析；可信就把結果寫回知識庫。
5. 重複到取得目標參數；過程中隨時可能要問使用者或被使用者打斷。

接手使用者工作、分析舊資料、排查問題時，還需要：知道之前發生了什麼、誰做的、某個值從哪裡來、這次和上次好的那次差在哪。

## 設計原則

| # | 原則 | 具體做法 |
| --- | --- | --- |
| P1 | **名詞對齊量測，不對齊 UI** | agent 的名詞是 Experiment（模板）、Measurement（一次執行的不可變紀錄）、Context（知識庫）、Device、Operation。tab 是 Measurement 在 GUI 上的投影，agent 不需定址 tab/pane。 |
| P2 | **一次回覆足以做下一個決定** | `measure` 完成時直接帶回擬合結果、品質指標、圖檔路徑、建議寫回的 current→proposed；不需再串三個 getter。 |
| P3 | **環境用宣告，不用步驟** | `setup` 描述「應該是什麼狀態」，冪等，回報做了哪些改變；重複呼叫安全。 |
| P4 | **非同步只有一種** | 所有長操作回傳 Operation；只有一個 `wait` 與一個 `cancel`。`wait` 也會因使用者訊息、錯誤或使用者介入而提早返回。 |
| P5 | **與人共用控制權** | 每個改動帶 origin（[[0052]]）並出現在 timeline；衝突錯誤直接附上最新狀態；不覆蓋使用者尚未套用的編輯。 |
| P6 | **安全靠邊界，不靠 agent 謹慎** | 儀器輸出上下限、實驗安全範圍由 GUI 端強制；超出時結構化拒絕，只有使用者能放寬。 |
| P7 | **錯誤可行動** | 錯誤帶 `reason`（closed set）、`message`、`hint`，以及做下一步所需的狀態摘錄。 |
| P8 | **省 context** | 預設精簡，細節用 `include=` 要；圖一律回檔案路徑；陣列預設降採樣或匯出檔案。 |
| P9 | **每個值都有出處** | Context 值記錄由哪個 Measurement／哪次手動寫入產生；可追溯、可比較。 |
| P10 | **變更可重試** | 會改狀態的 tool 接受 `request_id`；同一 `request_id` 重送回傳第一次的結果，不重複執行。 |

## Decision：核心物件

```text
Experiment   模板：name, summary, requires[context paths], provides[context paths],
             cfg schema（預設值由 context 解析）, capabilities（analyze / post / interactive）

Measurement  一次執行的不可變紀錄（使用者或 agent 發起都會產生）
  id           "m12"
  experiment   "t1"
  origin       "agent" | "user"
  status       queued | running | finished | failed | cancelled
  context      "flux_0p5"              執行當下的 context label
  based_on     "m9" | null             由哪個 measurement 衍生
  cfg_changes  {path: value}           相對於 experiment 預設值的差異（完整 cfg 用 include 取）
  data_file    "…/2026-09-25/Q1_t1_001.hdf5"
  analyses     [{id: "m12.a1", stage: primary|post, status, fit, quality, figure}]
  proposed     [{target: "md.t1_us", current: 32.1, proposed: 35.4, from: "m12.a1"}]
  warnings     [...]

Context      知識庫，單一路徑語法：md.<key>（MetaDict 純量或陣列）、ml.<module>.<path>（ModuleLibrary）
             每個值帶 provenance：{value, set_by: "m12.a1" | "user" | "agent", at}

Operation    長操作：{id, kind: measure|analyze|device|setup, status, progress, eta_s}
             measure / analyze 的 operation id 即 measurement / analysis id
```

## Decision：Tool 集合（24 個）

### A. 連線與全局狀態

**`connect(port?, launch = "if_missing" | "never" | "new", clean = false)`**
接上 GUI；`launch="if_missing"` 找不到就啟動，`"never"` 用於接手使用者的 GUI（絕不改變任何狀態），`"new"` 用於開發驗證。回傳 `status()` 的內容與程式碼版本。

**`shutdown()`**
只關閉由本 MCP 啟動的 GUI；接上的使用者 GUI 回 `reason="not_owner"`。

**`status()`** — 唯一的定位讀取，回傳：

```text
{
  environment: {project, soc, context, devices: [{name, value, unit, output}], predictor},
  ready: {can_measure: bool, missing: ["soc"]},
  running: [{id, kind, experiment, origin, progress, eta_s}],
  user: {active_recently: bool, last_action: "set twotone.freq.stop", at, focus: "m15"},
  drafts: [{tab: "Twotone 2", experiment: "twotone", cfg_changes: {...}, origin: "user"}],
  inbox: [使用者訊息、錯誤診斷、重要事件，自上次讀取以來]
}
```

`drafts` 讓 agent 看到使用者已準備但尚未執行的設定；`user.focus` 是使用者目前正在看的 measurement。

**`timeline(since?, limit = 30, origin = "all" | "user" | "agent")`**
依時間列出發生過的事：開實驗、改 cfg、執行、分析失敗、寫回 context、儀器變動，每筆帶 origin 與相關 measurement id。接手時第一個呼叫。

### B. 環境宣告

**`setup(project?, soc?, context?, devices?, predictor?, request_id?)`**
宣告期望狀態，冪等：

```text
setup(
  project = {chip: "Q12_2D", qubit: "Q1", resonator: "R1"},
  soc = {kind: "remote", address: "192.168.10.8"},
  devices = {flux: {type: "YOKOGS200", address: "GPIB0::1::INSTR", limits: {value: [-2e-3, 2e-3]}}},
  context = {label: "flux_0p5", create_from: "flux_0p4"},
)
→ {changed: ["soc", "devices.flux"], unchanged: ["project"], problems: [], status: {...}}
```

- 已相同的部分不動作；不同的部分只在不影響進行中操作時才改，否則回 `reason="busy"`。
- `devices.<name> = null` 表示斷線。
- `limits` 在儀器首次登錄時可設定，之後 `setup` 只能收緊；放寬只能由使用者在 GUI 操作，`setup` 嘗試放寬回 `reason="limit_change_requires_user"`（見 P6）。

### C. 實驗知識與 context

**`experiments(name?)`**
不帶 `name`：列出所有實驗 `{name, summary, requires, provides, ready: bool, missing: [...]}`，`ready` 依目前 context 判斷。帶 `name`：回傳 guide、cfg schema（預設值已由目前 context 解析）、分析參數、典型耗時、常見失敗樣態。

**`context_get(paths? , history = false)`**
不帶 `paths` 回傳整個 context 摘要；`paths` 支援萬用字元（`md.*`、`ml.pi_q.*`）。`history=true` 附上每個值的變更鏈。

**`context_set(values: {path: value}, reason, request_id?)`**
寫入 context，`reason` 必填並記入 provenance；回傳 `{path: {before, after}}`。與使用者同時修改同一值時回 `reason="conflict"` 並附最新值。

### D. 量測

**`measure(experiment, cfg? , based_on?, from_draft?, analyze = "auto" | "none" | {params}, wait_s = 10, dry_run = false, request_id?)`**

- `cfg`：只寫要改的 path；未寫的由 experiment 預設值（context 解析）或 `based_on` 的 cfg 補齊。
- `based_on="m9"`：沿用 m9 的完整 cfg 再套 `cfg`，用於「縮小掃描範圍重跑」。
- `from_draft="Twotone 2"`：執行使用者在 GUI 準備好的設定。
- `analyze="auto"`：完成後自動做 primary 分析；互動式分析則停在 `awaiting_user`。
- `dry_run=true`：只驗證 cfg、檢查安全範圍與前置條件，回傳解析後的完整 cfg 與預估耗時，不碰硬體。
- 在 GUI 開一個對應的 tab 讓使用者看見；資料完成即自動存檔（`data_file`），不需另外 save。
- `wait_s` 內完成回傳完整 Measurement；否則回傳 `{id, status: "running", eta_s}`，之後用 `wait`。

**`analyze(measurement, stage = "primary" | "post", params?, request_id?)`**
對既有資料重新分析，產生新的 analysis 版本（`m12.a2`），不覆蓋舊版；回傳 fit、quality、figure、proposed。

**`commit(measurement, analysis?, items = "all" | [targets], edits? , request_id?)`**
把 proposed 寫入 context（預設取最新 analysis）；`edits` 可在寫入前修改個別值；回傳 `{target: {before, after}}` 與寫入的 context label。圖檔隨 commit 一起存檔。

**`measurements(filter?, limit = 20)`**
搜尋本次 session 與磁碟上的歷史量測：`filter = {experiment, qubit, context, since, origin, status, text}`。歷史檔案第一次被引用時才載入，之後可直接 `analyze` / `get`。

**`get(measurement, include = ["summary"])`**
`include` 可選 `cfg`（完整）、`fit`、`figures`、`proposed`、`log`（該次執行的錯誤與警告）、`environment`（執行當下的儀器與 context 快照）。

**`data(measurement, role?, max_points = 200, export = false)`**
回傳降採樣後的軸與數值（複數拆成 I/Q 或 amp/phase）；`export=true` 改回 `.npz` 路徑，供 agent 用 Python 自行分析。

**`diff(a, b)`**
比較兩個 measurement 的 cfg、執行當下的儀器狀態與 context 值、分析參數；排查「昨天可以今天不行」用。

### E. 非同步

**`wait(ids? , timeout = 60)`**
等待指定（或全部 agent 發起的）operation。以下任一發生即返回：

- 任一 operation 結束（回傳該 Measurement / 結果）
- 使用者在 GUI 送出訊息或按下「暫停 agent」
- 出現 error 診斷
- 使用者修改了 agent 正在依賴的資源（例如同一 context 的值）
- timeout（不是錯誤，回傳進度與 eta）

回傳 `{settled: [...], running: [{id, progress, eta_s}], inbox: [...]}`。

**`cancel(id, reason?)`**
取消任何 operation（measure、analyze、device ramp）；已結束則回 `reason="already_settled"`。

### F. 儀器

**`device_set(name, values: {field: value}, request_id?)`**
例如 `{value: 0.5e-3}`、`{output: true}`。超出 limits 回 `reason="limit_exceeded"`，附 limits 與「需使用者放寬」的 hint；ramp 類為 Operation。

**`devices(name?)`**
不帶 `name` 為清單；帶 `name` 回完整欄位、limits、最近錯誤。

### G. 預測

**`predict(at: {device_value} | {flux}, transitions = [[0,1]], include_dispersive = false)`**
回傳各躍遷頻率、所用模型來源（params.json 或載入檔）與可信區間（超出擬合範圍時標示 `extrapolated`）。模型安裝走 `setup(predictor=...)`。

### H. 使用者與排查

**`ask_user(question, choices?, blocking = true, timeout = 600)`**
`blocking=false` 時立即返回，答案之後出現在 `wait` / `status` 的 inbox。

**`notify_user(text, level = "info")`**
在 GUI 狀態列與 agent 活動面板顯示 agent 目前在做什麼，不阻塞。

**`screenshot(target = "window" | measurement id | dialog)`**
回傳 PNG 路徑；以 measurement id 為目標時擷取該量測的圖面。

**`diagnose(target? = measurement id | device | "soc" | "app")`**
彙整：最近錯誤（含 traceback 摘要）、GUI log 尾段路徑、SoC 與儀器健康檢查、與同實驗最近一次成功量測的自動 `diff`。

## 情境演練

### 情境一：給環境參數，多輪量測到取得 T1

使用者：「Q1、SoC 在 192.168.10.8、flux 用 GPIB0::1 的 Yoko，偏在 0.5 mA，幫我量 T1。」

```text
connect()                                                   → ready.can_measure = false, missing: [soc, devices]
setup(project={...}, soc={kind:"remote", address:"192.168.10.8"},
      devices={flux:{type:"YOKOGS200", address:"GPIB0::1::INSTR", limits:{value:[-1e-3, 1e-3]}}},
      context={label:"flux_0p5", create_from: "latest"})     → changed: [soc, devices.flux, context]
device_set("flux", {value: 0.5e-3})                         → operation d1 (ramp)
wait(["d1"])
experiments()                                               → t1: ready=false, missing: [md.q_f, ml.pi_q]
                                                              twotone: ready=true（md.r_f 已有）
predict(at={device_value: 0.5e-3})                          → 0-1: 842.7 MHz
measure("twotone", cfg={"qub_pulse.freq": {center: 842.7, span: 40}})
                                                            → m1 finished, fit q_f=845.1 MHz, quality ok
commit("m1")                                                → md.q_f: null → 845.1
experiments("rabi")                                         → 需要 md.q_f ✓
measure("rabi", analyze="auto")                             → m2 finished, quality.flags: ["under_sampled"]
measure("rabi", based_on="m2", cfg={"gain.sweep.stop": 0.4, "gain.sweep.expts": 101})
                                                            → m3 finished, pi_gain=0.213, quality ok
commit("m3")                                                → ml.pi_q.gain: 0.25 → 0.213
measure("t1", wait_s=5)                                     → m4 running, eta_s=240
notify_user("T1 量測中，約 4 分鐘")
wait(["m4"], timeout=300)                                   → m4 finished, t1_us=35.4±1.2, quality ok
commit("m4")
```

每一輪 agent 只需要看 `quality` 與圖決定「重跑、調參、還是寫回」；`based_on` 讓調參不用重寫整份 cfg。

### 情境二 a：使用者做到一半，請 agent 接手

使用者：「我在調 twotone，你接著做完到 rabi。」

```text
connect(launch="never")
status()     → user.focus = "m15"; drafts: [{tab:"Twotone 2", cfg_changes:{"qub_pulse.freq.span": 10}}]
timeline(limit=10)
             → user 執行 m14 twotone（寬掃）→ user 執行 m15 twotone（窄掃）→ m15 分析失敗 fit_diverged
get("m15", include=["fit","figures"])
data("m15", max_points=150)                 → agent 看到峰值貼在掃描邊緣
measure("twotone", based_on="m15", cfg={"qub_pulse.freq.center": 848.0})
                                            → m16 finished, fit ok
ask_user("m16 找到 q_f=848.3 MHz（見圖），要寫回並繼續做 rabi 嗎？", choices=["是","否"])
commit("m16") → measure("rabi") → ...
```

接手完全不需要知道使用者開了哪些 tab。

### 情境二 b：分析舊資料

使用者：「上週 flux 0.3 那批 T1 看起來怪怪的，幫我看。」

```text
measurements({experiment:"t1", context:"flux_0p3", since:"2026-09-15"})   → m-h41, m-h42, m-h43
get("m-h42", include=["fit","figures","environment"])
data("m-h42", export=true)                  → /…/m-h42.npz（agent 用 Python 自行擬合雙指數）
analyze("m-h42", params={"model": "double_exp"})                          → m-h42.a2
diff("m-h41", "m-h42")                      → readout gain 不同、flux 值差 0.02 mA
```

### 情境二 c：排查問題

使用者：「twotone 突然都量不到峰。」

```text
diagnose("m20")
  → soc ok；flux device 最近一次讀值 0.0（output=false）；
    與上次成功 m11 的 diff：devices.flux.output true → false
ask_user("flux 源輸出目前是關閉的，要打開並回到 0.5 mA 嗎？")
device_set("flux", {output: true, value: 0.5e-3})
measure("twotone", based_on="m11")          → 峰值恢復
```

## 對現有架構的需求（缺口）

| 需求 | 現況 | 需要做的事 |
| --- | --- | --- |
| Measurement 紀錄 | 結果掛在 tab 上，重跑會取代 | GUI 端 measurement registry：每次 run 一筆不可變紀錄，含 cfg、環境快照、analyses |
| requires / provides | adapter 宣告 writeback items；requires 未顯式宣告 | adapter 宣告依賴的 context path；或由 cfg 中的 context 參照推導 |
| timeline | EventBus 已有 `EventMeta(seq, origin)`（[[0052]]） | 持久化的 activity log 與查詢介面 |
| provenance | 無 | Context 寫入附來源 measurement 或 reason |
| 儀器 limits | 未發現 | Device 欄位上下限，GUI 端強制，僅使用者可放寬 |
| 自動存檔 | 需手動 save | run 完成即寫 experiment data file；圖隨 commit 存 |
| 冪等 `request_id` | 無 | MCP session 或 GUI 端記錄近期 request_id 與結果 |
| 使用者→agent 訊息 | `gui_prompt_user` 單向阻塞、feedback wakeup（[[0025]]） | GUI 側 agent 面板：訊息、暫停 agent、顯示 `notify_user` |
| 歷史量測搜尋 | result scope 與資料檔存在 | 以 data file metadata 建立索引 |
| `dry_run` 與耗時估計 | 無 | adapter 提供耗時估計；cfg 驗證可沿用 CfgSchema 邊界（[[0011]]） |

## 與 [[0059]] 的關係

- [[0059]] 的七類 workflow tool 清單由本設計的 24 個 tool 取代。
- [[0059]] 的 `rpc.catalog`、guard policy 宣告與 `gui_rpc_*` 保留，但只在開發模式（例如 `ZCU_MCP_DEV=1`）暴露，供開發 agent 做 e2e 驗證；量測 agent 看不到。

## Alternatives considered

- **維持 UI 名詞（tab/pane/editor），只減少數量**：agent 仍需在每一步做 UI 翻譯，接手與排查時也無法回答「之前發生什麼、值從哪來」。
- **一個 `run_workflow(goal)` 全自動 tool**：把判斷藏進 GUI，agent 失去看圖、調參與問使用者的機會；量測品質判斷正是 agent 的價值所在。
- **Code execution（Blender 式）**：繞過 permit、guard 與儀器安全邊界，不可接受；`data(export=true)` 提供離線分析的出口即可。
- **分開的 save tool**：資料保存不是決策點；自動存檔讓每次量測都可追溯，commit 才是需要判斷的動作。

## 待決問題

1. Measurement registry 的保存範圍：只保留本 session，或所有量測都進入可搜尋的歷史索引。
2. 自動存檔是否為預設：磁碟用量與「失敗的量測也保存」的取捨。
3. 儀器 limits 的來源與管理介面：由使用者在 GUI 設定、寫入 project 設定檔，或兩者皆可。
4. `commit` 是否一律需要使用者確認，或只在值變動超過門檻時才經 `ask_user`。
5. 實作分期：建議先做 Measurement registry、`measure`／`analyze`／`commit`／`wait`，再做 timeline、provenance、diagnose。

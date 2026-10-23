# 用 startup script 建立 entry 與 alias

**讀者：** lab-author、agent。指定 startup 腳本，建立 entry、point 與 alias 表。

讀完本篇，你能指定一份 startup script，建立自己的 entry，並讓表單透過 alias 讀取目前 point 的元件。先使用 [FTF kickstart](../kickstart/FTF/README.md)。這份腳本不連線 SoC 或 device，也不建立模擬環境。

## 先分清 startup 與 entry

Startup script 是定義 `startup(ctx)` 的可信 Python 檔。GUI 顯示後會在背景執行它，狀態列顯示進度，完成前不能 Run。腳本透過 `ctx.setup` 建立或開啟 entry、選取 point，再透過 `ctx.context` 加入元件與套用 alias rows。可用能力與錯誤處理見 [Measure 啟動腳本](../../docs/adr/0064-process-startup.md#measure-啟動腳本)。

Entry 保存實驗參數與結果的工作範圍。Setup 是新 point 的元件範本，每個 point 有獨立的元件、module library 與 alias 表。改 setup 不會同步改既有 point。複製規則見 [文件與 seed](../../lib/zcu_tools/resources/entry/README.md#文件與-seed)。Startup 只是操作這些資源的程式，不是 entry 的檔案格式。

FTF 的 [startup.py](../kickstart/FTF/startup.py) 第一次會建立 `FTF` entry，讀取相鄰的 [entry-template.yaml](../kickstart/FTF/entry-template.yaml)，把元件加入 setup，建立並選取 `initial` point，最後保存該 point 的 aliases。範本不含 module library 或校準完成的量測輸入，建立成功不代表可以 Run。

## 指定腳本啟動

以下 shell 指令都在 repo root 執行。先建立一個既存的絕對目錄，讓新 entry 不與目前專案混在一起：

```bash
mkdir -p /absolute/my-ftf-project
uv run --directory . --no-sync -- python scripts/run_measure_gui.py \
  --project-root /absolute/my-ftf-project \
  --script zcu_lab/kickstart/FTF/startup.py
```

把 `/absolute/my-ftf-project` 換成自己的目錄。`--project-root` 改變資料根目錄，不改變程式碼或腳本路徑的定位方式。省略時資料根目錄是 repo root。等狀態列結束 startup，再確認目前 entry 是 `FTF`、point 是 `initial`。

使用 Measure MCP 時，改呼叫下面的 `connect`，不要同時啟動兩個 GUI 操作同一份 entry：

```python
connect(
    launch="new",
    script="/absolute/repo/zcu_lab/kickstart/FTF/startup.py",
    project_root="/absolute/my-ftf-project",
)
```

這是 MCP tool 呼叫，不是可以直接 import 的 Python 函式。把 `script` 換成 repo 中腳本的絕對路徑。`launch="new"` 要求啟動新 GUI；attach 不重跑 startup，attach 時提供 `script` 或 `project_root` 會報錯。Connect 回覆只表示 RPC 已就緒，不保證背景 startup 已完成。選項與 attach 限制見 [MCP startup scripts](../../lib/zcu_tools/mcp/measure/README.md#startup-scripts)。

`--script` 完全取代框架的預設腳本 `lib/zcu_tools/gui/app/measure/default_startup.py`，不先還原上次 session；要還原時在自訂腳本內 `from zcu_tools.gui.app.measure.default_startup import startup` 並呼叫它。`--clean` 不執行任何 startup，不能與 `--script` 同用。預設腳本的保存與關閉行為見 [保存與關閉](../../docs/adr/0064-process-startup.md#保存與關閉)。

## 修改自己的 entry-template

先把 kickstart 的兩份檔案複製到自己的目錄，再改副本：

```bash
mkdir -p /absolute/my-startup
cp zcu_lab/kickstart/FTF/startup.py \
  zcu_lab/kickstart/FTF/entry-template.yaml /absolute/my-startup/
```

把 `--script` 或 MCP 的 `script` 改成 `/absolute/my-startup/startup.py`。腳本依自身檔案位置找 YAML，不依 shell cwd 找。要改 entry 名稱或第一個 point 的 label，修改自己 startup 副本中的對應名稱，建立與重開分支要一起改。

[FTF 範本](../kickstart/FTF/entry-template.yaml) 的結構如下，修改時保留外層與內層各自的 header：

- `format: zcu.entry-template` 與 `format_version: "1.0"` 標示範本格式。
- `components` 把 `Q1`、`Q2`、`coupler`、`R1`、`R2`、`MG12` 等名稱對應到元件資料。每個元件的 `kind` 選擇已註冊 model，其餘欄位由該 model 驗證。
- `MG12.wiring` 填 `gen_ch: 1`、`ro_ch: [1, 2]`，逐項對應 `MG12.resonators: [R1, R2]`。列表長度不同會在載入時報錯。R1／R2 自己的 `wiring.ch: 0`、`wiring.ro_ch: 0` 是單 tone 接線，不是 mux ADC。Qubit 的 `freq` 保留 null，接線不能替代未知的量測頻率。
- `aliases` 是完整的內層文件，使用 `format: zcu.aliases`、`format_version: "1.0"` 與另一層 `aliases` mapping。內層 mapping 保留列順序與原始字串，不先解析連結。

完整格式與載入限制見 [可攜 entry 範本](../../lib/zcu_tools/resources/entry/README.md#可攜-entry-範本)。新增 kind 的流程見 [component 教學](01-components.md)。

## 綁定 alias rows

FTF 範本先放 `qubit: Q1`，再放 `resonator: $qubit.resonator`。後一列讀取 Q1 的 resonator 欄位，得到 `R1`。表單在 Open／Reset 從 `$qubit.flux_source` 取得 `f1_bias`，不需要 flux 裝置 alias。軟連結只能引用排在前面的 alias，不能引用自己或後面的列。

Aliases 的 Value 欄接受具體名稱或前列 alias 的欄位連結，兩者的差別如下：

| Alias row | Value | 解析結果與後續變更 |
| --- | --- | --- |
| `qubit` | `Q1` | 直接綁定具體 component `Q1`。 |
| `resonator` | `$qubit.resonator` | 讀取前列 `qubit` 所指元件的 `resonator` 欄位，此時得到 `R1`。改綁 `qubit` 後會跟隨該欄位。 |
| `f1` | `Q1` | 同樣直接綁定 `Q1`，不隨 `qubit` 列的修改而變更。 |

若要讓 `f1` 與 `qubit1` 指向同一元件，兩列都填具體名稱 `Q1`。不要填 `f1: $qubit1`，Aliases 不接受裸 alias 作為另一列的連結。`$qubit.resonator` 才是可接受的欄位連結，而且 `qubit` 必須排在 `resonator` 前面。這是 Aliases 的儲存語法，不是表單 expression 中裸 `$alias` 的求值語法。

在 GUI 開啟 Inspect 的 Aliases 頁，把 `qubit` 改成 `Q2`，再按 Apply。`resonator` 會解析成 `R2`。重新 Open／Reset 後，flux 表單從 Q2 的 flux_source 取得 `f2_bias`。`mux: MG12` 仍指向同一個 group，不隨 `qubit` 切換。元件的 source 欄位與表單的裝置名稱都不會連線或建立 device。

Apply 保存目前 point 的 `points/initial/aliases.yaml`。切到另一個 point 會載入另一張表，缺檔時是空表，不回退到 setup 或 preset。Alias 解析與保存細節見 [context 與 Run snapshot](../../lib/zcu_tools/gui/session/README.md#contextsession-environment-與-run-snapshot)。

既有 tab 不會因 alias 改變而更新 input。重新 Open 或 Reset，表單才會從新的 alias 表擷取初值。Alias rows 也不是 `Template.bindings`，不替表單補模板佔位符。範本與初值的 author 契約見 [measure adapter authoring](../v2/measure-authoring.md#目錄佈局)。

## 關閉後再開一次

正常關閉 GUI 後，用同一個 project root 與同一份 FTF startup 再啟動。腳本只重開 `FTF` 並選取 `initial`，不再套用 YAML，也不覆寫已改的元件或 alias。改了 entry-template 後重開既有 entry，不會更新它；請在目前 point 明確修改，或換新的 project root 建立另一份 entry。

若 `initial` 不存在或 entry 不完整，腳本會報錯。GUI 保留已完成操作，不自動回滾，也不改跑預設腳本。先按錯誤訊息檢查資料，再決定修復或另建專案，不要把重啟當成重設。

需要比較「建立 entry」與「安裝模擬資源」時，另讀 [FTF 模擬 startup](../simulation_envs/FTF/README.md#建立-entry-與啟動)。它會安裝離線 SoC 與 devices，與本篇 kickstart 的用途不同；本篇不要求先準備模擬環境。

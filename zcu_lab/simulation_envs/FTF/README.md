# 模擬環境

**讀者：** operator、lab-author、agent。準備 FTF 模擬參數與 entry，並從 GUI 或 Notebook 啟動模擬環境。

**Last updated:** 2026-10-23，mux group wiring

這裡放 FTF 模擬參數、entry 樣板與可信 GUI startup 範例。Startup 與 Notebook 直接呼叫公開 `zcu_tools.experiment.ftf.ftf(...)`，不另設 environment 薄包裝。Startup 讀取相鄰 `ftf.yaml`；Notebook 明示 YAML 路徑。指定檔案不存在或內容有誤時直接報錯，不改讀另一份設定。

`default.yaml` 是納入 Git 的樣板；`ftf.yaml` 是各自的工作副本，不納入 Git。第一次使用時複製一份：

```bash
cp zcu_lab/simulation_envs/FTF/default.yaml zcu_lab/simulation_envs/FTF/ftf.yaml
```

缺少 `ftf.yaml` 時，`ftf(...)` 直接報錯。完整欄位、單位、合法範圍、三迴路映射與 entry 接線對照見 [FTF YAML v2 欄位指南](field-guide.md)。

## 建立 entry 與啟動

在 repo root 準備好上面的 `ftf.yaml`，再用 startup script 啟動。不需先執行 entry CLI：

```bash
mkdir -p /absolute/new-ftf-project
uv run --directory . --no-sync -- python scripts/run_measure_gui.py \
  --project-root /absolute/new-ftf-project \
  --script zcu_lab/simulation_envs/FTF/startup.py
```

`--project-root` 須指向既存的絕對目錄。省略時使用 repo root。
Script 透過公開 ctx 在該專案建立 `ftf-demo`，從相鄰 `entry-template.yaml`
加入 Q1、Q2、coupler、R1、R2 與 mux group MG12，再建立並選取第一個 point `initial`。
MG12 使用 `MuxGroupSchema`（`group/mux`），依序綁定 Q1／R1 與 Q2／R2。
Script 用 `ctx.context.apply_alias_rows` 初始化該 point 的 alias，框架保存到 `points/initial/aliases.yaml`。
新 point 從 setup 複製這些元件。`ftf-demo` 已存在時，script 不套用 template，
直接開啟該 entry 並選取 `initial`，保留之前的修改；`initial` 不存在就報錯。
`result/` 與 `Database/` 只剩其中一個 `ftf-demo` 時是不完整的 entry，script 報錯並指出缺哪一邊，
不重用也不覆寫；兩邊都刪除即重新建立。
首次建立中途失敗可能留下部分 entry，沒有跨檔交易或自動 rollback。

Script 接著建立離線 FTF 環境，安裝 SoC，登記 f1_bias、coupler_bias、
f2_bias 三個 live device，GUI 即可從 New Tab 開實驗。毋須手動 Open 或選 point。
Template 不含 module library 或校準值；tab 可編輯，但執行量測前仍須設定有效輸入。
Script 用 `workers=1, warmup=False`，不在啟動時預編譯或執行 acquisition。

MCP 的等效入口是
`connect(launch="new", script="/absolute/path/startup.py", project_root="/absolute/new-ftf-project")`。
Attach 不重跑腳本；在 attach 傳 script 會報錯。自訂 script 取代預設腳本，
不先執行預設 startup。`--clean` 跳過所有腳本，不能與 `--script` 同用。

可將 `startup.py`、`entry-template.yaml` 與工作副本 `ftf.yaml` 複製到自己的目錄，
再指定該處的 startup。範本與設定按 startup 檔案位置定位，不跟隨 shell cwd。
要改 entry、point 或 tab 名稱，編輯自己的 startup 副本。

模擬用 `entry-template.yaml` 省略 qubit energies 與 R1／R2 的 `freq`，只提供接線、元件關聯與 aliases。物理模型的 energies 與 resonator 頻率由 `ftf.yaml` 提供，不會自動填入 entry。Qubit 頻率隨 bias 改變，範本保留 null，不推算頻譜。量測前仍須填入有效的頻率與其他輸入，範本不提供 module library 或校準完成的量測參數。[Kickstart 範本](../../kickstart/FTF/entry-template.yaml) 另外預填示意 energies 與 R1／R2 頻率，不代表模擬 entry 也有這些值。若修改工作副本的接線，也要調整 entry 範本或已建立的 entry。

Startup 腳本建好初始 alias。`mux` 指向 MG12，mux 表單透過 `$mux` 讀取同序的 qubits 與 resonators。
R1／R2 的 wiring 保留 pulse gen0／ADC0。MG12 的 mux wiring 填 `gen_ch: 1`、`ro_ch: [1, 2]`，依序對應 R1／R2，不讀 resonator 的 pulse ADC。
範本不附 library assets，也不預填 module key。八張 mux 表單缺 readout slot 時，Open／Reset 從 group wiring 補通道，tone gain 補 0.1，其餘缺失的必填輸入仍保持 Invalid。有 slot 時沿既有 library 路徑，細節與核對步驟見[接線對照](field-guide.md#generatoradc-與-entry-接線對照)。
初始 alias `qubit` 指向 Q1，`resonator` 與 `flux_dev` 分別讀取其 resonator 與 flux_source。把 `qubit` 改為 Q2 後，兩個連結會跟著切換。`qubit1`、`qubit2` 分別指向 Q1、Q2，`coupler` 是普通 alias，指向 qubit/transmon 元件，不是 Notebook role。

## 設計初猜與 predictor

環境作者可在 `ftf.yaml` 的 `design:` 填寫獨立標稱值。這是設計圖的頻率初猜，不是
`chip` 或 DC 校準真值，也不是量測結果。樣板為 Q1、Q2 各填一筆，數值刻意與物理模型不同。
每筆必填 `ej_ghz`、`ec_ghz`、`el_ghz`、`flux_unit`、`flux_half`、
`flux_period`、`flux_bias`。欄位意義與單位見[欄位指南](field-guide.md#design設計初猜)。

Startup 以 entry-template 的 qubit 名稱驗證 design，並檢查 `flux_unit` 與其
`flux_source` 的 device 單位一致。未知 qubit 或單位不符會報錯。
Startup 經 resources 的 `save_predictor_file` 建立 entry 的 `predictors/Q1.json`、
`predictors/Q2.json`，只寫缺失的檔案。既有 predictor 與手動調整保留不變。
標稱值一對一交給 predictor，不從物理模型或 DC 校準推算或補值，也不填入 entry 的頻率。

省略 `design` 或填 `design: {}` 時，環境照常啟動，不建立 predictor。
狀態列會顯示「design 未提供，未建立 predictor」。既有檔案不會刪除。
提供的每筆若缺欄位，`ftf()` 載入時會指出欄位路徑，例如 `design.Q1.flux_period`。

MCP 先呼叫 `predictor.list_files` 發現檔案，再使用
`predictor.predict(filename="Q1.json", device_value=1.2e-3)` 取得標稱頻率。
樣板的 device 原生單位是 A，這個輸入表示 1.2 mA。初猜仍須由量測驗證。

## 工作副本與調參

`default.yaml` 使用 `format: zcu.ftf`、整數 `version: 2`。V1 文件不能載入，不自動轉換既有工作副本，也不自動更新既有 entry。三個 DC 電源與校準位於 `dc_bias.sources`，樣板使用 A。環境每 round 讀取 live DC snapshot，保留靜態磁通掃描。`drive_lines` 的每條線以 `operator: charge | flux` 選電荷或固定 DC 小訊號磁通導數，省略為 charge。Flux operator 不是逐時重算 Hamiltonian 的大振幅 fast-flux 控制。

FTF 的 flux drive 只在 DC 附近線性化，不建構大振幅 fast-flux controls；`flux_lines` 即使為空 map 也按未知欄位拒絕。未綁定 generator 沿用既有處理。共享 `simulate` 引擎仍保留完整 flux-pulse 能力，但 FTF 不使用該路徑。採用新樣板時請自行保留或重新填入需要的設定，不覆寫尚未保存的本機調整。舊工作副本或預覽若仍含 `flux_lines`，須自行移除。

啟動腳本會執行 Python，只載入可信來源。修改腳本或 YAML 後須重啟 GUI。腳本失敗時，GUI 保留已完成操作並顯示腳本位置與錯誤，不自動改跑預設腳本。

`default.yaml` 的 circuit energies、couplings 與 resonator frequencies 採論文 Device A，其餘包含本機手動設定。它不是通用校準結果，也不保證截斷或 solver 收斂。Channel 數量由最大索引加一推導，不是綁定條數。樣板使用 pulse readout gen0／ADC0、mixmux8 mux readout gen1／PFB ADC1 與 ADC2，以及 f1_charge gen2、f2_charge gen3。這份佈局推導出 4 個 generator 與 3 個 readout。兩條 charge line 的主迴路係數為 0.1 GHz/gain，coupler 串擾為 0.01，對另一 fluxonium 為 0。這些係數是模擬假設。Entry 範本的 Q1/Q2 分別綁 gen2/gen3，既有 entry 不會自動更新。完整訊號路徑與接線表見 [欄位指南](field-guide.md)。更換樣品時可複製 startup 與 YAML，並調整 Notebook 的 YAML 路徑。

[Notebook 範例](../../../notebook_md/single_qubit.md) 呼叫同一個公開 `ftf(...)` builder。成功回傳後，Notebook 負責關閉傳回的 device manager。Startup 在交付前登記 `on_close`。成功登記的 driver 由 GUI 關閉，環境 manager 在 runtime 只借用它們；callback 先移除這些借用，再關閉尚未交付的 driver。Startup 中途失敗仍保留 callback。完整組裝與數值限制見 [FTF 契約](../../../lib/zcu_tools/experiment/ftf/README.md)。

## 演化與重播

FTF 預設使用分段引擎的快速算法。GUI startup 明確傳 `workers=1, warmup=False`，不啟動數值預編譯。Notebook 明確傳 `workers=max(1, (os.cpu_count() or 1) - 2), warmup=True`，保留兩核並在建立環境後啟動單一背景 worker 預編譯常用數值路徑。核心數未知時使用 1 個 worker，八核主機使用 6 個 Notebook workers。公開 `ftf()` 的預設是 `workers=1, warmup=False`；`workers` 必須為正整數。

預編譯會占用 CPU 與記憶體，只把編譯時間與其他工作重疊，不縮短總編譯成本。同一個 process 內啟用 warmup 的環境共用同一個 future。

啟用預編譯時，`env.warmup.status` 提供進度與錯誤。停用時 `env.warmup` 為 `None`。`env.wait_ready()` 等待完成；若要讓計時排除預編譯，先等 ready 再啟動計時。Acquire 也會自動等待，失敗則回到 on-demand 編譯。背景工作不抽樣、不切換正式 device bias，也不占用正式引擎的 prepare 快取。結束 process 時會等當前編譯完成，不強制中止 compiler。完整介面見 [FTF 契約](../../../lib/zcu_tools/experiment/ftf/README.md#startup-preparation)。

整個 evolve 期間 native BLAS 限為 1 個執行緒。重疊的 evolve 共用 process 層級的 reference count，第一個進入時設限，最後一個離開時還原。其他程式碼不要在 evolve 期間自行更改 threadpoolctl 限制。沒有演化工作、只有一條 timeline 或只有 1 個 worker 時，不開 joblib pool。模型與 prepare 快取只保留在記憶體，不需要離線建庫或 GPU，也不保存 exact density records。

Builder 可用 `ftf(..., private_replay=absolute_directory)` 保存私有輸入。`replay_round(round_path)` 在相同數值、RNG 與相依套件版本下重算一個 round，不讀取演化資料庫。私有檔案含模型真值，不放入一般 Run metadata。

# ZCU-Tools：一個用於 ZCU216 平台的量子測量工具包

以下為AI生成:

ZCU-Tools 是一個專為 ZCU216 FPGA 平台設計的綜合性 Python 工具包，用於執行和分析量子位元測量。它為實驗控制、數據採集、即時視覺化和數據後處理提供了一個結構化的框架。

## 核心功能

* **實驗自動化**: 提供標準化的類別，用於運行各種量子實驗，如OneTone和TwoTone光譜、拉比振盪等。
* **遠端控制**: 基於 qick 構建，允許在伺服器（ZCU board）和客戶端之間遠端執行實驗和處理數據。
* **即時繪圖**: 在數據採集過程中即時視覺化實驗數據，支援 Jupyter Notebook 。
* **數據分析**: 一套基於 `scqubits` 和 `scikit-learn` ，用於數據處理、擬合（例如諧振、衰減曲線）和進階分析的工具。
* **模擬功能**: 提供模擬量子系統和預測實驗結果的工具。

## 目錄結構

以下是本專案主要目錄的概覽：

* `lib/zcu_tools/`: 工具包的主要原始碼。
  * `experiment/`: 包含定義和運行不同類型實驗的邏輯。
  * `program/`: 處理 QICK SOC 的底層編程。
  * `liveplot/`: 提供即時數據視覺化功能。
  * `notebook/`: 包含在 Jupyter 環境中進行數據分析、擬合和持久化的工具。
  * `remote/`: 使用 Pyro5 實現客戶端-伺服器架構。
  * `simulate/`: 包含模擬量子系統的程式碼。
* `Database/`: 儲存原始實驗數據的預設目錄。
* `notebook_md/`: 包含保存為 Markdown 格式的筆記本，常用於分析和文檔記錄。
* `script/`: 存放工具腳本，例如用於啟動 Pyro 伺服器（`pyro_server.py`）。

## 安裝指南

ZCU-Tools 可以根據您的不同需求安裝不同的依賴項。

### ZCU端

在要連接到 ZCU216 的上運行：

```bash
pip install ".[server]"
# or
uv sync --extra server
```

**注意**: `qick` 庫需要手動安裝，請參考其[官方儲存庫](https://github.com/openquantumhardware/qick.git)。

### 電腦端

#### 數據分析

如果您只需要數據分析：

```bash
pip install ".[fit]"
# or
uv sync --extra fit
```

#### 實驗控制

如果您需要從電腦控制實驗：

```bash
pip install ".[client]"
# or
uv sync --extra client
```

#### 安裝所有功能

```bash
pip install ".[all]"
# or
uv sync --extra all
```

## 基本使用

### 啟動 Pyro 伺服器

若要允許遠端連線到實驗設備，請在ZCU xilinx環境中執行腳本：

```bash
# 或者使用jupyter notebook開啟script/start_server.ipynb
python script/pyro_server.py --port xxx --soc v2
```

這將啟動一個 Pyro5 伺服器，使硬體控制物件可供遠端客戶端使用。

### 建立Jupyter Notebook

```bash
python sync.py md2nb # 建立預設notebook
```

### 開始量測

以下說明專案中幾個常用 Notebook 的典型流程與資料流向，方便你從「量測」一路走到「擬合與物理分析」。

#### 1. 單量子位基礎實驗：`notebook/single_qubit.ipynb`

* **建立結果與資料夾結構**  
  在開頭設定 `chip_name`、`res_name`、`qub_name` 後，Notebook 會：
  * 在 `result/<chip_name>/<qub_name>/` 底下建立結果資料夾（圖檔、擬合結果等）。  
  * 在 `Database/<chip_name>/<qub_name>/...` 底下建立對應的 HDF5 原始數據資料夾（透過 `create_datafolder`）。

* **連線 ZCU216 與量測儀器**  
  * 使用 `zcu_tools.remote.make_soc_proxy()` 與 ZCU216 (SOC) 建立連線，取得 `soc` / `soccfg`。  
  * 使用 `pyvisa` 與 `zcu_tools.device.GlobalDeviceManager` 連線如 `YOKOGS200` 等儀器，並可透過 `dump_device_info` 將設定存成 `device_info.json`。

* **設定通道與實驗參數**  
  * 透過 `MetaDict`（例如 `md.res_ch`, `md.qub_1_4_ch`, `md.ro_ch` 等）統一記錄讀出通道、量子位通道等實驗設定。  
  * 之後的實驗 Notebook（如 `autofluxdep.ipynb`、`overnight.ipynb`）都會重用這些設定。

* **執行標準實驗**  
  * 使用 `zcu_tools.experiment.v2` 中的實驗類別（如 OneTone、Rabi、T1、T2 等）進行掃頻與時間序列量測。  
  * 每個實驗會自動將結果存成 HDF5 檔案到 `Database/...`，並由 `ExperimentManager` 幫你管理實驗標籤與紀錄。

#### 2. 自動磁通掃描：`notebook/autofluxdep.ipynb`

* **載入既有的磁通校正與參數**  
  * 使用 `ExperimentManager.use_flux(...)` 從先前的磁通掃描結果讀入 `ModuleLibrary (ml)` 與 `MetaDict (md)`，並以 `FluxoniumPredictor` 載入 `params.json` 中的參數，作為接下來自動掃描的預測模型。

* **設定磁通掃描與任務組合**  
  * 透過 `FluxDepExecutor` 設定一組磁通掃描點（例如 `flx_values = np.linspace(...)`），並加入多個任務：  
    * `QubitFreqTask`：尋找每個磁通點的量子位共振頻率。  
    * `LenRabiTask`：自動調整 Rabi pulse 長度與增益，以得到穩定的 \(\pi\) 脈衝。  
    * `T1Task`、`T2RamseyTask`、`T2EchoTask`：自動量測壽命與相干時間。  
    * `MistTask`：針對 Mist 讀出進行增益掃描與單次讀出特性量測。

* **執行與資料儲存**  
  * `executor.run(...)` 會實際控制 ZCU216 與儀器進行完整掃描，並把所有資料存入 `Database/...`。  
  * Notebook 會同時建立一個 `*_snapshot` 目錄，內含：  
    * `measure_code.py`：實際執行的測量程式碼快照。  
    * `device_info.json`：量測當下的儀器連線與設定。  
    * `module_cfg.yaml`、`meta_info.json`：對應的模組與實驗參數備份。  
  * `executor.save(...)` 會在 `Database/<chip_name>/<qub_name>/...` 下建立對應的結果檔，並附上文字註解。

#### 3. 長時間穩定性量測：`notebook/overnight.ipynb`

* **重用既有模組與裝置設定**  
  * 同樣透過 `ExperimentManager.use_flux(...)` 與 `reconnect_devices(...)` 讀入先前的 `ml` / `md` 與 `device_info.json`，確保夜間量測與白天校正使用一致的設定。

* **使用 `OvernightExecutor` 週期性量測**  
  * `OvernightExecutor(num_times=..., interval=...)` 會根據你設定的次數與時間間隔，自動重複執行一組單次讀出（Mist 單次讀出）的測量任務。  
  * 典型任務組合包含：  
    * 基態/激發態的 Mist 單次讀出掃描（`mist_g`, `mist_e`）。  
    * 長時間探測讀出穩態（`mist_steady`）等。  
  * 用來追蹤讀出與量子位參數在數小時到一整夜內的漂移情況。

#### 4. 結果擬合與物理分析 Notebook

以下三個 Notebook 主要針對上述量測資料進行擬合與物理解讀：

* **磁通依賴與能階擬合：`notebook/analysis/fluxdep_fit.ipynb`**  
  * 讀取原始 flux 依賴光譜（HDF5，通常由 one-tone / flux 掃描實驗產生），並利用互動工具 `InteractiveLines`、`InteractiveFindPoints` 手動挑選共振點。  
  * 使用事先透過 `script/generate_data.py` 產生的 Fluxonium 模擬資料庫（預設存於 `Database/simulation/fluxonium_*.h5`，內含隨機掃描的 \((E_J, E_C, E_L)\) 參數與對應能階），搭配 `calculate_energy_vs_flx` 搜尋最佳的 \(E_J, E_C, E_L\) 參數，使模擬能階與實驗點對齊。  
  * 產生並儲存：  
    * `result/<qub_name>/params.json`：包含最佳擬合參數、磁通中心 `mA_c`、週期 `period`、以及允許的躍遷集合 `allows`。  
    * `data/fluxdep/spectrums.hdf5` 與 `selected_points.npz`：整理好的光譜與手動選點資料。  
    * 對應的圖檔與網頁（`image/fluxdep_fit/*.png`、`web/fluxdep_fit/*.html`）。

* **讀出腔 Dispersive 擬合：`notebook/analysis/dispersive.ipynb`**  
  * 讀入 `params.json` 與 one-tone flux 掃描資料，先自動校正電纜延遲與 readout IQ 圓圈，得到正規化後的 phase map。  
  * 使用 `search_proper_g` 與 `auto_fit_dispersive` 擬合 readout 腔與量子位的耦合強度 \(g\) 以及腔頻率 \(r_f\)，並與實驗的相位圖疊圖比較。  
  * 輸出：  
    * Dispersive shift 圖（`dispersive.png` / `dispersive.html` 等）。  
    * 更新後的 `params.json`，在 `dispersive` 欄位中寫回最佳的 \(g\) 與 \(r_f\)。

* **T1 曲線與雜訊通道分析：`notebook/analysis/T1_curve.ipynb`**  
  * 讀取 `params.json` 與 `samples.csv`（內含各磁通點的 \(T_1\)、頻率與誤差），將磁通掃描與 T1 資料對齊。  
  * 使用 `scqubits.Fluxonium` 以及多種雜訊通道模型（電容耦合、電感耦合、準粒子、Purcell 等）計算與擬合 \(T_1(\Phi)\)。  
  * 繪製並輸出：  
    * 各雜訊通道對應的 \(Q(\omega)\) 圖與擬合結果。  
    * 實驗點與理論 \(T_1\) 曲線的比較圖（含有效 \(T_1\) 與 Purcell 修正）。  
  * 這些圖會存放在 `result/<qub_name>/image/` 與 `web/` 目錄下，方便後續報告與比對。

#### 5. 典型使用流程總結

1. 使用 `single_qubit.ipynb` 進行初始頻譜掃描與基礎 T1/T2 量測，建立 `result/...` 與 `Database/...` 結構。  
2. 以 `fluxdep_fit.ipynb` 對磁通依賴光譜進行擬合，取得可靠的 Fluxonium 參數與磁通校正。  
3. 利用 `autofluxdep.ipynb` 自動掃描多個磁通點，批次取得 \(f_q(\Phi)\)、\(T_1(\Phi)\)、\(T_2(\Phi)\) 等。  
4. 使用 `dispersive.ipynb` 擬合 readout 腔的 dispersive shift 與耦合強度。  
5. 視需求以 `overnight.ipynb` 進行長時間穩定性量測。  
6. 最後透過 `T1_curve.ipynb` 分析 T1 的限制機制與各雜訊來源，完成從「裝置參數擬合」到「壽命機制分析」的完整流程。

----

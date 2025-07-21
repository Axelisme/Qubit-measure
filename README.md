# ZCU-Tools：一個用於 ZCU216 平台的量子測量工具包

以下為AI生成:

ZCU-Tools 是一個專為 ZCU216 FPGA 平台設計的綜合性 Python 工具包，用於執行和分析量子位元測量。它為實驗控制、數據採集、即時視覺化和數據後處理提供了一個結構化的框架。

## 核心功能

* **實驗自動化**: 提供標準化的類別，用於運行各種量子實驗，如單音和雙音光譜、拉比振盪等。
* **遠端控制**: 基於 Pyro5 構建，允許在伺服器（連接到硬體）和多個客戶端之間遠端執行實驗和處理數據。
* **即時繪圖**: 在數據採集過程中即時視覺化實驗數據，支援 Jupyter Notebook 和獨立腳本。
* **數據分析**: 一套用於數據處理、擬合（例如諧振、衰減曲線）和進階分析的工具，並與 `scqubits` 和 `scikit-learn` 等庫整合。
* **模擬功能**: 提供模擬量子系統和預測實驗結果的工具。
* **模組化硬體編程**: 為 ZCU216 上的 QICK（量子儀器控制套件）提供結構化的編程方法。

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

### 基本安裝

基本安裝包含核心功能所需的依賴：

```bash
pip install .
```

### 伺服器擴展

如果您需要伺服器功能（例如，在連接到 ZCU216 的主機上運行）：

```bash
pip install ".[server]"
```

**注意**: `qick` 庫需要手動安裝，請參考其[官方儲存庫](https://github.com/openquantumhardware/qick.git)。

### 客戶端擴展

如果您需要從遠端客戶端電腦控制實驗：

```bash
pip install ".[client]"
```

**注意**: `labber` 和 `qick` 庫需要手動安裝。

### 數據分析與擬合擴展

如果您需要進階的數據分析和視覺化功能：

```bash
pip install ".[fit]"
```

### 完整安裝

若要安裝所有功能：

```bash
pip install ".[server,client,fit]"
```

## 基本使用

### 啟動 Pyro 伺服器

若要允許遠端連線到實驗設備，請執行伺服器腳本：

```bash
python script/pyro_server.py
```

這將啟動一個 Pyro5 伺服器，使硬體控制物件可供遠端客戶端使用。

---

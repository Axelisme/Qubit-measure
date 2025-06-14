# ZCU-Tools

ZCU-Tools 是一個專為 ZCU216 平台上的量子比特測量設計的工具包。

## 安裝指南

ZCU-Tools 可以根據您的不同需求安裝不同的依賴項。

### 基本安裝

基本安裝包含核心功能所需的依賴：

```bash
pip install .
```

這將安裝以下核心依賴：

- numpy
- tqdm
- matplotlib

### ZCU伺服器擴展

如果您需要伺服器功能，可以使用：

```bash
pip install ".[server]"
```

這將安裝基本依賴外加：

- myqick
- dill

其中依賴`myqick`需要自行手動安裝，請參考[myqick](https://github.com/Axelisme/myqick)。

### 客戶端工具擴展

如果您需要客戶端功能（遠端測量），可以使用：

```bash
pip install ".[client]"
```

這將安裝以下依賴：

- myqick
- labber
- numpy (<=1.19.5)
- h5py
- ipykernel
- ipywidgets
- ipympl
- pyyaml
- pandas
- scikit-learn
- scipy
- dill

其中依賴`labber`和`myqick`需要自行手動安裝，請參考[labber](https://github.com/Axelisme/labber_api)和[myqick](https://github.com/Axelisme/myqick)。

### 數據擬合擴展

如果您需要進階的數據擬合和分析功能（包括可視化和數據處理），可以使用：

```bash
pip install ".[fit]"
```

這將安裝以下依賴：

- h5py
- ipykernel
- ipywidgets
- ipympl
- pandas
- scqubits
- plotly
- kaleido
- scipy
- nbformat (>=4.2.0)
- joblib
- numba
- requests

### 全部功能安裝

如果您想安裝所有功能，可以使用：

```bash
pip install ".[server,client,fit]"
```

## 系統需求

- Python 3.8 或更高版本

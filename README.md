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
- myqick
- h5py
- numpy
- tqdm

### 資料伺服器擴展

如果您需要資料伺服器功能，可以使用以下命令安裝相關依賴：

```bash
pip install ".[data]"
```

這將安裝基本依賴外加：
- flask

### 客戶端工具擴展

如果您需要客戶端功能（遠端測量），可以使用：

```bash
pip install ".[client]"
```

這將安裝以下依賴：
- labber
- ipykernel
- ipywidgets
- matplotlib
- pandas
- scikit-learn
- scipy

### 數據擬合擴展

如果您需要進階的數據擬合和分析功能（包括可視化和數據處理），可以使用：

```bash
pip install ".[fit]"
```

這將安裝以下依賴：
- scqubits
- matplotlib
- plotly
- kaleido
- scipy
- ipympl
- nbformat (>=4.2.0)
- joblib

### 全部功能安裝

如果您想安裝所有功能，可以使用：

```bash
pip install ".[data,client,fit]"
```

## 系統需求

- Python 3.8 或更高版本

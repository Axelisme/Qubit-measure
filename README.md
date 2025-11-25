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

## 上傳實驗結果至 Google Drive (OAuth 2.0)

為了更方便地將實驗結果上傳到您個人的 Google Drive，此腳本已更新為使用 **OAuth 2.0** 進行身份驗證。這種方法會透過瀏覽器授權，將檔案直接上傳到您的帳戶，避免了服務帳號 (Service Account) 的儲存空間配額問題。

### 功能

*   **直接上傳**: 將檔案直接上傳到您的個人 Google Drive 帳戶。
*   **一次性授權**: 首次執行時，只需在瀏覽器中授權一次，後續腳本會自動刷新憑證。
*   **斷點續傳**: 上傳過程支援斷點續傳，適合大型檔案。

### 設定指南

#### 1. 安裝額外依賴

如果您尚未安裝，請安裝 `gdrive` 擴展套件：

```bash
pip install ".[gdrive]"
```

#### 2. 設定 Google OAuth 2.0 憑證

1.  **啟用 Google Drive API**:
    *   前往 [Google Cloud Console](https://console.cloud.google.com/)。
    *   確定您已選擇或建立一個專案。
    *   在 API Library 中搜尋 "Google Drive API" 並啟用它。

2.  **設定 OAuth 同意畫面 (Consent Screen)**:
    *   前往 "APIs & Services" > "OAuth consent screen"。
    *   選擇使用者類型為 **External**，然後點擊 "Create"。
    *   填寫必要的應用程式資訊（應用程式名稱、使用者支援電子郵件等）。
    *   在 Scopes 和 Test users 頁面，您可以直接點擊 "Save and Continue" 跳過，因為此腳本在開發模式下運作良好。

3.  **建立 OAuth 2.0 用戶端 ID (Client ID)**:
    *   前往 "APIs & Services" > "Credentials"。
    *   點擊 "Create Credentials"，然後選擇 **OAuth client ID**。
    *   在 "Application type" 中，選擇 **Desktop app**。
    *   為它命名，然後點擊 "Create"。
    *   一個包含 Client ID 和 Client Secret 的視窗會彈出。點擊 **DOWNLOAD JSON**。
    *   將下載的檔案重新命名為 `client_secret.json` 並將其放置在專案的根目錄下，或您選擇的其他安全位置。

#### 3. 設定環境變數

在專案的根目錄下，根據 `.env.example` 檔案建立一個 `.env` 檔案，並填入以下資訊：

```env
# .env
# 指向您下載的 OAuth 2.0 client secret JSON 檔案的路徑
GOOGLE_CLIENT_SECRET_FILE="client_secret.json"

# 貼上您希望將結果上傳到的 Google Drive 目標資料夾的 ID
# 例如，網址 https://drive.google.com/drive/folders/SOME_LONG_ID 的 ID 就是 SOME_LONG_ID
GOOGLE_DRIVE_PARENT_FOLDER_ID="your-google-drive-folder-id"
```

### 使用方法

#### 首次執行 (授權)

第一次執行腳本時，它會：
1.  自動在您的預設網頁瀏覽器中開啟一個新的分頁。
2.  要求您登入您的 Google 帳戶。
3.  要求您授權該應用程式存取您的 Google Drive。
4.  授權完成後，腳本會在本機專案目錄下自動建立一個 `token.json` 檔案。此檔案儲存了您的授權憑證，以便未來執行時不再需要手動授權。

**重要提示**: 請確保不要將 `token.json` 檔案分享或上傳到公開的程式碼儲存庫。

#### 上傳指令

授權完成後，您可以通過以下指令上傳結果。假設您要上傳 `result/Si001` 目錄下的所有內容：

```bash
python script/upload_result.py Si001
```

腳本會自動在您指定的 Google Drive 父資料夾中建立一個名為 `Si001` 的新資料夾，並將本地的所有內容複製過去。


# Jupytext 工作流程說明

## 核心理念

- **主要編輯區**: `notebook/` 目錄下的 `.ipynb` 檔案。
- **版本控制區**: `notebook_md/` 目錄下的 `.md` 檔案。
- **同步工具**: `sync.py` Python 腳本。

## 檔案結構

```text
├── notebook/              # 在此編輯 .ipynb 檔案
│   └── ...
├── notebook_md/           # .md 檔案，用於版本控制
│   └── ...
└── sync.py                # 跨平台的同步腳本
```

---

## 工作流程

### 步驟一: 日常開發與提交變更 (ipynb -> md)

完成 Notebook 編輯後，將變更同步到 Markdown 以便提交。

**1. 編輯 Notebook:**

- 在 `notebook/` 目錄下編輯 `.ipynb` 檔案。

**2. 執行同步腳本:**

- 在終端機中執行以下指令：

  ```bash
  python sync.py nb2md
  ```

- 腳本會逐一檢查每個 Notebook。如果目標 Markdown 不存在，會**自動建立**。如果已存在但內容不一致，它會**停下來詢問您是否要同步該檔案**。

**3. 提交到 Git:**

- 在確認並同步完所有必要的變更後，提交 `notebook_md/` 目錄。
- `git add notebook_md/`
- `git commit -m "更新 notebook"`
- `git push`

### 步驟二: 從版本庫更新 (md -> ipynb)

從 git 拉取更新後，將 `.md` 檔案的變更同步回您的 `.ipynb` 工作區。

**1. 拉取 Git 更新:**

- `git pull`
- 如果有合併衝突，請在 `notebook_md/` 目錄下的 `.md` 檔案中手動解決。

**2. 執行同步腳本:**

- 在終端機中執行以下指令：

  ```bash
  python sync.py md2nb
  ```

- 腳本會逐一檢查每個 Markdown。如果目標 Notebook 不存在，會**自動建立**。如果已存在但內容不一致，它會**停下來詢問您是否要覆寫本地的 Notebook**。

**3. 繼續工作:**

- 在同步完所有遠端變更後，您 `notebook/` 中的檔案已是最新狀態，可以安全地繼續開發。

---

## 版本控制

- ✅ **追蹤**: `notebook_md/`
- ❌ **忽略**: `notebook/` (請確保已加入 `.gitignore`)

## 注意事項

- **停用自動同步**: 請確保 `pyproject.toml` 或其他 Jupytext 設定中已停用自動配對同步，以避免與本手動流程衝突。
- **環境**: 執行腳本前，請確認已啟用包含 `jupytext` 的 Python/Conda 環境。

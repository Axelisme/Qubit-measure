# Jupytext 工作流程說明

## 核心理念

- **主要編輯區**: `notebook/` 目錄下的 `.ipynb` 檔案。
- **版本控制區**: `notebook_md/` 目錄下的 `.md` 檔案。

## 檔案結構

```text
├── notebook/              # 在此編輯 .ipynb 檔案
│   └── ...
├── notebook_md/           # .md 檔案，用於版本控制
│   └── ...
├── sync_nb2md.sh          # (Linux/macOS) 同步 ipynb -> md
├── sync_nb2md.cmd         # (Windows)     同步 ipynb -> md
├── sync_md2nb.sh          # (Linux/macOS) 同步 md -> ipynb
└── sync_md2nb.cmd         # (Windows)     同步 md -> ipynb
```

---

## 工作流程

### 步驟一: 日常開發與提交變更 (ipynb -> md)

完成 Notebook 編輯後，將變更同步到 Markdown 以便提交。

**1. 編輯 Notebook:**

- 在 `notebook/` 目錄下編輯 `.ipynb` 檔案。

**2. 執行同步腳本:**

- **Linux/macOS**:

     ```bash
     ./sync_nb2md.sh
     ```

- **Windows**:

     ```cmd
     .\sync_nb2md.cmd
     ```

- 此腳本會將 `notebook/` 的內容單向同步到 `notebook_md/`。

**3. 提交到 Git:**

- `git status` 檢查 `notebook_md/` 中的變更。
- `git add notebook_md/`
- `git commit -m "更新 notebook"`
- `git push`

### 步驟二: 從版本庫更新 (md -> ipynb)

從 git 拉取更新後，將 `.md` 檔案的變更同步回您的 `.ipynb` 工作區。

**1. 拉取 Git 更新:**

- `git pull`
- 如果有合併衝突，請在 `notebook_md/` 目錄下的 `.md` 檔案中手動解決。

**2. 執行同步腳本:**

- **Linux/macOS**:

     ```bash
     ./sync_md2nb.sh
     ```

- **Windows**:

     ```cmd
     .\sync_md2nb.cmd
     ```

- 此腳本會安全地將 `notebook_md/` 的內容同步回 `notebook/`，並在偵測到衝突時提供警告與選項。

**3. 繼續工作:**

- 現在 `notebook/` 中的檔案已是最新狀態，可以安全地繼續開發。

---

## 版本控制

- ✅ **追蹤**: `notebook_md/`
- ❌ **忽略**: `notebook/` (請確保已加入 `.gitignore`)

## 注意事項

- **停用自動同步**: 請確保 `pyproject.toml` 或其他 Jupytext 設定中已停用自動配對同步，以避免與本手動流程衝突。
- **賦予執行權限 (Linux/macOS)**: 首次使用時，需要為 `.sh` 腳本添加執行權限：

  ```bash
  chmod +x sync_nb2md.sh
  chmod +x sync_md2nb.sh
  ```

- **環境**: 執行腳本前，請確認已啟用包含 `jupytext` 的 Python/Conda 環境。

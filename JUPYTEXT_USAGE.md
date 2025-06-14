# Jupytext 使用說明

## 檔案結構

```
├── notebook_md/           # Markdown 原始檔案 (版本控制)
│   ├── single_qubit.md
│   ├── specfic.md
│   └── analysis/
│       ├── T1_curve.md
│       ├── design.md
│       └── ...
├── notebook/              # 生成的 Jupyter notebook (不版本控制)
│   ├── single_qubit.ipynb
│   ├── specfic.ipynb
│   └── analysis/
└── sync_notebooks.sh      # 同步腳本
```

## 工作流程

### 1. 環境準備
```bash
# 激活有 jupytext 的環境
mmba activate qb13  # 或其他有 jupytext 的環境
```

### 2. 編輯 Markdown 檔案
- 編輯 `notebook_md/` 目錄下的 `.md` 檔案
- 使用任何文本編輯器或 IDE

### 3. 同步到 Jupyter Notebook
```bash
# 執行同步腳本
./sync_notebooks.sh
```

### 4. 在 Jupyter 中使用
- 打開 `notebook/` 目錄下的 `.ipynb` 檔案
- 執行、查看輸出
- **不要直接編輯 .ipynb 檔案**

## 安全機制

同步腳本包含以下安全機制：

1. **環境檢查**: 確認 jupytext 可用
2. **衝突檢測**: 檢查是否有本地 .ipynb 修改
3. **備份選項**: 可選擇備份現有 .ipynb 檔案
4. **差異預覽**: 可查看變更內容再決定是否同步

## 版本控制

- ✅ **追蹤**: `notebook_md/` 目錄下的 `.md` 檔案
- ❌ **忽略**: `notebook/` 目錄下的 `.ipynb` 檔案
- ❌ **忽略**: `backup_ipynb_*/` 備份目錄

## 手動命令

如果需要手動操作：

```bash
# 單個檔案同步
jupytext --sync notebook_md/single_qubit.md

# 查看差異
jupytext --diff notebook_md/single_qubit.md

# 強制從 .md 生成 .ipynb
jupytext --to ipynb notebook_md/single_qubit.md --output notebook/single_qubit.ipynb
```

## 注意事項

1. **只編輯 .md 檔案**: 所有修改都應該在 `notebook_md/` 目錄進行
2. **定期同步**: 編輯後記得執行同步腳本
3. **環境一致性**: 確保使用正確的 Python 環境
4. **備份重要**: 如有疑慮，選擇備份選項 
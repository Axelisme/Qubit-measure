#!/bin/bash
# sync_nb2md.sh - 將 Jupyter Notebooks 同步到 Markdown (.ipynb -> .md)

set -e

echo "=== Sync Notebooks to Markdown (.ipynb -> .md) ==="
echo "此腳本用於在開發後，將 .ipynb 的變更同步到 .md 以便版本控制。"
echo

# 檢查 jupytext
if ! command -v jupytext &> /dev/null; then
    echo "❌ 錯誤: 找不到 jupytext！請先啟用對應的環境。"
    exit 1
fi

echo "✅ 使用 jupytext: $(which jupytext)"

# 衝突檢查函式: 檢查 .md 是否比 .ipynb 新
check_conflicts() {
    local ipynb_file="$1"
    local md_file="${ipynb_file/notebook/notebook_md}"
    md_file="${md_file/.ipynb/.md}"
    
    if [[ -f "$md_file" && "$md_file" -nt "$ipynb_file" ]]; then
        echo "⚠️  警告: Markdown '$md_file' 比 Notebook '$ipynb_file' 更新。"
        echo "   這可能表示您有來自 git 的變更尚未同步到 Notebook。"
        echo "   建議先執行 sync_md2nb.sh 將變更同步回來。"
        return 1
    fi
    return 0
}

# 遍歷所有 .ipynb 檔案檢查衝突
conflicts_found=false
echo "🔎 正在檢查衝突..."
all_ipynb_files=$(find notebook -name "*.ipynb")

if [ -z "$all_ipynb_files" ]; then
    echo "ℹ️ 在 'notebook/' 目錄中找不到任何 .ipynb 檔案。"
    exit 0
fi

while read -r ipynb_file; do
    if ! check_conflicts "$ipynb_file"; then
        conflicts_found=true
    fi
done <<< "$all_ipynb_files"

if $conflicts_found; then
    echo
    echo "❗️ 發現衝突！繼續操作可能會覆蓋 .md 中的重要變更。"
    read -p "您確定要強制從 .ipynb 覆蓋 .md 嗎？ (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "同步已取消。"
        exit 0
    fi
fi

# 執行同步
echo "🔄 正在同步 .ipynb -> .md..."
mkdir -p notebook_md/analysis

while read -r ipynb_file; do
    path_in_notebook_dir=${ipynb_file#notebook/}
    md_file="notebook_md/${path_in_notebook_dir%.ipynb}.md"
    
    mkdir -p "$(dirname "$md_file")"
    echo "   - $(basename "$ipynb_file") -> $(basename "$md_file")"
    
    # 使用 --to md 進行單向同步
    jupytext --to md "$ipynb_file" --output "$md_file"
done <<< "$all_ipynb_files"

echo
echo "✅ 同步完成！"
echo "現在您可以提交 notebook_md/ 中的變更了。" 
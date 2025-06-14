#!/bin/bash
# sync_md2nb.sh - 從 Markdown 更新 Jupyter Notebooks (.md -> .ipynb)

set -e

echo "=== Sync Notebooks from Markdown (.md -> .ipynb) ==="
echo "此腳本用於在更新 .md 檔案 (例如 git pull) 後，將變更同步到 .ipynb 檔案。"
echo

# 檢查 jupytext
if ! command -v jupytext &> /dev/null; then
    echo "❌ 錯誤: 找不到 jupytext！請先啟用對應的環境。"
    exit 1
fi

echo "✅ 使用 jupytext: $(which jupytext)"

# 衝突檢查函式: 檢查 .ipynb 是否比 .md 新
check_conflicts() {
    local md_file="$1"
    local ipynb_file="${md_file/notebook_md/notebook}"
    ipynb_file="${ipynb_file/.md/.ipynb}"
    
    if [[ -f "$ipynb_file" && "$ipynb_file" -nt "$md_file" ]]; then
        echo "⚠️  警告: Notebook '$ipynb_file' 比 Markdown '$md_file' 更新。"
        echo "   這表示您在 .ipynb 中有未同步的變更，可能會被覆蓋。"
        return 1
    fi
    return 0
}

# 遍歷所有 .md 檔案檢查衝突
conflicts_found=false
echo "🔎 正在檢查衝突..."
all_md_files=$(find notebook_md -name "*.md")

if [ -z "$all_md_files" ]; then
    echo "ℹ️ 在 'notebook_md/' 目錄中找不到任何 .md 檔案。"
    exit 0
fi

while read -r md_file; do
    if ! check_conflicts "$md_file"; then
        conflicts_found=true
    fi
done <<< "$all_md_files"

# 如果發現衝突，提供選項
if $conflicts_found; then
    echo
    echo "❗️ 發現衝突！您的部分 .ipynb 檔案有尚未同步的本機變更。"
    echo "建議先執行 sync_nb2md.sh 來同步這些變更。"
    echo
    echo "請選擇操作："
    echo "1. 中止同步 (建議)"
    echo "2. 強制同步 (放棄 .ipynb 的變更，從 .md 覆蓋)"
    echo "3. 備份並同步 (將現有 .ipynb 備份後再從 .md 覆蓋)"
    read -p "請選擇 (1/2/3): " -n 1 -r
    echo
    
    case $REPLY in
        1) echo "同步已取消。" && exit 0 ;;
        2) echo "將強制從 .md 同步..." ;;
        3)
            backup_dir="backup_ipynb_$(date +%Y%m%d_%H%M%S)"
            mkdir -p "$backup_dir"
            echo "正在備份 .ipynb 檔案到 '$backup_dir'..."
            find notebook -name "*.ipynb" -exec cp --parents -t "$backup_dir" {} +
            echo "✅ 備份完成。繼續同步..."
            ;;
        *) echo "無效選擇，離開。" && exit 1 ;;
    esac
fi

# 執行同步
echo "🔄 正在同步 .md -> .ipynb..."
mkdir -p notebook/analysis

while read -r md_file; do
    path_in_md_dir=${md_file#notebook_md/}
    ipynb_file="notebook/${path_in_md_dir%.md}.ipynb"
    
    mkdir -p "$(dirname "$ipynb_file")"
    echo "   - $(basename "$md_file") -> $(basename "$ipynb_file")"
    
    # 使用 --to ipynb 進行單向覆蓋同步
    jupytext --to ipynb "$md_file" --output "$ipynb_file"
done <<< "$all_md_files"

echo
echo "✅ 同步完成！"
echo "您的 Jupyter notebooks 已從 Markdown 來源更新。" 
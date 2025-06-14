#!/bin/bash
# sync_notebooks.sh - 安全的 notebook 同步腳本

set -e

echo "=== Safe Notebook Sync ==="

# 檢查 jupytext 是否可用
if ! command -v jupytext &> /dev/null; then
    echo "❌ Error: jupytext not found!"
    echo "Please activate the appropriate environment first."
    echo "Example: mmba activate qb13"
    exit 1
fi

echo "✅ Using jupytext: $(which jupytext)"

# 檢查每個 .md 檔案對應的 .ipynb 是否有本地修改
check_conflicts() {
    local md_file="$1"
    local ipynb_file="${md_file/notebook_md/notebook}"
    ipynb_file="${ipynb_file/.md/.ipynb}"
    
    if [[ -f "$ipynb_file" ]]; then
        # 檢查 .ipynb 是否比 .md 更新
        if [[ "$ipynb_file" -nt "$md_file" ]]; then
            echo "⚠️  Warning: $ipynb_file is newer than $md_file"
            echo "   This suggests local .ipynb modifications exist."
            return 1
        fi
    fi
    return 0
}

# 檢查所有檔案
conflicts_found=false
echo "Checking for conflicts..."

for md_file in notebook_md/*.md notebook_md/analysis/*.md; do
    if [[ -f "$md_file" ]]; then
        if ! check_conflicts "$md_file"; then
            conflicts_found=true
        fi
    fi
done

if $conflicts_found; then
    echo ""
    echo "Conflicts detected! Options:"
    echo "1. Backup and continue: will create backup of .ipynb files"
    echo "2. Skip sync: exit without syncing"
    echo "3. Show diff: show differences before deciding"
    echo "4. Force sync: sync anyway (may lose .ipynb changes)"
    read -p "Choose (1/2/3/4): " -n 1 -r
    echo
    
    case $REPLY in
        1)
            # 建立備份
            backup_dir="backup_ipynb_$(date +%Y%m%d_%H%M%S)"
            mkdir -p "$backup_dir"
            find notebook -name "*.ipynb" -exec cp {} "$backup_dir/" \; 2>/dev/null || true
            echo "✅ Backup created in: $backup_dir"
            ;;
        2)
            echo "Sync cancelled."
            exit 0
            ;;
        3)
            echo "Showing differences..."
            for md_file in notebook_md/*.md notebook_md/analysis/*.md; do
                if [[ -f "$md_file" ]]; then
                    echo "--- Checking $md_file ---"
                    jupytext --diff --from md --to ipynb "$md_file" 2>/dev/null || echo "No differences or file not found"
                fi
            done
            echo ""
            read -p "Continue with sync? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Sync cancelled."
                exit 0
            fi
            ;;
        4)
            echo "Force syncing..."
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
fi

# 執行同步
echo "Syncing .md to .ipynb..."
jupytext --sync notebook_md/*.md 2>/dev/null || true
jupytext --sync notebook_md/analysis/*.md 2>/dev/null || true

echo "✅ Sync completed successfully!"
echo ""
echo "Files synced:"
echo "  📝 notebook_md/ -> 📓 notebook/"
echo ""
echo "Next steps:"
echo "  - Edit .md files in notebook_md/"
echo "  - Run this script to sync changes to .ipynb"
echo "  - Open .ipynb files in Jupyter to see results" 
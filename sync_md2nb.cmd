@echo off
setlocal

:: sync_md2nb.cmd - 從 Markdown 更新 Jupyter Notebooks (.md -> .ipynb)
echo === Sync Notebooks from Markdown (.md ^-> .ipynb) ===
echo 此腳本用於在更新 .md 檔案 (例如 git pull) 後，將變更同步到 .ipynb 檔案。
echo.

:: 檢查 jupytext
where jupytext >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ 錯誤: 找不到 jupytext！請先啟用對應的環境。
    exit /b 1
)

echo ✅ 使用 jupytext:
where jupytext
echo.

:: 警告並請求確認
echo ❗️ 警告: 此操作將會用 'notebook_md\' 的內容覆蓋 'notebook\' 中的 .ipynb 檔案。
echo    這會清除您在 .ipynb 中任何未同步到 .md 的變更。
set /p "choice=您確定要繼續嗎？ (y/N): "
if /i not "%choice%"=="y" (
    echo 同步已取消。
    exit /b 0
)

echo.
echo 🔄 正在同步 .md -> .ipynb...

:: 確保目標目錄存在
if not exist "notebook\analysis" mkdir "notebook\analysis"

:: 遍歷 notebook_md 目錄下的所有 .md 檔案
for /r "notebook_md" %%F in (*.md) do (
    call :process_file "%%F"
)

goto :eof

:process_file
    set "md_path=%~1"
    
    :: 替換路徑
    set "ipynb_path=%md_path:notebook_md\=notebook\%"
    set "ipynb_path=%ipynb_path:.md=.ipynb%"
    
    :: 確保目標子目錄存在
    for %%D in ("%ipynb_path%") do (
        if not exist "%%~dpD" mkdir "%%~dpD"
    )
    
    echo    - %~nx1 -> %ipynb_path:notebook\=%
    
    :: 執行同步
    jupytext --to ipynb "%md_path%" --output "%ipynb_path%"
    goto :eof

:eof
echo.
echo ✅ 同步完成！
echo 您的 Jupyter notebooks 已從 Markdown 來源更新。
endlocal 
@echo off
setlocal

:: sync_nb2md.cmd - 將 Jupyter Notebooks 同步到 Markdown (.ipynb -> .md)
echo === Sync Notebooks to Markdown (.ipynb ^-> .md) ===
echo 此腳本用於在開發後，將 .ipynb 的變更同步到 .md 以便版本控制。
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
echo ❗️ 警告: 此操作將會用 'notebook\' 中的 .ipynb 內容覆蓋 'notebook_md\' 中的 .md 檔案。
set /p "choice=您確定要繼續嗎？ (y/N): "
if /i not "%choice%"=="y" (
    echo 同步已取消。
    exit /b 0
)

echo.
echo 🔄 正在同步 .ipynb -> .md...

:: 確保目標目錄存在
if not exist "notebook_md\analysis" mkdir "notebook_md\analysis"

:: 遍歷 notebook 目錄下的所有 .ipynb 檔案
for /r "notebook" %%F in (*.ipynb) do (
    call :process_file "%%F"
)

goto :eof

:process_file
    set "ipynb_path=%~1"
    
    :: 替換路徑
    set "md_path=%ipynb_path:notebook\=notebook_md\%"
    set "md_path=%md_path:.ipynb=.md%"
    
    :: 確保目標子目錄存在
    for %%D in ("%md_path%") do (
        if not exist "%%~dpD" mkdir "%%~dpD"
    )
    
    echo    - %~nx1 -> %md_path:notebook_md\=%
    
    :: 執行同步
    jupytext --to md "%ipynb_path%" --output "%md_path%"
    goto :eof

:eof
echo.
echo ✅ 同步完成！
echo 現在您可以提交 notebook_md\ 中的變更了。
endlocal 
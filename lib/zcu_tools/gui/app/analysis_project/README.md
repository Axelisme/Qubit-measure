# Analysis entry selection

**Last updated:** 2026-10-28

**讀者：** framework-dev、agent。兩個分析 GUI 共用的 entry/component 選擇與 native 光譜載入接口。

`entry.AnalysisEntry.open(project_root, entry)` 只開 entry 並列合法 components，不解析 alias。
沒有合法 component 時報錯。`.select(component=None)` 選目標，None 才解析 qubit alias。
`AnalysisEntry.list_names(project_root)` 供 dialog 列 entry，同一個 owner 推導 Result root。
`selection.AnalysisSelection(project_root, entry="", component=None)` 保存未解析的選擇，不讀檔。
空 entry 讓 behavior 保持未開啟，component 沒有 entry 時報錯。
`entry.AnalysisTarget.open(selection)` 委派上述兩步，使用
`selection.project_root/result` 與 `selection.project_root/Database` 開啟既有 ResultEntry。
此入口要求非空 entry；兩個 analysis behavior 在註冊 lab components 後才呼叫它。
Component 省略時，以 AliasTable 解析 setup 的 qubit alias。
缺 entry、component 或未綁 alias 會報錯。Kind 必須是 qubit/fluxonium 或其子 kind。

AnalysisTarget 保留 absolute result_root、entry、component 與 ResultEntry resource。
`predictor_path` 回傳選定 component 的 predictor JSON 路徑。
分析只使用 setup、predictor/dispersive 路徑與 ledger，不呼叫 Database Save。
`target_payload` 產生 entry/component/Result 路徑投影，未開 entry 時欄位為 null。
`require_target` 在未開 entry 時以 no_entry 拒絕寫入。

`entry_dialog.EntryDialog` 提供 Project root、Entry 與 Component 選單。
Open entry 讀入全部合法 component，有效 qubit alias 只作預選。
Alias 缺或錯時顯示原因並保留空選擇，OK disabled，使用者明示選定後才可提交。
沒有合法 component 時顯示錯誤。不自選第一個，不補 alias，不安裝失敗候選。
`selected_target()` 回傳所選 target。`set_writable` 停用編輯與 OK，保留 Cancel。

`launcher.add_analysis_entry_cli_options` 加入 --project-root、--entry、--component。
`analysis_selection_from_args` 只將 CLI 選項轉成 selection，不開啟 entry。
Component 沒有 entry 時報錯。兩個 analysis behavior 接收 lab 與 selection，
先呼叫 lab 的 registrar，再開啟 target。Framework 不 import lab。

`flux.ComponentFluxProvider` 接受 `ComponentSchema`，回傳 analysis 的 `ComponentFluxContext`。
`component_flux_context(target, provider)` 從 target 的公開 snapshot 取得 component，再呼叫 provider。
這個入口要求已開 target。未注入或 provider 拒絕時，錯誤指名 component 與 kind。
`frame=None` 表示 offset 尚未校準，不補零值 frame。Context 提供儀器選項、預設名稱與顯示單位。
組合根注入 lab provider，framework 不解讀 lab schema 的欄位。

`native.read_spectrum_source` 按檔案 header 讀取 Container 或 LabberData。
宣告為 native 的檔案不在失敗後改讀 Labber。Container 保留同一來源的陣列與 metadata。
`NativeSpectrumSelection` 明示 variable、device_axis 與 device，不讀實驗 cfg。
`native_dialog.NativeSpectrumDialog` 提供變數／軸預選、儀器選擇與預覽。
Caller 經 `NativeDeviceChoices` 傳入儀器單位、預設名稱與可見錯誤。
共用 dialog 不取得 component 或 predictor。Fluxdep 自己保留 Type、Inherit 與 alignment overlay。
Dispersive 以所選 predictor 提供儀器清單，fluxdep 以注入的 component context 提供。

這個 owner 不提供 chip/qubit project、result-scope 掃描或 params.json CLI。
Notebook 的既有 params.json owner 不屬於此 package，兩個分析 app 不使用它。

共用 GUI、session、remote、measure 與 workflows 不依賴這個分析 owner。
方向由 shared-no-analysis-project contract 保護，見 ADR-0072。

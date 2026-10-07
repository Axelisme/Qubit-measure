---
name: run-fluxdep-gui
description: 透過 fluxdep-gui MCP 分析既有 flux spectra，載入、定線、選點、跨譜篩選、DB search 或匯出；恢復 GUI 分析與接手使用者修改時使用。
---

# Run Fluxdep GUI

透過 MCP 操作同一個 GUI analysis owner。這個流程不操作儀器。

## 開始與恢復

1. 確認分析目標、可讀的原始資料、search database 與可寫的輸出路徑。原始資料保留不變。若背景不足以判斷 axes／資料型別，先問使用者。
2. 第一次使用先讀 MCP server instructions。呼叫 fluxdep_connect 接既有 GUI；只有需要新 GUI 且取得授權時才 fluxdep_launch。Disconnect 或 MCP server 結束都保留 GUI。
3. 核對 fluxdep_state_check、fluxdep_project_info、fluxdep_spectrum_list、fluxdep_fit_result。讀每個現有 spectrum 的 fluxdep_spectrum_snapshot，再讀 fluxdep_selection_snapshot。恢復時以 GUI 現況為準，不從舊圖或記錄推定身份、完成或 fit。
4. 呼叫 fluxdep_interactive_read 看現有 input，核對 context_id、kind、spectrum_name、state、commands、can_undo 與 image。Inactive 回 null，不會自動開 editor。完成條件是目前資料、published resources 與可用 input 都已核對。

## Observations 與 GUI 接手

完整 published-resource reads 建立本連線的寫入 baseline。State check、pointcloud、interactive read/open/PNG、operation status/await 都不代替完整讀取。工具不隱藏預讀或重試。

Load 要先讀 project、collection 與全部 current spectra。Picker 寫入要先讀精確 spectrum snapshot。Joint selection 要先讀 collection、全部 spectra（含零點）與 selection。Fit set_params 要先讀 fit_result；search 要先讀 project、fit、collection、全部 spectra 與 selection。Export 按工具宣告讀其依賴。Remove 或更名後，曾觀察的 retired names 若出現在 stale 診斷，也須明示讀 absence snapshot。

使用者可隨時操作 GUI。收到 stale／identity mismatch 時停下 mutation，讀現況與相關 full snapshots，再決定下一步。Open 可重用有效 identity，context_id 不是 edit revision。Closed receipt 描述舊 identity，不是同步 successor；要看新 input 時另呼叫 interactive_read。

## 載入、定線與選點

1. 需要時以 fluxdep_project_setup 套 project。Project database_path 是 raw-data root，不是 fit search file。
2. 以 fluxdep_spectrum_load 載入 OneTone／TwoTone；需要 processed restore 時用 fluxdep_spectrum_load_processed。譜名是 basename 的 literal identity。Load 可 replacement，同名不是新 baseline；成功的新譜也要明示讀 snapshot。
3. 讀 collection 後用 fluxdep_spectrum_set_active 選定譜。讀該 spectrum snapshot，再用 fluxdep_spectrum_interactive_open(name, kind) 開 line／onetone／twotone input。
4. 從 image 的 axes、刻度、單位與 native state 判斷座標。頻率使用 GHz；device 軸依原始資料與圖上 labels 核對。筆寬是 normalized radius，不是物理寬度或 diameter。送 stroke 前先從當前 commands 的 schema 確認頂點參數、mode 與 width，不猜 command 內部欄位。
5. 用 fluxdep_spectrum_interactive_command 傳 literal name、當前 context_id、command 與其 params。每次核對 native effect.changes 與同次 capture 的 image，再選下一個動作。
6. can_undo=true 時可送參數為空的 undo，回上一次 committed change；不是多層歷史。確認圖與差量後再繼續。Picker 的 finish 發布結果，cancel 關閉 input。空 Finish 仍是已完成選點；零點不代表可供 fit 的資料。
7. 要重做 published alignment／points，用 reset_alignment／reset_points；先讀 snapshot，成功後重新核對 native state，不用 cancel 冒充 reset。

PNG 在 MCP image content。文字 context.figure 只有 MIME/bytes metadata，不是持久路徑。圖像解碼或傳輸失敗可能發生在 publication 之後；先讀現況，不重送原命令。

## 跨譜篩選與搜尋

1. 所有需要的譜完成後，讀 collection、全部 spectrum snapshots 與 selection_snapshot。用 fluxdep_selection_interactive_open 開 joint input。依 commands 的 schema 做 stroke／settings／undo；apply 發布但保留 input，cancel 關閉。Input selected mask 與 downsample 後 selected_count 不一定相同。
2. 用 fluxdep_fit_set_params 套完整 fit inputs。EJb／ECb／ELb 是 finite numeric bounds（GHz）；transitions 沿工具 schema 與 GUI validation。Fit database_path 是 search database file。
3. 完整讀 search 所需資源後呼叫 fluxdep_fit_search，保留回傳的 app token。它與 context_id 不同。Operation status 不帶 token 可找 GUI／agent latest activity，明示 unknown token 是錯誤。
4. 用 fluxdep_operation_await(token, timeout) 接手，timeout 範圍 0–30 秒。Timeout 不取消，不建立 observation。Failed outcome 是搜尋結果，不是 invocation error；不要因此重啟。用 fluxdep_operation_cancel 提 stop request，再 await/status 核對 terminal。Cancel receipt 不是 cancelled proof。
5. 完成後明示重讀 fit_result，核對 params 與資料來源。Search terminal 不自動更新本連線的 fit baseline。Stdio await 會佔用此 server；client deadline 要超過 35 秒 method budget 並留傳輸開銷。

## 匯出與交接

讀工具宣告的完整依賴後，用 fluxdep_export_spectrums 與 fluxdep_fit_export_params。Spectrums 預設 create-only，只有明確需要取代時才 overwrite；params 沿 native JSON merge。工具回傳 native path；失敗可能留下已寫入前綴，不保證 rollback。

把目標、資料來源、實際輸出、待處理譜、當前 input 身份、search token／outcome 與下一步寫入目前任務的 gitignored .agent_state 記錄。恢復時重新核對 GUI。Timeout、disconnect 或回覆失敗都不證明 mutation 沒執行；先觀察，再選擇後續動作。

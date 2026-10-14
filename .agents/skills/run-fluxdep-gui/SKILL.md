---
name: run-fluxdep-gui
description: 透過 fluxdep-gui MCP 對既有 flux spectra 做聯合分析，判斷資料、對齊、採樣、躍遷假設、驗收、剔除與匯出；接手或恢復 GUI 分析時也用。
---

# Run Fluxdep GUI

人與 agent 共駕同一個 GUI。這個流程不操作儀器，原始資料只讀。命令、參數範圍與 transition 定義在工具描述裡。`<repo>` 是分析 session 的 repo cwd。

判讀或沉澱經驗前，先讀 `<repo>/measure_knowledge/README.md`。

## 開始與恢復

1. 確認分析目標、原始資料路徑、search database 與輸出目錄，缺少會影響下一步的資訊時問使用者。知識庫不可用時標示缺口，依下列最小判準繼續可判斷的工作。初始化留給使用者另行安排。完成條件是下一步的資料、授權與目的地已明確。
2. 第一次操作前讀 MCP server instructions。用 `fluxdep_connect` 接既有 GUI，需要新 GUI 且使用者授權時才 `fluxdep_launch`。回覆包含專案、每張譜的階段與草稿、fit freshness 及 control。恢復時以 GUI 現況為準。完成條件是說得出每張譜停在哪個決策點及下一步，並核對寫入是否由本連線持有控制權。

## 決策點

各步可重做，順序不固定，J4–J6 是迴圈。階段工具直接進入對象的草稿，finish 才發布。每次用數值差量決定是否繼續編輯，finish 回的圖用來驗收。

### J1 資料可用性

`fluxdep_spectrum_load` 回原始譜與已發布線／點的圖，`fluxdep_spectrum_snapshot` 用來重看。判準是有可辨認且隨 flux 變化的線。軸向不對時以 `transpose_axes` 重載，不可用時 `fluxdep_spectrum_remove`。完成條件是保留的譜有可辨認的線，軸向與單位已核對。

### J2 對齊

完整覆蓋 half 與 integer 時，以 `fluxdep_align` 移線到大致位置，跑 auto_align，數值合理後 finish。只覆蓋部分時，載入用 `inherit_from` 繼承已對齊譜，或移線到已知位置再 finish。判準是兩條線落在對稱軸且角色符合譜線特徵，onetone 平坦點通常是 integer、陡峭點通常是 half，twotone 以 plasmon-like／fluxon-like 特徵及反交錯區辨認。完成條件是 finish 圖的軸位與角色一致，判不出時問使用者。

### J3 採樣

用 `fluxdep_sample` 選可辨認的線，onetone 調 threshold，twotone 沿線 stroke、調 settings，畫錯用 undo。判準是每條可辨認的線都有點，雜點留給 J6。只用來定週期的譜可略過採樣；twotone 重新採樣從空遮罩開始。完成條件是 finish 圖上點覆蓋目標線，採完後用 `fluxdep_export_spectrums` 存 checkpoint。Restore 用 `fluxdep_spectrum_load_processed`，同名衝突會整批拒絕，checkpoint 不保存 Filter 遮罩。

### J4 躍遷假設與 search

`fluxdep_database_info` 查 DB 範圍與能階數，依線的數量與走向選 transitions 和 bounds，呼叫 `fluxdep_fit` 等搜尋結束。判準是假設能解釋主要線，bounds 在 DB 支援範圍內。省略的輸入沿用 GUI 當前值，明確 null 清除 r_f／sample_f。完成條件是取得 params、每條 transition 的殘差與 J5 圖，或記錄 failed／cancelled 原因並選定修正方向。

### J5 驗收

用疊圖、各 transition 點數與殘差判讀，殘差偏大的點以空心表示。判準是模型貼著主要線，各 transition 平均殘差在線寬內；只配到一兩個點的 transition 可考慮移除。完成條件是接受結果並寫明依據，或選定回 J4 換假設／bounds、回 J6 剔除的理由。`fluxdep_fit_result` 可重看結果，過期、無結果或無 recorded quality 都不回疊圖，先處理 freshness 或重跑 search。

### J6 剔除

用 `fluxdep_filter` 擦掉沒有躍遷能解釋的點或雜訊，finish 後重跑 fit。判準是剔除點有雜訊或不屬於目標線的證據，不為降低殘差擦掉可信線。完成條件是 finish 圖上的保留／剔除合理，並已回到 J4–J5 取得新結果。

## 匯出與記錄

接受且未過期後，用 `fluxdep_fit_export_params` 寫 params.json 給 dispersive。過期時先重跑 fit，再驗收與匯出。

把目標、資料來源、對齊值、接受的參數與殘差、實際輸出、剩餘不確定性及下一步寫進任務的 `.agent_state` 記錄。完成條件是輸出已確認，或未完成原因與接續步驟有記錄。

## 使用者接手與失敗

同時只有控制權 holder 能寫入。使用者接手、命令被拒或回覆未確認時，用 `fluxdep_spectrum_list` 讀階段、Filter 與 fit 現況，需要看譜時用 snapshot。Timeout、斷線或圖像失敗可能發生在發布之後，先核對現況再決定，不自動重送。Disconnect 保留 GUI。完成條件是下一步依目前 GUI 狀態與有效控制權決定，而非舊記錄或舊圖。

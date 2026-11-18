---
date: 2026-11-18
status: active
superseded_by: null
related:
  - ../adr/0066-operation-lifecycle.md
  - ../adr/0067-gui-application.md#results-browser-唯讀檢視
  - ../../lib/zcu_tools/gui/app/measure/services/results_index.py
  - ../../lib/zcu_tools/gui/app/measure/services/results_index_owner.py
basis:
  repo_commit: 9a09218c725426f435da63a2707c9a4add9bf84d
  verification_base: 1ae2449966675122bc7a6b896247b691e248e7dd
  external_sources: none
---

# Results Browser 開窗與背景索引

## 結論

使用者於 2026-11-18 裁定「1B 2照掃 3單獨開，但歸你與designer管」。這次移除掃描重工，改在背景讀取，保留 `legacy-Axel.bak`。使用者另選「A」保留 cfg 搜尋，不把讀 cfg 延到掃描之後。

Results Browser 顯示後，Measure app 逐個 Database entry 交付清單。關閉瀏覽器只隱藏，同一個 app 程序重開時沿用原掃描。designer 於同日代裁採用 app owner，不把掃描算成 operation；這項邊界待使用者覆核。現行規則見 [ADR-0067](../adr/0067-gui-application.md#results-browser-唯讀檢視)。

## 問題

在使用者的實驗室工作區，首次開窗兩次都約 32 秒。同步索引 23 個 entry、5,948 筆 run 約花 31 秒；畫面與縮圖約 1.4 秒。讀 metadata 與 cfg 花 15.8 秒，路徑解析約 5.5 萬次、共花 11.0 秒。單是 `legacy-Axel.bak` 的 1,661 筆 run 就花 8.2 秒。

## 考慮過的選項與比較

只去掉重工能縮短索引總時間，但仍要等完成才看得到視窗。背景索引配合去重工，可以先畫空視窗，再交付可搜尋、可開啟的已掃成果。持久索引還要定義跨程式執行的失效規則；這次不採用。

檔案讀取沿用 datafile 的驗證，一次開啟 native container 讀兩份文件。單份文件失敗不遮住另一份。掃描重用 lexical 目錄前綴，不沿用舊的檔案系統檢查結果。遇到連結或 junction 時走既有範圍檢查；這並沒有消除檢查與開檔之間的競態。

背景 owner 有兩個選擇。一個是 app owner 借用既有 executor 與 owner scheduler，將每個 entry 投遞到 owner thread；另一個是用 OperationRunner 取得既有 handle，直接參與 operation 確認與取消。app owner 遵循 [ADR-0066](../adr/0066-operation-lifecycle.md) 的 owner delivery、合作取消和 drain，也沿用 [ADR-0067](../adr/0067-gui-application.md#results-browser-唯讀檢視) 的唯讀邊界。既有 Controller 關閉時會清理 file watcher，作為非 operation 工作由 app 收尾的先例。

## 被否決的選項與理由

- 只去掉重工仍讓 Qt owner thread 等到掃描完成。
- 持久快取、依名稱略過 `.bak`、以及掃描後才補 cfg 搜尋都不符使用者取捨。
- 註冊 OperationRunner handle 時，關閉主窗會把唯讀掃描當成「Operations in progress」。掃描沒有待保存的成果，不應單獨讓使用者確認。取消在每筆 run 之間檢查，否則單一 entry 就可能超過既有的 5 秒 drain 上限。

## 留下的問題

實驗室工作區的前後比較由使用者安排。合成工作區不能代表實驗室的檔案系統速度、HDF5 內容或圖檔大小。既有 Open admission、縮圖解碼策略與 RPC wire 不變。

## 依據

- 實驗室重現數字由 2026-11-18 的使用者工作區報告取得，原始工作檔未入庫。
- 第一輪 scanner 比較用 23 個合成 entry、5,948 筆 run、5,850 個 HDF5，同一 CPU slot 未插樁各跑三次。掃描中位數由 8.516 秒降到 4.069 秒，唯讀投影相同。這輪 PNG 是 placeholder，沒有用於開窗驗收。單獨量 cfg dataset 讀取與 JSON 解碼，5,850 次共 0.648 秒和 0.616 秒，並未把這筆插樁讀數從總時間扣除。
- 背景取消的[公開契約案例](../../tests/gui/app/measure/services/test_results_index_shutdown.py)在大 entry 第 32 筆 native read 暫停，經 `Controller.close()` 的 owner loop 釋放讀取，不掃後面的 runs。沒有取秒數作測試斷言。[元件情境](../../tests/gui/app/measure/ui/test_results_browser_progress.py)用合成資料驗證頁面逐批進展與主窗關閉。

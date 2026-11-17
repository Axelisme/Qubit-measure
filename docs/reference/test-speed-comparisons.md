# 測試成本比較紀錄

**讀者：** framework-dev、agent。查閱 worker 數、等待與 fixture 調整的歷史量測、方法與限制。

執行方式與現行測試規則見 [tests README](../../tests/README.md#執行測試套件)。以下數據只描述各次比較，不是目前選集的基準。

## Worker 數比較

4 workers 的建議來自同機 8 核心的單次矩陣。非模擬在 2／4／6／8 workers 的 receipt wall
為 458／306／321／322 s，模擬為 219／166／165／181 s。
6 workers 沒有明確優於 4，4 也留下核心給其他 lane。
背景負載與 cache 順序沒有完全控制，這不是穩定加速率或其他機器的最佳值。
原紀錄未提供此矩陣的日期，不與下一節的比較合併計算。

## 2026-10-22 等待與 fixture 比較

兩組互不重疊的局部選集已量得約 8.705 s 的 worker elapsed 節省：

- Fluxdep 兩個 busy-poll 案例以外部 `sleep` 的本地 binding 替身去掉真實等待，
  保留正式每秒 poll 策略、兩次 poll 與結果／context 斷言。
  三組 old/new 比較中，先算每案例各自 old/new call median 的差，再相加，得到 4.001 s。
- 純 snapshot 消費 fixture 改為每案例建立獨立的 typed model，省去不必要的真 storage；
  真 storage／reload 案例保持原樣。完整 126 案例的 setup/call/teardown 加總，
  三組 paired old 減 new 差值的 median 為 4.704 s，包含 50 個未改案例的量測波動。

合計是 `4.000834279 + 4.704295066 = 8.705129345 s`，不是 CPU time 或整套 wall time。
兩項的 median 算法不同，合計也不是整套的 median；不再加各 module median 或 process／import 節省。
不能除以 worker 數推算整套 wall 改善。

配置只試一輪，沒有採用變更。Default 的 `worksteal` 改 `load` 單 pair wall 為
264.306 → 260.697 s，1.366% 的小差與背景負載變化不足以支持更改排程。
`-p no:cacheprovider` 的 collection wall 為 25.044 → 25.623 s，沒有收益。
完整 import self time 前十沒有能保留公開 import／decorator 契約的安全 lazy 候選。
預設配置不變，執行指南繼續使用 `--dist=worksteal`。

下表是相同 production 與 dependency tree 的單次參考比較。Old 只還原
`tests/mcp/fluxdep/test_align.py`、`tests/gui/session/test_parameter_context.py` 與
`tests/resources/entry/test_roles.py` 的等待／fixture 變更，其餘檔案與依賴保持相同。
每個選集各跑 old/new 一次，四筆均為 `-n 2 --dist=worksteal`。

| 選集 | Old wall s | New wall s | Old 減 new s | 差值占 old |
| --- | ---: | ---: | ---: | ---: |
| Default，兩筆各 11849 passed | 256.538714 | 246.988530 | 9.550184 | 3.723% |
| Full，兩筆各 11852 passed | 267.610523 | 262.417422 | 5.193101 | 1.941% |

Default 沿用 `pyproject.toml` 的 `not slow and not sim`，full 使用 `not sim`，包含 slow。
兩個 pair 各自的完整有序選集相同，沒有 skip 或 xfail。對應的 pytest 命令如下，
執行時仍須按當次任務授權經共用重工作佇列：

```bash
uv run --directory <worktree> --no-sync -- python -B -m pytest \
  -n 2 --dist=worksteal -q --durations=50
uv run --directory <worktree> --no-sync -- python -B -m pytest \
  -n 2 --dist=worksteal -q --durations=50 -m "not sim"
```

四筆在同一 high/P-core slot、CPU 2、3 執行，GUI offscreen，數值函式庫 threads 為 1。
Wall 包含 uv／Python／pytest 啟動與退出；兩邊的 instrumentation、JUnit／duration 輸出與
每次重建的使用者 cache 隔離一致，這些觀測成本沒有扣除。
每個 pair 都是 old 先、new 後，且有共享負載與 OS page cache 影響。
因此只作單次參考，不宣稱穩定加速，不外推到其他選集或環境，也不拿跨時段基線宣稱改善。

當日比較沒有採用以下三項變更：

- MCP initial grace 需要新的公開 clock/deadline 契約，不 fake wait、不 patch `Condition`、不改 grace 值。
- Editor GC policy 涉及真實顯式 `collect` 的 lifecycle 驗證，不能自行減少收集次數。
- 數值 fit 資料量變更須保留真擬合與驗證精度，不能降級驗證來換速度。

既有成本診斷只用來找候選，不是加速證據；整個案例的秒數也不是可省時間的上界估計。

## 2026-11-17 預設選集第一波整理

本次在本機8核心經 `tools/run_heavy.py` 執行預設選集
`pytest -n 4 --dist=worksteal`。同一 worktree、同一 tree 先完整跑一次預設選集暖機，
這次不計入樣本，再在同一批次連續跑3次取牆鐘 median。
GUI cache 仍每個 process 從空目錄開始，不以跨 process 的 GUI cache 當暖態。

基準為 HEAD `de8245e3ff4ff958a05663771bf15dc30bbfeaff`，每次17,150 passed。
`run_heavy` receipt 的 elapsed wall median 為263.51 s。該批背景噪音大，
只保留其量法與數值，不把它當成無干擾的對照。

整理後在 tree `86ffd001634fa79dfc804e319bccfb6be2819e38` 量到以下結果，
每次17,323 passed。Child wall 由 wrapper 以 child 啟動至退出的時間取得，
包含 uv／Python／pytest 啟退，不含 runner 排隊或事後 CSV 分析。

| 正式 run | Child wall s | Pytest 自報 s |
| --- | ---: | ---: |
| 1 | 183.98 | 181.87 |
| 2 | 185.08 | 182.95 |
| 3 | 185.70 | 183.55 |
| Median | 185.08 | 182.95 |

交付 tree `9ceabc70a93670bbe809525b13db172d87cc1674` 在後續 owner guard 修正與
README 定稿後沒有重新量測，不能把上表當成它的新量測結果。

基準與整理後的時鐘邊界、tree、選集數都不同，也不是同批 old／new 成對比較。
兩個 median 的差值不當成提速收益。Setup／call／teardown 的 phase 加總是四個 worker
各自 phase 的耗時相加，不是整套牆鐘，也不能除以 worker 數推算可回收時間。

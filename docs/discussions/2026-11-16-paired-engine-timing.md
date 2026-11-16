---
date: 2026-11-16
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
basis:
  repo_commit: c82f24522e8009defb7567a17abffe677a85821e
  external_sources:
    - Numba 0.65.1, numba/core/dispatcher.py, accessed 2026-11-17
---

# 成對 Engine 計時的 process、身份與統計

## 結論

唯一成對入口是 `tools.paired_benchmark`。Controller 直接管理兩側 child。
Warm 每側保留一個 process。First 與 cold 每筆建立新 process。
每筆 first 只呼叫一次既有 `SampleRunner(mode="first")`，不套 Pareto 的慢 family 規則。
Session、history 與 clock 仍由共同量測 owner 管理。
現行責任見 [ADR-0072](../adr/0072-module-boundaries.md#開發工具的數值交接)。

兩側先核輸入格式、快取格式、工具協定與逐筆 record 身份。
數值演算法版本只作 provenance，否則無法比較合法的數值優化。
格式 owner 見 [ADR-0078](../adr/0078-evolution-module-boundary.md#凍結數值輸入與參考解交接)。

## 問題

長駐 process 內只換 Engine/session，不等於新 process 第一次 evolve。
先前載入的 Numba signatures 會影響下一筆首次成本。
兩個版本若用不同 disk-cache 狀態，編譯或載入成本就會混進版本差異。
計時工具也不能因記錄 family CPU 的開銷而讓兩側用不同 accounting 模式。

身份核對原本沒有定義場景版本。Physical hash 只綁序列化內容，不綁 reader 的解碼語意，physical 記錄內也沒有版本欄。
當時 `SCENE_CACHE_VERSION` 管 index JSON 與快取目錄，單筆容器另用字面 1，兩者沒有連動。
`SEGMENT_NUMERIC_VERSION` 則管數值結果相容性，不能用它拒絕正要比較的數值實作差異。

依 lockfile 的 [Numba 0.65.1 dispatcher](https://github.com/numba/numba/blob/0.65.1/numba/core/dispatcher.py)，compile 先重用 process 內已有的 signature，沒有才載入 disk cache 或編譯。
因此長駐 child 內連續量 first，狀態會依先前走過的 signature 改變，cold 也不能靠換 session 得到。
Repo 要求在首次 Numba 工具 import 前設定 cache root，之後改 root 須新 process，見 [cache namespace](../../lib/zcu_tools/simulate/README.md#numba-machine-code-cache)。

## 考慮過的選項與比較

Controller 直接建立 measurement child，只需管理一層 process 生命週期。
Warm 保留 child，first/cold 則在每筆後關閉。
First 先在每側跑同 workload 的丟棄暖機 child，讓之後各筆載入同側既有 namespace。
Cold 每筆用新的空 Numba cache parent，child 結束後清理。
Process 啟動與 import 都不進 evolve timer。

Warm 近似 GUI 開著連續 acquire；first 近似 GUI 剛開、disk cache 已有時的第一次 acquire，不含 import 與 GUI 建構；cold 近似升級後空 disk cache 的第一次。
這是有條件的類比。GUI environment 持有常駐 Engine，並不為每次 acquire 換 Engine。
[Simulate README](../../lib/zcu_tools/simulate/README.md) 的歷史比較記錄 disk cache 命中省 5.5–17.4 秒，不含 import，不能當本次計時實跑。
早先否決每筆新 child 是因 warm 要重建、啟動成本難排除。這裡 warm 仍長駐、啟動在 timer 外，原理由不再適用。

握手要求工具協定版本、physical 解碼版本、`SCENE_CACHE_VERSION` 與有序逐筆四欄身份都相等。
Metadata hash 也在身份內，因為 companions 與 companions_by_target 決定 whitelist，expected_routes 決定歸因。
兩側各自經 codec 讀取核身份，共用快取與顯式 scene-cache 覆寫都適用。
例如一側把 ns 改成 µs 卻沒 bump physical 版本，hash 相同仍可能量錯；有 bump 就在樣本前因版本不同停止。
兩格式版本維持 1，不改 bytes、hash 或 reference key，不作遷移。日後 bump 的 owner 須在同一變更中讓 producer 重建場景並重解受影響的 reference。

ABBA 區塊以兩筆 candidate 總秒數除以兩筆 baseline 總秒數。
Wall 與 process 分開計算 median 和線性 IQR。
兩側各自執行 off/off 與 off/on 區塊，四個 overhead median 都在相應 noise IQR 內才共同啟用 telemetry。
超出時兩側都關，另跑同座標、空 history 診斷。
這些診斷不解釋成 telemetry-off 原樣本的 family 分解。

區塊比值、median 與 IQR、校準和最終比較共用估計量，以及兩側共用 telemetry 模式，使用者於 2026-11-16 答「1確認」。
四項 clock 判定的合併方式、process 邊界與 first 每 child 一筆，是 designer 的實作與量測方法裁定，不記為使用者追認。
2026-11-16 的設計裁定採上述 process 邊界，並將 first 收窄為每 child 一筆。
它不增加共同 owner 的 API，也不改 Pareto 原有慢 family 協定。
Process 邊界這項是 designer 的介面裁定，不記為使用者追認。

## 被否決的選項與理由

- 長駐 child 再轉送短命 measurement process 會多一層轉送與清理，controller 已能直接管理。
- 長駐 process 內重建 session 不能對齊新 process 的 Numba 載入狀態。
- First 留在已暖機 child 量的是機器碼已在 process 內的新 session，沒有 consumer 需要，還依賴暖機走過的 signature。
- 把 Pareto 的慢 family 補 warm 規則搬到成對 consumer，會複製 owner 的門檻並擴大單筆 first 介面。
- 在 owner 加 first 後同 session 補 warm 的公開操作，沒有 consumer 讀補量結果。補量失敗會讓成功 first 請求失敗，每個慢 child 還多一次超過 60 秒的 evolve，並要再改共用 owner。現有 SampleRunner 量完就關 session，不能用現有 API 組合。
- 成對工具自己抄 60 秒門檻，會有兩份門檻。
- 拿掉場景版本會把 hash 相等誤當解碼語意相同。
- 把版本放進被雜湊的 physical payload，會改所有 hash 與 reference key，使舊 reference 失效。
- 用數值演算法版本當格式相等條件，會拒絕原本要量的數值優化候選。
- 每筆記 spawn 到 ready 的 wall 作診斷，沒有判定讀它。
- 校準與最終比較使用不同估計量，會讓 noise floor 與速度比值不再指同一件事。
- 兩側各選 telemetry 或只看 wall，會讓 process CPU 比值混入不對稱觀測開銷。
- 只看 process 則漏掉 wall 上的記錄開銷，例如同步 I/O。兩個都是輸出結果，都要在雜訊內才開 telemetry。

## 留下的問題

數值驗收仍需兩側含共同 committed 工具，以同版本量 noise floor 和同次獨占診斷。
Synthetic 協定測試不替代這項驗收。實際 CLI、K 與成本要在執行前凍結並取得資源授權。
這篇紀錄不宣稱完成任何吞吐量或冷快取重建驗收。

有 consumer 要讀昂貴 first 緊接的同 session warm 時再開題。
屆時才考慮在 owner 加公開操作，讓 measure_modes 與成對工具共用，而不是現在補量無人讀取的樣本。

## 依據

- [共用計時 owner](../../tools/simulator_measurement.py) 與 [成對工具](../../tools/paired_benchmark.py)。
- [工具用法與 timer 邊界](../../tools/README.md#paired-engine-benchmark)。
- 上列 Numba 0.65.1 原始碼於 2026-11-17 重讀，版本取自 uv.lock。
- [共用模擬器計時討論](2026-11-16-shared-simulator-measurement.md)。遵循 prepared session 的職責分界。
- [Frozen inputs 討論](2026-11-16-frozen-simulator-inputs.md)。遵循 codec 擁有格式與 solver-free cache 的分界。

原始設計工作檔未入庫。本篇保存採用與否決的理由，現行契約由上述 ADR 和公開 API 擁有。

---
date: 2026-11-23
status: active
superseded_by: null
related:
  - ../adr/0063-persistence-ownership.md
  - ../adr/0069-experiment-core-and-frontends.md#native-可選變數與-labber-投影
  - ../adr/0072-module-boundaries.md#分層
  - 2026-11-23-coherence-native-shot-moments.md
  - 2026-11-23-mist-trace-propagated-sem.md
  - ../../migrations/README.md
basis:
  repo_commit: cd3ed7ab19150fbbfc0af41b98ab4d74e8755353
  external_sources: none
---

# 舊 MIST gain Labber 的同版轉換

## 結論

Designer 代裁，使用者 2026-11-23 追認。只在離線遷移宣告 `mist/gain / PowerDepCfg` 的歷史單一 `data`，軸為 Drive Power，訊號為 Signal，兩者都是 `a.u.`。沿用既有 Labber 版式辨識及嚴格驗證；轉成 Native 時，軸是 float64、資料是 complex128。以當代 Native 規格驗證轉出結果，cfg 維持 `3.0`，讀回缺少的 `iq_moments` 為 `None`。

同版只限本次已證等價的輸入。保留 fluxonium `1.28` 和 transmon `1.29`，不改 manifest 的版號與既有資料驗證。

## 問題

歷史 reader 原本從 Native 變數推導預期 log。新增可選六矩後，這個推導要求舊單 log 檔有兩個變數。原先可轉換的檔案在同一版號下變成 pending；當代 Labber 匯出本身仍是單一 log。

## 選項與比較

- A。不改宣告。同版接受的輸入縮小，原有 manifest 可能無法接續轉出本來能轉的 MIST 資料。
- B。固定原 MIST 宣告，兩 profile 都證等價才同版。重用 [coherence 的明列方式](2026-11-23-coherence-native-shot-moments.md)，另驗 MIST 自己的歷史格式。採用。
- C。固定同一宣告，但升兩個 mapping revision。不必證同版行為，已有 manifest 則要用舊 revision 的工具續跑或重驗。
- D。把整個歷史 registry 和當代 spec 拆開。每個歷史身分要考證原版式、驗證及版號，超過本次 MIST 的範圍。

## 未採用

不把當代 Labber 匯出選集當成歷史證據。它不能證明舊檔的版式、標記或身分。也不為恢復離線轉換而撤回 Native 的六矩。C 是等價不能證明時的另案選項，不自動升版。

## 留下的問題

證據只含合成資料及 manifest，不代表實際歷史檔曾使用 marked 版式，也不涵蓋故障注入或真實資料遷移。此結論不擴及其他歷史身分。

## 依據

以含 coherence、未改 MIST 的 `61f76a1f2` 對 `cd3ed7ab1`，與 `d79113199` 核對原 MIST 宣告。fluxonium 與 transmon 各 9 次 `migrations.migrate_storage` 真 CLI，合計 18 次，全部 exit 0。合成輸入有未標記單 log、較窄 numeric dtype、marked 正例及五個負例。比較 Native 值、dtype、軸、單位、cfg、metadata、拒絕診斷與原 Labber hash。

基線只轉 parameters，再用候選補 data，確實發布並驗證 Native。基線已發布 data、未清理時，候選續跑保留 run assignment 和既有檔案指紋，並能 typed Load 舊目的。對已清理的基線，候選回原報告，不改寫檔案；這一例只驗 guard，不算 parser 或 Native validator 重驗。

原始合成檔與收據未入庫。[遷移 README](../../migrations/README.md)和 [schema 宣告](../../migrations/migration_experiments.py)記現行 owner。

---
date: 2026-11-23
status: active
superseded_by: null
related:
  - ../adr/0063-persistence-ownership.md
  - ../adr/0069-experiment-core-and-frontends.md#native-可選變數與-labber-投影
  - ../adr/0072-module-boundaries.md#分層
  - 2026-11-22-amp-rabi-shot-moments.md
  - 2026-11-23-coherence-run-only-shot-sem.md
  - ../../migrations/README.md#legacy-resultdatabase--native-storage
basis:
  repo_commit: a077ffbd4f0b27b9a0fc9a1344174dab7d056fe8
  external_sources: none
---

# Coherence 的 shot 矩寫入 Native 檔

## 結論

2026-11-23，使用者說：「0250，我認為可以將誤差存進檔案裡，這樣才一致」，並選「0250A」。[Mux Ramsey](../../zcu_lab/v2/mux/t2ramsey/guide.md)、[Mux Echo](../../zcu_lab/v2/mux/t2echo/guide.md) 與[單 qubit Echo](../../zcu_lab/v2/twotone/time_domain/t2echo/guide.md) 把同次 Run 的校正 IQ 六矩另寫為可選的 `iq_moments`。每 tone、每個 requested delay 保存六個 float64，順序為 `[N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²]`。Mux 保留 Run 的 tone 順序，單 Echo 沒有 tone 軸。

重載後用相同 Analyze 選項，按同一選點、投影和公式重算 SEM。舊 Native 檔缺矩仍可重載和擬合，不畫 SEM。Labber 只匯出原資料，不新增矩。

使用者同日選「那就c2」。離線轉換保留三個歷史 Labber 輸入宣告，用現行 Native 宣告驗證轉換結果。合成輸入與同版 manifest 驗收等價後，維持 fluxonium `1.28` 和 transmon `1.29`。這項例外只限本次，不是一般免升版規則。

## 問題

[前一個決策](2026-11-23-coherence-run-only-shot-sem.md)只在記憶體交接矩。當次 Run 能畫 shot SEM，同一份資料存檔後卻畫不出來。Native 加可選變數後還有一個問題：離線轉換器原本從 Native 規格推導歷史 Labber 輸入，會把新矩當成必填 log。舊檔沒有那個 log，轉換因而列 pending。

## 選項與取捨

保存原矩沿用 [Amp Rabi](2026-11-22-amp-rabi-shot-moments.md) 的單一 Native spec 與獨立 Labber 選集，也保留前一個決策的同批 shots 和同次交接。這次只改保存和還原，不改 SEM 定義、投影、略點或 Echo 的 Stop 判準。共同保存規則仍由 [ADR-0063](../adr/0063-persistence-ownership.md) 與 [ADR-0069〈Native 可選變數與 Labber 投影〉](../adr/0069-experiment-core-and-frontends.md#native-可選變數與-labber-投影) 管轄。

離線轉換不從現行 Labber 匯出選集反推歷史輸入。依 [ADR-0072](../adr/0072-module-boundaries.md#分層)，三筆宣告留在 [`migrations/migration_experiments.py`](../../migrations/migration_experiments.py)。其餘歷史宣告與 runtime Load 不變。

| 歷史 `source_tag` | `cfg_type` | 原 log 順序 |
| --- | --- | --- |
| `twotone/ge/t2echo` | `T2EchoCfg` | `data` |
| `mux/time_domain/t2echo` | `MuxT2EchoCfg` | `signals`、`actual_detunes`、`source_frequencies` |
| `mux/time_domain/t2ramsey` | `MuxT2RamseyCfg` | `signals`、`actual_detunes`、`source_frequencies` |

這些宣告保留既有軸、單位、dtype 及嚴格驗證，不加入 tag alias。轉出的 Native 檔讀回時，`iq_moments` 是 `None`。

## 未採用

- 直接存 SEM 會綁死 Run 當下的投影。Analyze 改用另一個方向時，不能從舊方向的 SEM 重算。
- 維持只在當次 Run 交接，會保留重載後的差異。
- C1：同樣只固定三筆宣告，但升兩個 profile 版本，會讓同版 manifest 無法用新版續跑或重驗。這只在等價無法證明時另行裁決。
- A：通用改用現行 Labber 選集並升版，會改到其他歷史輸入。現行匯出不是歷史格式的證據。
- B：通用改用現行 Labber 選集但不升版，會讓同版 manifest 對應不同輸入規則。
- 原樣：保留推導會使舊 coherence Labber 缺 log 而列 pending。

## 留下的問題

舊檔未留矩，無法從平均 IQ 重建 shot SEM。等價驗收只使用合成檔和合成 manifest，沒有讀寫真實歷史資料。

## 依據

以 `a077ffbd4` 對候選 `39baf330` 驗離線 CLI；fluxonium 測的是同一實作的排版前 tree `0eaa6b508eeecfbac929192303f95973c9a96b39`。兩個 profile 各對照完整 81 筆 registry schema。每個 profile 的合成樣本包含單 Echo 一 log，以及兩個 Mux 各一份有 marker 的 grouped 檔和一份無 marker 的多 log 檔。五份都轉換成功，Native 值與歷史 metadata 一致；錯 log 數與錯單位各一份，都維持 pending，未發布 Native。

兩個 profile 皆以基準 CLI 建立部分完成和已完成的 manifest，再用候選續跑或重驗。每個 profile 的 12 個已發布 Database 產物保留 SHA256、inode 與 mtime，5 筆 run assignments 與 entry identity 不變。Resume guard 維持原規則。等價驗收後，本次只再修空 schema 的報錯時機，並移除 Labber Save 不需要的矩數值檢查。這兩項修正不改三筆非空歷史宣告、歷史 reader 或發布。

當時的合成輸入及執行收據未入庫。合成回歸測試位於 [`tests/migrations/test_legacy_labber.py`](../../tests/migrations/test_legacy_labber.py)，涵蓋歷史 Labber schema 不會納入現行 Native 的可選矩。

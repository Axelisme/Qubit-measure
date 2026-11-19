---
date: 2026-11-20
status: active
superseded_by: null
related:
  - 2026-11-02-herald-transfer-matrix.md
  - 2026-10-29-heralding-post-selection.md
basis:
  repo:
    main: 2b76fe05f
---

# T1 二階矩實驗以 t1_cov 恢復

## 結論

- 恢復單 qubit T1 的二階矩實驗，adapter key 是 `twotone/t1_cov`，recipe 是 `t1_cov`，存檔 tag 是 `twotone/ge/t1_cov`。
- 這個方法改稱 cov。heralding 只指分類後的 post-selection。
- 兩次讀出叫 `pre_readout` 與 `post_readout`，存檔變數 `sums` 是 float64 `(N, 6, 6)`，特徵軸叫 `pre_readout_features` 與 `post_readout_features`。
- 自校準 helper 移到 `zcu_tools.analysis.cov_transfer`，公開 `selfcal_transfer` 與 `TransferResult`，行為和移除前相同。
- 實驗只收單 tone `readout/pulse`。可選的 reset 是脈衝類 `Reset.Cfg`，用來先穩定 g/e 分佈。播放順序是 reset、`pre_readout`、π pulse、delay、`post_readout`。
- 舊的 `heralding/t1` 資料沒有載入路徑，也沒有 alias。
- Amp Rabi、Len Rabi、Ramsey、Echo 四個實驗不恢復，需要時再加。
- 裁決記在 [ADR-0074〈Shot sums 與 heralding〉](../adr/0074-flat-readout-module.md#shot-sums-與-heralding) 的 2026-11-20 註。

## 問題

2026-11-09，使用者說「舊 heralding 我完全沒用過也沒讀過，可以直接刪掉。」，commit `188409d1d` 隨之刪掉五個單 qubit 實驗、它們的 recipe，以及 `zcu_tools.analysis.herald_transfer` 和它的測試。

這五個實驗當時都叫 heralding，但它們用的不是分類後的 post-selection。它們讀兩次，把 `[IQ, IQ]` 的二階矩總和存下來，再從總和自校準出轉移矩陣與指標 C = 1 − tr J，不需要 classifier。方法見 [2026-11-02 的討論](2026-11-02-herald-transfer-matrix.md)。

2026-11-19，使用者在檢討初始化策略時澄清：「可以加回來，應該說，我刪掉的不是cov版，而是heralding版」。使用者也要求改正名稱：「把前一种基于共变因素的实验，把它纠正其义，不要再叫其 hurdling，呃，改成加上什么 COV 的 postfit 之类的。」

刪除之後，執行層與 Form、recipe 的 API 都改過，所以不能直接 revert，要依現行介面重寫。盤點時確認，舊 T1 用到的 `IQ`、`shot_sums`、`run_acquire_summed`、Form helper 與 recipe API 都還在，呼叫方式沒變。

## 考慮過的選項與比較

### 恢復範圍

- **C2（採用）：** 先恢復 T1，連同 transfer helper 與它的測試。其他四個需要時再加。
- **C1：** 五個一次恢復。

使用者 2026-11-20 回答「舊題都採用建議」，採 C2。

### 命名

| 選項 | Adapter key | 目錄 | Helper |
| --- | --- | --- | --- |
| A（採用） | `twotone/t1_cov` | `zcu_lab/v2/twotone/time_domain/t1_cov/` | `analysis/cov_transfer.py` |
| B | `twotone/cov/t1` | `zcu_lab/v2/twotone/cov/t1/` | `analysis/cov.py` |
| C | `twotone/t1/cov` | `zcu_lab/v2/twotone/time_domain/t1/cov/` | `analysis/cov.py` |

存檔 tag 沿用普通 T1 的慣例：`twotone/t1` 的 tag 是 `twotone/ge/t1`，所以這裡用 `twotone/ge/t1_cov`。

兩次讀出原名 `herald` 與 `signal`。使用者改成 `pre_readout` 與 `post_readout`。

### Readout 與 reset

- Readout 只收 `readout/pulse`，單 tone，和移除前相同。
- 移除前沒有 reset 欄位。使用者要求加上可選 reset，用來穩定 g/e 分佈。

使用者 2026-11-20 的回答：「關於t1_cov，1A，名字叫pre_readout/post_readout比較明確。2採用建議。3我認為可以加可選reset，作為穩定ge分佈使用。」

### 先例與立場

- **2026-11-09 的刪除裁決（有拘束力）：** 區別。當時要刪的是 heralding。被刪的程式其實是二階矩方法，只是名稱叫 heralding。使用者 2026-11-19 的澄清確認了這一點。
- **2026-11-02 的二階矩自校準決策，記在 ADR-0074 第 1 至 6 項（有拘束力）：** 遵循。六分量 IQ、自校準、只用 C 的規則不變，只換讀出名稱與存檔軸名稱。
- **普通 T1 的 key 與 tag 對應（參考）：** 遵循，tag 加 `ge/`。
- **2026-11-19 合併單 qubit RB 與 IRB 時不留 alias（有拘束力）：** 遵循，舊 `heralding/t1` 資料不提供載入路徑。

## 被否決的選項與理由

- **C1：** 其他四個實驗目前沒有使用者。依小功能原則，需要時再加。
- **命名 B：** 只為一個實驗開一個新的 `cov` 家族目錄。使用者選 A。
- **命名 C：** 把 cov 實驗放成普通 T1 目錄下的子目錄。使用者選 A。
- **保留 `herald`／`signal`：** heralding 現在只指 post-selection，沿用這兩個名稱會混淆。使用者說 `pre_readout`／`post_readout` 比較明確。

## 留下的問題

- FTF 用預設的模擬偏壓時，資料不滿足二態辨識的秩門檻，analyze 會拒絕。離線驗收把未驅動的 spectator 偏壓移離 half-flux，並縮短讀出積分，才得到可辨識的資料。T1 擬合結果是 169 ± 13 µs（線性 delay）與 175 ± 12 µs（衰減格點加脈衝 reset）。換偏壓後沒有重校 π pulse，所以這次只驗收流程與資料契約，不代表物理精度或實機結果。
- 另一個進行中的工作會把 herald 做成 reset 模塊。屆時 `t1_cov` 的 reset 欄位仍只收脈衝類 reset：遮罩會改變聯合二階矩，而 cov 本身就用第一次讀出自我校準。

## 依據

- Repo：main `2b76fe05f`；刪除 commit `188409d1d`。
- 恢復前的盤點與離線驗收紀錄放在本機工作檔，未入庫。本文已寫入結論與必要的事實。

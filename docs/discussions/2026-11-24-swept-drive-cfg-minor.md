---
date: 2026-11-24
status: active
superseded_by: null
related:
  - 2026-11-23-form-ref-not-copy.md
  - 2026-11-22-two-qubit-native-axis-cutover.md
basis:
  repo:
    previous_writer_reader: 2b8f628c31a181dfd9fe187859408c9f4d236c7e
    review_candidate: 32ced495aab2ae3e80ec0287ca54045cbe153872
    verified_candidate: 151b4c30dfd5ab571d1e00d1e5a0aedd285103de
---

# 掃頻 drive cfg 的值域擴張升 minor

## 結論

五個掃頻入口各升一個 cfg minor。Chevron 寫 5.1，Conditional phase 與其 heralded 入口分別寫 7.1、4.1，Conditional spectroscopy 與其 heralded 入口分別寫 7.1、3.1。每個 writer 對所有新 cfg 用同一版號。保存兩條 drive 的完整來源值，不改 cfg 欄位、結果軸、tag 或 reader。

Designer 依使用者睡前授權代裁；方向由 liaison 依使用者同日原話代答。這項裁定待使用者覆核，尚未升為 ADR-0069 的通則。

## 問題

改成 ref 後，兩路 drive 在 library 裡可以是不同的 carrier。Run 仍用同一個掃頻點播放兩條線。Native 檔的 cfg 應保留兩個來源值，而非播放頻點或人造的 0／0。舊 `QubitDriveCfg` 對二者驗證相等；同 major 的 reader 會比對 cfg type 並用自己的 validator，不會僅因 minor 改變而拒讀。問題是如何留下新值域的版本標記，而不讓原先可讀的 inline cfg 失去型別。

## 考慮過的選項與比較

| 選項 | 結果 |
| --- | --- |
| 保持版號 | 新讀舊和舊讀新的行為與升 minor 相同，但同一標記代表不同合法值集合。 |
| 升 cfg minor | 標出 writer 契約擴張，舊合法 inline 不需遷移。舊 reader 可讀等頻新 cfg；異頻 cfg 仍依舊 validator 成為不可用。 |
| 升 cfg major | 兩代讀者都不還原對方的 cfg，即使兩路相等。只要 tag、軸和結果相容，仍可讀 Result 和 metadata。 |
| 按值打標或另存最低 reader | 同一 writer 會產生兩套版本語意；舊 reader 不解釋新增的標記。 |
| 改成等頻快照 | 滿足舊 validator，但抹掉了原設定。另存一份相容投影或 decoder 會增加第二套讀寫責任。 |
| 另加 Native 標記或動容器 | cfg 外的文件、metadata、變數、groups 或根屬性不參與 cfg 驗證。改容器身分或 major 會讓既有讀者連結果也無法讀。 |
| 分開來源與播放型別，或加外部 schema 門禁 | 前者要改五個入口與欄位格式；後者能檢查發版，不能讓已部署的 validator 接受異頻。 |

## 被否決的選項與理由

不改版號無法區分值域，升 major 又切斷任務要保留的舊 inline 回載。拒絕相容投影與其他標記，因為 `/docs/cfg` 已是來源快照正本。這次遵循 [ADR-0069](../adr/0069-experiment-core-and-frontends.md#cfg-欄位改名與-schema-版本) 的欄位改名先例，但本案沒有改名；也不同於 [Two-qubit 的 Native 軸切換](2026-11-22-two-qubit-native-axis-cutover.md)。通用容器依 [ADR-0080](../adr/0080-datafile-generic-container.md) 不認實驗 cfg 的相容史。

## 留下的問題

舊版載入器對異頻新檔能保留結果，不能據此重做 typed Analyze。要重分析請用能讀該 cfg 的新版。要追溯設定或數組，保留未重寫的 `data.h5`，必要時參照 Git 中的實驗定義。這次不打包 decoder，也不承諾重建舊環境。通用版號規則等使用者覆核後再處理。

## 依據

使用者 2026-11-24 表示「讓agent解析或是經Load看，前者是當資料格式不兼容時使用」，以及「其實是可以依賴程式碼的，至少是當前的對應實驗，只要能讀得懂就好」。這是對找回資料的方向，不是使用者親自選定這五個 minor。代裁依據是睡前的「我要睡了，之後你代裁」。原始討論未入庫。

離線驗證用舊版 `2b8f628c3` 和候選版 `151b4c30d`。各版用實際五個實驗的 typed Cfg、Result 和 Native writer，共寫 15 份合成 HDF5，各五份舊合法 inline、新等頻和新異頻。等頻是 4123／4123 MHz，異頻是 4123／4234 MHz，共用掃描設定是 4100／4101／4102 MHz。新版五份舊 inline 完整回讀，五份新異頻完整往返；舊版五份新等頻仍可回讀，五份異頻都保留 Result 和 metadata 並回 `cfg=None`／`cfg_issue`。同一份 Chevron 異頻檔經 [Measure adapter](../../zcu_lab/v2/two_qubit/chevron/gui.py) 離線驗證。舊版拒絕 Analyze、data item 不補 cfg；新版用原 cfg 產生 Analyze 圖。合成樣本未觸發擬合，這不是模擬或實機驗證。

讀取及 cfg 邊界見 [Native persistence](../../lib/zcu_tools/experiment/native_persistence.py) 和 [Chevron 核心](../../zcu_lab/v2/two_qubit/chevron/core.py)；來源快照契約見 [ADR-0063](../adr/0063-persistence-ownership.md)。

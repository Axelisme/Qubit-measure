---
date: 2026-11-17
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
basis:
  repo: 89d125ae9f06063501a7e99f0b2566f0082f6aa8
  external: none
---

# 共用清單的 Pareto 選集與明確新增場景

## 結論

本次將 Pareto 預設選集與共用凍結清單的成員分開。
工具旁的名稱宣告保留既有 19 個場景，缺名就失敗。
場景作者只有明確指定更新與單一新名稱，才能把已宣告的場景加到原 category 尾端。
Tests 側 reference producer 加入嚴格單列選取，先選清單列，再讀數值及求解。
生效契約由 [ADR-0072](../adr/0072-module-boundaries.md#開發工具的數值交接) 與
[ADR-0078](../adr/0078-evolution-module-boundary.md#凍結數值輸入與參考解交接) 擁有。

## 問題

同兩份凍結清單需要容納 Pareto 場景與效能 profile 輸入。
若預設量表把所有清單列都當成決策輸入，新增 profile 列就會改變量表的選集，也會要求額外 payload 與 reference。
原 prepare 要求單選名稱已在清單內，不能讓已宣告的新 builder 發布第一列。
原 reference producer 只有整批模式，準備一列仍須載入其他列的數值。

## 考慮過的選項與比較

保留清單全選作預設，會讓存放成員隱含決定 Pareto 範圍。
新增一套 Pareto 專用數值清單則會重複保存同筆身份。
工具持有名稱集合可以沿用兩份清單與相同 cache，且不改四欄身份或 reference key。
顯式 index 仍全選，scenario 仍可單選非 Pareto 列。

自動以 inventory 同步整份清單，可以省去新增列的明確操作，卻會把缺少宣告解釋成刪列或換組。
本次沿用 update-index 與 scenario，將新增限制在一個已宣告且兩份清單皆無的名字。
既有列保留原位置，新列只加到宣告 category 的尾端。

Reference 的單列選取留在 tests producer。
呼叫端繼續明示每份清單對應的 solver，選取不讀 metadata，也不將 solver 放進 codec。

## 被否決的選項與理由

不採預設名稱集合與現有列的交集。缺列表示輸入未備妥，不應悄悄縮小量表。
不在失敗後換 solver，也不為未選列補建 payload。這些操作會改變明確選取的成本與失敗語義。
不將 profile 列當成 family 去留證據。新增 Pareto 場景要同時修改名稱宣告與清單列。

## 留下的問題

本次只驗選取、發布與 cache I/O 契約，沒有重建真實 inventory、準備真實 reference 或執行 benchmark。
完整冷重建的可重產性與數值收益仍需各自的授權及證據。

## 依據

遵循 ADR-0072 的工具來源與身份接縫，選取不進計時 owner。
遵循 ADR-0078 的 solver-free codec 與 tests 側獨立求解，格式版本、hash 及 key 不變。
[既有凍結交接討論](2026-11-16-frozen-simulator-inputs.md) 記錄了這個分工。
[場景資料說明](../../tools/data/family_pareto/README.md) 與
[工具用法](../../tools/README.md#freeze-numeric-scenes) 保存目前的作者操作。

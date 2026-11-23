---
date: 2026-11-24
status: active
superseded_by: null
related:
  - 2026-11-23-cz-optimize-labber-selection.md
  - 2026-11-24-cz-optimize-search-space.md
basis:
  repo: afe83cadce8c19ee1674131cfef2bf4c2db82181
  external: null
---

# CZ 最佳化的具名 Native 與 Labber 欄位

## 結論

Native 用獨立的具名欄位保存候選參數。每欄有自己的單位，旋轉角用 rad，起點時長用 s。Labber 選同一組具名參數與 `peak`，不匯出計數、起點包絡或無軸欄位。Native 是可重開的正本。

評估 0 直接播原 pulse cfg。Native 保存它的顯示包絡、表示和時長，分析不必重新讀資產來畫包絡與頻譜。Load 仍走 cfg 還原；若無法還原，保留結果及 metadata，回報 `cfg_issue`。此時 Analyze 仍可畫圖，但不畫來自 cfg 的避頻帶，也不提寫回。

## 問題

舊 [Labber 選集](2026-11-23-cz-optimize-labber-selection.md) 匯出 `params` 矩陣，而一個 Variable 只能有一種單位。矩陣同時包含無因次節點、共同倍率和兩個旋轉角。它不能用單位欄逐一說明這些值。

起點也不能從 CMA 的初始參數反推。內建 DRAG、非對稱 arb、ramp 與 DDS 平台不同高的 flat-top，都可能擬合出與原波形不同的包絡。保存完結果以後，原資產還可能移走或被同名檔取代。

## 比較過的選項

- 用既有的 `VariableZSpec.index` 把參數依 evaluation 投影成具名欄位，Load 再重組。這讓 Native 與 Labber 共用欄名和單位，且不保存第二份矩陣。
- 同時保存矩陣與欄位。兩份表示可能不一致，reader 得再決定哪份為準。擴框架 schema 也能給矩陣欄位個別單位，但會影響其他實驗的讀寫，這次不採用。
- 用樣條擬合播放評估 0，或拒收樣條無法重現的起點。前者會改變校準脈衝，後者會排除原已允許的 flat-top。最初考慮保存編譯後樣本供回播，後來改用原 cfg 播放，只另存顯示包絡；不用為顯示圖新增播放介面。
- 改共享 cfg 驗證，讓 Load 一律不讀資產。這會牽動 RB 和 chevron。保留現有驗證，在不能還原 cfg 時走 [ADR-0069](../adr/0069-experiment-core-and-frontends.md) 的降級路徑。

## 留下的問題

新版不加入歷史欄位的 reader。舊檔保留，用原寫檔版本開啟。2026-11-24 使用者談舊 Native 檔時說：「我不想做太多兼容的場景，這會造成維護負擔，照讀或是拒絕選一個好維護的就好」。CZ 當時只有模擬測試資料，採用原有的 [two-qubit 格式換版做法](2026-11-22-two-qubit-native-axis-cutover.md)，不把這個選擇外推到有實機資料的家族。

## 依據

[ADR-0063](../adr/0063-persistence-ownership.md) 區分 Native 正本與衍生匯出。[ADR-0069](../adr/0069-experiment-core-and-frontends.md) 定義 cfg 降級與 Labber 投影。實際欄位由 [CZ axes spec](../../zcu_lab/v2/two_qubit/cz_optimize/_result.py) 宣告；[操作指南](../../zcu_lab/v2/two_qubit/cz_optimize/guide.md) 說明資產驗證與寫回。原始調查與裁決工作檔未入庫，這裡保留影響選擇的事實。

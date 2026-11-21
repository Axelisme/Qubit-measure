---
date: 2026-11-22
status: active
superseded_by: null
related:
  - 2026-10-14-artifact-persistence.md
  - 2026-11-20-experiment-naming-and-display.md
  - 2026-11-22-two-qubit-identity-and-neutral-positions.md
basis:
  repo:
    before_task: 47fbcf5e8024372cf2655faf28e4325877b0c389
---

# Two-qubit 新 native 軸名與舊格式

## 結論

2026-11-22，使用者選擇新檔用中性軸名，舊軸不符就按現有 Load 契約拒絕。舊檔原樣保留，用對應舊版讀取。這個選擇只適用於已確認只有模擬器測試資料的家族。規則見 [ADR-0069](../adr/0069-experiment-core-and-frontends.md#只有模擬測試資料時的舊結果格式)。

## 問題

前一次同意的是 cfg 改名升 major，仍讀回相容的 Result 和 metadata。本次也要改 HDF5 的位置軸。Native restore 先檢查結果 layout 才處理 cfg，所以同 shape 的 `terminal_f1` 與 `terminal_qubit1` 不是可互換的宣告。只升 cfg major 無法讓兩份 layout 一起讀入。

## 考慮過的選項與比較

保留舊軸，僅改 cfg、表單與圖，可以維持原檔讀取，但新 HDF5 還是顯示 F1/F2。同時支援兩種 layout 則需由實驗聲明歷史 layout，共用 restore 依完整軸和 variable 辨識，不能拿 cfg major 當結果版本。直接改新檔且保留嚴格 reader 的改動最小，代價是舊檔需在舊版開。

裁定前曾以合成 joint_readout 結果驗證原 spec 的 36 個計數可完整 roundtrip；只換宣告後，舊檔在結果軸驗證就被拒絕。這一點區別於 [ADR-0069](../adr/0069-experiment-core-and-frontends.md#cfg-欄位改名與-schema-版本) 的 Local Z cfg-only 改名例。既有 Load 在 layout 相容時仍回傳 Result、metadata 與 `cfg=None`。實驗擁有映射，通用容器不認實驗歷史，沿用 [ADR-0063](../adr/0063-persistence-ownership.md) 與 [ADR-0080](../adr/0080-datafile-generic-container.md)。

使用者答「我沒有舊資料，可以直接換到新格式」，並說 chain、herald 和 two_qubit 沒有在實機上跑過，只有模擬器測試資料。這收窄了舊檔直接 Load 的承諾，沒有取消 cfg 版本或原檔保存規則。

## 被否決的選項與理由

- 保留舊軸：會捨棄已確認的新檔命名。
- 加歷史 reader 或離線轉換：目前沒有實機歷史結果需要它。歷史 layout、錯序或混合宣告會多一組維護與驗證責任。
- 新增另一套持久 key 和顯示 label 宣告：不能免去已存在的 layout 差異，且本次不需要通用格式機制。

## 留下的問題

有實機資料或不知道資料來源的家族，不能直接套用本次取捨。原決定沒有批准改 chain 或 heralding 的格式。Two-qubit 只改位置軸名，不改 tag、variables、raw Result 欄位、軸序、座標、單位、shape 或 dtype。

## 依據

原始調查與合成檔未入庫。舊 joint_readout 的保存宣告可查 basis commit。對外的舊檔處理步驟在 [two-qubit guide](../../zcu_lab/v2/two_qubit/guide.md#openreset-與人工-apply)。

---
date: 2026-11-20
status: active
superseded_by: null
related:
  - 2026-11-19-single-qubit-rb-irb-merge.md
basis:
  repo:
    light/n1-two-qubit-display: 6536e2d32
---

# v2 實驗命名與顯示：先對齊 two_qubit 的圖

## 結論

使用者在 2026-11-19 回覆「全采用建议」。整個 `zcu_lab/v2` 採用一套分類，從 `two_qubit` 起按 family 分批實作。先定使用者看得到的名字，再對應目錄、registry key、GUI、cfg 與 MCP 名稱。Tab ID 與預設 run 檔名由 key 推出。顯示的範圍是 liveplot 與 Analyze 的排版、標記、顏色、背景格線及兩者的一致性。

使用者推翻了風格指南原本的「不為命名一致性改動現有 cfg 欄位」。改名不留 alias。既有 cfg 欄位改名時，沿用單數改名先例，提升 cfg schema major，不遷移舊 cfg。若存檔 tag 與結果資料仍相容，舊 run 保留 Result 與 metadata，但不還原 cfg。取捨原則寫在 [ADR-0069](../adr/0069-experiment-core-and-frontends.md#cfg-欄位改名與-schema-版本)。

2026-11-20 代裁，待使用者覆核。依使用者同日的授權「STDO:我去睡了，我看過目前規劃方向沒有問題，請你之後幫我代裁設計與推進任務，都不需要問我，你自己決定」，第一批只改 `two_qubit` 的圖和 guide 首標。9 個普通版保留描述性首標，9 個 heralded 版統一以 `Heralded` 開頭，後接普通版首標。這 18 個 registry key 原已和各自的 `AXES_SPEC.tag` 相同，兩者都保留；cfg、Result 欄位和分析數值也不變。[風格指南第 4 節](../../zcu_lab/v2/experiment-style.md#4-live-plot-與-analyze-圖)記下全 v2 的顯示規則，本批只套用到 `two_qubit`，沒有改共用繪圖工具的預設值。

## 問題

使用者在 2026-11-19 指出：「liveplot 跟 Analyze 有时候差异极大」。盤點的 116 個 key 中，有 111 個同時有 liveplot 和分析圖，含 GUI 的互動分析圖。有 22 個配對在 liveplot 明確關了格線，分析圖卻沒同樣關掉。多數圖由實驗分別繪製，前端不會對齊兩處。Liveplot 可能顯示量測點，Analyze 卻先畫擬合或另一個診斷量。

命名也混用了分類依據：`mux` 按讀出方法分組，`two_qubit` 按 qubit 數量，`mist` 則按物理現象。畫面或 guide 的文字可以單獨調整。registry key 和存檔 tag 的改動則可能影響舊 run 的 Load，cfg 欄位改名還會改變可還原的設定。

## 考慮過的選項與比較

**同一套分類，分批改。** 全 v2 一次改完可以同時收斂命名，但要同時處理舊檔識別與 cfg 版本。各 family 自定分類則保留現況的混用。使用者選全 v2 共用分類。第一批選 `two_qubit` 的圖和首標，因為這兩項不用改持久資料，且這批沒有 key 和 tag 不一致的問題。

**Analyze 首圖跟 liveplot 的量。** 顯示規則第 5 條的做法也是代裁：把既有量測圖移到 Analyze 第一張，或補一張與 liveplot 同量、同軸的量測圖。其餘診斷保留。例如 Local Z 把 X/Y 量測圖提到相位診斷之前。唯一反過來改 liveplot 的是原本以多條曲線疊畫的二維掃描：chevron 和 constructive phase 的普通及 heralded 版都換成 Analyze 已用的熱圖，直接以兩個掃描量作為座標。這個例外只換畫法，不換量測量、Result 或擬合。

**沿用單數改名先例。** 例如 Local Z 曾把 cfg 的 `gate_counts` 改成 `gate_count`，並提高 schema major；Result 和存檔軸未改。Load 仍還原舊 Result，但把舊 major 的 cfg 標為不可用。[實驗作者指南](../../zcu_lab/v2/experiment-authoring.md#cfg-名稱順序與取數欄位)記錄了該批改名。

## 被否決的選項與理由

保留「不為命名一致性改動現有 cfg 欄位」會讓既有同義欄位一直用不同名字。使用者准許改名，但沒有選 alias 或遷移。若把改名當成同 major 的小改動，讀取器可能忽略舊 key，靜默套用新預設值。升 major 會明確標示舊 cfg 不可用。

讓所有 liveplot 跟著 Analyze 的原首圖，會讓 liveplot 從量測進度改去呈現擬合或次要診斷。讓所有 Analyze 診斷都照 liveplot 畫，則會丟掉分析才需要的資訊。只要求首圖能接續 liveplot，兩者各自需要的圖仍可保留。

## 留下的問題

其他 family 的 key 和 tag 不一致，以及重複 tag，留待後續批次。盤點時，116 個 key 有 33 個與存檔 tag 不同。兩組重複 tag 各有兩個 key：`singleshot/t1_tone_sweep_gain` 與 `singleshot/t1_tone_sweep_freq` 都存為 `singleshot/t1/t1_with_tone_sweep`；`twotone/zigzag_scan/gain` 與 `twotone/zigzag_scan/freq` 都存為 `twotone/ge/zigzag_scan`。Load 自動配對時，先找完全相符的 registry key，再以 tag 找 adapter。若 tag 對應到多個 adapter，配對會報歧義；改掉 tag 也可能讓舊 run 找不到 adapter。後續需逐 family 決定名稱及舊檔處置，不能把 tag 改名當成顯示文字修正。

## 依據

- `6536e2d32` 包含第一批實作與[風格指南](../../zcu_lab/v2/experiment-style.md#4-live-plot-與-analyze-圖)的變更。[`Registry.resolve_experiment`](../../lib/zcu_tools/gui/app/measure/registry.py)是上述 key 和 tag 的查找入口。
- 原始命名與顯示盤點未入庫。盤點讀的是程式碼，不是 GUI 截圖，也沒有比較像素。
- 另用 CPU 上的 FTF 模擬逐一跑 18 個 key 的 Run 與 Analyze，不開 GUI、不接硬體。13 個 key 首輪完成，5 個 heralded key 用剪枝重試完成。逐張檢視 124 張 PNG，核對軸標與單位、格線、marker 與擬合線、colormap、Analyze 首圖。原始 probe 紀錄未入庫；剪枝設定只用於看圖，不作校準。

## 後記

2026-11-20，使用者覆核本文的代裁。agent 提議全部確認，使用者回覆「確認」。第一批只改 `two_qubit` 的圖與 guide 首標，維持原決定。

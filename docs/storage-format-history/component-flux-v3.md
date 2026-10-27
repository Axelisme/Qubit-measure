# Component flux v3，offset 與 periods

**讀者：** operator、lab-author、framework-dev、agent。辨識 v3 flux 模型與 v1→v2→v3 的轉換邊界。

## 現行形狀

Runtime、entry template 與新建 component 使用 `schema_version: 3`。
YAML 容器的 `format_version` 不變。Qubit 的校準為：

```yaml
schema_version: 3
components:
  Q1:
    kind: qubit/fluxonium
    flux_dev: Q1_bias
    flux:
      offset: 0.5
      period: {Q1_bias: 0.002, coil_bias: 0.01}
```

這是相關欄位節錄，不是完整可載入文件。
公式為 `φ = offset + Σ V_d／period_d`。Offset 是所有儀器輸出 0 時的 flux，以 Φ0 表示，不做 mod 1。
Periods 使用各儀器原生單位／Φ0，保留方向。未校準形狀為 `{offset: null, period: {}}`。
有 period 但當下缺值的儀器不貢獻，period 保留；明寫無效值仍報錯。
Qubit、mux、JPA 使用 `flux_dev` 作主要／預設控制儀器，不參與公式，也不必有校準。

## 保留 v2 的歷史語義

[V2 歷程](component-flux.md) 保存 main 的每台儀器 `{half, integer, period}` 與 `flux_source`。
V2 的參考條件為其他儀器停在各自 integer，不將它重新定義成 v3。
現行長期升級鏈是 v1→v2→v3，讀取端不增加 legacy 相容層。

V2→v3 以 `flux_source` 為主要儀器。有該 entry 時：

```text
offset = 0.5 - half_main / period_main - sum_other(integer[o] / period[o])
```

公式保留主要儀器存下的 half 分支，不做 mod 1。
例如 v2 的 `Q1_bias: {half: 0, integer: -0.001, period: 0.002}`、
`coil_bias: {half: 0.005, integer: 0, period: 0.01}`，來源為 Q1_bias，轉成上例 offset=0.5。
沒有 `flux_source` 或主要儀器缺 entry 時 offset=null，仍保留所有 signed periods。
`flux: {}` 轉為未校準形狀。此步將 qubit、mux、JPA 的 `flux_source` 改名為 `flux_dev`。

## 現行 v1 中間步驟與報告

V1 自己的線保留原 half／integer／signed period 轉換。
Mux coil 只保留 period，在 v2 中間形狀寫 integer 0、half = period／2，原 half／int 列在報告。
下一步不恢復已捨棄的 coil offset 貢獻。
Native 單檔依自己的線、明示指定 coil、唯一 coil 選主要儀器。
多 coil 未指名時不填來源，v3 offset=null，全部 periods 保留。
報告列來源與原校準候選 offset，提醒重校。候選不同可能來自 flux jump，不視為腐敗。

缺名 local／coil 分別用 `unnamed_device`／`unnamed_coil`。
長期 upgrade CLI 與 `migrate_storage` 都逐實際目的位置提醒改名，不把缺名列為手動處理失敗。
Native 自己的線可用佔位指名主要儀器。Storage 每批與最終的 `flux_dev` 只保留明示來源，否則留空。
Storage 按完整 overlay 的 v1 結構角色算 offset，依自己的線、唯一 coil、否則 null，不靠名稱推測，不新增 finalizer。
明示來源包含有效繼承來源，原資料明示的 `unnamed_*` 也照常保留。
Point 完整有效且明示來源的校準覆寫繼承 offset 與該儀器 period，其他 periods 保留。

## 操作與未改範圍

先跑 [升級指南](../flux-migration.md) 的 dry-run，核對 dropped coil 值、缺名位置與重校提醒。
合法 v3 重跑跳過，runtime 不自動升級。名稱語法驗證不代表 device 已連線。
Component 升級不改歷史 records、不補 frame／component。
Predictor v1、standalone flat row schema 與 app reference anchors 不變。
[Record frame](flux-frame.md) 另說明 v3 frame、缺席儀器與 fallback。

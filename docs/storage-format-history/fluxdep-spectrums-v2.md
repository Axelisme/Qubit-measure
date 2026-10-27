# Fluxdep spectrums v2

**讀者：** operator、framework-dev。辨識 processed spectrum 格式並明確升級舊檔。

## 版面與意義

檔案根 attrs 為 `format=zcu.fluxdep-spectrums`、`version=2`。
每筆光譜 group 的 attrs 為 `offset`、`period`、`device`、`unit`。
`offset` 是被掃儀器值為零時，這筆數據的 reduced flux。
它包含當時其他儀器的固定貢獻與 flux jump，不代表 component 的目前校準。
`period` 是帶號且非零的原生儀器值／Φ0。
換算式為 `flux = offset + dev_value / period`。

`spectrum/` 與 `points/` 保留原有陣列與換算後的 `fluxs`。
選填 `type` attr 保存 OneTone／TwoTone。
選填 `device_snapshot/` 每台儀器一個 scalar dataset，dataset attr `unit` 記錄單位。
Snapshot 不再加到上述換算式，避免被掃儀器重複計算。
沒有 snapshot 的檔案省略整個群組，有記錄但沒有儀器的 snapshot 保留空群組。
外部 Labber 與舊檔升級使用 `device=unnamed_device`、`unit=native`。

App 仍以 half／integer 線對齊。
存檔使用 `offset = 0.5 - flux_half / flux_period`。
讀取以 `device_value_at_flux` 投影 φ=0.5 與 φ=0 還原兩線。
兩線可能與舊檔相差一個週期，光譜與點位的 flux 陣列不變。

## 明確升級

Reader 只接受 v2。舊檔會報錯並列出升級命令。
在 checkout 中先預覽，再套用。

```bash
uv run --directory . --no-sync -- python scripts/upgrade_spectrums.py /path/spectrums.hdf5
uv run --directory . --no-sync -- python scripts/upgrade_spectrums.py /path/spectrums.hdf5 --apply
```

CLI 逐位置列出變更。預設不寫檔。
套用時保留 period 與所有陣列，計算 `offset = 0.5 - flux_half / flux_period`，移除舊 calibration datasets。
CLI 先驗證候選檔，再建立相鄰且不覆寫的 `.bak` 原始 bytes 備份，最後原子替換原路徑。
已是 v2 的檔案通過驗證後跳過。壞檔維持原樣，其他輸入繼續處理，任一拒絕使命令回傳 1。
還原時先停止使用該檔案，再將報告中的 `.bak` 複製回原路徑。

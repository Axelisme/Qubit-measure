# `zcu_tools.analysis` 模塊重點文檔

**Last updated:** 2026-10-27

**讀者：** framework-dev。分析核心、純數值工具與前端 adapter 的分工速查。

`zcu_tools.analysis` 放置不依賴 notebook widget 或 Qt widget 的分析核心。各 notebook
與 GUI 套件只負責把使用者事件、圖表容器、與 workflow 狀態轉成這裡的純函式或互動狀態機呼叫。

Package root 在 caller 選用 `fluxdep` 時才載入它。直接使用 `analysis.fitting` 不載入
fluxdep 的互動 line picker，也不要求 QICK。

`spectrum.py` 提供兩個分析 GUI 共用的原始頻譜資料型別，以及 Hz→GHz 與座標遞增排序。
其中 `fluxs` 是座標軸，不是 flux 校正參數。

`failure.AnalysisFailed` 表示數值 owner 已辨識的無法產出分析結果，與任意程式錯誤分開。
Experiment 公開同一型別，GUI analyze work 經分類後才讓 recipe 交付 failed 值。
分類範圍由各數值 owner 文件列明，不以 ValueError／RuntimeError 的文字猜測。

## 多 device flux 座標

`analysis.flux` 擁有線性 flux 數學，不讀 component、record 或檔案，也不畫圖。
`DeviceFluxFrame(integer, period, unit="native")` 描述一台儀器在其他 device 都停在 integer 時的參考點。
Period 是每 Phi0 的 signed 儀器值差，必須有限且非零。
`FluxFrame(devices)` 保存非空的 device 名稱到具名 frame 的 mapping。
Device keys 與數學入口的 `device_name` 共用 `cfg_model.validate_device_name`。
它拒絕 ASCII 點號、空字串與全空白，其餘名稱原樣保留。
各 device 使用自己的原生單位，工具不換算單位。

`flux_from_device_values(frame, values)` 計算 `Σ(value - integer) / period`。
`values` 是 device 名稱到當下有限儀器值的 mapping。
缺任何 contributing device 就報錯，不猜零。
Booleans 不算數值。

`DeviceFluxPoint(integer, half)` 保存同一 device 的整數與半整數 flux 儀器值。
`reference_flux_points(fitted, device_name=..., frame=..., values=...)` 把擬合點換回其他 device 的 integer 參考點。
Caller 在 frame 的 selected device 填本次擬合的 period 大小與保留的方向，其他 device 保留當時的參考校準。
函式對兩個點都加上 selected period 乘以其他 device 的 flux 貢獻。
`actual_flux_points(reference, ...)` 減去同一平移量，得到其他 device 停在當下值時的掃描點。
兩個入口只要求其他 device 的當下值，不讀被掃 device 的值。
它們檢查點對的間距為 `abs(period) / 2`，允許 half 在 integer 任一側。

Record 轉成這些具名輸入的責任留在 resources。
Notebook 的第二軸也呼叫同一數學入口。
這個方向讓分析不依賴儲存容器。

## 子模組

- `fluxdep/`：Flux-Dependence Analysis 的選點、filtering、line selection、one-tone peak detection
  共用 kernel。`fluxdep/models.py` 擁有頻譜校正／點位資料、transition 型別與能譜到躍遷頻率的共用轉換；
  `fluxdep/io.py` 讀寫 HDF5 頻譜集合；`fluxdep/search.py` 負責共用 database search。
  診斷圖由 plotting 提供，顯示與 export workflow 由 notebook/GUI adapters 管理。
- `dispersive/`：色散分析的前處理、electrical-delay refinement 與 auto-tune 數值核心；契約見 [dispersive README](dispersive/README.md)。
- `fitting/`：擬合模型、shared/fixed fitting、singleshot 與 resonance fitting。契約見 [fitting README](fitting/README.md)。

`singleshot.py` 擁有純數值 shot 分類、Rabi 分類比例及 population 換算。QICK raw buffer 的 accumulator normalization 留在 caller 的實驗 helper；本模組不讀 program metadata。

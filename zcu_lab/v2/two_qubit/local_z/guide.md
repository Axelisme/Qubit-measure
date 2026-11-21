# Local Z

**讀者：** operator、agent。量兩個 target 的分類 local-Z slope，審閱並套用 correction pair。

Cfg schema 為 6.0，掃描欄位為 `gate_count`。新版 Load 拒絕軸名不相容的舊結果檔，不改動原檔。要讀取原檔，請使用對應舊版。

## Run 前

表單先列 reset、X90、雙線 drive 與 mux readout。接著設定外層 `gate_count`，最後填取數欄位。

選 `two_qubit/local_z`。綁定 scalar aliases `qubit1`、`qubit2`、`coupler`、`mux`。兩個 qubit 位置須綁到不同元件，且都列在 `$mux.qubits`。沿用舊 point 時，重綁這兩個位置，並將 CZ slot 與 pulse library key 改為 `cz_qubit1`、`cz_qubit2`。

先校準 X90、CZ 與完整 mux readout。Qubit1/Qubit2 的原 tone 都需要 circle classifier 與 GE Post radius。缺 classifier、非 circle 或缺 radius 時，Run 會要求先做 GE Post。其他 tone 不需要 classifier。修改 readout 設定後，重新校準。

Readout 有 `$mux.module.readout` slot 時引用完整 library 設定。缺 slot 時手填 blank mux 範本的必要欄位。Open／Defaults 捕捉角色、library 與 tone 位置。Refresh 保留原稿，並用新 capture [核對完整 tone 數](../guide.md#綁定位置與讀出-tones)。要用新來源重填初值時，按 Defaults 並核對輸入。

Form 複製兩角色 X90 與 `$coupler.module.cz_qubit1`／`cz_qubit2`。缺 CZ slot 時必須補填。兩 CZ pulses 用不同 generator，frequency、duration、pre-delay 必須相同。保留 signed gain 比例、relative phase 與 waveform。X90 generator 必須對應自己的角色。幅度與 frequency 選定後再量 local Z。

設 `gate_count`、`reps`、`rounds` 與等待。n 至少三點，必須是連續、嚴增、非負整數，預設 `[0,1,2,3,4]`。n=0 不發射 CZ，直接量基線。Rounds 在取數端累加，不保留 host round 軸。`reset=None` 不發射 reset，填入 reset cfg 才會在每個 shot 最前面發射。

## 序列與保存

同一次 Run 分別量 Qubit1、Qubit2。每個 shot 依序做可選 reset、target X90、n 次雙線 CZ、projection、terminal mux readout。另一 qubit 保留原初態。Target 與 quadrature 用硬體 loop。只有 gate_count 改變序列結構，留在 host。X/Y projection 使用不同 shots。Preparation phase 是 φ，X projection 用 φ，Y projection 用 φ+90°。

終端 Qubit1/Qubit2 在取數時分類為 g/e/other。`sums` 是 float64 `(2,2,Nn,3,3)`，依序為 target、projection、n、terminal Qubit1、terminal Qubit2。Target 座標 0/1 表示 Qubit1/Qubit2，projection 0/1 表示 X/Y。Class 順序是 g/e/other。保存總和原值，不保存逐 shot IQ，也不先換算比例。未量點保持 NaN。

取消保留已完成 round 的總和。StopSignal 只承載停止請求與 reason。取數失敗由 Run 直接拋出例外；停止請求後正常返回的 partial 仍須按分析完整性規則判讀。Native Save／Load 保留 sums、axes 與來源 cfg。Load 不修改目前 Form。Analyze 只用保存來源，缺來源 cfg 時停止分析。缺 sums 的舊逐 shot 檔案會明確拒絕載入。

## Phase 與 slope

每個 projection 的 `sigma=(N_e-N_g)/N_total`，other 留在分母。Logical phase 是 `atan2(sigma_Y,sigma_X)`，degree wrap 到 `[-180,180)`。Contrast 是兩個 sigma 的向量長度。分析不再線性投影 IQ。

每個 phase 要求 X/Y 兩個 projection 都有已完成 rounds 的計數，且總和大於零。空或缺失 projection、零 contrast 都保留原因，不猜 phase。每個 target 必須有整列 phase 才做 unwrap 與 fit。微弱但非零 contrast 仍可分析，操作員須看 quadratures 與 contrast 判斷是否重測。

分析沿 n 做相鄰主值 unwrap，再用含 intercept 的計數加權最小平方擬合 `phase_deg=intercept+slope*n`。兩 target 各自擬合。Slope 是每 CZ 的 logical Bloch phase accrual，correction 是它的負值，不重新 wrap。零 slope 可以有效，缺失 correction 維持缺失。

解讀假設相鄰真實增量滿足 `|Δθ|<180°`。每 gate 額外的 360° 無法由連續 n 採樣辨認。Live X/Y 圖標明完成點數。`phase_points` 圖顯示 contrast 與 wrapped phase。`phase` 圖顯示 unwrapped phase、密網格加權 fit、slope、反號 correction 與 fit 品質。`quadratures` 圖顯示兩 target 的 X/Y sigma。X/Y 的 ±1 shot SEM 由該投影實際保留的分類計數算出；啟用 mux herald 時不納入被丟棄的 shots。Slope 的 1σ 是擬合參數不確定度，不是單點 shot SEM。R² 是 fit 解釋的 phase 變異比例。Normalized RMS 把 fit residual 的 RMS 除以觀測 phase 的範圍。Relative fit SE 把參數標準誤除以參數絕對值。這些品質數值沒有額外選取門檻。Summary 的 `fits` 按 Qubit1、Qubit2 排列，只保留各 target 的 intercept、slope、correction、slope 1σ error、fit_quality 與不可用原因。逐點 phase、quadrature、counts 和曲線只在分析內供畫圖使用。每個 sigma 的 variance 是 `(P_e+P_g−sigma²)/N`，包含 g/e 的 multinomial covariance，other 留在 N。獨立 X/Y 的 variance 經 atan2 導數傳播為 phase variance，fit 使用它的倒數作權重。Covariance 是 `(XᵀWX)⁻¹`，不按 residual 重新縮放。缺失、半圈 unwrap 歧義或無正值 phase variance 時不擬合，也不提供該 correction。

普通版仍量 ensemble，local phase 可能偏離線性模型。純態 phase 與 gate fidelity 須另行驗證。

## Preview 與 Apply

只有兩 target 都有有效 correction，才提出一筆 `$coupler.ext.cz_phase`。內容沿用 `qubit1`、`qubit2` component 名稱與 `correction_qubit1_deg`、`correction_qubit2_deg`。任一分量缺失就不提案。Preview 後勾選該列，再按 Apply Selected。

寫回使用同次 Run／Load 的固定 capture 取得元件名與 coupler，不讀取目前的 aliases。沒有 capture 時，寫回失敗。後續使用 correction pair 時，表單會核對元件名與順序。

Correction 用於後續 single-qubit virtual-Z frame。Pulse phase 正負須依既有 convention 核對。Apply 只更新 phase pair，不改 CZ constructive relative phase、pulse 或 slot。修改 amplitude、frequency、duration、waveform、relative phase 或 DC 後，要重測 local Z。

完整校準順序見 [FTF bring-up guide](../guide.md)。

## 用 mux herald 篩選 shot

Reset 選 `reset/mux_herald`，readout 與終端讀取用同一份設定。Qubit1、Qubit2 位於完整 readout 的第 0、1 個 tone 時，填 `keep: {'0': g, '1': g}`。用原始 tone 位置，不用 ADC 編號。

機率是通過 herald 的 shot 裡的條件機率，分母是該格的計數總和。接受率等於計數總和除以 reps × rounds，不代表基態純度。沒有 shot 通過時，該格機率為 NaN。中途停止仍可用已完成 rounds 的計數。原本獨立 heralded 實驗的存檔已無法載入。


## Notebook

Use a validated `LocalZCfg` as `cfg` and fill `qubit1_tone` and `qubit2_tone` with the zero-based positions in the full readout. Notebook roles use `qubit1` and `qubit2`, but are separate from GUI aliases and do not bind them automatically. Supply the notebook's existing handles and device manager. `point_view` selects the save destination.

The GUI uses the Run or Load capture's component names in the analysis legends. If the run has no captured qubit pair, analysis uses `Qubit1` and `Qubit2`, but writeback still requires the fixed bindings. Notebook analysis uses the same position labels by default. Supply `LocalZExp.Options(qubit1_label=..., qubit2_label=...)` for custom labels.

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.local_z.core import LocalZExp

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(LocalZExp())
record = exp.run(cfg)
analysis = exp.analyze(None)
saved = exp.save_run(point_view, source=record)
```

## Calibration order and field units

Calibrate each role's Xπ/2 slot with amplitude and length Rabi before using it.
Confirm pulse frequency in MHz, signed gain in [-1, 1], and duration in us.
Use lookback to set mux timing, then mux onetone to set frequencies and gains.
Confirm readout timing in us and gains in [-1, 1].
Locate the dual-line CZ pulse with conditional spectroscopy and chevron.
Confirm the coupler's `cz_qubit1` and `cz_qubit2` slots before copying them into the Form.

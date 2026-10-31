# Local Z

**讀者：** operator、agent。量兩個 target 的分類 local-Z slope，審閱並套用 correction pair。

## Run 前

選 `two_qubit/local_z`。綁定 scalar aliases `f1`、`f2`、`coupler`、`mux`。F1/F2 必須是不同 fluxonium。Coupler 是 transmon，mux 的 qubits 必須包含兩者。

先校準 X90、CZ 與完整 mux readout。F1/F2 的原 tone 都需要 circle classifier 與 GE Post radius。缺 classifier、非 circle 或缺 radius 時，Run 會要求先做 GE Post。其他 tone 不需要 classifier。修改 readout 設定後，重新校準。

Readout 有 `$mux.module.readout` slot 時引用完整 library 設定。缺 slot 時手填 blank mux 範本的必要欄位。Open／Reset 捕捉角色、library 與 tone 位置。Refresh 保留已填輸入，換來源後要 Reset 並核對輸入。

Form 複製兩角色 X90 與 `$coupler.module.cz_f1`／`cz_f2`。缺 CZ slot 時必須補填。兩 CZ pulses 用不同 generator，frequency、duration、pre-delay 必須相同。保留 signed gain 比例、relative phase 與 waveform。X90 generator 必須對應自己的角色。幅度與 frequency 選定後再量 local Z。

設 gate_counts、reps、rounds 與等待。n 至少三點，必須是連續、嚴增、非負整數，預設 `[0,1,2,3,4]`。n=0 不發射 CZ，直接量基線。Rounds 在取數端累加，不保留 host round 軸。`reset=None` 不發射 reset，填入 reset cfg 才會在每個 shot 最前面發射。

## 序列與保存

同一次 Run 分別量 F1、F2。每個 shot 依序做可選 reset、target X90、n 次雙線 CZ、projection、terminal mux readout。另一 qubit 保留原初態。Target 與 quadrature 用硬體 loop。只有 gate_count 改變序列結構，留在 host。X/Y projection 使用不同 shots。Preparation phase 是 φ，X projection 用 φ，Y projection 用 φ+90°。

終端 F1/F2 在取數時分類為 g/e/other。`sums` 是 float64 `(2,2,Nn,3,3)`，依序為 target、projection、n、terminal F1、terminal F2。Target 座標 0/1 表示 F1/F2，projection 0/1 表示 X/Y。Class 順序是 g/e/other。保存總和原值，不保存逐 shot IQ，也不先換算比例。未量點保持 NaN。

取消保留已完成 round 的總和。先檢查 Run 的 stop/error 狀態，再判斷 partial data 是否可用。Native Save／Load 保留 sums、axes 與來源 cfg。Load 不修改目前 Form。Analyze 只用保存來源，缺來源 cfg 時停止分析。缺 sums 的舊逐 shot 檔案會明確拒絕載入。

## Phase 與 slope

每個 projection 的 `sigma=(N_e-N_g)/N_total`，other 留在分母。Logical phase 是 `atan2(sigma_Y,sigma_X)`，degree wrap 到 `[-180,180)`。Contrast 是兩個 sigma 的向量長度。分析不再線性投影 IQ。

每個 phase 使用兩個 projection 已取得的總和，不要求完成全部預定 rounds。空或缺失 projection、零 contrast 都保留原因，不猜 phase。每個 target 必須有整列 phase 才做 unwrap 與 fit。微弱但非零 contrast 仍可分析，操作員須看 quadratures 與 contrast 判斷是否重測。

分析沿 n 做相鄰主值 unwrap，再用含 intercept 的計數加權最小平方擬合 `phase_deg=intercept+slope*n`。兩 target 各自擬合。Slope 是每 CZ 的 logical Bloch phase accrual，correction 是它的負值，不重新 wrap。零 slope 可以有效，缺失 correction 維持缺失。

解讀假設相鄰真實增量滿足 `|Δθ|<180°`。每 gate 額外的 360° 無法由連續 n 採樣辨認。`phase` 圖顯示 contrast、wrapped phase、unwrapped phase 與 fit。`quadratures` 圖顯示兩 target 的 X/Y sigma。Summary 保留 acquired counts、每點原因、intercept、slope、correction、slope 1σ error 與 fit_quality。每個 sigma 的 variance 是 `(P_e+P_g−sigma²)/N`，包含 g/e 的 multinomial covariance，other 留在 N。獨立 X/Y 的 variance 經 atan2 導數傳播為 phase variance，fit 使用它的倒數作權重。Covariance 是 `(XᵀWX)⁻¹`，不按 residual 重新縮放。缺失、半圈 unwrap 歧義或無正值 phase variance 時不擬合，也不提供該 correction。

普通版仍量 ensemble，local phase 可能偏離線性模型。純態 phase 與 gate fidelity 須另行驗證。

## Preview 與 Apply

只有兩 target 都有有效 correction，才提出一筆 `$coupler.ext.cz_phase`。內容沿用 `f1`、`f2` component 名稱與 `correction_f1_deg`、`correction_f2_deg`。任一分量缺失就不提案。Preview 後勾選該列，再按 Apply Selected。

提案只讀來源 cfg 與同次 Run／Load capture。F1/F2/coupler 與來源不同時拒絕提案，包括交換 F1/F2。後續使用 correction pair 時也要核對身分與順序。

Correction 用於後續 single-qubit virtual-Z frame。Pulse phase 正負須依既有 convention 核對。Apply 只更新 phase pair，不改 CZ constructive relative phase、pulse 或 slot。修改 amplitude、frequency、duration、waveform、relative phase 或 DC 後，要重測 local Z。

Herald 版是獨立的 `two_qubit/local_z_heralded` 入口。完整校準順序見 [FTF bring-up guide](../guide.md)。


## Notebook

Use the validated LocalZCfg as `cfg`. Supply the notebook's existing handles and device manager. `point_view` selects the save destination.

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.local_z.core import LocalZExp

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(LocalZExp())
record = exp.run(cfg)
analysis = exp.analyze(None)
saved = exp.save_run(point_view, source=record)
```

# Conditional Ramsey 與 idle ZZ

**讀者：** operator、agent。設定非 heralded conditional Ramsey，判讀 idle ZZ 擬合頻差與限制。

`two_qubit/idle_zz` 比較 spectator I、Xπ 的兩個名目分支。Target 使用小寫 `f1` 或 `f2`。每個 shot 做可選 reset → spectator I/Xπ → target π/2 → delay → phase projection → terminal mux readout。reset 留空時不發射 reset。

## 開啟前確認

先依 [FTF bring-up 指南](../guide.md) 校準 F1/F2 的 Xπ、Xπ/2 與 mux readout。Aliases `f1`、`f2` 指向不同 fluxonium，`coupler` 指向本對 transmon，`mux` 指向含 F1/F2 的 MuxGroup。Group 的有序 qubits 與 resonators 必須逐項對應。

Open／Reset 固定捕捉角色與原始 tone 位置。每個角色的 Xπ 與 Xπ/2 使用同一 generator，F1/F2 使用不同 generator。Readout 引用完整 `$mux.module.readout` slot，缺 slot 時手填 blank mux 範本。F1/F2 必須有含 radius 的 circle classifier，缺少時先完成 GE Post。其他 tone 保留原順序，不要求 classifier。

Refresh 更新來源與選項，保留已填輸入。修改來源或 aliases 後按 Reset 重新取得初值，並核對角色、pulses 與 tone 對應。Reset 可能覆蓋尚未保存的輸入。

## 設定與 Run

1. 選 target `f1` 或 `f2`，另一個是 spectator。需要 reset 時填入已校準的 reset cfg。
2. 人工填 signed detune，單位 MHz。其絕對值須明顯大於預期 abs(ζ) 加 qubit 頻率誤差，使兩分支留在 detune 選定的同側。單一 projection 的資料不能證明此前提。
3. 設 delay 的 µs 範圍及點數。分析至少需要 6 個不同、等距的實際 delay，正的時間跨度，並解析數個 fringe 週期。量化後 delay 不可重複。
4. 只掃 delay 時，DC source 留空，各 source 不變。結果的單點 DC 座標 0 代表該軸停用。要掃 DC 時，填直接 device 名稱、driver 的 working unit 與已授權的 source values 範圍。支援 A、V、native，不做單位轉換或 F1/F2 補償。
5. 核對 reps、rounds、等待時間與完整 mux 積分窗口，再 Run。每個 DC 點用一支 program 在硬體掃 spectator_states 與 delay。DC 要設定外部儀器，因此留在 host。硬體 delay sweep 會產生逐點等待差。若編譯指出收尾 WAIT slack 大於 relax_delay，增加 relax_delay 後再 Run，不把 delay 改回 host scan。Live 每條線是一個 DC／branch 的 target ⟨σ⟩。

DC 操作也可能影響其他 loop。工作點與補償由操作員設定並記錄。結束後 source 維持最後設定，取消或失敗後先核對裝置狀態。

## FIT 與符號

第二個 target π/2 的 phase 是第一個 pulse 的 phase 加上 `360 * detune * actual_delay`，單位 degree。取數時分類 terminal F1、F2 的 g/e/other，每點保存 9 類計數。分析以目標 qubit 的 `⟨σ⟩ = (N_e − N_g) / N_total` 對 delay 做 free-phase decay cosine fit。Other 留在分母，不旋轉或重分類 IQ。擬合用計數推得的 shot-noise error 加權，飽和點的 error 下限為 1/N。

Fitter 給非負 fringe magnitude m，單位 MHz。在兩分支同側的前提下，`f = sign(detune) * m`，`ζ = f_Xπ − f_I`。負 detune 也使用 signed frequency 求差。共同人工 phase ramp 在頻差中抵消。

每個 DC 點有兩分支 Ramsey／fit 圖，另有 ζ 圖。ζ error 是兩支 frequency errors 的平方和開根號。Analyze 只發布圖並回傳 `None`。逐 DC 的 branch fits、ζ、誤差與原因只在分析內供畫圖使用。

ζ 是名目分支的單頻模型擬合頻差，不是純化的 Hamiltonian ZZ。熱 population 可能混合多個 conditional 頻率。操作員須比較量測曲線與擬合曲線，並檢查 ζ 圖的誤差和不可用原因。有限 fit 不代表模型偏差已排除，不套自動統計拒絕門檻。

### 符號辨識限制

正負 detune 都在 magnitude 座標比較 m 距離 0 與 `2 * abs(detune)`。距任一邊界不超過 `max(1/span, frequency_1σ_error)` 時，signed frequency 與 ζ 不可辨識。span 是實際 delay 最大值減最小值，單位 µs，1/span 是 MHz。

Detune 為 0、點數不足、span 非正、無 Ramsey contrast、非有限不確定度或數值 fit 失敗，也會保留原因而不給 ζ。先保存圖與原因，再人工調整範圍或校準。

## 取消、保存與重分析

Result 的 sums 是 float64 `DC × spectator_states × delay × 3 × 3` 原始計數。spectator_states 存整數 [0, 1]，分別是 spectator I、Xπ。末兩軸是 terminal F1/F2，各為 g/e/other。Rounds 在取數端累加，不除以 reps 或 rounds，不保存個別 shots 或 IQ。

取消保留已完成 rounds 的計數，未量點保持 NaN。Live 可見部分曲線，fit 只接受所有 delay 都完成要求 rounds 的分支。兩支 fit 可用時才算 ζ。總數為 0 的比例是 NaN。

Save 保存 sums、全部座標與本次 cfg。Native Load 不改目前表單。Analyze 只用所載 cfg，不用目前 library；缺 cfg 時拒絕分析。普通版 cfg schema 為 5.0，target 的舊大寫值不作相容轉換。舊 raw-IQ 檔缺少 sums 時明確報錯。

本實驗不寫回 calibration，也不自動調整 detune 或 DC。

## Notebook

`cfg` 使用與表單相同的 IdleZZCfg。經 NotebookAdapter 保留 RunRecord 與保存流程。
`point_view` 是儲存目的地已綁定的 `PointView`。保存時以本次 `run()` 回傳的 `record` 作為 `source`。

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.idle_zz.core import IdleZZExp

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(IdleZZExp())
record = exp.run(cfg)
exp.analyze(None)
exp.save_run(point_view, source=record)
```

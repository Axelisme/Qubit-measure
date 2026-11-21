# Conditional spectroscopy

**讀者：** operator、agent。設定 two-qubit conditional spectroscopy 並判讀四種名目準備態的頻譜。

Cfg schema 為 7.0。掃描欄位使用 `dc`。

## 量測序列

每個 shot 依序執行可選 reset、名目 B00/01/10/11 準備、雙線 weak probe、可選 Xπ⊗Xπ contrast、terminal mux readout。準備軸是 `prepared_qubit1`、`prepared_qubit2`，各為 0＝g、1＝e。Qubit1 在外層，兩顆的 Xπ 在獨立 timeline 同時準備。未使用 reset 時，名目準備仍包含熱布居。

Qubit1/Qubit2 的 terminal 讀出在取數時分成 g、e、other，保存每個聯合類別的計數。分析顯示兩顆 qubit 的 P_e，不再顯示全 tone mean IQ。不做 confusion correction、peak fit 或 writeback。

## 開啟前確認

在目前 point 的 Aliases 表綁定 `$qubit1`、`$qubit2` 到不同元件，並綁定 `$mux` 到包含兩者的 MuxGroup。舊 point 也要新增這兩個位置 alias；不會自動重綁。GUI aliases 與 Notebook 的同名內建角色分開設定。 不需要綁定 `$coupler`。依 [校準步驟](../guide.md) 準備兩顆的 Xπ 與 mux readout。

Open／Reset 根據有序的 `$mux.qubits` 設定隱藏的 tone index，並帶入 pulse 初值。有 `$mux.module.readout` 時引用完整 library readout。缺 slot 時須填妥 blank mux。完整 tone 順序保持不變，Qubit1/Qubit2 之外的 tone 不參與本實驗的 P_e 分析。

Qubit1/Qubit2 tone 都必須有 circle classifier 與 radius。缺少校準或使用 linear 時，Run 前會要求先做 GE Post。改動讀出設定後須重新校準，不可直接沿用舊 radius。Refresh 保留原稿，並用新 capture [核對完整 tone 數](../guide.md#綁定位置與讀出-tones)。要按新 aliases 或 library 重填初值時，按 Reset 並核對 cfg。

## 設定與 Run

表單先列 reset、Xπ、雙線 drive 與 mux readout，再列 contrast。
外層 DC 的 device、unit、values 在內層 frequency 前。取數欄位在最後。

1. 核對兩顆 qubit 的 preparation Xπ pulses。每條 probe 使用該 qubit 的 charge generator，Qubit1/Qubit2 generator 必須不同。
2. 填 `drive.qubit1` 與 `drive.qubit2`。如有綁定 `$coupler`，表單會複製 `cz_qubit1`、`cz_qubit2` slots。舊 point 若要沿用這些脈衝，請在 coupler 上設定同名 slots 與 pulse library keys。這些 pulse 不一定適合 weak probe。缺 slot 時，表單提供 cosine 包絡，channel 取自各 qubit 的 wiring。請填 gain、duration 與 Nyquist zone。
3. 兩線須填相同 base frequency、envelope duration 與 pre-delay。Gain 可帶符號，gain 比例及 relative phase 由 operator 決定。兩線平行播放，等待兩線及各自 post-delay 結束。頻率用 MHz，時間用 us，phase 用 degree。
4. 填共同 probe frequency 掃描範圍。每點同時替換兩線 frequency，其他 pulse 設定不變。保存軸是要求的 MHz 座標，實際硬體頻率會量化。
5. Double Xπ contrast 開啟時，probe 後依序播放 Qubit1、Qubit2 Xπ。關閉時直接接 terminal。比較兩種設定時分別 Run 並保存 cfg。
6. 只掃頻率時，`dc_device=None`。實驗不動任何 DC source，結果 DC 軸為單點 `[0]`，只作停用哨兵。需要 DC 掃描時填直接 device 名稱、working unit 與已授權範圍。支援 A、V、native，不補償 Qubit1/Qubit2，也不換算單位。
7. `reset=None` 不發射 reset。需要 reset 時明填其 cfg。核對 reps、rounds、等待時間與 mux 積分窗口後 Run。

只有 DC 留在 host，因為每點要設定外部儀器。每個 DC 的一支 program 用兩層 2 點準備態 loop 與 frequency 硬體 sweep 取完四種準備及所有頻率。每點的 rounds 在取數端累加，不逐 round 保存。DC source 留在最後設定，取消或失敗也不自動恢復。繼續前先確認裝置當下值。

## 看圖與分析

Analyze 只發布圖並回傳 `None`。激發機率、每點 shot 數與完整性只在分析內供畫圖使用。

Live 每個 DC／branch 顯示兩條 P_e 曲線，第一條為 Qubit1，第二條為 Qubit2。曲線由當下累計計數更新。FIT 分別提供 Qubit1、Qubit2 的四分支頻譜，各 DC 一條線。啟用 DC source 時，另有各 qubit 的四分支 DC × frequency 熱圖。

P_e 等於該 qubit 判為 e 的計數除以全部 terminal 計數，other 留在分母。零計數或未量點為 NaN。`complete` 表示該 DC／準備態的每個頻率都至少取到一輪。Shots 圖列出各頻率保留的 shot 數、要求值與零接受計數的診斷。

這些圖顯示分類後的激發機率。它們不直接給出高階態 population、leakage、Hamiltonian 或 CZ frequency。峰的位置與下一步 probe 窗口由 operator 判讀。

## 保存與取消

Result 保存 float64 `sums`，形狀為 `DC × prepared_qubit1[2] × prepared_qubit2[2] × frequency × 3 × 3`。最後兩軸為 terminal Qubit1、Qubit2，順序皆為 g、e、other。數值是總和，不是比例，檔案大小不隨 reps 或 rounds 增長。

取消保留已完成 rounds，未量點維持 NaN。StopSignal 只承載停止請求與 reason，不承載錯誤。取數失敗由 Run 直接拋出例外；停止請求後正常返回的結果可以是合法 partial。Save 保存座標、計數與本次 cfg。Load 不改目前表單，Analyze 只讀保存的 cfg 與 sums。新版 Load 會拒絕位置軸名不符的舊檔，不改動原檔。要讀原檔，請用與檔案格式相符的舊版。 不轉換舊 cfg。

## 用 mux herald 篩選 shot

Reset 選 `reset/mux_herald`，readout 與終端讀取用同一份設定。Qubit1、Qubit2 位於完整 readout 的第 0、1 個 tone 時，填 `keep: {'0': g, '1': g}`。用原始 tone 位置，不用 ADC 編號。

機率是通過 herald 的 shot 裡的條件機率，分母是該格的計數總和。接受率等於計數總和除以 reps × rounds，不代表基態純度。沒有 shot 通過時，該格機率為 NaN。中途停止仍可用已完成 rounds 的計數。原本獨立 heralded 實驗的存檔已無法載入。

## Notebook

建立 `SpectroscopyCfg` 時，按完整 readout 的順序填 `qubit1_tone` 和 `qubit2_tone`，index 從 0 起算。環境 handles 由 caller 提供。`point_view` 是保存目的地的 bound ParameterView。Notebook 經 adapter 管理 scope、plots 與 RunRecord。

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.conditional_spectroscopy.core import SpectroscopyExp

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(SpectroscopyExp())
record = exp.run(cfg)
exp.analyze(None)
saved = exp.save_run(point_view, source=record)
```

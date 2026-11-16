# Heralded conditional spectroscopy

**讀者：** operator、agent。本頁說明獨立實驗 `two_qubit/conditional_spectroscopy_heralded`。Probe、contrast、DC source、角色捕捉與校準步驟見 [普通版本](guide.md)。

Cfg 掃描欄位使用 `dc`，此實驗的 cfg schema 為 3.0。載入舊主版本的 run 時保留 Result 與 metadata，cfg 為 None，並發出 `CfgUnavailableWarning`。這些舊 run 不支援重新分析，也不提供 cfg 遷移或欄位 alias。

## 序列與校準

每個 shot 執行可選 reset → herald mux readout → 兩顆準備 → 雙線 weak probe → 可選 Xπ⊗Xπ contrast → terminal mux readout。

`prepared_f1`、`prepared_f2` 都是 2 點硬體 loop，0 不發射 pulse，1 發射該 qubit 的 Xπ。F1 在外層。準備的兩條 timeline 從 reset 與 herald 的結尾 fork，各做準備，join 後才 probe。兩條 probe 同時播放。Contrast 保留原先的 F1、F2 依序 Xπ。

Herald 與 terminal 重複使用同一個 mux 物件、gain、積分窗口與 circle classifier。Readout 的 post_delay 是 herald 後的等待。F1/F2 tone 都須含 radius，缺少時先做 GE Post。`reset=None` 不發射 reset。Herald 不重設 qubit，也不決定是否執行後續 pulse。

Frequency 在硬體掃描。只有 DC 留在 host，因為需要設定儀器。每個 DC 一支 program 取完兩個準備軸與 frequency，rounds 在取數端累加。

## 先驗準備，再找條件頻移

1. 先核對 F1/F2 的 tone mapping、Xπ pulses，以及符合本次 readout 設定的 circle classifier 與 radius。
   讀出設定改變後先重做 GE Post，不以 herald gg 比例代替分類器校準。
2. 固定已授權的 DC，不做 DC 掃描。將 `drive.f1.gain`、`drive.f2.gain` 都設為 0，
   `contrast=false`，frequency 只留一個哨兵點，保留原來的 probe duration 與等待時間。
   四種名目準備 B00、B01、B10、B11 仍會全部執行。
3. 從保存的 81 類 `sums` 選 herald gg，對每個 Bxy 取 terminal 的九類計數。
   將各類計數除以該 Bxy 選中的全部 shot 數，得到 `P(ab | Bxy, herald=gg)`，other 留在分母。
   零選中計數時不能判定 prep，先取得可用計數。檢查 B00 的 P00、B01 的 P01、B10 的 P10 與 B11 的 P11。
   比較 `P11 | B11` 與 `P(F1=e | B10) × P(F2=e | B01)`。
   P11 與乘積在 shot 統計不確定度內相符，且其餘名目準備對應到預期 terminal 類別時，才繼續找條件頻移。
   若 P11 明顯偏低，先查 prep 的條件失諧與讀出分類，不用 contrast、relative phase 或強 probe 解釋缺失的準備。
   此檢查支持準備分支存在，不扣除分類、herald 擾動與等待造成的系統誤差，也不是純態或 gate fidelity 門檻。
4. Prep 通過後恢復待測線的弱 probe，先保持 `contrast=false`。
   用長弱 probe 讓有效線寬小於要辨別的條件頻移，頻率步距也要能解析該頻移。
   Pulse 時長 T 的 Fourier 尺度約為 1/T，MHz 與 us 互為倒數；功率展寬與退相干也會限制解析度。
   例如比較 F2 在 B00 與 B10 的譜線位置時，可將 F1 probe gain 保持 0。
   未解析時記錄線寬與步距所限的解析度，不報零頻移或零耦合。
   找到條件線後才比較 contrast、relative phase 或較強 probe，並為各設定分別 Run、Save。

## 成本、進度與 Stop

每個 DC 都執行 `4 × N_frequency` 個準備態／頻率座標。
要求的 shot 總數為 `N_DC × 4 × N_frequency × reps × rounds`，每 shot 都含 herald 與 terminal 兩次 mux readout。
估硬體時間時，將各 shot 的 reset、準備、probe、contrast、兩次 readout 與等待時間加總，
再計入 program 啟動、資料傳輸與 DC 設定成本。要求的 shot 數不代表已完成的 shot 數。

FTF mock 的 heralded 計算成本主要隨頻率點數與四個準備分支增加，每個座標需處理 herald 後的演化與讀出。
降低 reps 不會按比例降低這部分成本。這是 mock 的成本觀察，不表示真實硬體時間與 reps 無關。
先用同一 adapter、相同 pulse 時長、reps 與 rounds，在單一 DC 跑完 1–3 個頻率點的牆鐘 pilot。
記錄牆鐘時間、要求的硬體時間估算，以及實際返回的每點 shot 數，再估完整窗口的成本。
不要把數百或數千點寬掃當作 pilot，也不要靠中途 Stop 取代少點測量。

進度的外層是 DC host loop，frequency 與四種準備在同一支 program 內，不逐頻點回報進度。
`rounds=1` 時沒有 rounds 進度條，因此外層 DC 長時間不動不代表沒有執行。
`rounds>1` 時可看到已完成 round 的進度；live 圖與計數也只在完整 round 返回後更新。

Stop 是合作停止請求，不保證立即返回。當前未完成 round 的計數不進 Result，已完成 rounds 與先前 DC 的資料保留。
若 `rounds=1` 且在第一個 DC 的首輪完成前停止，沒有可用計數，`sums` 保持 NaN。
若已有 DC 完成，不能將當前 DC 的空資料誤認為整次 Run 都沒有結果。
Stop 返回後先核對 Shots 圖與保存資料，再決定是否重跑。DC 不自動恢復，繼續前確認裝置當下值。

## 依實際初態比較 gg 與 ee

一般結果按 Analyze 後會發布圖並回傳 `None`。頻譜、shot 數、完整性與 herald 比例只在分析內供畫圖使用。

若 `sums` 全為 NaN，Analyze 仍發布空圖，但改回傳 `error`，說明沒有完整 acquisition round 的資料。MCP 讀取分析結果時會在 `summary.error` 取得這個原因。Save 仍可保存原始 NaN 計數與本次 cfg。

分析與 live 都使用 herald 判為 gg 和 ee 的兩組 terminal 計數。兩組保留各自的 P_e，不合併，也不擬合。Other 留在分母。零選中計數或未量點為 NaN。

每個 DC 各有 F1、F2 頻譜，每張圖的四格是 probe 前的實際準備態。Herald gg 的名目準備 `(f1,f2)` 與 herald ee 的名目準備 `(1-f1,1-f2)` 疊在同格。這裡的翻轉只指 probe 前的準備 Xπ。Probe 後的 contrast 會改變 terminal 分配，圖標題列出是否啟用。

分析內的 `gg`、`ee` 頻譜以名目 `prepared_f1`、`prepared_f2` 排列。畫圖時才對齊實際初態。兩者的 `acquired_shots` 都是全部 81 類的每點總數，`complete` 以每個頻率是否取得 reps × rounds 判定，不要求每個 shot 都通過 herald。

Shots 圖按名目準備顯示全部 shot 數、選中 gg 與 ee 的數量，以及要求的 shot 數。取消後仍顯示已完成 rounds 的計數，不丟掉部分掃描。

## Herald 九類比例

Herald 圖按 DC 與名目準備顯示各頻率的九類聯合比例。類別依 F1、F2 的 g、e、other 排列。Ge、eg 及含 other 的 shot 都保存，但不進 gg/ee 頻譜。

`pass_fraction` 是 herald gg 計數除以全部 shot 數，**不是 ground-state purity**。它包含 classifier 的誤判與 readout 效應。沒有 shot 的點為 NaN。

## 保存與 DC

Result 保存 float64 `sums`，形狀為 `DC × prepared_f1[2] × prepared_f2[2] × frequency × 3 × 3 × 3 × 3`。末四軸是 herald F1、herald F2、terminal F1、terminal F2，各為 g、e、other。保存全部 81 類總和，不丟棄未選中的類別，也不換成比例。舊 per-shot IQ 與單一 branch 軸格式拒絕載入。

`dc_device=None` 保持單點 DC 哨兵 `[0]`，不動 source。啟用具名 DC 時不補償、不恢復，停止後先確認裝置當下值。Save 保留本次 cfg。Analyze 不從目前表單或 live library 補歷史設定。

## Notebook

先建立同樣通過分類器驗證的 `SpectroscopyHeraldedCfg`，並取得已授權的環境 handles。`point_view` 是保存目的地的 bound ParameterView。Notebook 經 adapter 管理 scope、plots 與 RunRecord，不直接呼叫 core。

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.conditional_spectroscopy.core_heralded import SpectroscopyHeraldedExp

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(SpectroscopyHeraldedExp())
record = exp.run(cfg)
exp.analyze(None)
saved = exp.save_run(point_view, source=record)
```

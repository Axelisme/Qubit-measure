**讀者：** operator、agent。設定雙線 conditional spectroscopy，從四種名目準備的終端激發頻譜選擇下一次掃描窗口。

# Conditional spectroscopy

每發做「可選 reset → 名目 g,g／g,e／e,g／e,e 準備 → 雙線弱 probe → 可選雙 Xπ 對比脈衝 → mux 終端讀出」。準備和讀出標籤都按 Qubit1、Qubit2 排列；g 用 I，e 用 Xπ。兩顆都要翻轉時，準備脈衝同時開始。兩條 probe 各用自己的 drive line，以同一個 RF 頻率並行播放。沒有 reset 時，四種名目準備仍包含熱布居。

## 設定表單

1. 在 point 綁定兩個不同元件到 `$qubit1`、`$qubit2`，並將 `$mux` 綁到包含兩者的 MuxGroup。[核對 qubit 與 resonator 順序](../guide.md#綁定位置與讀出-tones)。不需要 `$coupler`。Refresh 保留原稿並以新來源核對；Defaults 按目前 bindings 和 library 重建初值。
2. **Qubit1 Xπ** 和 **Qubit2 Xπ** 分別從各自的 `x180` slot 複製。缺 slot 時留空。用 amplitude／length Rabi 校準的翻轉脈衝。**Mux readout** 從 `$mux.module.readout` 引用完整讀出；缺 slot 時手填 blank mux。保留完整 tone 順序，用同一設定的 [mux GE 和 GE Post](../../mux/singleshot_ge/guide.md) 校準兩顆 qubit 的 g/e circle center 與 radius。改 readout frequency、gain 或積分時間後重校。
3. **Two-qubit spectroscopy probe** 內的 **Qubit1 drive**、**Qubit2 drive** 是頻譜 probe，不是準備用 Xπ。若有 `$coupler.module.cz_qubit1`、`$coupler.module.cz_qubit2`，表單複製兩個 slots 作為起點。CZ 脈衝的 gain 或時間不一定適合弱 probe。缺 slot 時給 cosine 包絡與各 qubit 的 drive channel，另填 **Gain** (a.u.)、長度 (µs) 與 **Nyquist zone**。兩線用各自 Xπ 的 channel，包絡長度和開始前等待相同；兩線的 gain 和相位可不同。相位用度數，gain 的符號保留。
4. **Common probe RF frequency** 的起訖初值空白，點數初值 101。填有望涵蓋目標頻線的 RF 區間 (MHz)。每個 frequency 點覆蓋兩條 probe 原本的 RF；其他 pulse 設定不變。圖和存檔以要求的頻點為座標，不是獨立量到的 RF 頻率。
5. **Double Xπ contrast** 預設開啟。開啟時，probe 結束後依序播放 Qubit1、Qubit2 Xπ，再讀出。關閉時，probe 直接接讀出。縱軸永遠是終端 P(e)，不推回 contrast 脈衝前的布居。
6. **Coupler DC source** 初值空白，不動外部 DC。要掃 bias 時，選連接的 source、**DC source unit**（安培、伏特或 native）和以該單位填寫的 **Coupler DC bias sweep**。此掃描初值為單點 0，source 留空時忽略。實驗不補償其他 loop，不在結束或取消時恢復 source，繼續操作前先讀回裝置值。
7. **Reset** 留空時省略；可選已校準的準備 reset 或下述 herald filter。**Reps** 初值 1000，是每個準備與頻點每輪嘗試的 shots。**Rounds** 初值 10，在每個 bias 點重複全頻掃描並疊加計數。**Initial wait** 在每輪第一發前，**Relaxation wait** 在每發後。兩者初值 1 µs，應按已量到的弛豫與腔場放空時間核對。

## 讀 Run 與 Analyze 圖

兩個 qubit 的 P(e) 都是該 qubit 判為 e 的次數除以當點保留的終端 shots。分母包含 g、e 和 other。這是分類後的邊際機率，不是平均 IQ，也不是純態、經混淆校正的激發布居。Run 與 Analyze 的一維頻譜用折線連接量測點，並以同一頻點保留 shots 的分類計數估計 P(e) 的 shot SEM。少於兩發時不畫誤差棒。

- **Terminal P(e)** 每個 bias 點一張，四個 panel 分別是 g,g、g,e、e,g、e,e。橫軸是兩線共用的 RF (MHz)，兩種顏色是 Qubit1 和 Qubit2。看同一顆 qubit 的峰谷是否隨另一顆的名目準備改變。Run 和第一張 Analyze 用同一組座標、顏色與比較尺度。Run 另列已取樣的準備／頻率點數與保留 shots；有取樣但沒有通過 herald 時，該點仍不畫機率。
- **Terminal P(e) map** 在掃過多個 bias 點時，分開畫 Qubit1、Qubit2。橫軸是 RF，縱軸是 DC source 座標。四個準備分支共用 0–1 色階，可沿著亮帶或暗帶追蹤同一條頻線隨 bias 的移動。
- **Retained shots** 每個 bias 點一張，橫軸也是 RF，縱軸是對應準備的保留 shots (count)。點在 0 表示沒有 shot 通過；缺點代表未取數。用這張圖區分沒有資料和頻譜確實平坦，不把沒有保留 shot 的點當成零激發機率。

版式參考：Ding 圖 5-8(b)（頁 151）用二維色圖追蹤共振。本實驗圖掃 RF 與 DC source 設定，不掃 CZ pulse plateau；色階是所選 qubit 的終端 P(e)，不是指定高階態布居。

## 選下一次掃描

同一條頻線在 Qubit1 的 g,g 和 g,e panel 偏移時，先讀各自峰或谷的採樣頻率。兩者相減是在目前步距下看見的名目分支差，不是擬合值。若下次要確認一個候選頻率，以該峰谷的採樣頻率為中點，縮到仍涵蓋兩側基線，並保留足夠的點。例如目前步距 2 MHz，峰只佔 1 至 2 點，可把步距改為 0.5 MHz，中心不動；若最強的變化仍在掃描邊界，先往邊界外延伸。切換 Double Xπ contrast 時各跑一次，同一個準備和 qubit 比較，不把峰谷反轉先解讀成新躍遷。

沒有峰擬合或自動寫回。熱混合、Xπ 準備誤差與讀出分類都可能改變對比。這組資料不能單獨識別高階態、量化 leakage、決定 Hamiltonian 參數或給出可寫回的 CZ frequency。要採用候選，需用獨立校準確認躍遷並做後續 CZ 掃描。

## 篩選、部分取數與保存

Reset 可選 `reset/mux_herald`。使用和終端相同的 mux 完整讀出，`keep` 的鍵填 tone 的 zero-based 位置，不填 ADC 編號。例如 pair 在 0、1 可填 `{'0': g, '1': g}`。頻譜和 counts 只含通過篩選的 shots；接受率不是純度。

停止時保留完成的 rounds；未量點沒有機率也沒有計數值。Analyze 顯示來源 run 和保存的 Reps×Rounds，不用現在的表單重建過去資料。Save 保存四種準備、頻率與 DC 軸上的原始九類計數與本次 cfg，不保存單發 IQ。位置軸格式不同的舊檔、原先獨立 heralded 入口的舊檔，須用寫出該檔的舊版開啟。

## Notebook

建立 `SpectroscopyCfg` 時，以完整讀出中的 zero-based 位置填 `qubit1_tone`、`qubit2_tone`。NotebookAdapter 擁有 RunRecord 和作圖。`point_view` 是保存目的地。

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.conditional_spectroscopy.core import SpectroscopyExp

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(SpectroscopyExp())
record = exp.run(cfg)
exp.analyze(None)
saved = exp.save_run(point_view, source=record)
```

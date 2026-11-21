**讀者：** operator、agent。用延遲頻譜追蹤 fast-flux 脈衝前後的 qubit 共振，分辨頻率位移、有限探測時間與可能的長尾。

# Fast-flux 暫態頻譜

## 量測序列

開啟 `fastflux/distortion/freq`。每個 shot 先執行選填的 Reset，再從同一時間起排三條分支：

1. 等待 Flux pulse 的 Pre-delay，播放 flux 脈衝。
2. 等待 Qubit probe wait，再以 Qubit probe frequency 指定的 RF 載波探測躍遷。
3. 等待 Minimum readout wait。

三條分支都結束才讀出。Minimum readout wait 設的是下限，不是絕對讀出時刻。Relax delay 是每個 shot 後供 qubit 弛豫的等待。

## 選擇來源與掃描

先校準 qubit 頻譜與 resonator 讀出。Open 和 Defaults 會捕捉當時選定 point 的 `qubit`、`resonator` 及 library。切換來源後按 Defaults 重新取得初值，會捨棄表單編輯。

| 欄位 | 初值與設定用途 |
| --- | --- |
| Reset | 取 qubit 的 reset slot。缺少或停用時，序列不播放 reset。 |
| Flux pulse | 取 `flux_pulse` 範本。通道來自 qubit 的 fast-flux 接線。初值為 0 MHz 矩形脈衝、gain 1.0 a.u.、Nyquist zone 2、長度 0.5 µs。先確認接線與允許的 gain，設定長度及 Pre-delay。這些初值不是線路校準。 |
| Qubit probe | 取 qubit 的 microwave drive 通道。初值 gain 0.1 a.u.、長度 0.05 µs。頻率由掃描欄位指定，並非模組內的固定載波。 |
| Readout | 優先取 resonator 的 readout slot，缺少時用可編輯範本。選 resonator 的校準讀出，核對 ADC 積分窗完整落在接收脈衝內。 |
| Qubit probe wait | 從平行分支起點到探測脈衝開始前的等待。預設 0 到 1 µs，共 101 點。預留脈衝前的基線與回程後的尾巴。 |
| Qubit probe frequency | 絕對 RF 頻率 (MHz)。預設以 qubit 頻率為中心，兩邊各 1.5 倍 qubit 線寬，共 101 點。若沒校準值，手填起點、終點和點數。 |
| Minimum readout wait | 預設 2 µs。若要固定讀出時刻，至少覆蓋兩條脈衝分支的最晚完成時間。 |
| Reps、Rounds、Relax delay | Reps 預設 1000，是每輪每個掃描點的 shot 數。Rounds 預設 10，是取完整張圖再累積平均的輪數。Relax delay 預設 1 µs，應用足夠的等待回到可重現的初態。 |

掃描選 `linrange` 時指定 Start、Stop 和 Expts，Expts 是格點數 (count)。需要以中心與全寬輸入時選 `center`。選帶 `_step` 的方式時以 Step 指定格點間隔，而非 Expts。兩條掃描的 Start、Stop 和 Step 都用該軸的單位。

模組內的 Gen ch 選 DAC 通道，RO ch 選讀出 ADC 通道。Gain 是 DAC 滿量程的帶號比例，Phase 設載波相位，Pre-delay 和 Post-delay 控制播放前及播放後的等待。Waveform 決定包絡及長度。Readout 的 Frequency 是讀出與解調的 RF 頻率。Trig offset 是 ADC 開窗相對讀出播放的延遲，RO length 是積分窗。Mixer freq 只有該路由使用 mixer 時填入，NQZ 選相應的 DAC Nyquist zone。即使設定 Classifier，這個實驗也不使用分類結果。模組接線與 timing 有校準來源時沿用該來源，不從這張圖估算。

## 從圖找暫態

Run 和 Analyze 使用相同座標與色階量。橫軸 Probe-center time 是掃描的等待加半個探測脈衝長度。縱軸是探測脈衝的 RF 頻率。每一格先平均 complex IQ，再取 IQ 掃描資料最大散布方向的投影，單位 a.u.。它是未分類的讀出對比，不是態機率。投影隨目前資料更新，顏色深淺只在當前色階內比。Run 的圖標明已有 IQ 的格數。每輪取得整個矩形掃描，第一輪後格數可達全部，後續 rounds 繼續累積平均；以 Run 進度區看輪數。

Analyze 在同一熱圖上用黑點標每個時間切片的 Lorentzian 中心，豎線是中心的 fit 標準誤。標準誤由擬合共變異數估算，不是逐點 shot SEM。熱圖本身不畫誤差棒。分析跳過未取完的切片，不按殘差或 R² 自動選點。若擬合點不沿著連續亮帶或暗帶，或落在頻率邊界，不能只看標準誤就相信峰位；平坦的切片也可能產生無意義的中心。

綠線是矩形參考，不是對線路的擬合。跳變時刻取 Flux pulse 的 Pre-delay 與 Pre-delay + 波形長度。兩段高度取自擬合中心：從第一個中心算絕對位移，分別平均最小和最大的約一成。灰帶是邊緣 ± 半個探測脈衝長度。探測脈衝在這段會跨越邊緣，灰帶內的展寬或頻率中間值不能單獨當成線路失真。

先看共振帶是否在掃描窗內，再讀灰帶之外相對於脈衝前基線的偏移。要量回程的沉降時間，定出可接受頻差，例如 2 MHz。以連續數個已回到基線容許範圍內的探測中心時間，減去編程脈衝的結束時間，作為回程等待的量測依據。若掃描到 0.30 µs 仍未回到門檻、步距是 0.02 µs，下次把 Qubit probe wait 的終點加 0.10 µs，並加 5 點保留步距。若共振貼頻率邊界，先向該側擴至少一個觀察到的線寬，再判讀峰位。這張圖不能單獨決定補償係數。

## 保存與適用範圍

Save 保留等待與頻率座標、complex IQ 和來源設定。Load 不改目前表單，Analyze 需要保存的脈衝設定才知道時間原點與寬度。可見摘要列的是來源 run 的 Reps×Rounds，不是這次表單新填入的值。若設定缺失，資料仍可載入，但不能做時間對齊分析。

驅動 gain 到磁通要另做校準。未辨識的鄰近躍遷、探測造成的展寬和模擬的暫態，都不能當成實機 transfer function。此實驗沒有發布 recipe 或校準寫回。

**讀者：** operator、agent。設定 conditional Ramsey，用兩個 spectator 分支估計 idle ZZ 頻差。

# Conditional Ramsey 與 idle ZZ

選一顆 qubit 做 Ramsey，另一顆當 spectator。每發依序做「可選 reset → spectator I 或 Xπ → target π/2 → 自由演化 → target π/2 相位投影 → mux 終端讀出」。比較兩個名目 spectator 分支的 Ramsey 條紋頻率。若 Reset 留空，spectator I 或 Xπ 都作用在原來的熱布居上，不保證對應純 g 或 e 態。

## 設定表單

1. 綁定 `$qubit1`、`$qubit2` 到不同元件，`$mux` 到包含兩者的 MuxGroup。舊 point 要補位置 aliases；不需要 `$coupler`。[依家族導覽核對 qubit 與 resonator 的順序](../guide.md#綁定位置與讀出-tones)。Refresh 保留原稿並重查來源；Defaults 從目前 aliases 和 library 重建初值。
2. **Ramsey target** 初值是 Qubit1。選中的 qubit 播兩個 Xπ/2，另一顆的 Xπ 用於 spectator Xπ 分支。兩顆的 **Xπ**、**Xπ/2** 分別從各自的 `x180`、`x90` slot 複製；缺 slot 時留空。使用 amplitude／length Rabi 校準的脈衝。同一顆的兩個脈衝必須在同一 drive channel，兩顆使用不同 channel。頻率用 MHz、signed gain 用 a.u.、時間用 µs。
3. **Mux readout** 從 `$mux.module.readout` 引用完整讀出。缺 slot 時填好 blank mux。保留原有 tone 順序，並用同次讀出條件的 [mux GE 和 GE Post](../../mux/singleshot_ge/guide.md) 為兩顆 qubit 設定各自的 g/e circle center 與 radius。**Reset** 留空時略過；選已校準的準備 reset 或下述 herald filter，會在兩個分支發出。
4. **Ramsey free-evolution delay** 初值為 0–10 µs、101 點。實際量到的延遲必須等距、有正跨度且至少六個不同點。**Programmed Ramsey detuning** 初值空白。第二個 π/2 的相位是第一個 π/2 的相位加上 `360° × detuning (MHz) × actual delay (µs)`。這是相位斜率，不是改 pulse 的 RF。從近期 Ramsey 頻率誤差和預期的 `|ζ|` 估算，選絕對值明顯大於兩者之和的非零 detuning，讓兩分支條紋在同一頻率符號側。
5. 檢查取樣間距和掃描跨度。起點可用每個預期最快條紋週期至少 10 點，即間距小於 `1 / (10 × |f_max|)` µs。至少涵蓋數個週期；若還要用衰減包絡檢查模型，窗口可先取約 `3 × T2*`。例如預期最快條紋 2 MHz，間距不超過 0.05 µs。太疏可能把高頻看成低頻，窗太短時會看見不足一個週期或良好擬合但頻率誤差很大。
6. **Coupler DC source** 初值空白，此時不動任何 DC。要掃 bias 時選連接的 source，設定 **DC source unit**（安培、伏特或 native）及同單位的 **Coupler DC bias sweep**。初值是停用的單點 0；啟用前填已核對的區間。實驗不另換單位、不補償其他 loop，結束或取消後不恢復 bias。
7. **Reps** 是每個分支和時間點每輪嘗試的 shots，初值 1000。**Rounds** 在每個 bias 點重複整個延遲掃描並疊加計數，初值 10。**Initial wait** 是每輪開始等待，**Relaxation wait** 是兩發之間等待，兩者初值 1 µs。按量過的弛豫與讀出腔放空時間設定。

## 讀 Run 與 Analyze 圖

每個延遲和 spectator 分支有九類終端聯合計數。縱軸 `P(e) − P(g) = (N_e − N_g) / N_total` 是 target 的分類對比。`other` 包含在分母，單發對比值為 0；e 記 +1，g 記 −1。這不是 IQ 實部，也不做讀出誤差修正。每點誤差棒是這組單發值的樣本標準差除以 `√N_total`。`N_total` 只含保留的 shots。少於兩發時不畫棒。這不是單發散布或擬合參數誤差。擬合沿用原先的計數加權和飽和點 `1/N_total` 權重誤差下限，圖上的 shot SEM 不使用該下限。

- **Run 的 Ramsey contrast** 每個 bias 點一張，兩種顏色各代表 spectator I 和 Xπ。橫軸是實際延遲 (µs)。看兩支條紋是否已出現，以及標題中的分支時間點取樣數和保留 shots 計數。沒有保留 shot 的點不畫對比。
- **Analyze 的 Ramsey contrast** 保留同樣的量測點和誤差棒。實線是各自的衰減餘弦，不是額外取樣。圖下列各分支條紋頻率及其 1σ 擬合誤差。R² 用來衡量模型解釋量測起伏的程度。Normalized residual RMS 是殘差 RMS 除以量測對比的峰對峰範圍；越接近 0，模型殘差相對越小。Relative frequency error 是頻率 1σ 誤差除以頻率絕對值。沒有 R² 或殘差的自動通過門檻；對照實線、點和殘差是否有系統差異。圖下另列兩分支的 ζ 與誤差。
- **Conditional fringe shift** 在有多個 bias 點時顯示 ζ 對 bias 的關係，編號對應各張條紋圖。縱向誤差是兩分支頻率擬合誤差的平方和開根，不是該 bias 點的 shot SEM。缺 ζ 時回看該分支未能辨識的原因。

版式參考：Ding 圖 5-9(b,e)（頁 152）把 Ramsey 類序列和相位判讀放在一起。本實驗沿自由演化時間畫兩分支的終端分類對比，用條紋頻差估計 ζ，不掃 CZ 驅動頻率，也不提 π 相位的位置。

## 決定是否採用頻差

在同側假設下，`f = sign(detuning) × |f|`，`ζ = f(Xπ) − f(I)`。共同的人工相位斜率在兩頻相減時抵消。例如 detuning 為負，兩分支 `|f|` 為 1.80 與 2.10 MHz 時，ζ 是 `−2.10 − (−1.80) = −0.30 MHz`。不能只減兩個絕對值。若 `|f|` 到 0 或 `2 × |detuning|` 的距離不超過 `max(1/時間跨度, 頻率 1σ 誤差)`，原分析不給 signed `f` 或 ζ。圖上會列原因和辨識界限；有 fit 線也不代表頻率符號已辨識。

若窗長限制辨識，例如 3 µs 的跨度讓 `1/span ≈ 0.33 MHz`，要把此項降到 0.10 MHz，延長到至少 10 µs，並保留原取樣間距。若兩分支可能跨過零頻，先用新校準的 target RF 頻率降低已知誤差，或在仍可充分取樣的範圍內增大 detuning 的絕對值。若條紋可見但單頻模型有系統偏差，先檢查熱布居和兩顆的 Xπ 校準。ζ 是名目分支的單頻估計，不能由此排除混合頻率或宣稱量到純 Hamiltonian ZZ。多 bias 點可先用 `|ζ|` 和擬合誤差挑較小且已量過的候選，再在該 bias 複測；不從個別 ζ 自動推算偏壓步進或寫回校準。

## 篩選、部分取數與保存

在 Reset 選 `reset/mux_herald` 可只分析通過 herald 的 shots。用與終端相同的完整讀出，`keep` 的鍵填完整讀出的 zero-based tone 位置，不填 ADC 編號。位置 0、1 的 pair 可填 `{'0': g, '1': g}`。此時所有終端機率都以通過篩選的 shots 為分母；通過率不是純度。

停止時保留已完成 rounds，未量點留空。任一分支的必要延遲缺資料或沒有通過篩選的 shot，就不給該分支的頻率。Analyze 摘要顯示來源 run 與保存的 Reps×Rounds，圖分析使用保存的 cfg 和計數。沒有可還原的 cfg 時不以當前表單代替。Save 保留全部實際延遲與終端九類計數，不保存原始 IQ。不相容的舊位置軸或獨立 heralded 實驗的檔案須用寫出檔案的版本開啟。

## Notebook

以 `IdleZZCfg` 從完整讀出順序填 `qubit1_tone`、`qubit2_tone`，index 從 0 起算。經 NotebookAdapter 保留 RunRecord，`point_view` 是保存目的地。

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.idle_zz.core import IdleZZExp

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(IdleZZExp())
record = exp.run(cfg)
exp.analyze(None)
exp.save_run(point_view, source=record)
```

**讀者：** operator、agent。寬頻掃描 resonator 反射，選出凹陷位置供下一次窄掃。

# One-tone range find

每發 shot 在同一個絕對 RF 播放並解調 Readout，再積分 ADC window 得到 IQ。每輪掃完整條頻率軸，不打 qubit drive，也不做態分類。

## 設定掃描

- 綁定 resonator。先做 [lookback](../../lookback/guide.md)，核對 Readout 的 generator、mixer、ADC、gain 與時間窗。**Trig offset** 起點是 flight time + `5/(2π·κ)` µs，κ 以 MHz 計；無 κ 時只用 flight time。缺 flight time 時 trigger 與 pulse length 都是必填空欄。Pulse length 至少涵蓋 trigger 加 **RO length**，不短於範本。
- **Post-delay** 讓腔放空。從 `5/(2π·κ)` µs 開始；校準此等待時，比較 herald → 名目 Xπ 在此等待與更長等待下的結果。
- **Readout RF frequency** 的 Start／Stop 初始留空。填絕對 MHz，非 mixer offset；使用等頻距且至少 11 個不同頻率點供 Analyze 找谷。讓每個預期凹陷離邊界有餘裕、寬度跨數個點。Run 不需要 resonator 名單。

開啟表單、更新來源、採用與保存見 [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md)。

## 讀圖與選谷

Live 顯示 `|mean(I + iQ)|`（a.u.）與量化後的絕對 RF（MHz）。曲線是累積平均，不畫逐點 shot SEM。Stop 保留已完成 acquisition 的資料；Analyze 不接受部分或非有限曲線，需重掃完整頻窗。

**Resonators (JSON)** 在 Run／Load 後預設為 `["R1"]`，不是現用 alias。填非空、名稱非空白且不重複的字串清單。自動猜測最多取名稱數量的候選，再按頻率升冪配對。名稱改了須重開 Analyze。猜測不證明身分。

- **Min separation (MHz)** 限制候選最小間距。**Smoothing sigma (samples)** 以取樣點為單位，0 不平滑。**Max FWHM (MHz)** 篩除過寬候選。**Prominence noise factor** 留空時按預期谷數自動選倍率，**Actual noise factor** 顯示實際值。修改 controls 會重算候選與選點，只保留頻率未變的元件對應。
- 灰點是自動候選、紅星是選到的量測點。點曲線可新增最近的量測點；點星號或用 **Remove selected dip** 刪除。用 **Selected dip** 和 **Component** 指定元件或 **Unassigned**。同一元件改配另一點時，舊點會變成 Unassigned。
- **Done** 至少需一個有效選點，才發布結果；**Cancel** 保留前次成功結果。Done 後的 depth 是 prominence，FWHM 是半 prominence 寬度，score 是 depth / FWHM。這些不是擬合線寬或誤差。手選點不是目前的自動候選時，三個值留空。

## 採用頻率

若凹陷在邊界，延伸掃描；若只有一兩點寬，縮小步距重掃。先核對身分，再用量到的絕對頻率當 [one-tone 窄掃](../freq/guide.md)中心，另校準線寬。

Done 對每個具名 dip 預選 `<name>.freq`；Unassigned 不提案。採用時更新作用中文件組的頻率；名稱不存在則建立 resonator，所以要核對拼字。任一頻率列失敗，所選列整批不寫。若沒有歷史頻率證明目前排序，摘要與預覽提示身分待確認，但不封鎖寫回；即使有排序也須核對。

## Recipe 輸入

`onetone_range_find` 的 `freq`、`n` 和非空 `resonators` 不可省略或填 null；schema 在開 tab 前即拒絕。Analyze 仍會拒絕無效或重複名稱。Null `reps`／`rounds` 清空表單值；未以 overrides 補齊時在 Run readiness 失敗。省略 `min_separation` 不覆寫；非正或非有限值在輸入驗證失敗，null 則在 Run 完成後的 Analyze 參數驗證失敗。

Run 後用 `tab_interact` 審閱 picker，調整選點和對應並送 `done`，即使接受初始猜測也要送出。Done 成功才詢問寫回。Analyze 失敗不詢問寫回，仍保存完成的 Run。

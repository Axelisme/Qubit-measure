---
date: 2026-11-21
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - ../adr/0083-program-macros-and-timelines.md
  - ../adr/0084-simulation-integer-time-base.md
  - 2026-11-21-chain-readout-spacing.md
basis:
  integration_commit: 7df1ebc5910c5b993926704342112a4a60c46d62
  main_base: 6e44c21321f8099993e20346fce2ff49daac5123
  external_sources: none
---

# FTF 合法讀出時序：ADC image、drive admission 與到達時間缺口

## 結論

本篇所述的 task 裁決均由 designer 代裁。其中 raw 字的 DDS 同餘類與消費端的同餘距離，使用者同日覆核後選擇保留；其餘尚未覆核。使用者原則「模擬器要給出合法的預期結果，只要合法，不因為不合理就抱錯」（2026-11-21）是本題的約束。

- 具名 ADC config 保留作者指定的物理 image；真正沒有具名來源的 raw 頻率字按該 ADC 的 DDS 同餘類解調。
- 拿 readout 頻率和物理頻率比較的消費端，依同一個頻率語意標記選絕對距離或同餘距離。Six 依 raw 字選 cavity 時，同餘距離最近的 cavity 不唯一就拒絕，不默默任選。
- FTF routes 只表示設定中的實際 ADC 接線；不接 ADC 的 charge 與 fast-flux DAC 由 `drive_generators` 明示 admission。
- block 邊界按接收端到達時間判斷；在途 pulse 的到達區間與 ADC window 重疊、該 pulse 卻不在該 window 的 event 裡時，規劃時拒絕，不產生缺訊號 IQ。
- 整個 `TimelineTemplate` 沒有 readout pulse 時，pulse path 產生 receiver-only event；單一 template 內只有部分點沒有 pulse 時仍拒絕。
- 完整的讀出到達時間模型尚未實作；在途配對缺口由守門暫時攔住，trigger lookback 與跨 block 腔場記憶仍待設計。
- chain_length 的端到端 Run 移到落地後第 6 步的排他重測批次；本次只完成離線編譯與規劃驗收。

## 問題

FTF 原先把每個 generator 與每個 receiver 的笛卡兒積都當成 ADC routes。這讓 ADC0 看似接著 gen0、gen2、gen3、gen7，儘管 default settings 只有 gen0 接 ADC0、gen7 接 PFB ADC2／3，gen2／3 是 charge DAC。

`Outputs._routes` 在多個候選接線中依 window 內的發射事件挑 generator。它按發射時間比較，不把 receiver 的飛行時間加進去。

tProc 控制的 ADC 原先要求 window 內有 pulse carrier，並用 carrier 作 DDS word 解碼的物理參考。ADC config 在 window 起點已生效，即使該 window 沒有 carrier。

整段沒有 pulse 的 pulse path 原先不能形成 receiver-only event。跨 block 時，讀出 window 也可能延伸到下一個 pulse block 的發射時間。

按發射時間分配的 event 可能漏掉一個已發射、但正在飛向 ADC 的 pulse。漏配時照常計算會回傳少了該 pulse 的 IQ。

唯一一次原樹 chain_length Run 在 250 秒期限時發出取消請求，當時停在 projective evolution；每步有 24 個能階及 14／15／15 次 projective reads。之後沒有觀察到 Run 的終態，程序以 exit 134 結束，沒有產生三點的端到端結果。

## 考慮過的選項與比較

- routes 若照實際 FTF settings 配對，ADC route 就不再把 charge DAC 當作 ADC source。`SimulationSoc` 因此另收 `drive_generators`，讓宣告的純 drive DAC 可通過 admission，而不建立假的 ADC 接線。板卡範圍與每個 window 的 ADC route 檢查仍保留，見 [program/v2 README](../../lib/zcu_tools/program/v2/README.md)。
- ADC frequency provenance 分成具名與無來源 raw 兩種。具名 config、同來源的 WMEM／`r_wave` 修改與 sweep 沿用具名的語意頻率；無來源 raw word 傳 DDS 同餘類標記與 `f_dds`，由 ADC 將 `f_demod − f_ref` 折入 `[−f_dds/2, f_dds/2)`。這遵循 [ADR-0072](../adr/0072-module-boundaries.md) 的明示邊界傳遞與 [ADR-0078](../adr/0078-evolution-module-boundary.md) 的數值 owner 分工，也保留 [ADC README](../../lib/zcu_tools/simulate/adc/README.md) 記錄的絕對 shot-time 相位原點。
- 頻率語意標記不只給 ADC 用。Single、Six 等拿 readout 頻率和物理頻率比較的地方，明示 image 照舊比絕對距離，DDS 同餘類比 `min_m |f_phys − (f + m·f_dds)|`。同一個 raw ADC6 程式因此選回 Q4；平手時物理上分不出 ADC 該看哪個 cavity，依 ADR-0078 的契約外 fast fail 拒絕。
- R4 用「下一個 block 起點 + 該 receiver 的 time of flight」判斷前一個 window 是否已結束。下一個 pulse 在前一個 window 內抵達時，兩個 pulse 放進同一 event 相干相加；同 ADC trigger 不重疊的規則仍依 [ADR-0083](../adr/0083-program-macros-and-timelines.md)。
- 因為 R4 仍按發射時間配對，另加規劃守門：若 pulse 的發射區間加上 flight time 與 ADC window 重疊，但該 pulse 不在該 event，就報出 ADC、window、generator 與發射時間並拒絕回傳 IQ。
- 完整地把所有 pulse 依到達時間重新配對、處理早於 pulse block 的 trigger，並將先前 block 的腔場接入 ADC，會擴大 `ReadoutEvent`／`ReadoutBatch` 與 projective history 的資料交接。這些資料型別由多個 simulator 共用；本 task 先用守門避免已知漏訊號，將完整模型留待專題設計。
- chain_length 的 0.2／1.2／2.2 µs 三點改在最終樹做離線規劃；0.2 µs 預期被守門拒絕，1.2／2.2 µs 預期通過。端到端 Run 併入落地後的排他批次，沿用 wall-time 工具的單列期限與 stack dump。

## 被否決的選項與理由

- 對所有 ADC config 一律把頻差以 DDS 頻率取模，假設具名頻率與共用參考之差小於 `f_dds/2`。盤點推翻了這個前提：具名 MG12 ADC3 為 7079.6527 MHz，共用 reference 為 7189.9614 MHz，差值 −110.31 MHz；PFB 的 `f_dds/2` 只有 19.2 MHz。全域取模使具名 IQ 的 5520／5520 個分量改變，因此具名 config 必須保留明示 image。
- 只把 raw alias 換成 folded baseband、仍以絕對時間直接解調。6000 MHz 與 folded 的 1084.8 MHz 會讓 36 個複數樣點整體旋轉 257.14°，72／72 個 I/Q 分量不相等；旋轉也隨 trigger tick 改變。
- 沒有具名來源的 raw 字改成編譯時拒絕，拿掉同餘類標記、ADC 取模與消費端的同餘距離。這樣機制最少，但 raw 字是實機接受的程式。使用者覆核時否決：「那我选，K 吧，感觉有些少频率的，比如说像是 one tone frequency，如果不是等间距的话，要先计算 table，然后存起来。感觉就会出现模拟上需要解读弱智的问题。」（2026-11-21）。
- 以指令種類（static 或 tProc）代替 config provenance 分流。具名 runtime config 也有語意頻率，指令種類不能判斷作者是否指定 image。
- 允許所有板卡範圍內的 DAC 發射。這會改變其他 simulator 的 admission，且沒有證據證明未宣告的 generator 有可模擬的物理路徑。
- 漏配在途 pulse 時仍照常給 IQ。chain_length 的 0.2 µs 點會靜默遺失一個與 window 重疊的 pulse；拒絕比回傳錯誤資料安全。
- 在本 task 一次完成完整到達時間模型。跨 block 場需要和 projective outcome history 對齊，還要統一時間、相位與 flight-time handoff；這超出本次局部守門的範圍。
- 在本 task 把 chain_length 三點端到端 Run 當作驗收門檻。原樹唯一一次 Run 在 250 秒期限時發出取消請求，沒有觀察到 Run 的終態，程序以 exit 134 結束。projective 分支成本又不能從輸入規模直接估算；三點改排到落地後的排他重測。

## 留下的問題

完整到達時間模型仍要處理三個缺口：在途 pulse 的正確 event 配對、trigger 早於 pulse block 時的 lookback，以及前一個 block 的 cavity field 如何進入後續 ADC event。FTF 目前按發射時間配對；漏配守門只拒絕不完整 event，並未修正配對。

trigger 早於 pulse block 的讀出仍被拒絕。ADC event 也不繼承前一個 block 的腔場；R1 fixture 的 `post_delay=0 µs`、0.99 µs window 單模估計，在 window 起點漏掉約 9.5% 的穩態場振幅，到 window 末端約漏掉 0.4%。[FTF environment README](../../lib/zcu_tools/simulators/ftf/environment/README.md) 記錄了這個限制。

本次離線驗收中，chain_length 三點都完成編譯；0.2 µs 被守門拒絕，1.2／2.2 µs 完成規劃。兩個 heralded 實驗的 14 份離線 timelines 通過，幾何與原輸出相同。這些結果不證明 E2E 完成，也不證明最短窗訊號正確。

落地後的排他重測批次才跑 chain_length E2E；0.2 µs 預期以 `run_failed` 結束，其他耗時仍是該批次的定位對象。原始硬體 ADC phase epoch、reset 與 trigger 到首樣點的相位沒有 RTL 或實測證據；本次 DDS 同餘類契約不聲稱模擬這些相位來源，符合 [ADR-0084](../adr/0084-simulation-integer-time-base.md) 的既有範圍。

## 依據

本次的使用者原則原文為「模擬器要給出合法的預期結果，只要合法，不因為不合理就抱錯」（2026-11-21）。ADR-0078 區分合法但不合理的參數與具體模擬器正確性缺口；漏配守門屬於後者。[ADR-0078](../adr/0078-evolution-module-boundary.md)

[FTF environment README](../../lib/zcu_tools/simulators/ftf/environment/README.md) 記載實際 routes、純 drive DAC、arrival-time guard、receiver-only event 與未交接的前一 block 腔場。[program/v2 README](../../lib/zcu_tools/program/v2/README.md) 記載 routes、`drive_generators` 與 raw word frequency provenance。[ADC README](../../lib/zcu_tools/simulate/adc/README.md) 記載具名 image、DDS 同餘類與 `time_of_flight_ns` 的數值約定。

[讀出鏈的 pulse 長度與間隔](2026-11-21-chain-readout-spacing.md) 記載 chain_length 不再等 ADC window 的既有時序背景。相關跨模組界線分別見 [ADR-0072](../adr/0072-module-boundaries.md)、[ADR-0083](../adr/0083-program-macros-and-timelines.md) 與 [ADR-0084](../adr/0084-simulation-integer-time-base.md)。

原始 dossier、angle reports、task 工作檔與驗收收據未入庫；本快照已寫入必要數字與結論，不依賴那些檔案或本機路徑。

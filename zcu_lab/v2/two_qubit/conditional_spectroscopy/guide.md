# Conditional higher-level spectroscopy

**讀者：** operator、agent。從 GUI 或 MCP 執行雙線條件頻譜量測，選擇後續 CZ probe 窗口。

**Last updated:** 2026-10-21

Readout 有 `$mux.module.readout` slot 時引用 library 的完整 tones 與 scalar 設定。缺 slot 時從 blank mux 範本開始，須手填全部必要讀出欄位。

`two_qubit/conditional_spectroscopy` 量四個熱初態上的名目準備分支。每 shot 做 `B00/01/10/11 → 雙線 weak probe → 可選 Xπ⊗Xπ → 完整 mux readout`。F1 是 branch 的第一個 bit，F2 是第二個 bit。B11 仍含熱 population，所有 shots 都保留，直接分析原始 ensemble。

## 開啟前確認

先依 [FTF bring-up 指南](../guide.md) 校準 F1/F2 的 Xπ pulses 與 mux readout。綁定 `f1`、`f2` 到不同 fluxonium，`coupler` 到本對 transmon，`mux` 到包含兩者的 MuxGroup。Group 的 `qubits` 與 `resonators` 必須有序對應。

Open／Reset 固定捕捉角色、pulse 與 library。Tone 由各 component 在 `qubits` 的位置決定，畫面只供核對。完整 readout tone 列表維持原順序，包含 F1/F2 以外的 tones。缺 Xπ 或讀出校準時，先補與實測條件相符的有效設定。修改 aliases 或 library 後重新 Reset，再核對整份 cfg。

## 設定與 Run

1. 檢查兩個 preparation Xπ pulses。每條 probe 必須使用相同角色的 charge generator，F1/F2 的 generator 必須不同。Coupler 只設 DC bias。
2. 設定 `drive.f1` 與 `drive.f2`。有 `$coupler.module.cz_f1`／`cz_f2` 時，表單複製兩個普通 pulse 作起點。先人工降低並確認 gain，使 pulse 適合 weak probe。沒有 slot 時，表單提供 cosine 包絡與捕捉的角色 wiring channel；frequency、gain、duration 與 Nyquist zone 保持未填，須人工設定。所有初值都須依已授權硬體範圍審閱。
3. 兩線填相同 base frequency、envelope duration 與 pre-delay。兩個 waveform 都要明填；可保留不同形狀，但 duration 必須一致。Gain 可帶符號，兩線的 gain 比例與 relative phase 由操作員提供並保持固定。兩線同時間原點播放，等待兩線及各自 post-delay 結束後才接後續 pulse。頻率單位 MHz、時間 µs、phase degree。
4. 填 Common probe frequency 掃描範圍與點數。每個 frequency 點同時把兩線 frequency 換成該座標，保留 gain、relative phase 及 waveform。保存軸是共同要求的 MHz 座標；實際硬體頻率依裝置解析度量化。
5. Double Xπ contrast 開啟時，probe 後依序對 F1/F2 各播一次校準 Xπ。關閉時直接接 readout。比較 on／off 須分別 Run 並保留各自 cfg。兩次量測的對比只供診斷，高階態 population 仍須另行校準。
6. 只掃頻率時，Coupler DC source 留空，所有 DC source 維持原值，結果中的 DC 座標 0 表示該軸停用。需要外層掃描時，填直接 device 名稱、working unit 與已授權 source values。支援 A、V 或 native，直接使用所填單位；F1/F2 bias 須另行確認。
7. 核對 reps、rounds、等待時間及完整 mux 積分窗口後 Run。執行順序是 DC → B00/01/10/11 → frequency → host round。Live 圖顯示各 frequency 已取得 shots 的總數；FIT 顯示下節的診斷圖。

掃 coupler source 也可能影響其他 loop。F1/F2 bias 與工作點由操作員設定並記錄，結束後 DC source 維持最後設定。取消或失敗後先確認裝置當下值，再決定是否繼續。

## 看頻譜與熱圖

FIT 提供頻譜與熱圖診斷，峰的位置由操作員判讀。每個完整 tone 各有一張四分支頻譜，畫各 DC 點的平均 I 與 Q。標題保留原 tone index、label、F1/F2 對應及 contrast 開關。啟用 DC 掃描時，另有逐 tone 的 I、Q 四分支熱圖；橫軸是共同 probe frequency，縱軸是具名 source 與 working unit。非 F1/F2 tone 也保留並顯示。

每個 DC／branch 必須取得全部 frequency、rounds、reps，才計算該分支頻譜。Summary 列 complete、incomplete 或 not_started，以及 acquired／analyzed shots。Incomplete 分支的 analyzed shots 為 0，圖上將該分支標為缺失。已完成的其他分支仍可看。

從四分支圖人工選下一步要調查的 transition 與 probe 窗口。平均 IQ 與雙 Xπ 對比是診斷 proxy。高階態與 leakage population、bare coupler 能級、EC/EJ/J、Hamiltonian 及 CZ frequency 都需其他校準或證據。本實驗只提供圖與 summary，後續校準由操作員另行選擇。

## 保存、取消與重分析

Raw complex IQ shape 是 `DC × 4 branches × frequency × (rounds*reps) × full tones`。Shot index 依 host round 再依 program reps 排列。每個 shot 的完整 tones 一起保存，保留原讀出的 ADC units/sample 與 normalization。

取消保留已完成 rounds，未取得的完整 round slots 維持 complex NaN。部分取得的分支保留 raw，待資料完整才分析。已取得 IQ 中的非有限值或部分缺失 round 回報執行錯誤。

Save 保存 raw、座標及本次 cfg，包括角色、tone mapping、兩線 waveform、gain、phase、contrast 與具名 DC 設定。Native Load 保持目前表單原值。Analyze 只用載入來源的 cfg／raw。Cfg 不可用時仍保留有效 raw，但拒絕需要該 cfg 的頻譜分析。

實際 FTF 控制鏈與物理 gate 表現須用現場資料另行驗證。後續 constructive phase、chevron、amplitude、conditional phase 與 local Z 仍需各自量測與人工判讀。

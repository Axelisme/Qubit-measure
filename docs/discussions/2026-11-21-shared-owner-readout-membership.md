---
date: 2026-11-21
status: active
superseded_by: null
related:
  - ../adr/0078-evolution-module-boundary.md
  - ../adr/0083-program-macros-and-timelines.md
  - 2026-11-21-ftf-readout-device-time-pairing.md
  - ../../lib/zcu_tools/experiment/README.md
  - ../../lib/zcu_tools/simulate/adc/README.md
  - ../../lib/zcu_tools/simulators/ftf/environment/README.md
basis:
  - "本 repo merge base 163fbeb51d62ba21c671915d8bad584781a192bf"
  - "原幾何候選 19897d3464c7f7a5d9b3d7e69145525f4f37c756；review 候選 a8ece4a3d7089a9dca7abce793ffc0e9ea86dbea；修後幾何候選 3480a4ffe713178b7ad731affcca70c38c815d66"
  - "外部來源：無"
---

# 同 owner 的讀出窗共用脈衝歷史

## 結論

同一 point、ADC 與 owner 的有場事件，使用其各視窗相交 block 的脈衝聯集，每個 block 只收一次。各視窗仍保留自己的交集、owner、ADC grid 與輸出 column。聯集不從其他 ADC 或 owner 借入脈衝。這讓同一事件的早期成員脈衝把場延續到只與後續 block 相交的 grid，不必拆開事件或增加每個 grid 個別保存 pulse list 的資料結構。

這項聯集規則是 designer 依使用者授權所作的完整代裁，使用者未覆核，`Review: pending`。它明確修改了原先「每個視窗不相交的 block 不得進事件」的驗收條件，不是宣稱舊條文已被滿足。使用者覆核前，這條規則以代裁身分生效。若覆核推翻，另寫決策與討論快照，再修改程式。

## 問題

當時 `rpi_length` 用固定 `0.11 µs` 長度餘量。首點中，從零起算的 column 62 與 block 62、63 相交；column 63 只與 block 63 相交。兩窗最後相交的 owner 都是 terminal block 63。候選把兩個視窗放在同一個 ADC event，事件成員脈衝包含 block 63、62；原凍結驗收卻要求 column 63 不得因 column 62 的視窗而收到 block 62。程式行為與該驗收相矛盾。

一個 ADC `ReadoutEvent` 有一組 pulses 和多個 grids。此設計讓同一事件的場與 CTMC path 延續到各 grid。把 block 62 從後窗直接刪去，卻不傳遞該 pulse 留下的 cavity field，會把同 event 的 ringdown 一併刪掉。另一方面，CTMC path 可按 logical event 在多個 ADC 間共享；這不代表各 ADC 共用 pulse tuple。每個 ADC 仍有自己的 grids 與成員脈衝。

幾何收據顯示兩個 column 共用 owner 63；前窗交 block 62、63，後窗只交 63；候選事件的 pulse 清單是 `[63,62]`。收據只記錄離線規劃幾何，沒有執行 rounds 或 evolution，不能當作場效應的 I/Q 收據。

## 考慮過的選項與比較

- **U：保留逐窗交集，event pulses 取 union（採用的代裁）。** 每窗仍按裝置時間和 block interval 配對；相同 ADC／owner 的事件收集各視窗交集 blocks。這保留既有單一事件場路徑、CTMC path 與輸出拓撲，不改 `ReadoutEvent` 型別。代價是同一事件的較早 pulse 也會影響只與 owner block 相交的後窗。
- **W：每個 grid 使用自己的 pulse 集合，共享 CTMC path。** 這要求事件保存各 grid 個別的 pulse lists，或另有連續場交接。若只遮掉 pulse，後窗會丟失仍存在的 ringdown；若傳遞初場，則要擴充共享 ADC 與三個 simulator 的資料契約。
- **S：把各 window 拆成不同 event。** 各 event 可有獨立 pulse tuple，但會切斷既有事件共用的 CTMC 與場歷史，也可能改變 logical event／projection 與抽樣 stream 的對應。這不是單純整理資料。
- **R：同 owner windows 的交集不同就拒絕。** 當時 `rpi_length` 的預設已有這種幾何；拒絕會讓合法 caller 沒有結果。
- **N：改成時間連續的場與狀態模型。** 這可直接表示各時刻的 incident field，但需另定投影、ADC jump、state history 與場如何交接，並改動多個共用 caller。它超出本次規格矛盾修正，應另設計與驗證。
- **K：保持候選程式及原凍結條文。** 這會保留程式違反規格的狀態，review 無法判定預期契約。

## 被否決的選項與理由

- 逐 grid 遮掉所有不與該 grid 相交的 pulse，若不額外帶入初場，就會刪除同一物理 event 的既有 cavity ringdown。當時沒有只改 pulse tuple 的對照可證明這項貢獻大小，也沒有依據稱它為零或可忽略。
- 拆 event 會讓同一 logical event 的 ADC 視窗不再共享現行 CTMC path，且事件拆分可能改變 projection、seed stream 與輸出拓撲。
- 遇到 mixed intersections 就拒絕會擋掉當時的 `rpi_length`，與合法 caller 給出預期結果的原則不合。
- 完整連續場模型需要重定義更多事件與狀態邊界，不應假裝是小幅修補逐窗驗收。
- 維持原文會留下明確矛盾，因此不作為 promotion 結果。

## 留下的問題

當時的 R5 與跨投影近似表沒有單獨比較 column 63 使用 `[63,62]` 或 `[63]` pulse tuple 的差異。固定 `0.11 µs` 的幾何只用於歷史比較，不代表後來從讀出模組取得長度餘量的[表單預設](2026-11-21-rpi-length-fixed-margin.md)。跨投影數字固定相同 pulse 聯集，只改整窗 owner state 與依投影點分段的參考；R5 數字比較跨事件的 vacuum 起始與帶入殘場。不能用這兩組數字推論 block 62 對 column 63 的 I/Q 差為零或很小。

是否為未來設計量測該 pulse tuple 的條件 I/Q 增量，需先說明它回答什麼物理取捨。即使量測，也只能描述固定 model、grid、state、phase 與 receiver calibration 下的邊際貢獻，不能作成相對硬體的誤差或沒有門檻的 pass/fail gate。

FTF event 的 vacuum 起始與跨投影使用 owner state 仍是近似。此聯集規則不處理跨 event 殘場、不把 ADC jump 回饋至 qubit evolution，也不改投影時間。

## 依據

使用者在 2026-11-21 把這類細節授權 designer，原話為：「你只要能确保计算是正确的，然后机制是统一的，好维护，你知道存在这个东西，而且不会造成什么 bug 之类的就好了。那具体上做细节，我没什么意见。」使用者另選擇跨投影沿用一個 owner state，並表示投影時刻只影響 ADC；這沒有直接裁定同 owner 多個 grids 的 pulse tuple 粒度。故 event-wide union 在此標為 designer 代裁、使用者未覆核，`Review: pending`。

當時幾何、候選內容及誤差對照來自未入庫的離線收據與 probe。必要的幾何是：前窗交 block 62、63；後窗只交 63；兩者 owner 都是 63；候選事件成員為 63、62。既有近似表中的 `rpi_length` 數字是 owner-state 對分段-state，`chain/t1` 數字是 vacuum-start 對帶殘場；兩者都沒有改變 pulse tuple。

相關的時間座標、block interval 與共用 ADC 行為見 [FTF 讀出窗依裝置時間配對](2026-11-21-ftf-readout-device-time-pairing.md)、[experiment README](../../lib/zcu_tools/experiment/README.md)、[ADC README](../../lib/zcu_tools/simulate/adc/README.md) 與 [FTF environment README](../../lib/zcu_tools/simulators/ftf/environment/README.md)。本篇遵循 [ADR-0083](../adr/0083-program-macros-and-timelines.md) 的 ADC 獨立時間軸，但不把它當成使用者已裁定 pulse 聯集規則的依據。

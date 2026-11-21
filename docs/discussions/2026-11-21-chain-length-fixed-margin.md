---
date: 2026-11-21
status: active
superseded_by: null
related:
  - 2026-11-21-chain-readout-spacing.md
  - ../adr/0073-measure-form-derivation.md
  - ../adr/0083-program-macros-and-timelines.md
  - ../../zcu_lab/v2/twotone/ro_optimize/chain_length/guide.md
basis:
  repo_commit: 3d73c92afee38e41c783b10777a7ca248344617c
  external_sources: none
---

# Chain length 掃描固定 pulse 與 ADC window 的長度差

## 結論

使用者在 2026-11-21 選擇：「我会这样认为，这个 gap 常数零点零三应该是 real module 的 template 填的，而这个实验就采用固定长度差的方式。」`chain_length` 用每點 ADC window 加 `length_margin` 作 pulse 長度。Open 或 Reset 由擷取的讀出模組取得兩個長度之差，操作者可以改值。沒有模組才用 readout 範本的差值。唯讀飛行時間不再是 cfg 欄位，cfg schema 升到 6.0。

## 問題

[原長度規則](2026-11-21-chain-readout-spacing.md) 用 trigger offset、飛行時間與固定 0.11 µs 計算 pulse 長度。這在 pulse 傳到 ADC 後留了一段尾巴，卻要求操作員為掃描另外補一個不能編輯的校準值。使用者掃讀出長度時，已經用原模組的兩個長度定義他想維持的差值。

## 考慮過的選項

固定一個範本差值，能讓掃描輸入最少，但和模組的實際時序脫鉤。FTF fixture 的 R1 在 1.111 µs 開窗，飛行時間約 0.387 µs。若一律只加 0.03 µs，脈衝抵達 ADC 的尾端會比關窗早約 0.69 µs。

採用模組的差值，讓 pulse 和 window 同增同減。若原模組覆蓋 window，在 trigger offset 和 pre-delay 不變時會保留原覆蓋情況，差異只來自各自的時脈量化。這遵循 [ADR-0073](../adr/0073-measure-form-derivation.md) 的固定來源與可編輯初值規則。

## 被否決的選項

把飛行時間改成可編輯，仍保留一個計算相對長度不需要的欄位。固定全域差值則無法保留 R1 這類含 ring-up 的設定。兩者都不採用。

## 留下的問題

差值不會修正觸發太早、模組原本覆蓋不全或操作者輸入不合適的時序。Guide 說明要核對 ADC 開窗與關窗。讀出 pulse 後仍只等 `post_delay`，不再額外等 ADC 關窗，維持 [ADR-0083](../adr/0083-program-macros-and-timelines.md) 的 timeline 與 overlap 規則。本次不改 FTF 的讀出到達時間模型。

## 依據

- [FTF fixture](../../tools/data/experiment_wall_time/) 的 `ftf-demo/initial`。計算用 R1 的 pulse 模組與 resonator 飛行時間。
- 本文保留使用者對長度差的裁決與選項比較。原始任務工作檔未入庫。

---
date: 2026-11-21
status: active
superseded_by: null
related:
  - ../adr/0073-measure-form-derivation.md
  - ../adr/0083-program-macros-and-timelines.md
  - 2026-11-21-chain-readout-spacing.md
  - 2026-11-21-herald-wait-readout-post-delay.md
  - 2026-11-21-readout-template-window-margin.md
  - 2026-11-21-chain-length-fixed-margin.md
  - 2026-11-21-rpi-length-fixed-margin.md
basis:
  repo_commit: 3d73c92afee38e41c783b10777a7ca248344617c
  verified_integration: a80b1313bb1608777534ce11dc92c7feffc93aff
  external_sources: none
---

# Readout 時序預設：固定長度差與光子衰減等待

## 結論

普通 readout 範本在 ADC 視窗前後各留 0.015 µs。Trigger offset 為飛行時間加 0.015 µs，pulse 比 ADC 視窗長 0.03 µs。範本維持 1.0 µs pulse，所以普通 onetone 的視窗初值為 0.97 µs。DPM 範本和寫回也用 0.03 µs 的長度差。

Chain length 和 Rπ length 的每點 pulse 長度為 ADC 視窗加可編輯的 `length_margin`。Open／Reset 先取已捕捉 readout module 的 `waveform.length − ro_length`。沒有捕捉模組才取範本差值；解不出來時保留必填空白。`register_all` 把這次注入的 `registry.templates` 綁給兩個 Form factory，不讀全域範本表。Chain length 不再有 `time_of_fly` 欄位。兩個 cfg schema 分別由 5.0 升到 6.0、由 3.0 升到 4.0，不轉換舊 cfg。

Mux Open／Reset 的 `wait_for_ringup` 路徑用 `5/(2π·κ)` 給 `post_delay` 初值，κ 取所用 resonators 的最小線寬，單位 MHz。有捕捉模組時用已設的 κ 計算，並保留較長的原等待；全未設時維持原值。Blank 範本只有在 κ 齊全時才用公式，否則保留 0 µs。Herald 與 terminal 仍共用 `post_delay`。

## 問題

[舊讀出鏈規則](2026-11-21-chain-readout-spacing.md)把 chain length 的 pulse 設成 trigger offset 減飛行時間，再加視窗與 0.11 µs。這要求操作員另外提供飛行時間。Rπ length 用視窗加 0.11 µs，不管捕捉模組原本的差值。普通 readout 範本又留了 0.1 µs，三者不是同一個量測起點。

[Herald 等待的調查](2026-11-21-herald-wait-readout-post-delay.md)只比較 FTF 個案的 0 與 2 µs。2 µs 是那次比較選的值，不是線寬算出的起點。R2 線寬約 1.2 MHz 時，單個光子衰減時間約 0.13 µs，五倍約 0.66 µs。2 µs 約是十五個衰減時間，不能當成通用門檻。

## 考慮過的選項與裁決

- 長度差沿用模組。使用者在 2026-11-21 說：「我记得我在扫 Real Optimize 的时候。我是取 Real Module 的长度差，我固定的是长度差……我就不需要知道 Tile Fly」。範本則是：「我会这样认为，这个 gap 常数零点零三应该是 real module 的 template 填的，而这个实验就采用固定长度差的方式。」使用者同日說：「我通常会前后各留零点零一五，总共多出零点零三的长度。我认为这作为预设模板的生成常数值是可接受的。」將 trigger offset 加 0.015 µs 是從前緣餘量推得，已向使用者標明。Rπ length 同日得到確認：「不过它看起来问题是一样的，应该要比照办理。」
- 先用線寬給等待，再用實驗驗證。使用者同日說：「根据我的经验来说，这个等待 photon decay 的时间不会超过一百 nanosecond，就可以到达五倍的 decay time……这数量级差的有点大」。因此把 guide 的起點改為五個光子衰減時間，並在表單給同一公式的種子。Guide 依「guide 写给只懂物理的使用者」的要求，用線寬和光子衰減時間說明起點。操作員用 herald 後的名目 Xπ 比較這段等待和更長的等待。欄位歸屬遵循 [ADR-0074](../adr/0074-flat-readout-module.md)，起算點遵循 [ADR-0083](../adr/0083-program-macros-and-timelines.md)。
- 由 lab 組裝點綁範本目錄。這項接法是 2026-11-21 designer 代裁，使用者未覆核。依據是 [ADR-0073](../adr/0073-measure-form-derivation.md#持久範本目錄) 的目錄注入規則，以及使用者同日說的「一般实验用简单的做法就好了，有需求再做多余的设计」。目前只有兩個 factory 需要讀 seeds，由 `register_all` 綁定即可；第三個需要時再評估 `FormContext`。

## 被否決的選項與理由

- 每點固定加 0.03 µs，會丟失捕捉模組的長度差。FTF 的 R1 範例中，trigger offset 已含約 0.72 µs 的 ring-up；這樣算會少蓋住視窗尾端約 0.69 µs。
- 保留按飛行時間換算加 0.11 µs 的公式，仍需另一個 cfg 欄位，且沒有保留捕捉模組原來的長度差。間隔規則不需隨長度公式變更。
- Guide 單列 0／2 µs，操作者無法從自己的線寬選起點。只改 guide、不給初值，也會讓新 mux module 繼續一律從 0 µs 起步。
- 給所有 Form factory 的 `FormContext` 增加範本目錄，會為兩個 caller 改 framework 介面；把無捕捉時的 0.03 µs 寫死，會讓自訂目錄中選到的範本和長度差分離。

## 留下的問題

FTF 單列各試一次。Chain length 在 180 秒、Rπ length 在 360 秒逾時，都卡在第一個 acquire，沒有完成窗口。離線編譯驗證了每點長度，未證明其餘 FTF 點跑得完，也沒有實機驗證。這是模擬器的執行時間問題，不改本次的時序規則。

## 依據

- [範本與常數](../../zcu_lab/templates.py)、[Mux 表單種子](../../zcu_lab/v2/support/measure/mux_forms.py)及 [Lab adapter 組裝](../../zcu_lab/definitions.py)。
- [Chain length guide](../../zcu_lab/v2/twotone/ro_optimize/chain_length/guide.md)、[Rπ length guide](../../zcu_lab/v2/twotone/ro_optimize/rpi_length/guide.md)及 [Mux onetone guide](../../zcu_lab/v2/mux/onetone/guide.md)。
- [FTF fixture 與牆鐘工具](../../tools/data/experiment_wall_time/)。原始問答與單列診斷收據未入庫；裁決、數字和限制記在正文。

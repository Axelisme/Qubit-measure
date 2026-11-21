---
date: 2026-11-22
status: active
superseded_by: null
related:
  - 2026-10-14-artifact-persistence.md
  - 2026-11-22-shared-presentation-facts.md
  - 2026-11-22-two-qubit-native-axis-cutover.md
basis:
  repo:
    before_task: 85e48d392961e9d990797febc27b761ea4b30fca
    preflight_review: f50c4f4591ad1f775566c01b3f294813b6469b2f
    verified_candidate: 18214ab7cce221ee88bab411a83a67f48b7131c7
  external:
    qick: 0.2.436
---

# Amp Rabi 的 shot moments

## 結論

傳統 Amp Rabi 保留原來的平均 complex IQ，再保存同批保留 shots 的六個校正矩。Native 新檔可帶 moments，舊檔和 mean-only workflow 結果走同一 spec；Labber 明選原 `data`，仍是一個 log。Program 提供明示的校正 IQ reducer，datafile 公開單一 Variable 的無 I/O 結構驗證。這些實作選擇是 2026-11-22 designer 代裁，使用者未覆核。使用者確認的是誤差棒代表「平均标准差，也就是这个点的预期误差」。

## 問題

平均 IQ 檔缺保留數和二階矩，重開後無法畫 shot SEM。把新矩直接加入 Labber 會使普通單 log reader 拒絕；把新矩列為必填又會拒絕舊 native 檔和現行的 mean-only 校準產物。即使能存新欄位，`IQ(...)` 預設的未校正平均也不是舊 `signals` 的零點。最後一個缺口出現在 Labber 選集：如果先取子集才驗證，錯軸的 native 變數可因不匯出而躲過檢查。

## 考慮過的選項與比較

- **保存。** 同一份 grouped spec 保留必要的 `data`，在確實有六矩時才寫 `iq_moments`。另選 Labber 的輸出變數，可讓新 native 檔具備診斷資料，無須修改 Labber reader。可選只代表整個變數能缺席，不能使已存在的壞資料變合法。
- **零點。** 比較未校正 mean、校正 mean 配未校正 moments，以及同一批 shots 全部在 QICK 校正後加總。固定投影下，平移不改變中心化 SEM，但會改變平均的零點。第三種保留現有平均語意，live、SNR 早停與最終結果可用同一個 slot，不必存兩個 frame。選擇不靠另設數值精度門檻。
- **驗證。** Reader 不必驗沒使用的資料，和 writer 在此處要驗的來源不同。單檔與 artifacts 都從同一 RunRecord 產生輸出，所以由 experiment 在明選 Labber 子集前驗所有已映射 SI variables。Datafile package root 轉出兩個 writer 已共用的驗證器。它只檢查當下的單一 Variable，不宣布整份 Container 可寫。

Native 與 Labber 各有用途，遵循 [ADR-0063](../adr/0063-persistence-ownership.md)；單一實驗 mapping 與舊檔限制見 [ADR-0069](../adr/0069-experiment-core-and-frontends.md#native-可選變數與-labber-投影)。這裡區別於 [two-qubit 軸名換版](2026-11-22-two-qubit-native-axis-cutover.md)：Amp Rabi 有仍需讀取與寫出的 mean-only 結果。原 reducer 的預設與完成輪歸約不改，見 [ADR-0074](../adr/0074-flat-readout-module.md#選用校正後的-iq-矩) 與 [ADR-0083](../adr/0083-program-macros-and-timelines.md#round-在取數端歸約)。驗證沿用 [ADR-0080](../adr/0080-datafile-generic-container.md#介面) 的格式 owner。

## 被否決的選項與理由

- 全部 native 變數必填會拒絕舊檔和 mean-only 結果。另做歷史 loader，或讓各 caller 選 spec，則要維護兩份映射和路由。如果只保留平均而不存 native 矩，重開也畫不出誤差棒。
- 把所有變數匯出 Labber，或遇到多變數就切 independent。兩者都改掉普通 reader 目前能讀的單 log。
- 保留未校正 mean、額外擷取一批 shots，或把模擬器 compact mean 搭配 raw SEM。第一種改變既有零點；後兩種在估計不同批平均的抽樣誤差。只改 mean、讓 moments 留在原來零點也可計算，但 caller 得處理兩個 frame。
- 讓 experiment 直接匯入內部 validator 或重做檢查。前者把內部模組變成跨 package 約定，後者把同一結構交給兩個 owner。改成 `Variable` 建構即驗會改其他 caller 的失敗時點，也不能阻止陣列之後被修改。只驗匯出子集則漏掉本次要拒絕的 native 壞變數。

## 留下的問題

校正 opt-in 不擴到多 tone、多 read、遮罩或 edge counting。SEM 不含 IQ 投影方向或 fit 參數的不確定度，不作 fit 權重。保留每點少於兩發或統計缺失的均值時，該點不畫誤差棒。

## 依據

設計時以 `5+7j` 的原平均與兩分量各扣 `2` 的案例區分兩個零點。最終離線 probe 使用真 [Amp Rabi spec](../../zcu_lab/v2/twotone/rabi/amp_rabi/core.py) 與含 offset 23 的合成逐發資料。第一個完成輪保留 300／1200，backend 執行一次且 raw 只物化一次。校正 mean、六矩、SEM 與 SNR 早停均和同批資料相符。公開 Labber 匯出與普通 reader 另外核對單一 `data` log、Gain 及平均 IQ；native probe 核對有矩及 mean-only round trip。公開保存接縫的合成壞軸案例也驗證兩種 data outputs 失敗而其他 artifacts 仍可保存。原始調查、合成資料與 probe 圖未入庫，驗證未操作真實硬體。

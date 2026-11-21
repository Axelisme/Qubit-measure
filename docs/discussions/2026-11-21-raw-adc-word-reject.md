---
date: 2026-11-21
status: active
superseded_by: null
related:
  - 2026-11-21-ftf-legal-readout-timing.md
  - ../adr/0078-evolution-module-boundary.md
  - ../adr/0084-simulation-integer-time-base.md
basis:
  main_base: a42f7945ded58dcac09aff3cb61beb0c9ab086e0
  integration_commit: 8f36e1fc104b58698712c766f02323189356e4f4
  external_sources: none
---

# raw ADC 頻率字改成編譯時拒絕

## 結論

沒有具名來源的 raw ADC 頻率字若在 window 起點生效，模擬 frontend 在編譯時拒絕。訊息寫明 ADC 與 trigger 時間，請作者改用具名 ADC config。這次移除了 `DDSCongruence`、ADC 取模、Six 的同餘距離與同餘平手拒絕。具名 config 照原樣使用作者指定的物理頻率。規則見 [ADR-0078](../adr/0078-evolution-module-boundary.md) 與 [ADR-0084](../adr/0084-simulation-integer-time-base.md)。

## 問題

一個 raw 頻率字可能代表多個 DDS image。先前的做法讓 ADC 選最接近共用參考的 image，Six 則用同餘距離選 cavity。這套機制只有合成案例使用，沒有找到使用它的實驗。

使用者原先選 K：「那我选，K 吧，感觉有些少频率的，比如说像是 one tone frequency，如果不是等间距的话，要先计算 table，然后存起来。感觉就会出现模拟上需要解读弱智的问题。」[回歸測試](../../tests/program/v2/simulation/test_frequency_sweep_iq.py) 比對了四個非等距 table sweep 點與同頻率的 linear 讀出。Table sweep 保留具名來源，逐點 IQ 相等，不需要模擬器猜測 image。

## 考慮過的選項與比較

- K，維持原作法。合法的 raw 程式能編譯，但模擬器須在編譯、ADC 與 cavity 選路維護同餘語意。
- S，拒絕沒有具名來源的 raw 字並移除同餘機制。實機仍接受該程式；模擬 frontend 拒絕是因為缺少決定物理頻率的資訊，不是因為參數值不合理。2026-11-21，使用者改答「S」。

## 被否決的選項與理由

保留 K 的理由是 table sweep 可能失去具名來源。回歸測試證實這不會發生。沒有實驗需要維持同餘機制，不再為這條路徑保留型別及數值分支。

## 留下的問題

如果 window 起點的設定有具名來源，窗內 DDS word 與 phase 都相同的 raw 更新仍通過。起點已決定 image，等值更新不需要再解一次。如果後續 trigger 由該 raw 字提供生效設定，該 window 會被拒絕。

## 依據

驗收用相同的 [MG12 具名 fixture](../../tools/data/experiment_wall_time/fixture/result/ftf-demo/points/initial/library.yaml) 跑實際 MuxOnetone，固定 density 隔離演化。改動前後的 integrated IQ 與 ADC2、ADC3 各 115 個 decimated 樣本逐值相等，最大絕對差為 0。原始收據未入庫。

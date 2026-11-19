---
date: 2026-11-20
status: active
superseded_by: null
related:
  - lib/zcu_tools/simulators/ftf/environment/README.md
  - lib/zcu_tools/simulate/evolution/README.md
  - docs/adr/0078-evolution-module-boundary.md
basis:
  - "repo@f375b9421：cavity-drive Stark shift 讓相鄰態能量反轉時，FTF 以 point／segment 指名的錯誤停止 run"
  - "使用者 2026-11-19 的裁決與 2026-11-20 的代裁授權；原始工作紀錄未入庫，必要原話與結論寫在正文"
---

# FTF cavity Stark 反轉：從報錯改成警告

## 結論

Cavity drive 的 Stark 位移讓兩個相鄰態的能量順序反轉時，FTF 照原色散模型繼續算，不重排態。一次 acquisition 對每組反轉的態發一次 `UserWarning`，列出首個受影響的 point 與 segment、受影響的總點數與段數，以及讓順序不反轉的 cavity drive gain 共同縮放上限。單獨呼叫 Engine 時，警告給 Stark 差值的上限，單位 GHz。規則屬於 [ADR-0078〈模擬器不替使用者擋實驗參數〉](../adr/0078-evolution-module-boundary.md#模擬器不替使用者擋實驗參數)。

## 問題

一次 FTF 量測把讀出 gain 設成 0.05，整個 run 以 `cavity-drive Stark shift reverses energy ordering; outside dispersive model validity` 失敗。ADR-0078 的先例是「實驗參數設得不好時，模擬器照常給結果，以警告說明」，使用者 2026-11-15 的原話是「實驗參數設置不當就會發生的問題為什麼模擬器要擋？」。例外只有契約外的輸入與數值 guard。讀出 gain 0.05 是合法的實驗參數。

## 考慮過的選項

| 選項 | 內容 | 前提 |
| --- | --- | --- |
| A 照先例改成警告（採用） | run 照跑，結果附「超出色散模型適用範圍」的警告與可用上限 | 反轉後模型仍算得出數字，只是不準 |
| B 區別於先例，保留報錯 | 錯誤訊息補可用上限 | 反轉後模型沒有定義 |

建議是先查實作：能算出數字就選 A，不能就選 B，兩者都補上限。使用者選 A：「1cA」（2026-11-19）。實作確認反轉後兩條路徑（driven microwave 與 projective read）都能照常完成，所以 A 的前提成立。

## 上限的表示（代裁）

使用者 2026-11-20 授權代裁設計細節。上限有兩種呼叫點，各自報它看得到的量：

- FTF 層知道設定的 cavity drive gain。多個 cavity pulse 與 ringdown 疊加時，沒有唯一的單一 pulse 上限，因此報「所有 cavity drive gain 一起乘上的倍率上限」：固定 gain 比例與觀察到的腔場歷程，取所有受影響 segment 中最緊的值。
- 單獨的 Engine 只拿到 segment 的位移，報 Stark 差值上限（GHz）。

同一原因在一次 acquisition 內只發一次警告，不逐點洗版；跨 round 或 projection 歷程重複看到的同一 segment 只計一次。

## 被否決的選項與理由

- B：反轉後模型仍有定義，報錯會擋下合法參數，違反 ADR-0078 的先例。
- 重排態：會改變色散模型本身，不屬於本題。

## 留下的問題

- 色散模型沒有量測引起的躍遷，也不模擬 MIST；超過上限的結果只標示為超出有效範圍，不修正。

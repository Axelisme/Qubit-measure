---
date: 2026-11-15
status: active
superseded_by: null
related:
  - docs/adr/0078-evolution-module-boundary.md
  - lib/zcu_tools/simulate/evolution/README.md
  - docs/discussions/2026-11-14-common-frame-step-pricing.md
  - docs/discussions/2026-11-14-sampled-multitone-interim.md
  - docs/discussions/2026-11-16-constant-multitone-guard-diagnostic.md
basis:
  - "repo base@22ca5b0a05e647aa2bcc59aeb3de6a11402c5233"
  - "candidate@595716dba882be0a9277020e3066f84fcd8b4a4a"
---

# `common_frame` 的 coherent 門檻收窄與唯讀診斷

## 結論

`common_frame` 只收窄既有的 coherent 上限，從 0.03 降到 0.0018；bath 上限 0.01 不變，不加新的比較項。這個值高於 14 個留下的合格格中最大的 coherent 指標 0.001469411，低於最近的已知超標格 0.002623604。同一個門檻適用 `common_frame` 的整個 domain，包含兩 tone 的取樣列。被拒絕的列由既有價格在其他已放行的 family 間選，不強制走 `dop853`。

收窄前先新增 package-root 的唯讀查詢 `common_frame_diagnostic`，回傳 `CommonFrameDiagnostic`：admission 已算出的 coherent、bath 及五個分項數值、門檻與 `reason`。精度矩陣的 guard-edge 案例改放在新門檻內側（coherent 0.0017999982），舊的 8.066669e-3 strict xfail 移除。現行數值與欄位契約由 [evolution README](../../lib/zcu_tools/simulate/evolution/README.md#kernel-summary) 持有。

## 問題

約化格點的常數雙 tone 上，`common_frame` 自己的 guard 放行三個超標格：A_x3.5_T1600 的完整 density 誤差 7.30e-3，F_K6_g0.05_s0.5 為 2.28e-3，F_K6_g0.05_s2.0 為 2.15e-3。驗收標準是 2e-3。舊的精度矩陣也以 strict xfail 記著一個 8.07e-3 的 guard 邊緣案例。

[ADR-0078〈先保證正確，再優化瓶頸〉](../adr/0078-evolution-module-boundary.md#先保證正確再優化瓶頸) 要求 guard 在整個參數空間不放行超標結果。現有兩個指標怎麼收都無法同時保留 19 個合格格：只收 coherent 會多拒絕 5 格，只收 bath 會多拒絕 17 格。

收窄前，`common_frame` 的 guard 指標沒有公開讀法，呼叫者只看得到放行或拒絕，無法列出指標的實測數值與交界。

## 考慮過的選項與比較

- **A：只收窄 coherent 上限。** 改一個常數、診斷文字、一個精度案例與 README，可還原門檻退回。多拒絕的五格是 A_x3.5_T400、B_f0.25_g0.038_T1000、B_f0.25_g0.038_T250、B_f0.45_eps0.029_T400、C_g0.1_T512，都是合成格，沒有耗時紀錄顯示它們是瓶頸。五格中兩格原本預設走 `common_frame`，三格預設走 `multitone_rwa`。格點 row 計時上，五格走 `dop853` 約 0.471–2.978 s，走 `common_frame` 約 0.011–0.505 s；這是格點計時，不是正式 benchmark。
- **B：新增被省略項指標**（例如三階項）。只有 26 格可以校準，可分隔的間隙約 28%，還要另量獨立參考解；換回的只是合成格的速度。
- **C：補上被省略的高階項。** 沒有證據顯示一種修法能同時修好 A 型與 F 型，成本未知。

門檻取在 0.00147 到 0.00262 之間的下半段。0.0018 對留下的最大值有 22.5% 餘裕，對最近的超標格有 31.4% 距離。留下的 14 格歷史誤差最大 8.234e-4，低於 1e-3。

診斷沿用 single-tone 診斷的形狀：比較葉是 `RWAGuardMetric | RWAUncomputed`，沒算出的值回 `RWAUncomputed`，不以 0 代替。`common_frame` 另有 `reason`（`accepted`、`carrier`、`denominator`、`coherent`、`bath`），因為它的拒絕依固定順序發生，carrier 或 denominator 拒絕後後面的指標根本不會計算。查詢只建一次暫時係數、評估一次 admission，不演化 density、不選 family、不保留準備結果。

## 被否決的選項與理由

- B、C：理由見上一節；使用者選 A。
- 維持現狀：違反 2e-3 的驗收標準。
- 只用既有上限保留全部 19 格：數學上做不到。
- 把常數雙 tone 整個排除在 `common_frame` 之外：比需要的更窄，丟掉 O1 白名單 24／24 正確時的備援。
- 斷言三個超標格一律改走 `dop853`：那是強制路由。研究時 F_K6_s0.5 的 secular 指標 1.39e-4 低於 `multitone_rwa` 的 guard 門檻 1.476e-4，可能由 `multitone_rwa` 放行；去向交給價格。
- 舊邊緣輸入另寫拒絕測試：固定 `families=("common_frame",)` 時拋公開 `NoValidKernel` 的測試已涵蓋。
- 取樣列保留 0.03、只對常數雙 tone 用 0.0018：沒有證據顯示取樣列在 0.03 下 ≤2e-3。舊的取樣 fixture 只驗到 0.015，是標準的 7.5 倍寬。依 domain 分開門檻也等於新增比較項。

## 先例與立場

- [ADR-0078〈先保證正確，再優化瓶頸〉](../adr/0078-evolution-module-boundary.md#先保證正確再優化瓶頸)：遵循。不確定時 guard 從嚴，速度只對有紀錄的瓶頸優化。
- [common_frame 的步長成本歸 dispatch](2026-11-14-common-frame-step-pricing.md)：遵循。數值 guard 決定適用域，價格由 dispatch 持有；這次沒有改價格。
- [Sampled multitone 的過渡期拒收](2026-11-14-sampled-multitone-interim.md)：遵循。兩個 multitone family 拒收取樣列、`common_frame` 保留為取樣列的候選，三項都沒變；只有 `common_frame` 的門檻變嚴。精度矩陣的 `cheap_sampled_nearby` 仍走 `common_frame`。
- [single_tone_rwa 以 endpoint kick 保留](2026-11-14-single-tone-rwa-kick.md)：遵循它的診斷形狀（算過就報、沒算回 `RWAUncomputed`）。

## 裁決

使用者 2026-11-15 對「只收窄 coherent、接受多拒絕 5 個合成格、去向由價格選、guard-edge 改寫成新門檻內側案例」答「A」。

2026-11-17 land review 發現 task 驗收文字寫「取樣包絡的行為不變」，與單一門檻衝突。Designer 依使用者的代裁授權定案：單一門檻涵蓋兩 tone 取樣列，只改驗收文字與 README，不改程式。

## 留下的問題

- 0.0018 只在常數雙 tone 的約化格點校準。兩 tone 取樣列沿用同一門檻、沒有獨立校準；它比原本的 0.03 更嚴，但不代表 0.0018 以下的取樣列都在 2e-3 內。
- 收窄造成的變慢沒有量。成對 benchmark 與變慢超過 10% 的清單延到下一個 effort；若變慢來自這次收窄，屆時再考慮選項 B。

## 依據

格點數值、多拒絕格的計時與選項比較取自決策前的調查工作檔，原始工作檔未入庫，必要的數字已寫在正文。現行門檻、guard-edge 案例與診斷欄位見 [evolution README](../../lib/zcu_tools/simulate/evolution/README.md#common-frame-guard-diagnostics) 與 [common-frame owner](../../lib/zcu_tools/simulate/evolution/kernels/common_frame.py)。

---
date: 2026-11-24
status: rejected
commit: 0002badf503464563b03b3246220154e0d9c1986
modules:
  - lib/zcu_tools/simulators/ftf/environment
discussion: none
superseded_by: null
---

# FTF 依 timeline 段數分配誤差預算

**讀者：** framework-dev、agent。查五個 FTF 場景的正式速度配對，以及候選為何撤回。

## 核心思路

FTF 原先給每次演化固定的每段預算 `B=1e-3`。[雙載波 sampled RWA 探針的紀錄](2026-11-24-sampled-rwa-predicted-rejection.md#實測)顯示，有些 spot 的端點密度差低於這個預算，卻高於 `0.1B` 的採用門檻。候選希望短 timeline 能多採用近似，少走較慢的後援。

候選只改 FTF 傳給 Engine 的預算，改成 `B=clamp(1e-2/N, 1e-3, 1e-2)`。`N` 是一次呼叫中單點 timeline 最多的正長度段數，從該點演化起點算到 readout 起點，含套用傳播子的 idle。沒有正長度段時用上限，`N ≥ 10` 時仍給 `1e-3`。Sampled RWA spot、adiabatic map 和 Dyson 原本依賴 `B` 的門檻會跟著變；Engine 介面、kernel 公式、維度篩選門檻和 `SPOT_POLICY` 都沒改。

## 實測

### 正式速度配對

基準 A 是 `2ec6814affa1a624bd4990ee3f96ae76e1a53a60`，候選 B 是 `0002badf503464563b03b3246220154e0d9c1986`。2026-11-24 離線 FTF Run 用 `ftf-demo:initial` fixture、FTF seed 123、`time_scale=0`、CPU affinity 0–3、單 worker、單 native thread，關閉 OpenCL。每輪從全新的私有 cache 開始，不做 capture。各 block 的八份輸入檔 hash 在各輪一致。Block 1 的 manifest 為 `931fa4f599f4`，只跑 `len_rabi` 11 點；block 2 用 `c2b6b9e0522e`，其餘四列依表序跑 121、121、11、4 點。正式配對的各採用輪都完成。

下表單位為整次 Run 的 core 秒數，合計 compile、evolve、readout、materialize，不除以點數。同側差取該側兩輪的絕對差；減少量等於 A 平均減去 B 平均，未四捨五入前計算。Block 2 的 A1、B1 欄用 A1R、B1R。

| 場景 | A1 | B1 | B2 | A2 | A 平均 | B 平均 | A 同側差 | B 同側差 | 減少量 | 減少率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `twotone/rabi/len_rabi` | 19.575 | 19.251 | 19.869 | 19.003 | 19.289 | 19.560 | 0.572 | 0.618 | -0.271 | -1.406% |
| `two_qubit/chevron` | 88.191 | 87.799 | 91.373 | 89.999 | 89.095 | 89.586 | 1.807 | 3.574 | -0.491 | -0.551% |
| `two_qubit/constructive_phase` | 244.853 | 236.650 | 247.147 | 250.291 | 247.572 | 241.898 | 5.438 | 10.498 | 5.674 | 2.292% |
| `two_qubit/cz_amplitude` | 24.687 | 21.611 | 23.487 | 25.056 | 24.871 | 22.549 | 0.369 | 1.876 | 2.323 | 9.339% |
| `two_qubit/rb`（ordinary） | 14.556 | 14.263 | 14.452 | 14.477 | 14.517 | 14.357 | 0.079 | 0.189 | 0.159 | 1.097% |

五列平均加總為 395.344 → 387.950 core s，每側兩輪合計為 790.687 → 775.901 core s，減少 1.870%。去留看逐列門檻，不看總和。

另一個四 worker Pyright 在 17:43:43.468–17:44:16.717 +0800 與 block 2 原 A1 的 `constructive_phase` 重疊。整輪原 A1 排除。原 B1 在 Pyright 結束後才開始，B2 和 A2 也未重疊。主表採用事先選定的相鄰替換對，實際採用輪的順序是先 B2 → A2，再 A1R → B1R。敏感度檢查只把 B1R 換回確認乾淨的原 B1，仍不納入原 A1。原 B1 的四列 core 秒數依表序為 90.798、261.914、24.006、14.921。這時 block 2 最大減少仍在 CZ amplitude，平均 24.871 → 23.746 core s，減少 4.523%。兩種選法都沒有達標列。原始工作檔未入庫。

### 實際段數與參考解

段數來自另一次非排他觀測，沿用上表的 A、B 版本及各 block 輸入，基準五列各觀測一次，候選只觀測 CZ amplitude 和 ordinary RB，各一次。N 不由正式計時的累計 routing rows 推算。下表的基準預算均為 `1e-3`。

| 場景 | 基準實測 N | 候選 B |
| --- | --- | --- |
| `twotone/rabi/len_rabi` | 11 次 evolve：N=1 一次，N=2 十次。99 次 reuse：N=1 九次，N=2 九十次。manifest 的 Reset 和 preparation 關閉。 | 若分段相同，N=1 給 `0.01`，N=2 給 `0.005`。未觀測候選逐次 N。 |
| `two_qubit/chevron` | `7, 7, 8, 8`，無公開 reuse。 | 若分段相同，前兩次 `0.001428571`，後兩次 `0.00125`。未觀測候選逐次 N。 |
| `two_qubit/constructive_phase` | `8, 8, 8, 8`，無公開 reuse。 | 若分段相同，四次皆 `0.00125`。未觀測候選逐次 N。 |
| `two_qubit/cz_amplitude` | `7, 7, 7, 7`，無公開 reuse。 | 候選實測同 N，四次皆 `0.001428571`。 |
| `two_qubit/rb`（ordinary） | `31, 43`，無公開 reuse。 | 候選實測同 N，兩次皆 `0.001`。 |

在 block 2 的這次 A/B 觀測中，ordinary RB 的路由、終態 density/population 與公開 Result 逐位相同。CZ amplitude 的最終採用 family、prefix、維度及 basis 沒變，sampled RWA 仍未採用。兩側終態 density 和 population 的最大前後差均為 `1.58922e-5`。基準未存 CZ 的公開 Result 陣列，無法比對公開 Result。

CZ amplitude 只有第 10 點（從 0 起算）的 24 態終態有既有的同模型 DOP853 參考，覆蓋 11 個不同 physics keys 中的一個，兩個終態輪共用此 key。該參考的兩組容差終態 density 最大差為 `5.47385e-10`。在上表 A、B 的單次觀測，該點的最大絕對誤差，density 和 population 分別都是基準 `1.48387e-4`、候選 `1.58557e-4`。這不是全列誤差。其餘 key 未補解；len_rabi、chevron、constructive_phase 未比對候選終態或量誤差。五列以外也未補做分類或配對。

## 為何採用或否決

保留須有至少一列平均 core 減少達 10%，且減少的秒數大於該列兩側各自的同側差。正式五列都未達標。CZ amplitude 省 2.323 s，大於兩側的同側差，卻只省 9.339%；其他四列都更低。**速度不足、未補參考。** 撤回候選，FTF 保留原來的固定預算。

## 留下的線索

Chevron 和 constructive phase 的基準呼叫實測有 7 到 8 段，不是單段。若分段不變，預算只放寬到原值的約 1.25–1.43 倍。若再評估只改 spot 門檻或另用分配公式，這次配對不能替不同候選證明省時。這次量到的整次 Run 改善也不能說成 sampled RWA 的收益。

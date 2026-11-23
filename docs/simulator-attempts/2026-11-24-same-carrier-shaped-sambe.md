---
date: 2026-11-24
status: paused
commit: fd160e255
modules:
  - lib/zcu_tools/simulate/evolution
discussion: none
superseded_by: null
---

# 異形同載波的 shaped Sambe

**讀者：** framework-dev、agent。記錄異形同載波原型拒收的位置與重查條件。

## 核心思路

[同頻多線摺成單 tone](./2026-11-23-same-carrier-drive-merge.md)要求包絡同形。這個原型保留兩線各自的 I/Q 算符、複數包絡與相位。在同一載波座標中加總兩線的 Fourier block，沿時間插值 frozen-envelope propagator。原型只在固定 `families=("shaped_sambe",)` 的測試中加入候選。

## 實測

以下是 `fd160e255` 上各固定候選的一次 `Engine.evolve`。CPU 固定一核、一 worker，先暖機，再量含選路、prepare 和 apply 的 process CPU 時間。兩場都是 8 態的 CZ 代理模型、單段 100 ns、載波 +5.800 GHz。誤差是對同一輸入、budget `1e-7` 的 production DOP853 的完整 density 最大元素絕對差，不是獨立參考誤差。

| 輸入 | DP5(4) CPU s | DP5(4) density 差 | `multitone_gate` CPU s | `multitone_gate` density 差 |
| --- | ---: | ---: | ---: | ---: |
| CZ 兩線，FAST-DRAG／cosine | 0.592 | `1.095e-5` | 5.583 | `2.458e-4` |
| CZ 兩線，FAST-DRAG／FAST-DRAG，β 不同 | 0.603 | `1.026e-5` | 5.615 | `1.146e-4` |

兩場原型都在 modes 2、3、4 的時間插值或密集節點 Fourier 誤差檢查被拒收。收據沒有區分哪一項超限。原型未執行 apply，沒有可比較的 CPU 時間或 density。原型 commit 和原始收據未入庫。這些數字不代表完整 CZ 裝置、整場實驗或重用多列 prepare 的成本。

## 為何暫停

兩場都未證實原型收益。相同測量中，`multitone_gate` 各花約 5.1 CPU s 準備單列，DP5(4) 的整次演化約快九倍。先確認異形同載波列是否在代表工作負載中拖慢演化，再決定是否增加快速 family。不合入原型，也不放寬 guard。

## 留下的線索

1. 若速度普查把異形同載波列排進慢場景前段，先核對選路比價。`fd160e255` 上這兩次單列對照的 DP5(4) 約需 0.6 CPU s。檢查實際選到的 family、prepare 次數與可攤提的列數，別把固定候選的結果當成預設路由。
2. 若比價無法解釋慢場景，再拆開記錄 8→16、16→32 個時間節點的插值誤差，以及相同節點數下的 Fourier cutoff 誤差。先確認哪項超限，再決定採用時間分段或增加 modes。保留原 guard 門檻。

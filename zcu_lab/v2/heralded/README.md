# Heralded 實驗家族

**讀者：** operator、lab-author、agent。選擇保留多個 herald 類別的實驗，查找分析、存檔與共用模組的契約。

## 選擇入口

只需篩選初態時，在支援的普通入口選 `reset/herald` 或 `reset/mux_herald`。Pulse herald 用 `keep` 指定一個態。Mux herald 用 `keep` 指定一組 tone 條件，全部符合才保留該 shot。兩者都只留符合條件的一類，捨棄其他 shots，不保存 herald 分類。Mux 的 `keep` 鍵是完整 readout 的 tone 位置，不是 ADC 編號。設定範例見 [Two-qubit 操作指南](../two_qubit/guide.md#herald-reset-與獨立-heralded-入口)。

分析需要一類以上時，在 New Tab 選 **heralded**。這個選單和 **two_qubit** 並列。八個入口各自取得可見的 mux herald 與終端讀出，不再用 herald reset 篩選。Herald 與終端各有 Qubit1、Qubit2 的 g、e、Other，Native 保存每個已量座標的 81 格聯合計數，不保存逐 shot IQ。Herald Other 留在資料，不進主估計。各入口的 guide 解釋終端 Other、完成點與缺類如何影響分母和分析。

| Registry key／guide | 各類在分析中的用途 |
| --- | --- |
| [`heralded/rb`](rb/guide.md) | gg、ge、eg、ee 分別計同類回返。每臂共用衰減 `p`，各類保有自己的振幅與偏移。四類擬合和成對 seed 重抽樣的信賴區間都可用才發布主估計。 |
| [`heralded/conditional_phase`](conditional_phase/guide.md) | 對齊四類的實際 spectator 與 X/Y 符號。每個 spectator、每個頻點共用斜率，各類保有自己的 contrast 與偏移。缺類不改用 gg-only。整個掃頻窗的斜率差完整且有唯一 π 交點才提議寫回。 |
| [`heralded/conditional_spectroscopy`](conditional_spectroscopy/guide.md) | 先做名目 I／Xπ 準備，再依實測 herald 分開畫四支條件頻譜。沒有共用擬合或寫回。 |
| [`heralded/cz_amplitude`](cz_amplitude/guide.md) | 名目準備和 herald 之後，在 CZ 前後都對兩顆 qubit 打 Xπ。比較四類同類回返，只用 gg 擬合首週 gain，不寫回。 |
| [`heralded/cz_optimize`](cz_optimize/guide.md) | 固定單一 depth、interleaved arm 和同一批 seeds。先等權平均每個 seed 的四類回返率，再等權平均 seeds。缺類的評估無分，完成該次 seeds 後停止搜尋。起點與至少一筆非零包絡候選都有分數才提案。候選嚴格勝過起點才預選。不估 `p`。 |
| [`heralded/chevron`](chevron/guide.md) | 名目準備後依實測 herald 分成四張共色標的圖。目標是 ee → 推定 CZ 輸入 \|11⟩ → 終端 gg。其他三類只作旁觀對照，不進目標 duration fit。不寫回。 |
| [`heralded/local_z`](local_z/guide.md) | Target herald e 時將 X、Y 反號。Spectator herald g 的兩類共用主斜率；spectator herald e 的兩類共用另一斜率，只作條件相位對照。不寫回。 |
| [`heralded/idle_zz`](idle_zz/guide.md) | Target herald e 時反轉對比；spectator herald e 時對調名目 I／Xπ。每支實際 spectator g/e 只用 gg 與 target herald e、spectator herald g 的一類共用頻率。兩支都有有號頻率和標準誤才報差值 ζ。其餘類別作診斷，不寫回。 |

比較普通與 heralded 分析時，可給兩者相同的合成終端計數。這只核對計算，不證明實機採集或準備等效。Chevron 的對照是普通入口的 reset gg 與 heralded Chevron 的 ee 目標，兩者在這個對照都讀終端 gg。

## 開新 tab 與載入搬移前的結果

Registry key 是 `zcu_lab/v2/` 下的實驗目錄。GUI New Tab 與 MCP 的 `open_tab` 都用新 key。搬過來的三個入口保留原本的 Native tag，tag 用來識別檔案，不是另一個可開 tab 的 key。

| 從 New Tab 或 MCP 開啟 | Native tag（不變） |
| --- | --- |
| `heralded/rb` | `two_qubit/rb_heralded` |
| `heralded/conditional_phase` | `two_qubit/conditional_phase_heralded` |
| `heralded/conditional_spectroscopy` | `two_qubit/conditional_spectroscopy_heralded` |

以上三個舊 tag 也是搬移前的 registry keys。現在不註冊這些 key，不留舊 Python import 路徑，也不新增舊 schema reader。搬移不改 cfg schema 或 HDF5 映射。開新入口後，用既有 **Load** 讀入同 schema 的搬移前 Native 檔。不相容的舊格式不因此變得可讀。五個新入口各自的 Native tag 與自己的 key 相同。

既有 MCP recipes 改名為 [`heralded_rb`](../../recipes/two-qubit-guide.md#heralded_rb) 和 [`heralded_conditional_phase`](../../recipes/two-qubit-guide.md#heralded_conditional_phase)。RB recipe 仍只回報 gg、ee 兩類各兩臂的品質欄位，主估計仍用四類。Conditional spectroscopy 沒有對應 recipe，直接開 tab。

## 共用規則在哪裡

[ADR-0074](../../../docs/adr/0074-flat-readout-module.md#保留獨立-heralded-實驗的條件) 擁有 herald reset 與獨立家族的規則。[ADR-0072](../../../docs/adr/0072-module-boundaries.md#可執行投影) 擁有 import 邊界。`two_qubit` 與 `heralded` 不互相 import；兩邊直接使用中立的 `zcu_lab/v2/support` 或既有 lib owner，不透過另一家族轉匯出。

- [Paired counts](../support/paired_counts.py) 檢查有序 mux tones 的 circle 和 radius，並從保存的計數算對比等統計量。[Heralded projection](../support/heralded_projection.py) 依 target 和 spectator 的 herald 分類對齊投影，由 phase、local Z 與 idle ZZ 各自選要擬合的類別。
- [Paired RB sequence](../support/paired_rb_sequence.py) 與 [scheduling](../support/paired_rb_scheduling.py) 供普通 RB、heralded RB 與 CZ optimize 用。各入口寫自己的採集順序。[Paired RB analysis](../support/paired_rb_analysis.py) 匯出 `RBOptions` 與 `RBEstimate`；Notebook 從這裡 import 這兩個型別，RB recipes 的報告 helper 也從這裡取 `RBEstimate`。
- 普通與 heralded CZ amplitude 呼叫 [CZ return](../support/cz_return.py) 的首週回返擬合。Chevron 的 duration fit 由 [Rabi duration](../support/rabi_duration.py) 擁有。可重用的數值模型由 [`zcu_tools.analysis.fitting`](../../../lib/zcu_tools/analysis/fitting/README.md) 持有，例如 `fit_joint_decay`、phase 擬合及 Idle ZZ 的 `fit_weighted_decay_fringe`。Lib 不解讀 herald 類別、實驗軸或寫回資格。
- 固定的 [Measure support](../support/measure/README.md#mux-綁定驗證) 讀表單 capture，依 `$mux.qubits` 的次序找 tones。它不匯入可重載的 leaf 型別。各 Form 自己選來源與初值：RB 用既有 Slot，conditional phase 用 `source` 選 Slot 或 `readout/mux:blank`，conditional spectroscopy 沿用 `mux_readout_initial(ctx)`，不加 ring-up。這三個 Form 中，RB 和 phase 要求 `$coupler`，spectroscopy 不要求。修改固定的 `support/measure` 後要重啟 GUI。

各入口有自己的 cfg、Result、採集政策、Analyze 與 Figure。Form 和 GUI 接線也留在各入口，不共用完整 Form 工廠或分析圖。看特定入口的 gate 順序、寫回和缺點判讀時，從上表進 leaf guide。

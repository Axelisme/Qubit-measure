---
date: 2026-10-28
status: active
superseded_by: null
related:
  - ../../lib/zcu_tools/simulate/ftf/README.md
  - ../../lib/zcu_tools/simulate/evolution/README.md
  - ../adr/0078-evolution-module-boundary.md
basis:
  repo_commit: 84e9a420abe862b5f2ea38b86272fc4037f3ebb6
  external_sources: none
---

# FTF 固定 DC flux ramp 的截斷與成本界限

## 結論

本次結果是「未認證，僅界限」。M16 是觀測到的最大成功執行尺寸，不是已收斂的截斷。F1 從 DC 0.4 到 integer flux 0 的 reset，在同一組 DC 最低 K=8 態 projector 下，M8 與 M16 的 ground population 分別為 0.809094 與 0.626069。兩者相差 0.183025，K8 population L1 差為 0.359636，遠大於 1e-3 判準。不能推薦 M8 用於這個 0.4 Φ0 excursion，也不能把 M16 當真值。

主網格的 M24 第一點與唯一 M32 reference 都在 4096-node phase/gauge quadrature 上限失敗。其餘五個 M24 候選依成本與未認證界限裁決不執行，不能把它們標成失敗。三迴路、park 0.25 及 tau20 的 M8 覆蓋點均完成，但沒有較大 M 對照，不能認證任何幅度可用區。Native cutoff 沒有做收斂比較。

這組合成模型的成功點花 0.379 至 4.333 秒。長 hold 可以用常數段表示，不必隨 hold 增加 DAC nodes。現有結果不足以判定必須換新算法。下一步應先處理 quadrature 不收斂，再驗證固定 DC 投影的 M、local 與 native 收斂。本次沒有修改 engine、kernels、family 或 cache。

## 問題與輸入

評估要回答固定 DC 投影在大幅 flux reset 時保留多少人口資訊，以及 level 數、M 與 hold 長度的成本。這不是 UI 或 state-machine 展示，也沒有微波 drive。

模型是合成的非對稱 F1/C/F2，並非擬合的實驗裝置。單位為 GHz、ns、K、Φ0。

| 輸入 | 值 |
| --- | --- |
| DC | F1/C/F2 都是 0.4 |
| F1 的 EJ、EC、EL | 3.0、0.8、0.6 |
| F2 的 EJ、EC、EL | 3.3、0.7、0.55 |
| Coupler 的 EJmax、EC、asymmetry | 10.0、0.3、0.2 |
| g1c、g2c、g12 | 0.05、0.06、-0.04，乘完整 charge operator |
| 溫度 | 0.08 |
| Fluxonium native cutoff / coupler ncut | 24 / 8 |
| 每個 local circuit 的 levels | 4 或 5，parent dimension 為 64 或 125 |
| F1 baths | q_cap=1e4、q_ind=2e5、xqp=1e-7 |
| F2 baths | q_cap=1.2e4、q_ind=2.2e5、xqp=1e-7 |
| Coupler bath | q_cap=2e4 |
| 純退相干與 residual photons | 關閉，兩個 resonator 的 charge coefficients 與 thermal occupation 都是零 |
| 演化預算 / workers | 1e-4 / 1 |
| BLAS、OMP、MKL、Numba threads | 各 1 |

主序列為 F1 的 cosine ramp、constant hold、cosine return、2 ns DC terminal。每條 ramp 在 2 ns DAC 格點上取樣，ramp 長度是 8、32、128 ns。主網格 tau=0，park=0，M 為 8、16、24，local 為 4、5。M32 只安排 local5、ramp8 的一個 reference。

Hold 對所有主網格使用相同的 18662 ns。先建立八個 M/local 組合的公開 park model，取第一激發態 outflow lifetime 的最大值 3732.292 ns，再將五倍向上取到 2 ns 倍數。定義為 `1 / sum_i R[i,1]`，其中 `R[target,source]` 是公開 directional rates。這不是整個 multilevel 系統的 mixing time。

四個覆蓋點使用 M8/local4/ramp8。Coupler 與 F2 各從 0.4 到 0，另外兩點為 F1 park=0.25、F1 park=0 且 tau=20。各自從公開 rates 計算 hold，沒有沿用 F1 的 lifetime 假定。

Tau20 保留兩側各 40tau 的 causal settling，之後以兩個端點表示 constant plateau。省略的指數殘差小於 `0.4*exp(-40)=1.7e-18 Φ0`。這是明示的浮點近似，不是把濾波過渡改成階躍。

## 比較方法與結果

所有結果都投到 local5/native24/ncut8 模型在 DC 的同八個最低能態 projector。透過 `native_embedding()` 比較相同的 oscillator × charge × oscillator 座標，不用不同截斷各自的能態 index 冒充同一個 projector。人口不重正規化，K8 外的 mass 另報。

唯一 M32 reference 沒有成功。以下以同 ramp 的 M16/local5 作未認證比較基準，並排除對自身的零差。這些數字是模型間差異，不是對真值的誤差上界。

| 每條 ramp | M8/local5 p0 | M16/local5 p0 | K8 L1 差 | M16 的 K8 外 mass |
| --- | ---: | ---: | ---: | ---: |
| 8 ns | 0.809094 | 0.626069 | 0.359636 | 0.0309563 |
| 32 ns | 0.809094 | 0.626069 | 0.359636 | 0.0309563 |
| 128 ns | 0.810178 | 0.626874 | 0.360355 | 0.0307602 |

M8 與 M16 的初始 DC Gibbs 分布也不同。ramp8 的初始 K8 L1 差為 0.129075。比較 reset 前後變化量，K8 L1 仍有 0.355126。因此差異不只是初態 Gibbs 截斷的重正規化。

相同 M16、相同 ramp 的 local4/local5 最終 K8 L1 差為 3.843e-5 至 3.855e-5。這只是一次 local levels 對照，不能代替 M 或 native 收斂。M8 與 M16 都提高了本模型內的 p0，但這不構成截斷正確的證據。

| M8/local4 覆蓋點 | Hold ns | Hold / outflow lifetime | 最終 p0 | p0 增量 | Ordered final 對 park Gibbs L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coupler 到 0 | 20202 | 5.00010 | 0.584616 | 0.162209 | 0.001903 |
| F2 到 0 | 6618 | 5.00094 | 0.820889 | 0.398481 | 0.042425 |
| F1 到 0.25 | 8068 | 5.00055 | 0.563061 | 0.140654 | 0.007203 |
| F1 到 0，tau20 | 10346 | 5.00050 | 0.787744 | 0.365336 | 0.052531 |

五倍第一激發態 outflow lifetime 沒有保證全系統達到 Gibbs。上表的 Gibbs 差另外包含 ramp 路由與 terminal 段作用，不能單獨當作 plateau mixing 誤差。沒有較大 M 的覆蓋點 reference，所以本次不能判定 0.15 Φ0 excursion 已可用。

18 個成功的正式與成本診斷點最大 trace 誤差為 2.84e-13，最小 density eigenvalue 為 0.001877。這只檢查數值密度合法性，不證明近似、截斷或 reference 正確。沒有 clipping 或 renormalization，K8 外 mass 的約 -1e-14 值保留為浮點殘差。

## 成本與表示

以下為每點內部 wall timer，包含模型、timeline、levels、演化及結果保存，不含 process 啟動與 imports。每點都是 fresh process、fresh JointModel 與 Engine，但共用同版本的磁碟 Numba cache。各次 quadrature 工作量不同，不能從 level 數單獨擬合成本。

| M / local | ramp8 wall s | ramp32 wall s | ramp128 wall s |
| --- | ---: | ---: | ---: |
| 8 / 4 | 0.718 | 0.379 | 0.855 |
| 8 / 5 | 0.824 | 0.568 | 0.853 |
| 16 / 4 | 4.106 | 1.885 | 1.799 |
| 16 / 5 | 4.333 | 1.903 | 1.917 |

M8 三種 ramp 分別有 6、18、64 個 unique quantized levels，waveform nodes 為 11、35、131。開模花 0.003 至 0.010 秒，M8 level 建構花 0.017 至 0.286 秒。M16/ramp8 演化較慢，不能因為 rows 少就預期較快。

最初由 tau2 小 pilot 保守估計六個 M8 點需 90 至 246 秒，實測合計 4.196 秒。第二階段以 M8 演化乘 M³、level 成本乘 M²估計 M16 每點 2 至 6 秒，實測為 1.799 至 4.333 秒。M24 每點 6 至 18 秒只是成本外推；第一點提前失敗，不能把失敗 walltime 當成功演化成本。M32 的 40 秒上限估計也只得到提前失敗。

四個覆蓋點 walltime 為 0.395、0.432、0.427、2.091 秒。Tau20 有 811 nodes、90 unique levels、98 個 adiabatic_map rows 與 1 個 t4 park row。所有有效正式與 hold 診斷點合計 24.878 秒。加上 pilots、失敗與表示錯誤的所有演化嘗試，內部 wall 合計 41.234 秒，未達十分鐘。峰值 RSS 約 276 MiB，包含 imports 與 JIT，不能當作純模型陣列大小。

### Hold 長度

| M8/local4/ramp8 hold | Nodes / timeline bytes | Timeline build s | Evolution s |
| --- | --- | ---: | ---: |
| 18.662 µs，既有正式點 | 11 / 672 | 0.108 | 0.551 |
| 186.620 µs，成本診斷 | 11 / 672 | 0.154 | 0.503 |
| 1866.200 µs，成本診斷 | 11 / 672 | 0.137 | 0.445 |

這三次觀察沒有顯示演化成本隨 hold 增長。另有只建構的 tau20 對照，hold 為 4、40、400 µs 時都使用 811 nodes、29088 bytes，warm build 約 4 ms。不能由此推論任意其他 pulse 表示都具有相同成本。

### 八項 FIFO

公開 eigensystem 的兩遍順序查詢，在 7、8 個 unique levels 時，第二遍分別 reuse 7、8 個原 array。9、32 個 levels 時沒有 reuse，第二遍各花 1.83、5.69 ms。相同 workload 的 ModelSource 第二遍保留全部 level model，花約 4 至 19 µs，第一遍則為 19.8 至 93.8 ms。

這個觀察說明超過 FIFO8 可以讓直接 eigensystem 重訪重算，但 source model memo 沒有同樣的八項限制。Continuous quadrature 另以 FluxExpansion 建立瞬時 Hamiltonian，不能把該工作量都算成 JointModel FIFO miss。Identity reuse 只是此版本的觀察，不是新增 API 保證。本次不改 cache policy。

## 失敗邊界與被否決的做法

| 輸入 | 失敗階段 | 失敗演化成本 s | 總 wall s | 結果 |
| --- | --- | ---: | ---: | --- |
| 舊 pilot，M32/local4、F1 0.4→0.2、tau2、ramp8、hold20ns | Evolution prepare | 約 4.17 | 4.799 | Phase/gauge quadrature 到 4096 nodes 未收斂 |
| 正式 M24/local4、F1 0.4→0、tau0、ramp8、hold18662ns | Evolution prepare | 2.039 | 2.256 | 同類 quadrature 失敗 |
| 唯一 M32/local5 reference、F1 0.4→0、tau0、ramp8、hold18662ns | Evolution prepare | 2.515 | 2.952 | 同類 quadrature 失敗 |

這三項不是 active exact-degeneracy bath-rate raise。正式失敗點的 quantized level models 與公開 rates 已建好，之後才在演化 quadrature 失敗。本次已走過的三迴路 levels 沒有觀察到 bath exact-degeneracy raise，但沒有證明所有 flux 與所有 M 都安全。

另保留一次 scratch 表示錯誤。M8/local4/ramp128 的第一版合併相同 quantized level，遺失實際 ramp-end 邊界，127 rows 全走 adiabatic_map。該結果不算有效 reset 點。經使用者批准後，腳本加入公開 `segment_boundaries`，只補跑該點一次，原錯誤證據沒有覆寫，其他成功點不重跑。

沒有增加 quadrature 上限，沒有把 M32 改成多次試參數，也沒有強迫 ODE 作替代真值。本次沒有量 exact ODE 慢路徑；慢成本界限是上述走到 4096-node 上限後失敗的實測，不是 ODE throughput。較大的 M 不一定更正確，尤其尚未完成 local/native convergence。

## 既定近似與留下的問題

Continuous waveform row 的 adiabatic_map 保留 ordered populations，只衰 coherence，不做 population 弛豫。真正 constant park 走 t4。本次有效正式點都有獨立 t4 park，並非把人口保存當 reset。

當 ramp 遠短於 T1 時，忽略 ramp 中的 population 弛豫可以作為近似。Ramp 接近 T1 時會低估弛豫。Kernel 的 quadrature 收斂也不認證 slow-motion、noncrossing、nondegenerate 的物理 adiabatic 條件。本次沒有用精確非絕熱動力學驗證 8、32、128 ns ramp。

R8/r32 的 terminal 2 ns DC 段各有 t4，r128 的 terminal 段與最後 return motion 共用 row，仍走 adiabatic_map。因此不同 ramp 的人口差不能當作物理收斂曲線。共同 projector 的 M 對照都使用同 ramp、同物理序列與相同邊界表示。

後續 sim-performance 輸入應分開處理兩件事。

1. M24/M32 的 4096-node phase/gauge quadrature 不收斂，尚不知是截斷高態、能態排序、gauge transport 或積分條件造成。這是需重現的數值失敗，不是 bath degeneracy。
2. 同一 DC K8 projector 下 M8/M16 的 p0 差為 0.183025，需區分固定 DC joint 投影與更深 local/native 截斷。M16 沒有成為真值。

本次只交付這兩項具體輸入，不在產品中更換算法。

## 依據

程式依據為上述 basis commit 的 `simulate.ftf` 公開模型與 flux API、`simulate.evolution` 的 TimelineBatch、Engine 與 native embedding 契約。算法與 cache 語義以該 commit 的 module README 及 docstring 為準。

原始腳本、逐點 JSON/NPZ、CSV、PNG、命令與失敗 logs 沒有入庫。此快照已列出判讀所需輸入與數值。

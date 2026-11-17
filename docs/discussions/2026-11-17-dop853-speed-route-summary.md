---
date: 2026-11-17
status: active
superseded_by: null
related:
  - ../adr/0078-evolution-module-boundary.md
  - 2026-11-17-dop853-profile-scenario-routing.md
  - 2026-11-14-sampled-multitone-interim.md
  - ../../tools/README.md
  - ../../lib/zcu_tools/simulate/evolution/README.md
basis:
  repo: f2df2414f6903243c0485c07e507570ef3f6b754
  external: none
---

# DOP853 加速路線的證據與未驗項

## 結論

多 worker 是本輪唯一取得精度合格、按輪配對加速數字的路線。64 點 Gaussian 單段批次從 1 worker 改用 2、4、8 workers，wall speedup 中位數分別為 1.824922、3.170395、4.002307 倍。這些數字不適用於尚未量的任意取樣批次，也不決定環境預設。

取樣前導共享的擴充潛力未知。現行指定 Gaussian 掃描不觸發自動共享，省下的重複演化為 0，不能據此推論擴充後也沒有收益。`multitone_rwa` 則在准入階段拒收取樣重疊列，恢復取樣准入須另裁，不能當成只需放寬數值 guard 的路線。

載波積分器固定原型在原 14 組座標全部未通過完整 density 誤差門檻，沒有有效 2 倍加速結果。GPU 全 14 組比較未量。使用者於 2026-11-17 接受五小時收尾的範圍凍結，將 GPU discovery、環境、量法與共同 Session 文件放寬全部延到下一 effort。未量不表示未達 2 倍，更不表示 GPU 全面無益。

本文只整理既有證據與選項，不選 production 路線、不改 worker 預設、不新增量測或實作。本次收尾不等待新的路線裁決。

## 問題

取樣雙 tone 重疊列需要完整驅動的數值演化。要縮短耗時，可以分散獨立掃描點、少算重複前導，或更換數值傳播方法。這幾條路線的證據不同，不能只列速度而略過精度資格與准入限制。

本輪以自帶 single-fluxonium 模型的合成輸入比較，不使用實驗 YAML 或私有校準參數。每段有兩個同起同止的 tone。Gaussian 也以線性取樣包絡送入，另一種 sampled 包絡不是 Gaussian。這些輸入不是已校準的 STIRAP 轉移。

### 已量座標與比較邊界

原 14 組是下列 7 個座標各配 Gaussian、sampled 兩種包絡。D 是單段時長，L 是 retained levels，K 是每條包絡的 knots，N 是同一 timeline 串接同段的次數。Points 固定 1，總時長是 D×N。每次只改一軸。

| 座標 | D ns | L | K | N |
| --- | ---: | ---: | ---: | ---: |
| 基準 | 20 | 8 | 9 | 1 |
| 時長 | 40 | 8 | 9 | 1 |
| 能階 | 20 | 20 | 9 | 1 |
| 取樣密度 | 20 | 8 | 17 | 1 |
| 重複 2 段 | 20 | 8 | 9 | 2 |
| 重複 20 段 | 20 | 8 | 9 | 20 |
| 重複 100 段 | 20 | 8 | 9 | 100 |

Worker 掃描另用 Gaussian、D=20 ns、L=8、K=9、N=1、64 個獨立點，Pump gain 從 0.8 到 1.2，Stokes 不變。N 不是 worker 數，也不是獨立點數。

原 14 組 DOP853 的完整 density 對獨立 reference 最大誤差為 `5.70977479e-5`，低於 `2e-3`。跨 checkout 計時的兩側 simulate 實作相同，是 A/A 控制，不是 production 優化成果。Profile 的 14 組最高 self-time 項都是 `advance_endpoints`。Numba 編譯內部沒有拆帳，不能由此估算其內部某一步的收益。

計時包住 `Engine.evolve`，因此是 whole-Engine 成本，不是裸 kernel 成本。Profile 的 calling-thread CPU 與 logical-row 攤提值不能用來拆正式 telemetry-off 樣本。完整 density 與同維 oracle 相符，也不表示 retained-level truncation 已收斂。

## 考慮過的選項與比較

下表按證據可用程度排列，不替未知倍數排序。

| 路線 | 本輪可說的倍數或狀態 | 證據範圍 | 代價與風險 | 下一 effort 未驗步驟 |
| --- | --- | --- | --- | --- |
| 多 worker | 2/4/8 workers 的 wall speedup 為 1.824922/3.170395/4.002307 倍 | 一個 64 點 Gaussian 單段批次，8 輪配對，density 合格 | 可用既有 Engine 參數，但 process CPU 隨寬度上升，效益依批次而變 | 若要推廣到任意取樣或環境設定，仍缺各自的成本與精度證據 |
| 取樣前導共享 | 擴充潛力未知；指定場景現行自動共享省 0 | 輸入構造與共享條件的靜態查核，沒有有／無共享計時 | 改共享條件或由 caller 接續都有整合成本，需計入代表前導 | 序列層比整批演化與含前導成本的接續方案，量 wall/process 與完整 density 差 |
| 恢復 `multitone_rwa` 取樣准入 | 未知；現行取樣重疊列不可用 | 公開准入契約，加兩個代表場景的 4 次 inspect；其餘 13 場景是構造推論 | 需另裁准入與完整序列精度資格，不是調寬數值 guard 即可 | 若重開，仍須完整序列驗證與恢復原 19/20 次成本邊界驗收 |
| 載波積分器 | 固定原型 14/14 組精度不合格，無有效 2 倍結果 | Magnus1 加 Strang、每 knot 一步／兩步的固定兩網格原型 | 時間排序截斷、bath splitting、長序列累積誤差尚未解決，另有歷史 process 證據缺口 | 更高階或不同網格若另行研究，先取得精度資格再談速度 |
| GPU 批次 | 全 14 組比較未量，倍數未知 | 無本輪 GPU 速度或精度結果 | 裝置、FP64、batch 成本、精度與量測接線均未驗；CPU 準備不等於 GPU backend | discovery、環境、數值比較、共同量法與文件全部留下一 effort |

### 多 worker 的配對結果

同一 checkout、`families=("dop853",)`、budget `1e-3`。使用共同 SampleRunner 的 `warm_clear_history` 模式，每個寬度兩次暖機，另有一筆不保留開場。保留樣本按 `[1,2,8,4]`、`[2,4,1,8]`、`[4,8,2,1]`、`[8,1,4,2]` 順序跑兩遍，共 8 輪、32 筆，全部 telemetry off。最後每寬度一筆 telemetry-on 診斷不進計時。量測使用 CPU 0 至 7，native thread 設定為 1。

每輪先算 `T1/TN`，大於 1 表示 N workers 較快。表中 speedup 是 8 個按輪比值的 median，IQR 使用 linear Q25/Q75，不是兩個耗時 median 相除，也不是跨時段配對。IQR 描述分散，不是信賴區間。

| Workers | Wall median s | Wall T1/TN median | Wall 比值 linear IQR | 效率 speedup/workers | Process median s | Process T1/TN median |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 1.304899 | 1.000000 | [1.000000, 1.000000] | 1.000000 | 1.300756 | 1.000000 |
| 2 | 0.709484 | 1.824922 | [1.803721, 1.870863] | 0.912461 | 1.366777 | 0.947553 |
| 4 | 0.409913 | 3.170395 | [3.106459, 3.265609] | 0.792599 | 1.515028 | 0.859121 |
| 8 | 0.329593 | 4.002307 | [3.911054, 4.071161] | 0.500288 | 1.966037 | 0.660299 |

2、4、8 workers 的 wall 比值 IQR 都大於 1。Process clock 計整個 process 所有 thread 的 CPU，增加 workers 雖降低 wall，卻增加 CPU 總成本。8 workers 的效率約 0.50，不能把核心數當成線性收益。

32 筆保留樣本的完整 `64×8×8` complex density 與第一筆 W1 逐位相同，最大差 0。對同座標 reference 的最大誤差為 `8.31179654e-6`，reference 兩級收斂差為 `2.06024087e-12`。這同時提供 worker 一致性與獨立精度依據，不能只靠各 worker 彼此相同替代正確性。

### 取樣前導共享的現況與未知潛力

原 14 組只有一個 point，沒有跨 timeline 的共享。第 15 組 64 點的 Pump gain 從 t=0 起不同，也沒有相同前段。Repeats 只在同一 timeline 接同段，不形成跨 timeline 前導。

現行自動共享在沒有 caller 初態時，尋找 start>0、恰一個 active tone、包絡為 constant 的 probe 列。同一 probe 起點至少要有兩列，且之前的完整物理歷史相同。共享的是 probe 起點前的 density，前導本身可以含取樣 drive。這是當時實作的研究查核，不是新增公開契約。

Gaussian、drive-only STIRAP 在 σ=0.07 µs、lead=0.105 µs，以及 GUI Gaussian seed 的 σ=0.05 µs、lead=0.05 µs 下，都沒有合格 probe 分組。固定 Stokes detuning 的點雖可有相同 Stokes-only 前導，現行條件仍不觸發，GUI 也沒有使用 caller 接續。因此現況省下的重複演化為 0。這是條件推論，不是兩次計時相減或「未觀察到」。其他帶至少 10 ns 等值平段的包絡尚未判定，不能推廣到所有 gauss、cosine、drag、arb 設定。

另一個 8 能階合成 lead 掃描 `overlapping_shaped_tones` 只提供旁證。在預設選路、`first_evolve` 下，calling-thread CPU 的 apply median 為 DOP853 `19.6888043 s`、period_table `0.091967875 s`、shaped_sambe `0.085473192 s`。後兩者含單 tone 前導與尾段，合計約占已報 apply 的 0.9%。這不是共享潛力的上限，也不是預期加速。模型、包絡、單 tone 選路、guard 與 prepare 成本都可能改變。

下一 effort 的候選量法是在同一 checkout，用序列層 Gaussian STIRAP detuning 型 batch 比較整批 `evolve`，以及每個 Stokes detuning 先算一個代表前導，再以 `TimelineBatch.start_ns`／`initial_densities` 接續各點整列。代表前導的成本一起算，報 wall、process 與完整 density 差。序列總耗時不當作 kernel 成本。本文沒有執行這個對照，也不新增共享開關或診斷欄位。

### 取樣拒收是准入限制

[Kernel summary](../../lib/zcu_tools/simulate/evolution/README.md#kernel-summary) 當時只准 `multitone_rwa` 接受 fixed-flux 的兩個 constant 包絡，明文拒收 sampled rows。因此取樣重疊列的數值 guard 狀態是「未到達，准入條件拒收」。這是契約推論，不是 inspect 看到了 guard 內部路徑。

兩個代表場景是 `dop853_segment-gaussian-d20-l8-k9-r1-p1`、`dop853_segment-sampled-d20-l8-k9-r1-p1`。各做一次預設 families 與一次只允許 `multitone_rwa` 的公開 inspect，budget `1e-3`、workers=1，沒有演化 density。

| 代表場景 | 預設 inspect | Pinned inspect |
| --- | --- | --- |
| Gaussian，D=20 ns、L=8、K=9、N=1、P=1 | batch0/point0/segment0，start0 ns、duration20 ns；multitone_rwa rejected，dop853 accepted 並 selected | `NoValidKernel`，domain=multiple，attempted=[multitone_rwa]，points=[0]，segments=[0] |
| Sampled，D=20 ns、L=8、K=9、N=1、P=1 | batch0/point0/segment0，start0 ns、duration20 ns；multitone_rwa rejected，dop853 accepted 並 selected | 同上，首個失敗 group 是 point0/segment0 |

這是兩個代表場景的預設選路旁證，不是由 DOP853-pinned 計時推定預設，也不表示所有場景都選 DOP853。其餘 13 個場景依包絡類別與構造推論拒收，沒有逐場景 inspect。公開 inspect 不能逐列解釋拒收原因，不能排除內部錯走數值 guard 後仍拒收；`NoValidKernel` 只描述首個失敗 group。未列出的候選不當作 rejected，`deferred` 或 `untested_after_short_circuit` 也不當作數值拒收。

至少 10 ns 的等值取樣平段會切成 constant 列，兩個平段可進 `two`，不在這項取樣重疊拒收結論內。恢復取樣准入仍受[既有拒收討論](2026-11-14-sampled-multitone-interim.md)約束，須另裁並保留完整序列精度與 19/20 成本邊界驗收。本輪不恢復准入。

### 載波積分器的有限負結果

固定原型使用 exact-moment interaction-picture Magnus1 加 secular Strang。它解析積分每個線性包絡區間的載波與 Bohr 頻率，不做 RWA，但有限步數仍有時間排序與 splitting 誤差。每 knot 區間 coarse 一步、fine 兩步，每次重算兩網格並回傳 fine density。兩網格成本都在 timer 內，不用結果驅動 refinement，也不用單段矩陣冪代替 N 段演化。

A 是 `Engine(workers=1, families=("dop853",))`，telemetry on；B 是 scratch 原型。每座標各側兩次暖機後，量 3 個 ABBA blocks、各側 6 筆正式樣本。每 block 算 A 的兩筆時間和除以 B 的兩筆時間和，再取 median 與 linear IQR。這是 whole Engine 對原型，不是相同 production 接線的裸 kernel 比較。

以下誤差是各側 8 次 call 的完整 density 最大逐元複數差。每列 points=1，座標意義同前表。誤差門檻是 `2e-3`。

| D ns | L | K | N | Gaussian A 誤差 | Gaussian B 誤差 | Sampled A 誤差 | Sampled B 誤差 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 8 | 9 | 1 | 7.25252e-6 | 0.00765016 | 9.55623e-6 | 0.00712557 |
| 40 | 8 | 9 | 1 | 5.97655e-6 | 0.00427262 | 7.72358e-6 | 0.0180551 |
| 20 | 20 | 9 | 1 | 1.51266e-5 | 0.00764988 | 9.28288e-6 | 0.00712541 |
| 20 | 8 | 17 | 1 | 9.14882e-6 | 0.00289556 | 3.24415e-6 | 0.00314243 |
| 20 | 8 | 9 | 2 | 4.61093e-6 | 0.00832746 | 2.01975e-6 | 0.00837381 |
| 20 | 8 | 9 | 20 | 1.17901e-5 | 0.109769 | 9.27604e-6 | 0.137686 |
| 20 | 8 | 9 | 100 | 5.70977e-5 | 0.205189 | 3.15888e-5 | 0.135100 |

獨立 full-Lindblad interaction reference 的兩級 rtol/atol 為 `2e-10/2e-12`、`2e-12/2e-14`，14 組最大收斂差 `1.53583590e-10`。A 全部合格，B 全部不合格。即使 Gaussian 基準的 raw wall A/B median 為 `2.94338`，也不是有效 2 倍加速。取得合格精度所需網格及其耗時仍未知。

這個負結果只涵蓋上述 14 座標與固定兩網格版本，不否定所有解析載波方法。原型只處理靜態 DC 與完整重啟初態，未驗 continuation、continuous flux、segment shifts 或 production routing。當時數值函式約 131 行，整合與 guard 的另 200 至 400 行只是未實作估算，不能當成可直接移植的 patch。

原 measure 缺 numeric process 的 native corroboration，以及 transient children 與其資源觀察，這些仍 unknown。後續只讀 archive review 已確認 224 份 archive、560 個 complex128 矩陣與原 metrics 一致，但不能修復歷史 process 缺口。因此這是帶限制的失精度負結果，不是正式性能門檻或完整 CPU/process compliance 證明。

### GPU 保持未量

GPU 原定全 14 組 batch 比較沒有執行。2026-11-17 的使用者原話是「都同意，你给我最快效率地完成剩下的工作」，接受流程減重與範圍凍結，GPU 全部延到下一 effort。這個延期沒有產生任何 GPU 密度誤差或速度樣本。

共同 Session 的非 Engine 量法與文件放寬也未落地，不能宣稱已接通正式 GPU backend。既有合成 CPU 準備測試不構成 GPU 數值證據。下一 effort 仍缺 discovery、環境確認、具體量法與數值比較，現在不能為未執行的比較下速度結論。

## 被否決的選項與理由

不把多 worker 數字推廣成所有輸入的預設，因為任意取樣與實際序列沒有相應配對結果。也不把不同時段的控制計時拼成 speedup。

不以現行共享省 0 或另一場景的 0.9% 旁證替取樣前導定價，因為兩者都不是擴充方案的有／無對照。不以 cache hits 或 logical rows 反推 prefix 分組。

不把 RWA 取樣拒收描述成 guard 未查清，也不以 pinning 繞過准入。不以載波原型的失精度速度宣稱有效加速。不把 GPU 未量填成 0、未達 2 倍或全面無益。

本輪沒有選擇 production 實作。使用者已凍結收尾範圍，因此不補量測、不新增設計題或票券，也不等待 ruling。

## 留下的問題

可直接比較的收益目前只有指定 Gaussian 批次的多 worker 結果。其餘路線保留各自的未驗條件，沒有足夠證據排出一張預期倍數排行榜。

取樣前導共享的整體收益、任意取樣的 worker scaling、恢復取樣准入的資格、精度達標的載波算法，以及 GPU batch 的收益，都留下一 effort。序列整體影響須在多段 `TimelineBatch` 的 `Engine.evolve` 驗最終 density 與總耗時，不以單段結果代替，也不新增 program→acquire 的 e2e。

## 依據

原始研究工作檔未入庫。本文直接保留判讀結論所需的座標、數字、比較方向與限制，不要求讀者取得本機工作檔。

- 報告基準為 repo commit `f2df2414f6903243c0485c07e507570ef3f6b754`。既有 profile、workers 與代表 inspect 的 candidate 是 `2514d94bef8b6500e6cc3f4a4bbf4663616d7001`；載波原型的固定 repo HEAD 是 `e399448faa732fb66100d0f5abca34f03fd7a9ee`。這些 commit 記錄 repo 狀態，不表示一次性研究原型已入庫，也不是本輪重新量測。
- 遵循 [ADR-0078 的完整序列精度原則](../adr/0078-evolution-module-boundary.md#family-去留合格只比耗時)，先判精度再比速度。本文沒有用單段合格宣告 production family 去留。
- 遵循 [ADR-0078 的分層](../adr/0078-evolution-module-boundary.md#模擬器的測試與量測分層不做-e2e)，單段、dispatch inspect 與序列整體影響分開。本文是既有研究的彙整，沒有新增求解或驗收入口。
- 遵循[取樣 multitone 拒收先例](2026-11-14-sampled-multitone-interim.md)，拒收是現行 admission，不預告修完後恢復。公開資格與 inspect 的可觀測限制由 [evolution README](../../lib/zcu_tools/simulate/evolution/README.md#dispatch-inspection) 擁有。
- [既有 profile 接線快照](2026-11-17-dop853-profile-scenario-routing.md)記錄名稱與 producer owner，不重寫成效能報告。[專用單段用法](../../tools/README.md#dedicated-dop853-segments)與 [profile owner](../../tools/dop853_segment.py)記錄輸入宣告，宣告本身不證明已完成量測。
- [公開接續介面](../../lib/zcu_tools/simulate/evolution/README.md#public-entry-points)擁有 `start_ns`／`initial_densities` 契約。[Prefix 實作](../../lib/zcu_tools/simulate/evolution/_prefix.py)與 [planning](../../lib/zcu_tools/simulate/evolution/_planning.py)是本輪靜態查核的機制出處，不在本文新增共享公開承諾。

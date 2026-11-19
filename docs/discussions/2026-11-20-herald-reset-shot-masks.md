---
date: 2026-11-20
status: active
superseded_by: null
related:
  - docs/adr/0074-flat-readout-module.md
  - docs/adr/0083-program-macros-and-timelines.md
  - lib/zcu_tools/program/README.md
  - lib/zcu_tools/program/acquisition.py
  - lib/zcu_tools/program/_shot_masks.py
  - lib/zcu_tools/program/_shot_processing.py
  - lib/zcu_tools/program/v2/modules/herald.py
  - zcu_lab/v2/two_qubit/rb/guide.md
  - docs/discussions/2026-10-29-heralding-post-selection.md
  - docs/discussions/2026-10-16-ftf-simulation-speedup.md
  - docs/discussions/2026-11-20-t1-cov-restore.md
basis:
  - "repo@95064aa80：裁決時的 main。Herald 只存在於 9 個 two_qubit `_heralded` 實驗；reset 模塊只播脈衝；`acquire_summed` 收 `processor=`；SNR 由 `TrackerMixin` 與 `MomentTracker` 計算"
  - "模擬路徑的兩組量測在 repo@f0b01de4d 加上未提交的路由改動上執行，物理參數是未修改的 `zcu_lab/simulation_envs/FTF/default.yaml`"
  - "使用者 2026-11-19 至 2026-11-20 的問答。決策檔、模擬量測紀錄與 RB 離線驗收紀錄是未入庫的工作檔，必要數字與結論寫在正文"
---

# Herald 改為 reset 模塊與逐 shot 遮罩

## 結論

- Herald（分類後 post-selection）是 reset 模塊。實驗沿用既有 `reset` 欄位選 `reset/herald` 或 `reset/mux_herald`，以 `keep` 指定要保留的態。第一個使用者是 plain `two_qubit/rb`。
- 這推翻 2026-10-29「上層使用新實驗組，比如heralding/*」的放置選擇，也推翻 [2026-10-29 討論](2026-10-29-heralding-post-selection.md)結論的第 3 點。同一天的「讓底層支援post-selection」與不做 active reset 不變。
- 9 個 two_qubit `_heralded` 實驗保留，它們保存全部 81 類 herald×終端計數。
- 取數層每一輪替每個 shot 算一個 bool 遮罩，形狀 `(points, reps)`，所有 ro ch 共用。Program 可以註冊多個遮罩來源，照註冊順序以 AND 合併。沒有來源時，所有輸出和改動前逐位元相同。
- 所有下游都收遮罩。平均輸出 `Σ m·x / N`。分類計數只計保留的 shot，機率是條件機率。`run_acquire_shots` 回傳 shots 加遮罩，框架不套用遮罩。
- Herald 讀取對實驗隱藏，label、讀取索引與 converter 都不改。
- Tracker 刪除。`run_acquire_summed` 改收顯式的 `ShotReducer[State]`（`start`、`add`、`output`），關鍵字由 `processor=` 改為 `reducer=`。SNR 改從 `IQ(read, order=3)` 的十個總和計算。
- FTF 模擬沿用逐 shot 的 IQ 抽樣，統計抽樣的提案撤回。
- 現行規則寫在 [ADR-0074〈Herald reset 與逐 shot 遮罩〉](../adr/0074-flat-readout-module.md#herald-reset-與逐-shot-遮罩)與 [ADR-0083〈Builder 與 Schedule 的職責〉](../adr/0083-program-macros-and-timelines.md#builder-與-schedule-的職責)。

## 問題

Two-qubit 實驗要排除初始化不在指定態的 shot 時，只能改用另一份 `_heralded` 實驗。9 個 two_qubit 實驗各有一份 283–557 行的 `core_heralded.py`。一般實驗不能在原本的實驗裡選用 heralding。

當時的 reset 模塊只播脈衝，取數層不知道有 shot 要排除。平均、shot-sum processor、tracker 與 `run_acquire_shots` 各自處理資料，沒有共同的方式表達被排除的 shot。

使用者 2026-11-19 提出方向：「H我自己的想法是，如果是reset模塊，那就放棄可以取得多餘的資訊，比如說把分類非g的shot值都設定成0來使其對平均沒有貢獻。對於想要利用更多資訊的實驗，保持分化是合理的。」對取數機制，使用者說：「4這就是我最關注的，我考慮的是給program加上acquire的註冊處理shot機制，有別於顯式的summed，具體設計還要考慮。」

## 考慮過的選項與比較

### Herald 放在哪裡

| 選項 | 內容 | 代價 |
| --- | --- | --- |
| H1 herald reset 模塊（採用） | 新增 reset 模塊，實驗沿用 `reset` 欄位選用，取數層篩 shot | 模塊只留通過的 shot，放棄 herald 的類別資訊 |
| H2 只在 9 個 `_heralded` 內部改用模塊 | 模塊只服務現有 heralded 實驗 | 每個實驗仍有兩個版本，不符合目標 |
| H3 H1 落地後一次遷移全部 9 個 | 刪除 `_heralded` 版本 | 失去 81 種 herald×終端組合的條件分析，除非每個家族另留診斷路徑 |

使用者的判斷是 reset 模塊放棄多餘資訊，需要更多資訊的實驗保持分化，即 H1 加上保留 `_heralded`。2026-11-20，使用者對整批建議回答：「好，我沒問題了，全部採用建議」。

### Herald cfg

使用者先問：「也許herald cfg可以設定要留哪種。」（2026-11-19）看過 cfg 範例後，使用者問：「關於reset/herald，post_delay不是已經在readout裡就有了嗎？pre_reset是為什麼需要呢？……mux與一般的readout要用同一個module type還是不同的？我以為通常都是不同的。mux的module通常都是特化。」（2026-11-19）

- 去掉 herald 自己的 `post_delay`，herald 讀取沿用 readout 的 `pre_delay` 與 `post_delay`。
- 依 readout 分成兩個 type：`reset/herald` 配 `readout/pulse`，`keep` 是一個態；`reset/mux_herald` 配 `readout/mux`，`keep` 是 tone 位置到態的對應。兩個 type 各自固定 `keep` 的形狀，不需要條件式驗證。
- `readout` 必須和終端讀取相同，herald 與終端共用同一份 readout 宣告。省略 `readout`、自動沿用終端讀取，會讓 reset 模塊從實驗拿 readout，所有 reset 介面多一個參數。

第一版的 herald 前準備叫 `pre_reset`，只收脈衝類 reset。使用者看過預期程式範例後指出：「pre_reset應該可以不只是reset type，普通的qubit pulse或onetone pulse應該都可以達到穩定ge分佈的效果。」（2026-11-20）名稱與個數有三個選項：A `prepare` 可放多個；B `pre_reset` 放一個；C `prepare` 放一個。使用者回答：「C，目前沒想到需要多個prepare的場景，那就先簡單做法。」（2026-11-20）欄位也收一般 pulse 之後，`pre_reset` 這個名稱會誤導。

### Mux `keep` 的鍵

`keep` 原定寫成 `{0: g, 2: g}`。Library 為了和 JSON 相容，mapping 的鍵只收字串，在模塊驗證前就拒絕未加引號的 `0`。

| 選項 | 寫法 | 代價 |
| --- | --- | --- |
| 1 鍵加引號（採用） | `{'0': g, '2': g}`，模塊驗證時轉成整數位置 | YAML 要寫引號 |
| 2 和 tones 對齊的 list | `[g, null, g]` | tone 多時要數位置 |
| 3 storage 邊界轉換鍵 | 只對這個欄位把整數鍵轉成字串 | 通用 storage 要認得一個模塊欄位的特例 |

使用者回答「2-1」（2026-11-20）。

### 權重與自動過濾

使用者問：「你覺得用權重實現比較好，還是filter自動過濾shot點數比較好？run_acquire_shots不能多回傳權重或是Mask，或是過濾夠的點嗎？」（2026-11-19）agent 的第一個建議是混合做法：processor 逐點只收保留的 shot，平均與 tracker 用權重。另一案是所有下游統一收權重。

使用者選統一：「我傾向統一的處理方式，統一知道權重的存在，而也代表權重的Mixin要在比較底層，作為通用底層功能存在，而不屬於Module層級。」（2026-11-19）混合做法讓兩種處理並存。

把被排除的 shot 值記 0 再交給下游也不可行。Tracker 的矩與分群、shot 直方圖都會在 IQ 原點多一團。

### 遮罩的型別、來源個數與形狀

第一版是 float64 權重，一個 program 最多一個來源。使用者看過程式範例後提出：「2我認為可以支援多次register_shot_weights，照順序執行，並合併mask，考慮這這個機制，我傾向mask先只有0/1就好，用int?目前沒想出用float的需求，看你的想法。」（2026-11-20）

改成 bool 遮罩，多個來源照註冊順序以 AND 合併。Bool 是 agent 依「看你的想法」做的選擇：值只有兩種，不會出現 2 之類的非法值；合併直接用 `&`；乘上 IQ 時自動轉型。Float64 沒有需求。Int 0／1 的值域比 bool 寬，需要另外驗證。

使用者接著擔心形狀：「關於mask，我擔心的是mux與多個ro ch的多次readout，據我所知，不同ro ch儲存欄位是分開的，那mask的形狀就需要討論一下。」（2026-11-20）每個 ro ch 的 raw 是 `(points, reps, reads_ch, 2)`，只有 reads_ch 各 ch 不同。Points 與 reps 是 program 迴圈，所有 ch 共用。Mux 每個 tone 各有一個 ro ch。

| 選項 | 內容 | 代價 |
| --- | --- | --- |
| M1 以 shot 為單位、所有 ch 共用（採用） | `(points, reps)`，對每個 ch 廣播 | Mux 同時 herald 多顆獨立 qubit 時，接受率是各顆相乘 |
| M2 每個 ro ch 一個遮罩 | Herald 可選聯合或獨立 | 每個 consumer 都要決定用哪幾個遮罩；沒有 mux 獨立 herald 的實驗 |
| M3 介面每 ch 一個遮罩，herald 只產生聯合遮罩 | 為 M2 預留介面 | 沒有現存使用者，所有 consumer 現在就要處理 dict |

使用者回答「M1」（2026-11-20）。

### 平均公式

被排除的 shot 權重為 0，每點平均是 `Σ m·x / N`，N 是 reps 乘以已完成的 rounds。使用者 2026-11-19 說：「比如說把分類非g的shot值都設定成0來使其對平均沒有貢獻。」「由於掃flux本來就會正規化，所以都沒差……2我認為先不用存，但平均就能分析出來。」「平均公式可以調查一下設計張力，沒問題就採用。」

調查找到一個張力：在原始積分上記 0，扣 IQ offset 後被排除的 shot 會變成 −offset。所以 x 是逐 shot 長度正規化、扣 offset 之後的值，權重在這之後才套用。除以保留數會改變幅度的意義，而且使用者說接受率先不存，要能從平均分析出來。

### 分類計數的完整性（S4）

有 herald 時，每點的計數總和小於 reps×rounds。當時的 two-qubit 分析要求計數總和等於 reps×rounds，會把 post-selected 的點判為不完整。

| 選項 | 機率的分母 | 影響 |
| --- | --- | --- |
| S4a（採用） | 計數總和，即通過 herald 的 shot 裡的條件機率 | 點的完整性改成「每一輪都跑完」；接受率等於計數總和除以 reps×rounds |
| S4b | reps×rounds | 機率含接受率，現有分析的分母都要改 |

使用者回答「5a」（2026-11-19），選了這題的選項 a，即 S4a。例：RB 的 joint00 存活率是 herald 兩個 tone 都是 g 時的存活率。

拆票時再問一次範圍：A 只改 plain `two_qubit/rb` 的分析，其他 plain two-qubit 實驗接上 herald 時再改；B 依 contracts 原文一起改 5 個 plain 分析。使用者回答「Jiki 基本没问题，但我有发现一个设计上的取舍。」（2026-11-20）。「Jiki」是語音輸入，讀作「幾題」，表示三題都接受建議，採 A。那 5 個實驗收不到遮罩，現在改不會產生任何行為差異。

### Shots 的遮罩與補足

`run_acquire_shots` 回傳 shot 本身，被排除的 shot 不能像加總那樣以 0 透明處理。

| 選項 | 代價 |
| --- | --- |
| 回傳遮罩（採用） | 被排除的 shot 留在陣列裡，由實驗決定怎麼用 |
| 每點補足 reps 個保留的 shot（agent 原建議） | 框架替實驗決定丟棄與重跑，取數時間不固定 |
| 刪除被排除的 shot | 每點數量不同，陣列不再是方的 |
| 被排除的 shot 記 0 或 NaN | 直方圖、classifier 擬合或 `np.mean` 會出錯 |

使用者：「run_acquire_shots我傾向遮罩，應該說，怎麼使用遮罩應該是各實驗的自由。」（2026-11-19）實作的回傳型別是 `ShotResult`，含可見讀取的原始積分 IQ 與遮罩。

### Herald 讀取的隱藏

使用者要求：「3希望能設計的乾淨些，實作前請與我討論具體細節。」（2026-11-19）細節寫進 spec 的 contracts，使用者確認：「herald spec 的測試接縫和 contracts 我確認」（2026-11-20）。

- 硬體觸發次數照實計算，包含 herald 讀取。
- Readout label 與讀取索引在不含 herald 的邏輯讀取上解析，實驗的 converter、`shot_sums` 的 label 與讀取索引不改。
- Herald 讀取只交給遮罩來源。另一個做法是實驗自己跳過 herald 讀取，代價是每個實驗都要改讀取索引。

實作遮罩時遇到 contracts 沒寫的兩個邊界：

1. 某個 ADC 的讀取全被遮罩來源消耗時，平均、reducer 與 shots 的輸出都省略這個 ADC。保留零讀取的 ADC 會讓 converter 拿到的 ADC 欄數改變，單 ADC 的預設 converter 會報錯。
2. Program 有遮罩來源時，`run_acquire` 收到 QICK 原生 `threshold` 在進入點報錯。在 host 上逐 shot 套用 threshold 沒有需求，當時能選 herald 的 RB 也不走這條路徑。

使用者回答「B, 我傾向一次就設計完整。23可以」（2026-11-20），其中「2」同意這兩條寫進 contracts。

### 各下游的透明

草案曾列出 `run_acquire_shots`、tracker、IQ 二階矩與自訂 reducer 不支援遮罩。使用者問：「6為什麼不支援run_acquire_shots？為什麼不支援tracker？作為一個獨立於processor的機制，我希望能保持彼此透明，如果不能，我希望瞭解原因。」（2026-11-19）

所有下游統一收遮罩後，這些都能支援。只剩兩項不支援：

- Decimated 取數：讀到的是時間軌跡，沒有逐 shot 的積分值可分類。
- Herald readout 參數有掃描（頻率、gain、長度等）：classifier 只在一組設定下校準過。

### Tracker 與 processor 的定位重複

討論各下游怎麼收遮罩時，使用者指出 tracker 與 processor 重複。以下是 2026-11-20 的語音輸入原文；「城市」讀作「程式」，「stage」讀作 state，「显示」讀作「顯式」：

- 「Jiki 基本没问题，但我有发现一个设计上的取舍。我发现有 processor 机制之后，我好像不需要 tracker 机制，后者是不是可以删掉了？」
- 「但这两个东西的定位重复了，我不是很想要重复定位的东西出现在城市里，这对于使用者跟维护者来说都是负担。无单调缺可，就必须放弃。一些不可加的统计能力，比如说 K-means，K-means，所以这是我烦恼的。」
- 「那我觉得还是用比较显示的方法比较好，也就是一个 reducer，然后再一个 stage，然后者是状态，前者是累加器，这个抽象会比较合理一点，比较自然一点。」

當時用 tracker 的有 9 處：jpa 4 個、`ro_optimize` 4 個、`calibration_acquire`。它們用 tracker 的一、二、三階矩算 SNR 與 skew penalty。

| 做法 | 內容 | 結果 |
| --- | --- | --- |
| T1 `IQ` 加 `order=3`（採用） | `shot_sums` 多存 `I³, I²Q, IQ², Q³`，共十個分量；SNR 與 skew penalty 從總和算 | `IQ(read)` 預設仍是二階六分量，既有輸出不變 |
| T2 另寫 `moment_sums` processor | 一個專用 reducer | 當時模擬器統計抽樣的提案只涵蓋 `shot_sums`，T2 只能逐 shot 產生 IQ；提案後來撤回 |
| T3 保留 tracker | 不改 | 與 processor 定位重複 |
| 刪 processor，統一用 tracker | 反向收斂 | 要改約 50 處並推翻 ADR-0074 的兩條裁決；tracker 還得補聯合 ADC、結果回傳、取消語意與模擬器支援 |

刪掉 tracker 後，KMeans 這類不可相加的統計需要另一個表達方式：

| 做法 | 代價 |
| --- | --- |
| 顯式的 reducer 加 state（採用） | 介面多一個 state 型別參數 |
| 讓回傳型別自訂 `+` | 每輪看不到前面累積的 state |

採用的介面是 `ShotReducer[State]`：`start() -> State`、`add(state, shots, mask) -> State`、`output(state) -> (points, *axes)`。框架持有 state，每完成一輪呼叫一次 `add`，最終結果與 round hook 都用 `output(state)`。原本「全部輪次一起加總」與「給 round hook 的增量加總」兩條累加路徑合成這一條。

時機有兩個選項：在本任務併入遮罩的 reducer 改動，或延後。遮罩已經要改 reducer 的簽名，延後會讓同一個介面改兩次。關鍵字保留 `processor=` 時，名稱和型別 `ShotReducer` 對不上。對這兩項建議，使用者回答「采用建议」（2026-11-20）。

### 模擬器的統計抽樣：提出後撤回

使用者 2026-11-19 要求：「7不要逐shot模擬，太慢了，請使用一些統計手段來等效生成」。提案的做法如下：

- FTF 照舊按 projective outcome 分支，每個分支演化一次。
- 每點、每個分支 c，用模擬器已算好的無雜訊 IQ μ_c 與接收雜訊 covariance Σ。分支計數 n_c 抽 multinomial。
- Herald 的保留機率 p_c 是高斯落在保留區的機率，保留數 k_c 抽 Binomial(n_c, p_c)。
- 終端 IQ 加總是 k_c·μ_c 加上 variance 為 k_c·Σ 的常態雜訊。Classify 計數再抽 multinomial。二階、三階分量以 CLT 近似。
- 沒有遮罩來源的 `shot_sums` 也走這條路徑（p_c = 1），不維護兩套。其他 reducer 與 `run_acquire_shots` 改用抽樣的 IQ shot。
- 成本與 reps 無關。

實作這條路徑時量到兩組數字。量測程式與原始紀錄未入庫。

**讀取中的躍遷。** 取 `two_qubit/rb` 的終端讀取：depth 0、seed 71、無 reset、兩個 arm，保留 24 個 DC 態。讀取窗口從 emission 起點到最後一個接收樣本約 1688.9 ns。至少躍遷一次的機率以 `-expm1(-Σ outgoing rates × span)` 計算。每個分支、每種模型各抽 32768 shot。Circle classifier 用各 tone 的 g/e 條件中心，radius 是兩中心距離的一半。

| F1/C/F2 分支 | 至少躍遷一次 | 逐 shot 誤分類（ADC 2／3） | 高斯誤分類 |
| --- | ---: | --- | --- |
| 000 | 21.31% | 0.168%／0.201% | 0／32768 |
| 001 | 21.74% | 0.137%／0.299% | 0／32768 |
| 100 | 21.51% | 0.250%／0.244% | 0／32768 |
| 101 | 21.94% | 0.238%／0.354% | 0／32768 |

這四個 computational 分支佔該輪 1000 shot 中的 992 個。24 個分支的躍遷機率範圍是 0.96%–21.94%。逐 shot 模型的 16 個 I/Q 標準化三階矩絕對值在 0.58–46.99 之間，其中 14 個大於 7。高斯模型的 16 個值都在 0.041 以內。這組數字來自固定 seed 的 Monte Carlo，不是物理校準。

**逐 shot 路徑的耗時。** 取同一個 `two_qubit/rb`：depth 0、單點、兩個 arm、一個 host round。每種 reps 在獨立 process 跑四次，量 `run_stage` 的 wall time，包含 program 建構、取數與 Agg render。

| reps | 第一次 | 暖機後三次的中位數 |
| ---: | ---: | ---: |
| 1,000 | 4.69 s | 0.064 s |
| 100,000 | 4.91 s | 0.195 s |

第一次包含按需編譯與準備。量子演化本來就按 outcome 分支算一次，每個 shot 只做 IQ 抽樣。Heralded 實驗慢在演化，約每點 12 ms。

| 做法 | 結果 |
| --- | --- |
| A 每個分支一個高斯 | 三階矩與誤判率都和逐 shot 差很多，readout 優化的 skew penalty 會失效 |
| B 只考慮躍遷一次的混合模型 | 忽略約 2.5% 躍遷兩次以上的 shot，誤差沒有上界；補上兩次躍遷後實作與驗證更重，省下的只是零點幾秒 |
| C 讀取中躍遷機率高於門檻的點自動改走逐 shot | 門檻 1e-4 時，FTF 預設參數下每個點都走逐 shot，沒有加速 |
| E 不做統計快路徑（採用） | 沿用逐 shot 的 IQ 抽樣，herald 與終端讀取共用同一發的 outcome 歷史 |

使用者先回答「1BC我猶豫選哪個」。看過上面的數字與 E 之後，使用者回答：「shot抽樣就e吧」（2026-11-20）。使用者擔心的逐 shot 演化本來就沒有發生。統計抽樣的實作撤掉，FTF 上的 herald 改由 plain RB 的離線 probe 驗收。

### 先例與立場

- ADR-0074 的 2026-10-29「上層使用新實驗組，比如heralding/*」：有拘束力，由使用者推翻，見上方〈Herald 放在哪裡〉。同句的「讓底層支援post-selection」與不做 active reset：遵循。
- ADR-0074 的 2026-10-30「不要再有逐shot記錄的實驗了」：遵循。Herald 不增加逐 shot 存檔，RB 仍只存計數。
- ADR-0074 的 2026-11-02「先固定2階，3階有需要再考慮支援」：遵循。SNR 需要三階，所以 caller 明填 `order=3`，預設仍是二階。
- ADR-0074 的 classifier 隨 readout 保存：遵循。Herald 用 readout cfg 的 classifier，不另存。
- ADR-0083 的 builder 只記錄序列、取數由 Schedule 提供：遵循。Builder 只記錄遮罩來源，遮罩由 program 的取數層呼叫。
- [2026-10-16 FTF 模擬加速](2026-10-16-ftf-simulation-speedup.md)改成每點演化一次、再以統計方式產生 shot 雜訊：區別。那次的成本來自同一份演化重複 reps×rounds 次；這次演化已經每分支一次，剩下的逐 shot IQ 抽樣在 100000 reps 約 0.2 s。
- ADR-0075 的小功能原則（同一個問題多次出現才加）：遵循。第一版只接 plain `two_qubit/rb`，`prepare` 只放一個。

## 被否決的選項與理由

- H2 只在 `_heralded` 內部用模塊：每個實驗仍有兩個版本。
- H3 一次遷移 9 個 `_heralded`：失去 81 類條件分析。
- Processor 自動過濾、平均與 tracker 用權重：兩種處理並存，使用者要統一。
- 被排除的 shot 記 0 再交給下游：矩、分群與直方圖會在 IQ 原點多一團。
- Float64 權重：沒有需求。Int 0／1：值域比 bool 寬。一個 program 最多一個來源：使用者要多個。
- M2 每個 ro ch 一個遮罩、M3 預留每 ch 介面：見上表。
- 在原始積分上記 0：扣 offset 後變成 −offset。除以保留數：改變幅度的意義。
- S4b 除以 reps×rounds：現有分析的分母都要改。
- 一起改 5 個 plain two-qubit 分析：它們收不到遮罩，現在改沒有行為差異。
- Shots 補足、刪除被排除的 shot、記 0 或 NaN：見上表。
- 實驗自己跳過 herald 讀取：每個實驗都要改讀取索引。
- 保留零讀取的 ADC：單 ADC 的預設 converter 會報錯。在 host 上逐 shot 套用 threshold：沒有需求。
- 一個 herald type、`keep` 形狀隨 readout 變：需要條件式驗證。Herald 自己的 `post_delay`：與 readout 重複。名稱 `pre_reset`：欄位也收一般 pulse。`prepare` 放多個：沒有需要多個的場景。省略 `readout`：所有 reset 介面多一個參數。
- Mux `keep` 用 list 或 storage 轉換：見上表。
- T2 `moment_sums`、T3 保留 tracker、刪 processor 改用 tracker：見上表。
- 回傳型別自訂 `+`：每輪看不到前面累積的 state。延後 reducer 改動：同一個介面改兩次。保留 `processor=`：名稱和型別對不上。
- 統計抽樣的 A、B、C：見上表。原提案也否決過「只讓帶遮罩來源的路徑走統計、無來源的 `shot_sums` 照舊」，理由是兩條路徑並存；整個提案後來撤回。

## 留下的問題

- Mux 同時 herald 多顆獨立 qubit 時，M1 的接受率是各顆相乘。需要各自 herald 時，要改成每個 ro ch 一個遮罩，並改 reducer 的簽名。
- 其他 plain two-qubit 實驗與 CZ 脈衝優化實驗接上 herald 時，各自的分析要改成除以計數總和，完整性改成每一輪都跑完。
- 接受率沒有存檔欄位，由計數總和除以 reps×rounds 得出。
- `prepare` 只放一個；需要多個時改成清單。
- Active reset 與 FPGA feedback 不在範圍內。
- FTF 上的 herald 只有離線 probe 驗收，沒有實機資料。

## 依據

- 裁決時的程式與文件：repo@95064aa80 的 `lib/zcu_tools/program/acquisition.py`、`zcu_lab/v2/two_qubit/*/core_heralded.py`、[ADR-0074](../adr/0074-flat-readout-module.md) 與 [ADR-0083](../adr/0083-program-macros-and-timelines.md)。
- 使用者原話取自 2026-11-19 至 2026-11-20 的問答。決策檔是未入庫的工作檔，原話與理由已寫進正文與 ADR-0074。
- 模擬路徑的兩組量測見上方〈模擬器的統計抽樣〉。原始量測程式與紀錄未入庫。
- Plain RB 的離線驗收（原始紀錄未入庫）：
  - 使用本機 FTF 環境檔（未入庫），驗收前後檔案的 SHA256 相同。Model seed 123、RB seed 5、256 reps × 2 rounds。
  - 沒有 herald 的 plain RB 與 `rb_heralded` 的計數陣列，都和改動前逐值相同。
  - `reset/mux_herald` 保留 F1、F2 兩個 tone 都是 g 的 shot。Result 是 float64 `(2, 1, 2, 2, 3, 3)`，和沒有 herald 時相同。每格計數等於改動前 `rb_heralded` 對同一批 shot 存的 gg herald 切片。
  - 兩個 depth、兩個 arm 的接受率在 58.40%–63.09%，joint00 存活率在 75.85%–78.60%。
  - 兩組 circle 是獨立 idle read 的合成分類區域，不是 GE 校準。這組結果不驗證實機或 gate fidelity。

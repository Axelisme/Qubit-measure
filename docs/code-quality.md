# 程式碼品質

**讀者：** lab-author、framework-dev、agent。查閱實作及 review 的品質判準。

通過機械檢查是必要證據，不代表設計容易理解。
操作、測試與驗證流程以 [AGENTS.md](../AGENTS.md) 為準；工具使用方式見
[tools/README.md](../tools/README.md)。本文件不授權擴大任務範圍或改變既有架構。

## 使用方式

1. **實作前**：確認本次行為、既有規則的擁有者，以及呼叫者必須知道的契約。
   完成條件：能指出修改應落在哪個 module；責任不明或需要架構決策時，先請使用者決定。
2. **交付前**：逐項檢視下列品質判準。新增抽象或改變 interface 時，說明它減少了呼叫者的哪些負擔。
   完成條件：適用項目有程式碼或驗證證據支持；未解決項目有明確理由與決策者。
3. **Review 時**：沿一條受影響的主要流程與相關失敗路徑閱讀，再依「審查裁決」分類意見。
   完成條件：每項阻擋意見都有具體位置、違反的契約或品質判準、影響，以及可驗證的改善方向。

## 品質判準

### 讓主要流程可以直接閱讀

命名沿用 repo 的 domain language，讓讀者知道資料代表什麼，而不只知道它的容器型別。
同一概念使用同一名稱；單位、座標系或資料形狀有歧義時，在名稱、型別或契約中明示。

函式內維持一致的敘述層次。抽出的 helper 應命名一段有意義的工作，或集中需要獨立理解的細節。
長度與複雜度警告是重新檢視的起點；拆分後應更容易追蹤資料與控制流程。

Reviewer 若需作者額外解釋才能讀懂，將必要資訊放回命名、結構或註解。
註解記錄不直觀的原因、限制與取捨；interface 文件則說明呼叫者必須知道的行為。
Review 對話中的解釋不能代替這些資訊。

### 抽象服務目前需求

Module 是具有 interface 與 implementation 的程式單位。Interface 包含呼叫者必須知道的
型別、前置條件、呼叫順序、副作用與失敗語意，不只是函式簽名。

新增 module、Protocol、factory 或 registry 時，指出目前存在的變化需求，或它集中隱藏的複雜度。
若刪除一層只需改名、且沒有把責任散回呼叫者，優先使用直接呼叫。
單純轉接也可能合理，例如連接 Qt signal 與既有操作；保留理由是契約轉換，而不是層數本身。

不要為了降低檢查數字而搬移複雜度。拆檔、拆函式或引入參數物件後，確認讀者需要理解的概念
與呼叫順序確實減少，而非只讓單一檔案變小。

### 業務規則有單一權威來源

驗證條件、狀態轉移與錯誤分類由負責該行為的 module 定義。其他呼叫者重用它，或只做表示轉換。
判斷重複時看「是否表達同一項規則」，不只比對文字；長得相似但原因不同的邏輯可以分開。

例如，GUI 與 Notebook 若接受同一種量測設定，應共用該設定的有效性判斷；GUI 可以另外處理
輸入欄位與錯誤顯示。不要把兩套判斷各自維護成一致，或只為消除相似文字而綁住不同實驗。

### 在適當位置建立資料與錯誤契約

外部資料進入時完成必要驗證，內部依已建立的契約使用明確型別。
反射、無結構 dict 與第三方無型別資料若不可避免，將轉換集中在對應 adapter，避免每個呼叫者
重複猜測欄位。`cast` 說明的是型別主張，不提供執行期驗證。

必要的輸入與業務驗證使用明確分支與例外；它們必須在 Python 移除 `assert` 時仍然成立。
縮小 `try` 到預期會失敗的操作。捕捉例外的地方應有恢復、轉譯或隔離責任；無法處理則傳遞。
轉譯例外時保留診斷原因，或明確處理需要隱藏的細節。

廣泛捕捉可以用於 GUI 任務等隔離點，但必須保留失敗狀態與可追查資訊。
只有當契約本來就把失敗定義為可忽略時，才回傳預設值；量測失敗不能偽裝成成功的空資料。

Interface 有下列限制時，在其文件集中說明，不要求每個函式填寫相同模板：

- 單位、資料形狀、有效範圍，以及會改變輸入的操作。
- 執行緒要求、呼叫順序、資源所有權與釋放責任。
- 失敗、取消、部分完成的結果，以及呼叫者需要區分的例外。

### GUI 回呼與拆除

GUI 元件對 owner 的回呼、訂閱、timer、eventFilter 與 matplotlib 連線一律經
`gui.lifetime.LifetimeScope` 登記。Qt signal 使用 `scope.connect`，其他回呼使用
`scope.callback`，eventFilter 使用 `scope.install_event_filter`，退訂與資源釋放使用
`scope.add_cleanup`。可替換的 editor 或 row
使用自己的 scope，替換前先關閉。

拆除入口只呼叫 `scope.close()`，不在各個 callback 散放 deleted／RuntimeError 防禦。
Scope 先讓 callback 失效，再逆序釋放。清理失敗會拋出並保留未完成項供重試。
成功返回前不可刪除 children。Owner 的業務關閉許可必須先取得，被拒時不關 scope。
Qt parent 直接刪除 C++ child 的後備檢查集中在 scope，不取代顯式 close。
不把捕捉 owner 的 closure 接到 `destroyed` signal。

### 既有容忍項先嘗試修除，理由留在原地

新程式碼不新增 `cast` 或 suppression，不放寬型別配合錯誤。以下例外只適用於已明確授權清洗的既有項目，不能推廣到新程式碼。

既有未處置的 Ruff／Pyright 診斷先逐位置嘗試修除，確定保留後才加指定實際 rule 的 suppression：

- Ruff 用 `# noqa: PLR0913  # tolerated: <reason>`。
- Pyright 用 `# pyright: ignore[reportPrivateUsage]  # tolerated: <reason>`。

Suppression 留在診斷行。理由寫同行，放不下時放正上方的 `# tolerated: <reason>`。
禁止無 rule 的 noqa、type ignore 或 pyright ignore。理由說明該位置保留的具體限制，不能只寫「既有問題」。
既有 suppression 按 noqa、type ignore、pyright ignore 分組抽樣試清。整組必要時可整組保留並改成上述格式，有可清者再逐條展開。

既有 `cast` 與設定項也先嘗試清除。保留時分別用 `# tolerated(cast): <reason>` 與
`# tolerated(config): <reason>`，放原行行尾或正上方。保留設定值不變，不新增 `cast`，不新增或放寬設定。
File-size／radon／capability 保留項用 `# tolerated(<metric>): <reason>`，放相關行或檔案開頭。
理由只寫原地，不交容忍名冊。Review 逐項確認嘗試證據、指定標記與理由。

Ruff 的 RUF100 與 Pyright 的 reportUnnecessaryTypeIgnoreComment 檢查過期 suppression。
過期項要移除或收窄，不能關閉檢查或新增 suppression 掩蓋。

### 測試驗證可觀察契約

在模組接縫（seam）與對外契約（contract）驗證框架的可觀察行為。Notebook helper 與共用分析工具的公開契約也在測試範圍內，可測性取決於行為契約。測試從公開介面呼叫並觀察結果，讓介面後方的私有函式與實作結構可以獨立調整。

`zcu_lab` 的 adapter、workflow 與 recipes 是實驗定義，以直接審閱確認。框架測試用最小合成 workflow／adapter 驗證註冊、執行與結果交接，測試依賴限於框架與合成定義。Review 確認 tests 的 import 與此邊界一致，必要時以 grep 輔助。Pytest 在正常目錄布局下執行，維持 `zcu_lab` 目錄原位。測試的實驗輸入固定來自合成定義，`ZCU_LAB` 環境變數與 import hook 維持既有狀態。只因實驗定義改動而失敗的測試應刪除，讓實驗預設值由實驗定義自行維護。

程式入口採 humble shell，將行為放在可 import 的函式中。腳本行為測試在同一個 process 呼叫這些函式或 `main(argv)`。執行外部工具，或驗證 exit status、載入、跨 process lock 等 process 邊界時，才啟動獨立程序。測試平行化由測試 CLI 安排。

Notebook 的主要探索流程、文件措辭、設定值、靜態資料及其定位以直接審閱確認。測試範圍維持上述行為契約，既有案例也依此判準檢視。移除邏輯時，同步移除只服務該邏輯的測試。

依 [AGENTS.md](../AGENTS.md) 的測試契約選驗證方式；新增或搬遷案例時依
[tests/README.md](../tests/README.md) 找 owner、fixture 與搬遷驗收。Review 額外回答：
如果本次行為被拿掉或寫錯，哪個測試會失敗？斷言應能區分正確結果與相關的錯誤結果。

替身只替換驗證範圍外的協作者，被測邏輯仍經由公開 interface 執行。只有呼叫順序本身屬於契約時，才把順序當作驗收條件。

Production 可以提供中性的注入接口，即使目前只有測試替換它。中性接口同時符合四點：

- 以模組自己的概念表達，例如時鐘、對話框呈現或儀器驅動建立，內部結構維持封裝。
- 預設值就是 production 行為。
- 替換它只改變外部效果，被測邏輯照常執行。
- 型別與 docstring 說明該傳什麼。

協作者已有公開注入點時，測試經由該介面替換協作者。Qt modal dialog、時鐘、環境變數、
`subprocess`、網路等外部邊界缺少既有注入點時，可用 pytest `monkeypatch` 替換。
`monkeypatch.setenv`、`monkeypatch.chdir` 這類環境隔離一律可用。
被測模組自己的函式與私有 helper 保持原樣，私有狀態只經公開介面改變及觀察。
若案例需要 patch 目標函式或透過 `vars(obj)` 讀寫私有狀態，將驗證點移到公開介面。

例如，驗證量測取消時，斷言約定的終止狀態與資源釋放。測試是否有效取決於斷言能否識別相關錯誤，
測試行數、mock 數量、coverage 與 `assert` 數量只提供輔助資訊。

宿主的最高測試接縫是註冊介面。合成 workflow 經 caller 註冊清單交給宿主，再從 Window、RPC 或 MCP 觀察結果。
共用 registry、reload、分析與匯出機制由各框架 owner 使用合成定義驗證。Lab 實驗錯誤由 GUI 啟動或載入時直接回報。
這個邊界與使用者裁決見 [模組定位 ADR](adr/0072-module-boundaries.md#分層)。

### 讓變更可以獨立審查

一次變更聚焦一個目的，相關測試與實作一起交付。與目的無關的搬檔、改名與格式整理另外處理。
必要的小重構可以與功能一起提交，但要分清哪些是行為變更，哪些應保持行為不變。

檔案數與行數只提示 review 成本，不作通用硬門檻。若 reviewer 無法分辨變更的責任與驗證範圍，
按可獨立理解、驗證的目的拆分，而不是按行數切割。

## 審查裁決

| 分類 | 判準 | 處理方式 |
| --- | --- | --- |
| 必須修改 | 違反契約或本文件判準，且能指出具體的正確性、理解或維護成本 | 修正，或由適用的決策者確認例外後再接受 |
| 建議改善 | 有可說明的收益，但不是本次接受的必要條件 | 明示非阻擋；依任務範圍決定是否處理 |
| 個人偏好 | 沒有既定規範或具體影響支持的替代寫法 | 不阻擋接受 |

以本次變更是否改善或維持品質裁決，不要求順手清完歷史問題。

2026-10-24 使用者裁決：「請特別注意使用方的語義漏進工具中或是框架中，get_single_sweep_name就是很好的例子」。
lib 與 tools 不得假設 zcu_lab 的欄位名、段落名、實驗名或 cfg 形狀。
框架只依型別註解、FieldMeta 與 adapter 交來的值運作，不寫死尋找 sweep、modules 或 readout。
審查 lib 改動時，先 grep diff 中的字串常數，再逐項確認它來自公開契約而非實驗側名字。

門檻型指標只處理新跨過門檻的項目，包括原本未超標、改後超標，以及新增時就超標的函式或檔案。
這些指標包括參數、分支、statement 數、C901／radon 複雜度與檔案行數。
原本已超標、改後仍超標的項目不算退步，可以照常修改；不要求 statement／body 的 bytes 或 hash 稽核，
也不要求順手拆分。新超標依本文件判準拆出內聚職責，不為降低數字硬拆或加 suppression。
此判準可能讓既有超標項目繼續變複雜，這類債由專門的品質改善工作處理。
型別錯誤、import 規則等非門檻診斷不因此放寬，新增問題仍須修正。機械 gate 的 exit 如實回報，不改設定規避。

例外須記錄適用範圍、理由、風險與重新檢視條件，並由任務指定的決策者確認；未指定時請使用者決定。
持續適用的例外放在擁有該規則的文件，本次限定的例外放在任務決策紀錄。
例外不自動豁免機械 gate，也不授權修改其設定。

## zcu_lab 審查裁決

「我會希望zcu_lab中的內容有最大化的可讀性，上手時間最短，實作內容一看就懂」

這是 zcu_lab 的首要準則。在 zcu_lab 內，可讀性優先於 lib 的強型別、防禦分支與抽象規則。
衝突時選一眼就懂的寫法，型別與驗證由 lib 在邊界承擔。不要在實驗定義中重做框架的檢查。
lib 本身仍遵守前述品質判準。

2026-10-25 使用者裁決：「迴圈內每一次的執行應該直接放在主迴圈，只有每一個實驗的量測與分析可以放在 helper，因為耦合比較高的是 workflow 內部的流程，還有每個實驗內部的邏輯。」
使用者也指出：「workflow 超過門檻可能是沒辦法避免的…能包成 helper 的就包，不能的話還是保留在原函式」。

Lab workflow 主函式，亦即逐點依序呼叫各 stage 的流程，可以超過 Ruff 複雜度門檻
C901、PLR0912、PLR0915 與 radon。實驗的量測與分析抽成 stage helper；
流程步驟留在主函式，不為了門檻另拆。
同一校準目的的多個實驗可以構成複合 stage；它的函式體依序呼叫單實驗 helper，回傳具名校準結果，由主迴圈填 point record。複合 stage 不用 out-parameter，也不包整個 flux 點的校準與 lifetime 流程。
在函式定義行用 `# noqa: C901, PLR0912, PLR0915  # tolerated: <reason>` 標記，
radon 用 `# tolerated(radon): <reason>` 記錄流程必須放在一起的理由。
此例外只涵蓋 workflow 主函式，不涵蓋實驗 stage，也不放寬型別或 import 規則。
主體增長可能讓步驟更難追蹤；若無法直接讀出量測順序，重新檢視 stage 的實驗邊界，
不把逐點流程藏進另一個 helper。機械 gate 的 regression 與 exit 仍照實回報。

「加入自己的客製化修改也很自然」

lab-author 改一段脈衝、多一個欄位或換分析步驟，應像改腳本，不必先理解基底或 helper 契約。
lib 基底只提供可選的預設路徑，recipe 可以拆開使用，也可以直接插入或替換自己的步驟。
不以強制的 `run_all` 或模板方法流程限制實驗。

每個 cfg／表單／recipe 欄位要能用一句使用者術語說出它在這個實驗的用途，說不出就是殘留，review 時逐欄位問。

Review 以「新人多久能看懂這個 recipe／core」裁決。請 reviewer 從 recipe 函式說出量測順序，
或從 core 的 `run` 說出脈衝序列，再指出客製化改動要寫在哪裡。需要追進 helper 才能回答時，
把量測動作放回主體。以下六項用來找出閱讀障礙，不以 class 數或型別完整性取代閱讀判斷：

1. Recipe 主體是一串量測步驟。只讀函式就能知道開了哪張表單、改了什麼、量測與分析的順序、
   何時問答、寫回及保存。主體直接 select／command／set，逐筆套 overrides 後 check_ready，
   lib 只轉送 Form 操作，不替 recipe 選來源。
2. Recipe 不宣告分析結果的鏡像 model。直接引用分析 owner 匯出的型別，不重宣告
   FitQuality、QualityIssue 或 summary 的同名欄位，也不在 recipe 重驗 owner 輸出。
3. Recipe 的 class 只用於實驗輸入與報告。不要為串接幾個步驟增加 runner、policy 或 pipeline class。
   這不禁止 core 宣告實驗需要的 cfg 與結果型別。
4. Recipe helper 只做數字呈現等機械格式化，不藏量測判斷。掃描哪裡、用哪個脈衝、
   何時重測或接受寫回，留在 recipe 主體。
5. Core 的 class 數與每個抽象都納入 review，逐項說明它代表的實驗概念或隱藏的複雜度。
   不能說明用途就改成直接寫法，但不為降低數字拆檔或搬走脈衝序列。
6. Docstring 用使用者術語交代輸入、動作、結果與失敗。寫明單位、省略保留與 null 原樣交 owner、
   `n → expts` 或 `points → expts` 對應及分析參數的效果，不以「typed pipeline」等實作詞代替說明。

撰寫方式與目標骨架見 [Measure authoring](../zcu_lab/v2/measure-authoring.md#recipe-撰寫)。
本節不要求順手清理所有既有 class，也不改變 [模組定位](adr/0072-module-boundaries.md#分層)
的依賴方向或硬體授權。

## 參考來源

以下是判準的參考，不是額外必讀清單；外部指南不自動成為 repo 規則。

- [Google：What to look for in a code review](https://google.github.io/eng-practices/review/reviewer/looking-for.html)，設計、複雜度、測試與註解。
- [Google：The standard of code review](https://google.github.io/eng-practices/review/reviewer/standard.html) 與 [review comments](https://google.github.io/eng-practices/review/reviewer/comments.html)，接受標準與意見分類。
- [Google：Small CLs](https://google.github.io/eng-practices/review/developer/small-cls.html)，變更聚焦與可審查性。
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)，例外、動態技巧、命名與 interface 文件。
- [PEP 8](https://peps.python.org/pep-0008/)，例外處理與公開／內部 interface。

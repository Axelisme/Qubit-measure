# zcu_lab 程式碼規則

**讀者：** lab-author、agent。逐條核對 zcu_lab 的程式碼規則、檢查方式與契約來源。

修改 `zcu_lab` 時逐條核對本頁。每條列出檢查方式與來源，細節以連結的文件為準。
新增 component、實驗與 recipe 的步驟見 [tutorial](tutorial/README.md)。
本頁不授權操作硬體，範例能載入不代表校準齊全或可以 Run。

## 1. 在 core 寫出序列與分析政策

`core.py` 直接列出 pulse、delay、readout 與分析選擇，helper 不用實驗名稱或 mode 替 caller 選政策。
檢查：只讀 `run()` 與 `analyze()`，能依序指出量測步驟、用哪些資料擬合，以及明填給 helper 的參數。
來源：[core 的序列與分析政策](v2/experiment-authoring.md#1-在-core-直接寫出-pulse-序列與分析政策)。

## 2. 在 recipe 寫出操作流程

Recipe 函式本體直接列出開 tab、設定、Run、Analyze 與 writeback 問答，不把整段流程藏進 helper。
檢查：從 generator 本體依序指出實際採用的步驟，確認接受 writeback 後才呼叫 `accept()`。
來源：[recipe 操作流程](v2/experiment-authoring.md#2-在-recipe-函式本體寫出操作流程)。

## 3. Docstring 先說量測與結果

公開 docstring 先說量什麼、序列怎麼做、回傳什麼，再交代單位、資料形狀與必要失敗條件。
檢查：不看函式本體也能區分 raw IQ 與後續 fit，生命週期細節以連結查詢，不在各實驗重貼。
來源：[docstring 的內容](v2/experiment-authoring.md#3-docstring-說明量測序列與結果)。

## 4. 用註解與空行分開階段

Core 與 recipe 用空行分開操作階段，英文註解說明做什麼與原因，不逐行重述賦值。
檢查：沿著函式本體能找到準備、序列、取資料或流程問答的分界，註解與實際順序一致。
來源：[階段註解](v2/experiment-authoring.md#4-用註解與空行分開階段)。

## 5. Core 不依賴 GUI

Core 不直接或間接 import GUI，GUI adapter 在同目錄的 `gui.py` 接上 core。
檢查：審閱新增 import 的依賴方向，程式碼品質檢查中的 import-linter C15 必須通過。
來源：[core 與前端的責任](v2/README.md#v2-experiments)。

## 6. 共用 mechanics 不接管 caller 的選擇

`_support/` 放共用 mechanics，實驗序列、分析選擇與 writeback 政策留在 caller，具名共用初值 helper 沿用既有例外。
檢查：核對 helper 的 callers 與參數，Form 仍明選倍率或場景，不讓 core 經 `_support/measure/` 載入 GUI。
來源：[support ownership 與初值例外](v2/gui-contracts.md#ownership)。

## 7. Guide 寫給 operator

`guide.md` 描述準備、操作與結果判讀，mock 與 real 都當黑盒，不用模擬器內部真值代替掃描與看圖。
檢查：從 guide 能指出必要校準、Run 前要補的輸入、結果判讀與 writeback 審閱步驟，不要求 operator 讀 Python 類別。
來源：[operator guide 約束](v2/measure-authoring.md#operator-workflow-guide-約束) 與 [guide 的載入責任](v2/measure-authoring.md#目錄佈局)。

## 8. Docstring 使用量測者的術語

Docstring 沿用量測與元件的名稱，遇到單位、座標或資料形狀歧義時明說，不用容器型別代替數值意義。
檢查：核對同一概念的名稱是否一致，讀者能知道 IQ、delay、gain 等值代表什麼，而不是只知道它們是 array 或 dict。
來源：[domain language 與介面文件](../docs/code-quality.md#讓主要流程可以直接閱讀)。

## 9. Component 欄位交代 description 與單位

每個新 component 欄位用 `Field(description=...)` 說明意義，數值沿用共用單位型別，Flux 由 caller 明示座標與單位。
檢查：逐欄審閱 model 的 description 與型別，確認 Frequency 用 MHz、Duration 用 us、Energy 用 GHz，沒有把 Flux 標成固定單位。
來源：[欄位宣告](tutorial/01-components.md#1-宣告欄位)、[component 欄位單位](README.md#component-欄位單位) 與 [field specs](../lib/zcu_tools/resources/entry/README.md#元件與定義)。

## 10. 實驗定義不寫個別 pytest

Adapter、workflow、recipe 與實驗共用 helper 不新增個別 pytest，框架測試用最小合成材料驗證公開契約。
檢查：審閱變更未為 lab 定義新增測試或逐一走訪 lab registry，實驗載入與表單錯誤由 GUI 回報。
來源：[使用者實驗的測試](v2/experiment-authoring.md#使用者實驗的測試)。

## 11. Startup 腳本與 entry-template 各自獨立

每份 startup 腳本保持獨立，使用自己的 entry-template，不與其他 startup 共用範本。
檢查：審閱腳本的 import 與範本路徑，修改某份範本不會改到另一份 startup 建立 entry 的內容。
來源：[startup 與模擬環境的放置](../docs/adr/0072-module-boundaries.md#使用者元件與模擬環境的放置)。

## 12. 文件每句寫出具體行為

文件每句說明會發生什麼、產生什麼狀態或要做什麼，不用抽象評語代替機制與結果。
檢查：逐句核對讀者能指出操作、資料或狀態，刪掉無法對應具體行為的形容詞與空泛結論。
來源：[文件規範](../CLAUDE.md#文件規範)。

---
date: 2026-10-15
status: superseded
superseded_by: docs/discussions/2026-10-16-ftf-simulation-speedup.md
related:
  - docs/adr/draft/simulation-environments.md
  - docs/discussions/2026-10-13-simulation-environment-ftf-preset.md
basis:
  - "repo@57f3370d3e24301b7554bf90ef8180f20c9c6b23，文件整理時HEAD；研究來源與既有未提交文件不在此commit內"
  - "本機研究產物截至2026-10-15；Python 3.13.11、NumPy 2.4.6、SciPy 1.17.1、Numba 0.65.1、scqubits 4.3.1"
  - "既有GPU CZ研究：Linux Intel Arc、OpenCL、driver 26.35.39758、FP64；本輪未新增外部資料調查"
---

# 離線GPU建庫、分段演化與插值重用

## 結論

這次討論把重點從「每次模擬都要快速建表」轉為「離線建立可重用檔案，模擬時載入並插值」。使用者認為一輪使用中的simulation YAML通常不變，因此建庫成本可以跨多次掃描攤提。

使用者先確認：「但我覺得目前的路線可行：預先計算成檔案，模擬時載入並插值。」之後補充：「認同，通常模擬yaml參數在一輪使用中不怎麼會變。」最後要求整理「離線GPU平行建庫，模擬時插值演化，分段模擬以及快取等」。

當時CPU prototype已提供數值與速度證據，GPU平行建庫則是下一條路線，不是已完成能力。決策由[模擬環境ADR](../adr/draft/simulation-environments.md#離線演化表與模擬時重用)擁有。本文保存比較、使用者修正與留下的問題，不授權恢復已暫停的研究或正式實作。

## 問題

目標是支援two-qubit gate與取得gate參數的前置量測，同時保留可負擔的物理因素，讓完整模擬平均每點接近一秒。直接積分每個掃描點會反覆計算相同模型；但完全預算所有amplitude、frequency、phase與波形組合又會造成高維建庫成本。

當時研究逐步驗證了標準pulse、amplitude插值、phase接續、雙路amplitude及短平台片段的frequency重用。使用者追問後，Agent澄清：研究當時是一次建立選定範圍的表，再在記憶體中求值，雖然已輸出NPZ，尚未驗證正式載入與快取管理。

另一個澄清是「分段」的意思。建庫時仍按原本的連續envelope與carrier細步積分，每段保存一個演化矩陣，不把波形改成水平階梯。片段邊界不允許任意重設phase或量子態。耗散另以分段方式施加，其誤差與插值誤差分開。

## 考慮過的選項與比較

### 離線檔案與執行期工作

討論比較了每次執行重建、首次需要時建整張表、逐點lazy累積，以及離線預建檔案。使用者選擇的方向利用了固定YAML的使用模式，將建庫、首用載入及warm求值分開。

目前tensor插值需要指定格點才能形成一張完整表，但這不要求建立所有表。按pulse family或參數區塊建立小表、將未完成格點先存檔再續建，都可作為部分建庫方式。逐點lazy、adaptive refinement與區塊接縫尚未驗證，不能沿用完整表的精度宣稱。

磁碟表、記憶體快取與exact-setting快取服務不同用途。插值表可求新設定，exact快取只重用完全相同條件；兩者都不能重用上一個獨立shot的隨機輸出。固定YAML也不等於所有物理輸入固定，round間device flux仍須依實際模型核對。具體依賴與失效規則見ADR，不由本快照另訂。

### 精度與四／六階成本

雙amplitude研究中，四階用25個直接格點，六階用49個。係數分別約3.91及7.66MiB。兩者使用相同時間積分步長，階數指插值，不是solver階數。格點與係數約減半，不代表完整warm演化也快兩倍，因為density與bath成本不變；當時沒有公平的分階計時。

四階在512個LO-reset雙路pulses後，population最大絕對差約2.65e−7，trace偏離約7.49e−7。Agent原先依保守研究門檻判為未通過長sequence。使用者指出：「看起來還是夠小啊？相對誤差多少？那四階與六階的模擬成本呢？」

這個修正指出了驗收重點。2.65e−7相對0～1滿刻度只有.265ppm；相對實際population仍需對應分母，尤其近零leakage不能只看相對誤差。當時未保存逐項reference population，不能報實際最大相對誤差。研究門檻不是產品要求，最後應比較取得的frequency、π amplitude、RB／gate參數偏差與成本。四階仍是實用候選，並未正式採用或排除。完整數值記在雙 amplitude 報告中，該報告是本機工作檔，沒有入庫。

### 最新CPU表的成本

完整弱spectroscopy案例以Q1 .002GHz振幅、1000ns平台、兩個20ns sin²邊緣及±3MHz frequency範圍建表。係數約5.625MiB。三組新frequency／phase對完整直接解的最大U元素差約6.25e−9，population差約3.03e−10。

該次CPU建庫約63秒，warm平均約9ms／點。Warm包含一個thermal初態的完整演化、耗散及十個population effects，不含檔案載入、GUI、I/Q、sampling與packing。它不是完整acquire的一秒驗收。

先前將建庫攤入每次掃描時，25點約2.53秒／點、81點約.787秒／點；加初始化及JIT後分別約2.75與.857秒。再加本輪三組直接驗證則約7.32與2.27秒。這些是單次build與warm分項加總，不是獨立cold-start重測。使用者確認離線檔案路線後，建庫及驗證成本仍揭露，但不要求每次使用重付。載入成本當時尚未量測。

### GPU適合放在哪裡

較早的FP64 GPU CZ研究，81設定整批約11.106秒，每設定攤提.13711秒，同場CPU 2 workers為.23517秒，GPU約快1.72倍。這是批次演化，不是GPU建庫或單點latency。準備、分析與末端I/Q仍有CPU工作；沒有GPU utilization或occupancy百分比可報。

另一輪長平頂比較中，GPU全程逐步積分輸給CPU的週期重用。這支持先減少重複工作，再選硬體，而不是把所有積分照搬GPU。它也沒有否定尚未實作的GPU分段重用。

討論因而把GPU的下一個用途放在離線獨立格點批次建庫。當時最近幾輪建表與插值全在CPU；GPU常駐表、跨設定／初態批次演化及transfer成本尚待驗證。不能把既有1.72倍直接套到新建庫流程。

## 被否決的選項與理由

- 把每次模擬重建全部格點當成必要成本。固定設定與檔案重用讓它不必每次發生，但離線成本仍須揭露。
- 把完整tensor的指定節點需求解讀成不能部分建庫。可以先按family／區塊建立完整小表；零散插值則需要不同方法。
- 把分段當成envelope階梯化。這不是目前prototype的算法，會混淆積分精度與片段重用。
- 只因四階略超保守population門檻就排除它。實際用途應看observable、最終參數與成本，不能用研究門檻代替需求。
- 用兩個完整單路U直接組合取代雙路解。現有對照的population差約9.51e−4，不支持該替代。
- 用CPU warm的9ms或舊GPU batch數字宣稱整個模擬環境達標。兩者覆蓋的工作量都有限，載入與正式acquire尚未驗收。

## 留下的問題

表格式、model identity、版本與完整性、缺表行為、分區策略、4GB快取的host／GPU歸算及eviction仍待設計。NPZ只是研究格式。GPU建庫需要在相同格點及精度下和CPU比較，並記錄傳輸、初次編譯與峰值記憶體。

完整雙路amplitude／frequency／phase覆蓋、更大basis、長continuous-phase序列與既有flux動態目標仍有缺口。固定DC表不能宣稱支援flux pulse或任意waveform。四／六階還要比較校準參數偏差，不能從滿刻度ppm直接推定gate infidelity。

當時active goal已暫停。這次只有文件整理授權，沒有啟動新研究、修改正式程式、建立lanes或landing。

## 依據

- 現行方向與引用原話：[模擬環境ADR](../adr/draft/simulation-environments.md#離線演化表與模擬時重用)。
- 研究產物包括：task spec、離線表細節、本輪交接，以及單路 phase／amplitude、frequency 平台、完整 spectroscopy 原始 JSON、GPU CZ、CPU／GPU batch 比較等報告。它們都是本機工作檔，沒有入庫；本文的數字已寫在正文。

這些研究產物沒有隨 repo commit 保存。交接須另保留來源、JSON、NPZ及既有manifest。最新完整spectroscopy輪次尚未補專屬manifest；上述數字依原始JSON／log，不宣稱已有完整封存。本文的commit與外部版本見frontmatter，沒有新增第三方效能結論。

## 後記（2026-10-16）

使用者在 sim-speed 的討論中指出：「因為一定有耗散，離線U表機制可以刪除了，除非你找到可以支援耗散的算法」。之後同意刪除離線 U 表與 exact density record。

本文的離線建庫路線因此被取代：
- 固定 DC、constant plateau 改由 flux 值量化加 secular 解析傳播子處理，在 runtime 以 lazy 方式建立並快取，含耗散。
- gate 類區段改由 runtime 的分段 superoperator 快取處理。

比較與理由見[FTF 模擬加速](2026-10-16-ftf-simulation-speedup.md)。刪除會等 measure-simulation-environment 收票後才進行，模擬環境 ADR 也會同步更新。

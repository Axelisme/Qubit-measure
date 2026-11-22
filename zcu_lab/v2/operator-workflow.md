**讀者：** operator、agent。用 measure-gui 開啟實驗、更新表單來源、採用校準結果並保存資料。

# Measure GUI 共用操作流程

## 開啟實驗並設定表單

從工具列的 **Entry…** 開啟實驗所在的 entry。要量某個 point 時，從同一視窗選該 point，再按 **Switch to selected**。未切到 point 時使用 entry 的 Global。

從 **New Tab** 選擇實驗。開 tab 時會讀取當下的參數、aliases、可選模組與裝置資訊，並填入表單初值。

切換 point、改來源或寫回校準後，已開的表單不會自動更新。在 **Run** 頁選擇：

- **Defaults**：重新讀取目前來源並重算初值。確認後會丟棄表單編輯。要使用剛校準的值作為新初值時選這個。
- **Refresh**：重新讀取來源、更新可選項並重驗表單，保留已填值、選項與 expression 原文。Expression 會用新來源求值，已經填入的直接數值不會重算。

先照實驗的 **Guide** 完成必要校準並檢查欄位，再按 **Run**。要擬合或檢查結果時，在有提供的 **Analysis** 頁按 **Analyze**。若實驗另有 **Post-Analysis**，先完成 Analyze，再依實驗 guide 決定是否按 **Run Post-Analysis**。

## 審查與採用校準

分析有提出校準候選時，在該頁的 **Writeback preview** 對照結果圖、目前值、提案值和目的地。候選用該次 Run 或載入結果的來源解析；實際寫入的是按下按鈕時作用中的 entry 或 point。切換目的地後，重新核對預覽。

取消不採用的勾選，再按 **Apply Selected**。沒有按 Apply 就不會採用提案；預先勾選也不代表結果適合你的用途。核對哪些列已寫入；若報錯，先查看錯誤及已寫入的列，再決定怎麼處理。要讓既有表單用剛採用的校準重算初值，回 **Run** 頁按 **Defaults**。

## 保存結果

在 **Data** 頁確認要保存的 Run、分析成果和預覽路徑。按該項的 **Save**，或用 **Save All** 保存所有可保存的成果。保存後核對各項狀態及實際路徑；若有失敗，查看該項錯誤。只有看到 **Saved** 才算該項已保存。

**Apply Selected** 寫入選定的校準提案，不存 Run 或分析成果。**Save** 保存資料與成果，不採用校準提案。

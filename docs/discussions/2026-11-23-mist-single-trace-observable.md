---
date: 2026-11-23
status: active
superseded_by: null
related:
  - ../adr/0069-experiment-core-and-frontends.md#平均-iq-的投影慣例
  - 2026-11-23-inverted-error-bars-and-figure-facts.md
  - 2026-11-23-coherence-run-only-shot-sem.md
  - ../../zcu_lab/v2/mist/power_dep/single_trace/guide.md
basis:
  repo_commit: 7f51de892318b9d75685acff8dee438c51a810e6
  external_sources: none
---

# MIST single trace 保留平均 IQ 的位移大小

## 結論

2026-11-23，designer 夜間代裁（2026-11-23 授權），待使用者追認。MIST single trace 保留 `|mean IQ(gain) − reference IQ|`，讓 Live 與空白 Ground IQ reference 的 Analyze 使用同一複數參考。參考取掃描順序開頭 5% 的點，向下取整，至少一點。下掃也從開頭取，不按 gain 排序；如果首段無法形成參考，不從後段補一個。圖連接量測點，不分類、不擬合，也不畫 shot SEM。

## 問題

Live 本來用掃描首段的複數均值，Analyze 空白參考時卻用第一點。同次資料可能因參考不同而畫出兩條曲線。舊有徑向近似棒把投影的誤差畫在取模之後的距離上。先對複數 shots 求平均，再減參考並取模，得到的不是逐發距離的平均，不能用逐發樣本標準差除以 shot 數平方根來代替它的誤差。

## 選項與取捨

選首段複數均值保留原 Live 的參考，且不會強迫第一點的顯示值為零。例如相對參考偏移 `+1` 和 `+i` 都是 1 a.u.；只沿 I 軸看會把第二個偏移看成零。這是偏離[平均 IQ 主軸投影慣例](../adr/0069-experiment-core-and-frontends.md#平均-iq-的投影慣例)的理由。若想讓預設參考代表弱探測，就從弱探測區起步。不想用該段時，Analyze 可明填 Ground IQ reference。Excited IQ 仍只限制 y 上界，光子數係數仍只替換 x 軸，不影響 IQ 參考或分類。

採同一參考後，舊檔重新 Analyze 也用新規則，已存的圖不變。不改普通平均 IQ 取數、Result、Native、Labber 或 workflow。與[同次 coherence shot SEM](2026-11-23-coherence-run-only-shot-sem.md)不同，這裡不是因為重載後缺矩才沒有棒；同一次 Run 的觀測量也不是可直接套該 SEM 的線性平均。[一維反演量的傳遞規則](2026-11-23-inverted-error-bars-and-figure-facts.md)允許校正布居與計數相位從共變異推誤差，不會把這個幅值改成反演量。

## 未採用

- 只用 guide 說明兩個預設，仍會讓終格和 Analyze 不一致。全改用第一點雖能對齊，卻讓一點的波動全部進入參考，且第一個位移恆為零。
- 改成有號 PCA 或 G→E 投影，再用同批矩畫 SEM，能估一個方向的對比，卻會弱化正交的位移，並改變圖的物理量。G→E 還需要 Run 前就有方向，不是原有的可選 y 上限。
- 以共變異作徑向一階近似，需處理參考不確定度及首段與各點的相關；近零時也不能用穩定的徑向線性化。即使模型齊備，也不是本次允許的逐發平均 SEM。
- 新定距離的信賴區間或改畫逐發距離均值，都要重定估計量。後者還會把單發噪音帶進點值，不等於平均 IQ 到參考的距離。

## 留下的問題

這張圖只能指出平均 IQ 在某些 gain 的位移，不能識別布居或 leakage 能階，也不能用棒判定微小位移是否超出噪音。若需要區間推論，再為原幅值另定模型與參考的採樣方式。R0 的逐點 shot SEM 項在這個 observable 不適用，不因資料缺失而免除一個適用的棒；讀者仍須能說出量、參考、限制和下一步。

## 依據

- 2026-11-23，使用者對主軸投影說「有獨特理由」才另選；對反演誤差說：「反演出來的量如果可以從變異矩陣推導那也畫。」見[投影慣例](2026-11-23-mean-iq-plot-projection.md)與[一維反演量討論](2026-11-23-inverted-error-bars-and-figure-facts.md)。這些是代裁的先例，不是這次代裁的覆核。
- 原始選項調查與同源驗收工件未入庫。當時的成因、變更與不採用的方案記在本篇。實際操作及限制見 [MIST single trace guide](../../zcu_lab/v2/mist/power_dep/single_trace/guide.md)。

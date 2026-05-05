# tProc v2 ASM 指令集與暫存器速查筆記

> 來源：`qick/tprocv2_assembler.py` + `qick/asm_v2.py`
> tProc v2 指令寬度為 72 bits，每條指令佔一個 program memory word（WAIT 例外，展開為 2 words）。

---

## 一、硬體指令（Hardware Instructions）

### 1. `NOP`

- **用處**：空指令，消耗一個 tProc 時鐘週期。
- **屬性**：無。

---

### 2. `TEST`

- **用處**：對暫存器/立即數做 ALU 運算並更新 ALU flags，但**不寫入任何暫存器**（等同強制 `-uf` 的 `REG_WR`）。
- **屬性**：
  - `-op(A op B)`：ALU 運算式（支援完整 ALU 列表，見下方）。
  - `-if(COND)`：條件執行（通常不用於 TEST，但語法允許）。
  - `-wr(reg op/imm)`：可選，同時寫入一個附加暫存器。

---

### 3. `REG_WR dst src`

- **用處**：通用暫存器寫入，從多種來源（立即數、ALU、dmem、wmem、label address）寫入目標暫存器。
- **屬性**：
  - `dst`：目標暫存器（`sN`、`rN`、`wN`、或 `r_wave`）。
  - `src`：來源類型：
    - `imm`：立即數，需搭配 `#N` 字面量（32 bits）。
    - `op`：ALU 運算，需搭配 `-op(...)`。
    - `dmem`：從 data memory 讀取，需搭配 `[address]`。
    - `wmem`：從 wave memory 讀取到 wave registers（`r_wave`），需搭配 `[address]`。
    - `label`：將 program memory 地址寫入 `s15`（用於大程式 branch）。
  - `-op(A op B)`：ALU 運算式（`src=op` 時必填）。
  - `-if(COND)`：條件執行（`wmem` 來源時**不支援**條件）。
  - `-uf`：更新 ALU flags（`src=op` 才有效）。
  - `-wr(reg op/imm)`：同時寫入一個額外暫存器（side-data injection）。
  - `-ww`：同時將 wave registers 寫回 wave memory（`src=wmem` 時）。
  - `-wp(r_wave/wmem) pN`：同時送出到 wave port N（`src=wmem` 時）。
  - `@T`：指定輸出時間（`src=wmem` 時才支援）。
  - `[addr]`：dmem/wmem 的記憶體地址（支援 `&literal`、`sN`、`rN`，或 `base+offset` 格式）。

---

### 4. `DMEM_WR [addr] src`

- **用處**：寫入 data memory。
- **屬性**：
  - `[addr]`：目標地址（`&literal`、`sN`、`rN`，或 `base+offset`）。
  - `src`：`imm`（立即數 `#N`）或 `op`（ALU 運算 `-op(...)`）。
  - `-op(...)`：ALU 運算式（`src=op` 時必填）。
  - `-uf`：更新 flags。
  - `-wr(reg op/imm)`：side-data injection。
  - `-if(COND)`：條件執行。

---

### 5. `WMEM_WR [addr]`

- **用處**：將 wave registers（w0~w5）的當前值寫入 wave memory。
- **屬性**：
  - `[addr]`：目標地址（`&literal` 或 `rN`）。
  - `@T`：指定時間（此時**不能**搭配 `-wr` 或 `-op`）。
  - `-op(...)`：ALU 運算（無 `@T` 時可選）。
  - `-uf`：更新 flags。
  - `-wr(reg op/imm)`：side-data injection。
  - `-wp(r_wave/wmem) pN`：送出到 wave port。

---

### 6. `WPORT_WR pN src`

- **用處**：從 wave memory 或 wave registers 送波形設定到指定的 tProc wave port（連接 signal generator）。
- **屬性**：
  - `pN`：目標 wave port（N ≤ 15）。
  - `src`：`wmem`（從 wave memory 讀，需 `[addr]`）或 `r_wave`（從 wave registers）。
  - `@T`：指定輸出時間（立即數 32-bit）。
  - `-ww`：同時寫回 wave memory（`src=r_wave` 時）。

---

### 7. `DPORT_WR pN src data`

- **用處**：寫入資料到 data port（如觸發 readout），最多 4 個 port（p0–p3）。
- **屬性**：
  - `pN`：目標 data port（N ≤ 3）。
  - `src`：`imm`（立即 11-bit，`data` 欄填值 ≤ 2047）或 `reg`（`data` 欄填 `rN` 暫存器）。
  - `@T`：指定輸出時間。

---

### 8. `DPORT_RD pN`

- **用處**：從 data port 讀取 I/Q 累加結果到 `s_port_l`（I）和 `s_port_h`（Q）。
- **屬性**：
  - `pN`：來源 data port（N ≤ 7）。

---

### 9. `TRIG pN set/clr`

- **用處**：設定或清除觸發訊號 port（連接 readout trigger line）。
- **屬性**：
  - `pN`：trigger port（N ≤ 31；硬體加 32 offset）。
  - `set` / `clr`：設定 / 清除。
  - `@T`：指定觸發時間。

---

### 10. `JUMP [addr/label]`

- **用處**：無條件或有條件跳轉。
- **屬性**：
  - 目標：label 名稱（小程式）或 `s15`（大程式，pmem > 2^11）。
  - `-if(COND)`：條件跳轉（見條件列表）。
  - `-wr(reg op/imm)` + `-op(...)`：跳轉同時執行 side-data injection（用於 loop counter 遞增 + 比較 + 跳轉合併在一條指令）。
  - `-uf`：更新 flags。
  - 特殊 label：`PREV`（當前 -1）、`HERE`（當前）、`NEXT`（當前 +1）、`SKIP`（當前 +2）。
- **地址範圍限制**：
  - 立即數地址（`&N`）：**最大 2047**（11-bit unsigned，硬體 encoding 限制）。
  - 暫存器間接（`s15`）：無上限，需先用 `REG_WR s15 label X` 載入地址。
  - 程式超過 2048 words 時，所有 `JUMP`/`CALL`/`WAIT` **都必須**改用 s15 間接跳轉。Python 框架已自動處理（`pmem_size > 2^11` 判斷）。

---

### 11. `CALL [addr/label]`

- **用處**：呼叫子程序（Call-Return 機制，將 PC 存入 stack，跳到目標）。
- **屬性**：同 `JUMP`（但不支援 `-ww`）。

---

### 12. `RET`

- **用處**：從子程序返回（從 stack 取回 PC）。
- **屬性**：無。

---

### 13. `TIME C_OP [R1 / #N]`

- **用處**：控制時間基準暫存器（`s_out_time`）。
- **屬性**：
  - `C_OP` 選項：
    - `rst`：重置時間為 0。
    - `updt`：更新時間戳（從硬體讀取當前時間）。
    - `set_ref`：將 `s_out_time` 設為指定值（`R1` 或 `#N`）。
    - `inc_ref`：將 `s_out_time` 增加指定值（用於 `delay()` 實作）。
  - `R1`：來源暫存器（s/d 類型）。
  - `#N`：立即數（32-bit）。

---

### 14. `FLAG C_OP`

- **用處**：操作 external flag。
- **屬性**：
  - `C_OP`：`set`、`clr`、`inv`。

---

### 15. `ARITH C_OP R1 R2 [R3 [R4]]`

- **用處**：高精度乘加運算（非同步，需等待結果）。支援 MAC（multiply-accumulate）組合。
- **屬性**（`C_OP` 決定所需暫存器數）：

| C_OP  | 公式           | 暫存器     |
|-------|----------------|------------|
| `T`   | A×B            | R1=A, R2=B |
| `TP`  | A×B+C          | R1=A, R2=B, R3=C |
| `TM`  | A×B−C          | R1=A, R2=B, R3=C |
| `PT`  | (D+A)×B        | R1=D, R2=A, R3=B |
| `MT`  | (D−A)×B        | R1=D, R2=A, R3=B |
| `PTP` | (D+A)×B+C      | R1=D, R2=A, R3=B, R4=C |
| `PTM` | (D+A)×B−C      | R1=D, R2=A, R3=B, R4=C |
| `MTP` | (D−A)×B+C      | R1=D, R2=A, R3=B, R4=C |
| `MTM` | (D−A)×B−C      | R1=D, R2=A, R3=B, R4=C |

- 結果非同步寫回：低 32-bit → `s_arith_l` (s3)，需用 `WAIT div_rdy` 或輪詢 `s_status`。

---

### 16. `DIV NUM DEN`

- **用處**：整數除法（非同步），計算商與餘數。
- **屬性**：
  - `NUM`：被除數（`sN` 或 `rN` 暫存器）。
  - `DEN`：除數（暫存器或立即數 `#N`）。
- 結果：商 → `s_div_q` (s4)，餘數 → `s_div_r` (s5)，需輪詢 `s_status` 的 `bit_div_rdy`。

---

### 17. `NET C_OP [R1 [R2 [R3]]]`

- **用處**：QNET 網路外設控制（多 FPGA 同步）。
- **屬性**（C_OP）：`set_net`、`sync_net`、`updt_offset`、`set_dt`、`get_dt`、`set_flag`、`get_flag`。

---

### 18. `COM C_OP [R1]`

- **用處**：QCOM 通訊外設控制。
- **屬性**（C_OP）：`set_flag [0/1]`、`sync`、`reset`、`set_byte_1/2`、`set_hw_1/2`、`set_word_1/2`。

---

### 19. `PA C_OP [R1 R2 R3 R4]` / `PB C_OP [R1 R2 R3 R4]`

- **用處**：自定義外設 A / B。
- **屬性**：
  - `C_OP`：0–31 的操作碼（整數）。
  - `R1`~`R4`：最多 4 個暫存器運算元（R1/R2 為 data regs，R3/R4 為 addr regs）。
  - **不支援立即數** `#N`。

---

### 20. `WAIT C_OP [time/@T]`

- **用處**：阻塞等待（複合指令，展開為 2 條機器指令：TEST + JUMP）。
- **屬性**（C_OP）：
  - `time`：等到 `s_usr_time` (s11) 達到指定時間（`@T` 立即數，最大 24-bit）。
  - `port_dt`：等到 data port 有新資料（輪詢 `s_status` bit 15）。
  - `div_rdy`：等到除法器就緒。
  - `div_dt`：等到除法器有新結果。
  - `qpa_rdy`：等到 custom peripheral A 就緒。
  - `qpa_dt`：等到 custom peripheral A 有新結果。
- **注意**：`WAIT` 佔用 2 個 program memory words；`time` 的立即數最大約 24-bit（3 bytes）；更大的等待時間需手動使用 scratch reg + TEST + JUMP。

---

### 21. `CLEAR C_OP`

- **用處**：清除外設的 data-new flag（展開為 `REG_WR s2 imm`）。
- **屬性**（C_OP）：`arith`、`div`、`qnet`、`qcom`、`qpa`、`qpb`、`port`、`all`。

---

## 二、ALU 運算列表

### 完整 ALU（REG_WR `-op()` 專用）

| 符號 | 說明 |
|------|------|
| `+` | 加 |
| `-` | 減 |
| `AND` / `&` / `MSK` | 位元 AND |
| `OR` / `\|` | 位元 OR |
| `XOR` / `^` | 位元 XOR |
| `ASR` | 算術右移（帶符號）|
| `SR` / `>>` | 邏輯右移 |
| `SL` / `<<` | 左移 |
| `ABS` | 絕對值 |
| `MSH` | 取高 16-bit |
| `LSH` | 取低 16-bit |
| `SWP` | 高低 16-bit 交換 |
| `CAT` / `::` | 拼接（concat） |
| `NOT` / `!` | 位元非 |
| `PAR` | 同位（parity） |

> 移位量最大 15；`ASR`/`SR`/`SL` 的移位量用立即數指定（`#N`）。

### 精簡 ALU（`-wr()` side-data injection 專用）

`+`、`-`、`AND`、`ASR`

---

## 三、條件跳轉列表（`-if(COND)`）

| 條件 | 說明 |
|------|------|
| `1` | 永遠跳（無條件） |
| `0` | 永遠不跳 |
| `Z` | 等於零（ALU 結果 == 0） |
| `NZ` | 不等於零 |
| `S` | 負數（符號位 = 1，即 < 0） |
| `NS` | 非負數（>= 0） |
| `F` | external flag 為 1 |
| `NF` | external flag 為 0 |

---

## 四、暫存器系統

tProc v2 有三類暫存器：

### (A) 系統暫存器 `sN`（s0–s15，32-bit，唯讀 or 特殊用途）

| 別名 / 編號 | 用處 | 注意 |
|---|---|---|
| `s0` / `zero` / `s_zero` | 恆為 0 | 硬體固定，寫入無效 |
| `s1` / `s_rand` | LFSR 隨機數（32-bit PRNG） | 每次讀值會自動步進 |
| `s2` / `s_cfg` / `s_ctrl` | 設定暫存器（控制外設 clear / CSF）| 寫入指定 bit 觸發外設操作 |
| `s3` / `s_arith_l` | ARITH 結果低 32-bit | 只有 ARITH 指令完成後有效 |
| `s4` / `s_div_q` | DIV 商 | 只有 DIV 完成後有效 |
| `s5` / `s_div_r` | DIV 餘數 | 只有 DIV 完成後有效 |
| `s6` / `s_core_r1` | 核心讀取暫存器 1 | 一般 read-only（from external） |
| `s7` / `s_core_r2` | 核心讀取暫存器 2 | |
| `s8` / `s_port_l` | DPORT_RD 結果 I（低字）| DPORT_RD 後有效 |
| `s9` / `s_port_h` | DPORT_RD 結果 Q（高字）| DPORT_RD 後有效 |
| `s10` / `s_status` | 外設狀態旗標 | 各 bit 含義見下方 |
| `s11` / `s_usr_time` / `curr_usr_time` | 目前硬體時間（只讀）| 用於 WAIT time 比較 |
| `s12` / `s_core_w1` | 可外部讀取的寫暫存器 1（shot counter）| 可由 tProc 寫入，外部（Python）可讀取 |
| `s13` / `s_core_w2` | 可外部讀取的寫暫存器 2 | |
| `s14` / `s_out_time` / `out_usr_time` | 輸出時間暫存器（基準時間）| 所有 timed instruction 以此為參考 |
| `s15` / `s_addr` | Program address（大程式跳轉用）| 需先用 `REG_WR s15 label X` 載入地址再 JUMP |

#### `s10` (`s_status`) 各 bit

| Bit（alias）       | 含義 |
|--------------------|------|
| bit 0  `bit_arith_rdy` | ARITH 外設就緒 |
| bit 1  `bit_arith_new` | ARITH 有新結果 |
| bit 2  `bit_div_rdy`   | DIV 外設就緒 |
| bit 3  `bit_div_new`   | DIV 有新結果 |
| bit 4  `bit_qnet_rdy`  | QNET 就緒 |
| bit 5  `bit_qnet_new`  | QNET 有新資料 |
| bit 6  `bit_qcom_rdy`  | QCOM 就緒 |
| bit 7  `bit_qcom_new`  | QCOM 有新資料 |
| bit 8  `bit_qpa_rdy`   | Custom PA 就緒 |
| bit 9  `bit_qpa_new`   | Custom PA 有新資料 |
| bit 10 `bit_qpb_rdy`   | Custom PB 就緒 |
| bit 11 `bit_qpb_new`   | Custom PB 有新資料 |
| bit 15 `bit_port_new`  | Data port 有新資料 |

#### `s2` (`s_cfg`) 設定值（寫入觸發操作）

| 值（alias）              | 作用 |
|--------------------------|------|
| `cfg_src_axi`    `#h00`  | 資料來源設為 AXI |
| `cfg_src_arith`  `#h01`  | 資料來源設為 ARITH |
| `cfg_src_qnet`   `#h02`  | 資料來源設為 QNET |
| `cfg_src_qcom`   `#h03`  | 資料來源設為 QCOM |
| `cfg_src_qpa`    `#h04`  | 資料來源設為 PA |
| `cfg_src_qpb`    `#h05`  | 資料來源設為 PB |
| `cfg_src_core`   `#h06`  | 資料來源設為 core |
| `cfg_src_port`   `#h07`  | 資料來源設為 port |
| `ctrl_clr_arith` `#h1_0000` | 清除 ARITH new flag |
| `ctrl_clr_div`   `#h2_0000` | 清除 DIV new flag |
| `clr_all`        `#h7F_0000` | 清除所有 new flags |

---

### (B) 資料暫存器 `rN`（r0–r31，32-bit，通用）

- 用戶可自由配置，作為通用暫存器（計數器、臨時值等）。
- 程式框架（`QickProgramV2`）會用 `add_reg()` 自動分配；也可手動用 `r0`–`r31` 直接操作。

---

### (C) Wave 暫存器 `wN`（w0–w5，多 bit，對應 wave memory field）

Wave registers 是一個 6-field buffer，對應 wave memory 的一個 entry：

| 別名       | 編號 | 用處 |
|------------|------|------|
| `w_freq`   | w0   | 頻率 |
| `w_phase`  | w1   | 相位 |
| `w_env`    | w2   | envelope index |
| `w_gain`   | w3   | gain |
| `w_length` | w4   | 脈衝長度 |
| `w_conf`   | w5   | configuration flags（outsel, mode, stdysel, phrst 等）|

- 流程：`REG_WR r_wave wmem [&addr]`（讀進）→ 修改 `wN` → `WMEM_WR [&addr]`（寫回）→ `WPORT_WR pN r_wave @T`（送出）。
- `w0` (`w_freq`) **不能**作為 `REG_WR` 的 `dst`（硬體限制）。

---

## 五、記憶體系統

| 記憶體 | 指令 | 說明 |
|--------|------|------|
| Program Memory (pmem) | `JUMP`/`CALL` | 存放 ASM 指令；地址從 0 開始（address 0 固定為 NOP） |
| Wave Memory (wmem) | `REG_WR r_wave wmem`, `WMEM_WR`, `WPORT_WR` | 儲存波形參數（freq/phase/env/gain/length/conf）；每個 entry 168 bits（21 bytes） |
| Data Memory (dmem) | `REG_WR dst dmem`, `DMEM_WR` | 通用資料儲存，地址以 `&N` 格式存取 |

---

## 六、指令修飾符（modifier）彙整

| 修飾符 | 意義 | 支援指令 |
|--------|------|----------|
| `-if(COND)` | 條件執行 | JUMP, REG_WR（非 wmem）, DMEM_WR |
| `-uf` | 更新 ALU flags | REG_WR, DMEM_WR, TEST, JUMP |
| `-op(A op B)` | ALU 運算式 | REG_WR, DMEM_WR, TEST, JUMP |
| `-wr(reg op/imm)` | 同時寫入附加暫存器 | REG_WR, DMEM_WR, JUMP, CALL |
| `-ww` | 同時寫回 wave memory | REG_WR (wmem src), WPORT_WR |
| `-wp(src) pN` | 同時送到 wave port | REG_WR (wmem src), WMEM_WR |
| `@T` | 指定輸出時間（立即數）| WMEM_WR, WPORT_WR, DPORT_WR, TRIG |
| `[addr]` | 記憶體地址（`&N`, `sN`, `rN`, `base + &offset`）| REG_WR, DMEM_WR, WMEM_WR |

---

## 七、組合語言雜項

### 立即數格式

| 格式 | 含義 |
|------|------|
| `#N` | 有符號十進位（signed decimal） |
| `#uN` | 無符號十進位（unsigned decimal） |
| `#bN` | 二進位（binary） |
| `#hABCD` | 十六進位（hex，大寫） |
| `&N` | Program / memory address |
| `@N` | 時間值（用於 WAIT/timed instructions） |

### Directive（組合語言指令）

| Directive | 用法 | 作用 |
|-----------|------|------|
| `.ALIAS name reg` | `.ALIAS myloop r5` | 為暫存器取別名 |
| `.CONST name val` | `.CONST N #100` | 定義常數 |
| `.ADDR N` | `.ADDR 64` | 設定下一條指令的 program address（用 NOP 填充空隙） |
| `.END` | | 插入無限迴圈（程式結束標記） |

### Label 特殊保留字

`PREV`（當前 - 1）、`HERE`（當前）、`NEXT`（當前 + 1）、`SKIP`（當前 + 2）

---

## 八、硬體迴圈加速（Hardware Loop）

**結論：tProc v2 沒有專用硬體迴圈指令**，不支援類似 DSP 核心的 `LOOP`/`REPEAT` 指令。

迴圈完全由軟體實作：

1. 用 `REG_WR` 初始化計數器（如 `r0 = 0`）。
2. 設定 label（迴圈起點）。
3. 迴圈體。
4. 最後一條 `JUMP` 搭配 `-wr(r0 op) -op(r0 + #1) -if(NZ)` —— **遞增計數器 + 比較 + 跳轉可在同一條指令完成**（這是唯一的硬體加速點：side-data injection `-wr` 讓 inc + cmp + branch 合併為 1 word）。

```asm
// 範例：執行 100 次的迴圈
     REG_WR r0 imm #0          // 初始化計數器
LOOP:
     // ... 迴圈體 ...
     TEST   r0 - #99           // 比較 r0 == 99？（或用下面的合併寫法）
     JUMP   LOOP -if(NZ) -wr(r0 op) -op(r0 + #1)  // 遞增並跳回（合併）
```

Python 框架（`open_loop` / `close_loop`）就是這樣展開的，`CloseLoop` 展開結果：

```python
TEST   reg - #(n-1)         # 設定 flag
JUMP   label -if(NZ) -wr(reg op) -op(reg + #1)  # 跳回並遞增
```

> `-wr` + `-op` + `-if` 合併到 `JUMP` 是目前最緊湊的迴圈控制，比「先 TEST 再 JUMP」少一條指令，但每個迴圈底部仍需 1 條機器指令。這是 tProc v2 的設計上限。

---

## 九、時序系統重點

| 暫存器 | 別名 | 含義 |
|--------|------|------|
| `s11` | `s_usr_time` | 當前硬體時間（只讀，持續增加） |
| `s14` | `s_out_time` | 輸出基準時間（timed instruction 的絕對時間參考） |

- `TIME inc_ref #N`：`s14 += N`（`delay()` 的底層實作）。
- `TIME set_ref R1`：`s14 = R1`。
- timed instruction（`WPORT_WR @T`、`DPORT_WR @T`）的實際發送時間 = `s14 + T`。
- `WAIT time @T`：阻塞直到 `s11 >= (s14 + T - WAIT_TIME_OFFSET)`，其中 `WAIT_TIME_OFFSET = 10`。
- `WAIT` 只接受純量時間，**不支援 sweep**（Python 框架會自動取 max 並警告）。

---

## 十、Wave Memory 參數限制

### Pulse length 硬體範圍

| Generator 類型 | length 最小值 | length 最大值 |
|----------------|--------------|--------------|
| StandardGen（全速/插值）| 3 cycles | 2^16 - 1 cycles（65535）|
| MultiplexedGen（mux）  | 3 cycles | 2^32 - 1 cycles |
| Readout config         | 3 cycles | 2^16 - 1 cycles |

> 低於 3 cycles 的 length 在某些 generator 上會導致輸出錯誤，建議 envelope 末端補零以確保長度 ≥ 3。

### `w_conf`（w5）bit encoding

`cfg2reg()` 的 bit layout（`axis_signal_gen` 類型）：

| bits | 含義 | 選項 |
|------|------|------|
| [1:0] `outsel` | 輸出來源 | `product`=0（table×DDS），`dds`=1，`input`=2，`zero`=3 |
| [2] `mode` | 單次/週期 | `oneshot`=0，`periodic`=1 |
| [3] `stdysel` | 波形結束後輸出 | `last`=0（維持最後一個樣本），`zero`=1 |
| [4] `phrst` | 相位重置 | 1 = 重置 DDS 相位累加器（僅部分 gen 支援：v6, int4_v1, int4_v2）|
| [15:8] `tmux_ch` | tProc mux channel | 有 mux 時自動填入 |

### `flat_top` 三段展開

`flat_top` 一個 pulse 會展開為 **3 個 waveform entry**（int4 類型額外加第 4 個 dummy entry）：

| 段 | `outsel` | `gain` | envelope | 說明 |
|----|----------|--------|----------|------|
| ramp up | `product` | full gain | 前半段 envelope | 上升沿 |
| flat | `dds` | half gain × `maxv_scale` | 無 | 平坦部分，長度由 `length` 指定 |
| ramp down | `product` | full gain | 後半段 envelope | 下降沿 |

- **envelope 必須是偶數長度**（否則中間一個 cycle 會被跳過並警告）。
- flat 段的 gain 會乘以 `1/2 × maxv_scale`（Fraction 運算），確保與 ramp 段幅度匹配。
- `phrst` 只套用在第一段（ramp up）。
- **`flat_top` 不支援 `mode` 和 `outsel` 參數**（固定為 `oneshot` + `product`/`dds`）。

---

## 十一、資料暫存器（`rN`）使用注意

- 數量由韌體的 `tproccfg['dreg_qty']` 決定（通常為 32，即 r0–r31）。
- Python 框架用 `add_reg(name)` 自動分配地址，分配順序從 r0 開始。
- **框架內部也會消耗 rN**（loop counter、swept time register、scratch register），手動程式需留意衝突。
- `scratch` register 可用 `allow_reuse=True` 聲明，允許多處重複使用同一個地址（但不能有 init value）。
- 以下名稱為保留字，不能用於 `add_reg`：所有 `REG_ALIASES`（`w_freq`、`s_zero` 等）以及形如 `sN`/`rN`/`wN` 的地址字串。

---

## 十二、程式編譯兩階段流程

tProc v2 程式與 v1 不同，採用**兩階段編譯**：

```
階段一（make_program / 用戶呼叫 macro 方法）
  → 建立 macro_list（高階巨集）與 waves（波形參數）

階段二（compile()）
  → preprocess：確定 loop 結構 → 計算 sweep step → 計算 timeline → 分配 register
  → translate：展開 macro_list → 產生 prog_list（ASM 指令字典列表）
  → 產生二進位：prog_list → pmem binary，waves → wmem binary
```

- `get_pulse_param()` / `get_time_param()` 只能在 `compile()` 後呼叫（sweep step 在此時才確定）。
- `prog.asm()` 可印出展開後的文字 ASM，方便除錯。
- `prog.__str__()` 會同時印出 macro list、register 分配、pulse 定義、展開 ASM。
- 程式存檔（JSON dump）只保留低階資訊（prog_list + waves），高階 macro/sweep 資訊不保留。

### 版本相容性

此 Python 版本支援的 tProc v2 韌體 revision：**21–27**（`ASM_REVISIONS`）。
韌體版本可由 `soc.tproccfg['revision']` 查詢。

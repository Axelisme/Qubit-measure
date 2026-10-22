# 新增 component

**讀者：** lab-author、agent。新增 coupler kind，讓 entry 與表單讀取它的校準值。

這篇教學新增一個 `coupler` kind，讓 entry 保存它的校準值，並讓表單以 `$coupler.freq` 讀取頻率。Kind 是 schema 的註冊名稱，`C1` 則是某個 entry 內的元件名稱。新增 kind 不會替實驗加上 pulse 序列，也不會讓模擬環境自動支援它。

## 1. 宣告欄位

新增 `zcu_lab/components/coupler.py`，寫入以下完整 model。每個新欄位都用 `Field(description=...)` 說明數值的意義。

```python
from pydantic import Field
from zcu_lab.components.base import BuiltinComponent, Duration, Energy, Flux, Frequency

class CouplerSchema(BuiltinComponent):
    """Coupler calibration values; flux uses the caller's declared bias unit."""
    freq: Frequency = Field(default=None, description="Coupler transition frequency.")
    settle_time: Duration = Field(default=None, description="Bias settling time.")
    EC: Energy = Field(default=None, description="Coupler charging energy.")
    flux: Flux = Field(default=None, description="Bias in the caller's declared unit.")
```

`Frequency` 的工作單位是 MHz，`Duration` 是 us，`Energy` 是 GHz。`Flux` 沒有固定單位，caller 必須交代使用的座標與單位。這些型別允許 `None`，但 `default=None` 才讓欄位可以省略。不要把未知校準值填成 `0.0`。單位與 global／local flux 的區別見 [component 欄位單位](../README.md#component-欄位單位)。

`BuiltinComponent` 只提供 `ext` 與 `module`，不提供 wiring。需要通道的 kind 必須自己宣告 wiring model 與欄位，沒有讀取者就不加。內建 resonator 宣告 `WiringSchema(ch, ro_ch)`，qubit 宣告 `QubitWiringSchema(ch, lo_flux_ch)`，mux group 宣告 `MuxWiringSchema(gen_ch, ro_ch)`，其中 mux `ro_ch` 是與 resonators 同序、等長的列表。上面的 coupler 範例沒有通道讀取者，因此不宣告 wiring。

未宣告的輸入欄位，包括巢狀 model 裡的未知欄位，會移到 `ext.migrated` 的元件相對路徑，並寫出 warning。例如 `freqq: 4800.0` 會變成 `ext.migrated.freqq`，不會填入 `freq`。再次輸入同一路徑會覆蓋該 migrated 值，其他 ext 值保留。已宣告欄位的非法值仍報驗證錯誤。載入不立即改寫檔案，保存行為見 [保存與版本](../../lib/zcu_tools/resources/entry/README.md#保存與版本)。

## 2. 註冊 kind

在 `zcu_lab/components/__init__.py` 加上 `from .coupler import CouplerSchema`，再在既有 `register_all(registry)` 內加入：

```python
registry.register("coupler", CouplerSchema)
```

Kind 使用唯一且穩定的字串。現有 kind 以 `/` 分類，例如 `qubit/fluxonium`、`qubit/transmon`、`amplifier/jpa`。不要把新 model 註冊在已占用的名稱，也不要只為了取得 qubit 角色而把不同用途的元件命名成 `qubit/...`。重複註冊與未知 kind 的錯誤見 [元件與定義](../../lib/zcu_tools/resources/entry/README.md#元件與定義)。

GUI 的 alias 不要求 RoleSpec。若 Notebook 要用 `point.resolve(roles=["coupler"])`，在同一個 `register_all` 內另加 `registry.roles.register("coupler", RoleSpec("coupler"))`。`RoleSpec` 已由該檔 import。RoleSpec 的 kind 是 fnmatch pattern，`qubit/*` 因此接受兩種內建 qubit。四個內建角色與 shorthand 見 [components 索引](../components/README.md#角色與引用)。Notebook 的 explicit、focus 與 via 解析順序見 [角色與失敗](../../lib/zcu_tools/resources/entry/README.md#角色與失敗)。

Import model 不會執行註冊。GUI 組合根會呼叫 `register_all`，修改註冊後重開 GUI。Notebook caller 則在空 registry 上明確呼叫 `components.register_all(component_registry)`，不要重複 bootstrap 同一個 registry。

## 3. 建立元件並讓表單讀取

在自己的 entry-template 中新增元件，並在 aliases 內新增 `coupler: C1`。以下是可單獨載入的最小範本，數值只示範儲存格式，不是硬體操作建議。

```yaml
format: zcu.entry-template
format_version: "1.0"
components:
  C1:
    kind: coupler
    freq: 4800.0
    settle_time: 0.5
    EC: 0.3
aliases:
  format: zcu.aliases
  format_version: "1.0"
  aliases:
    coupler: C1
```

將範本交給自己的 startup 建立 entry 與 point。讀取範本本身不會建立 entry，完整格式見 [可攜 entry 範本](../../lib/zcu_tools/resources/entry/README.md#可攜-entry-範本)。已存在的 point 不會隨範本更新，setup 與 point 的關係見 [文件與 seed](../../lib/zcu_tools/resources/entry/README.md#文件與-seed)。

選取該 point 後，在 Inspect 的 Aliases 頁確認 `coupler` 解析為 `C1`，再 Apply 保存。修改已有 entry 時，也可以在 GUI 新增 `coupler` kind 的元件並填入這些欄位，再新增同一列 alias。表單只讀取選定 point 的表，不從元件名稱猜 alias。Alias 的保存與捕捉規則見 [context 與 snapshot](../../lib/zcu_tools/gui/session/README.md#contextsession-environment-與-run-snapshot)。

在自己的 `form.py` 使用 `from zcu_tools.cfg_model import Expr`，把對應的數值初值寫成 `Expr("$coupler.freq")`。例如 cfg 若宣告 `freq: float`，就在 `Form(..., tree={"freq": Expr("$coupler.freq")})` 放入該節點。普通字串 `"$coupler.freq"` 是 literal，不會當作 expression。Tree 與初值規則見 [Form 的形狀與初值](../../docs/adr/0073-measure-form-derivation.md#form-的形狀與初值)。

Open 或 Reset 會讀取 alias 指向的元件，將成功求值的初值存成 direct 數字。改 alias 後，既有 input 不會自動改值。Reset 才重新計算初值，缺必要來源時欄位保持未填且 Invalid。

Qubit 的 `resonator` 欄位填 resonator 元件名稱，`flux_source` 填供應 flux bias 的名稱，resonator 的 `amplifier` 填 amplifier 元件名稱。這些欄位存普通字串，不存 `$role.field`。完整對照見 [字串引用表](../components/README.md#角色與引用)。

例如 `qubit: Q1` 之後的 `resonator: $qubit.resonator`，會讀出 Q1 的 resonator 名稱。GUI softlink 只能引用前面的 alias 列，不會自動沿 RoleSpec 推論。Flux device alias 也可以指向 connected-device 名稱，不保證 entry 內有同名 component；實驗仍須取得已連線 driver 才能操作裝置。

```python
%load_ext autoreload
import os

import numpy as np

import matplotlib.pyplot as plt
from scqubits import Fluxonium, HilbertSpace, Oscillator
from joblib import Parallel, delayed
```

```python
r_f = 5.9
g = 0.1
params = (7.0, 1.1, 1.4)
flx = 0.0
```

```python
res_dim = 210
qub_cutoff = 40
qub_dim = 15


resonator = Oscillator(r_f, truncated_dim=res_dim)
fluxonium = Fluxonium(*params, flux=flx, cutoff=qub_cutoff, truncated_dim=qub_dim)
hilbertspace = HilbertSpace([fluxonium, resonator])
hilbertspace.add_interaction(
    g=g, op1=fluxonium.n_operator, op2=resonator.creation_operator, add_hc=True
)
hilbertspace.generate_lookup(ordering="LX")
```

```python
bare_esys_dag_array = np.array(
    [
        np.sqrt(j) * hilbertspace.bare_productstate((j, m)).dag().full()
        for j in range(qub_dim)
        for m in range(res_dim)
    ]
)
```

```python
os.makedirs("../../result/DesignR59/", exist_ok=True)
os.makedirs("../../result/DesignR59/image", exist_ok=True)
```

```python
params = (7.0, 1.1, 1.4)
r_f = 5.9
g = 0.1

```

```python
branchs = list(range(15))
populations = branch_population(hilbertspace, branchs, upto=100)
```

```python
for b in branchs:
    pop_b = populations[b]
    if np.ptp(pop_b) > 1.0:
        color = None
        label = f"Branch {b}"
    else:
        color = "lightgrey"
        label = None
    plt.plot(populations[b], label=label, color=color)
plt.legend()
plt.grid()
plt.savefig("../../result/DesignR59/image/int_branch_analysis.png")
plt.show()
```

```python
flxs = np.linspace(0, 0.5, 1001)
branchs = [0, 1]
```

```python
populations = branch_population_over_flux(
    flxs,
    params,
    r_f,
    qub_dim,
    qub_cutoff,
    res_dim,
    g,
    upto=res_dim - 10,
    branchs=branchs,
    batch_size=30,
)
```

```python
ground_populations = populations[:, 0, :]
excited_populations = populations[:, 1, :]

# calculate the critical photon number
ground_cn = np.argmax(ground_populations >= 2, axis=1)
ground_cn[ground_cn == 0] = ground_populations.shape[1] - 1
excited_cn = np.argmax(excited_populations >= 3, axis=1)
excited_cn[excited_cn == 0] = excited_populations.shape[1] - 1
```

```python
# plot the critical photon number as a function of flux
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))

df = flxs[1] - flxs[0]
ax1.imshow(
    ground_populations.T,
    interpolation="none",
    aspect="auto",
    origin="lower",
    extent=[flxs[0]-0.5*df, flxs[-1] + 0.5*df, 0, populations.shape[2]],
)
ax1.plot(flxs, ground_cn, label="ground", marker=".", color="r")
ax1.set_title("Ground state")
ax2.imshow(
    excited_populations.T,
    interpolation="none",
    aspect="auto",
    origin="lower",
    extent=[flxs[0]-0.5*df, flxs[-1] + 0.5*df, 0, populations.shape[2]],
)
ax2.plot(flxs, excited_cn, label="excited", marker=".", color="r")
ax2.set_title("Excited state")

ax1.legend()
ax2.legend()
fig.savefig("../../result/DesignR59/image/branch_analysis.png")
```

```python
np.savez_compressed("../../result/DesignR59/branch_analysis.npz", flxs=flxs, branchs=branchs, populations=populations)
```

```python

```

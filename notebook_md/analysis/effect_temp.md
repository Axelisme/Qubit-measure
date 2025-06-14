---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.13.2
---

```python
import os

%matplotlib widget
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import scipy.special as sp
import scipy.constants as sc
```

```python
os.makedirs("../../result/Eff_T", exist_ok=True)
```

```python
s_f = 7.48675e9  # GHz
R = 50  # attenuator impedence
log_fpts = np.linspace(6, 11, 1000)  # 1MHz to 100GHz
fpts = np.power(10, log_fpts)

att_4K = 20  # dB
att_100mK = 20  # dB
att_20mK = 20  # dB

length_in_4K = 0.2  # m
length_in_100mK = 0.2  # m
length_in_20mK = 0.2  # m
```

```python
# use attenuation at 4K as reference
# Reference: http://www.coax.co.jp/en/product/sc/086-50-cn-cn.html
SC086_4K_att_tb = [  # (Hz, dB/m)
    (0.5e9, 4.1),
    (1.0e9, 5.7),
    (5.0e9, 12.8),
    (10.0e9, 18.1),
    (20.0e9, 25.7),
]
SC086_att = np.interp(fpts, *zip(*SC086_4K_att_tb))

# for freq out of data table range, use A ~ sqrt(f)
first_f, first_att = SC086_4K_att_tb[0]
last_f, last_att = SC086_4K_att_tb[-1]
SC086_att[fpts < first_f] = first_att * np.sqrt(fpts[fpts < first_f] / first_f)
# SC086_att[fpts > last_f] = last_att * np.sqrt(fpts[fpts > last_f] / last_f)


plt.figure()
plt.plot(fpts, SC086_att, label="4K SC086")
plt.axvline(s_f, color="red", label="s_f")
plt.xscale("log")
plt.title("SC086 Cable Attenuation")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Cable Attenuation [dB/m]")
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()
```

```python
A_in_4K = att_4K + SC086_att * length_in_4K
A_in_100mK = att_100mK + SC086_att * length_in_100mK
A_in_20mK = att_20mK + SC086_att * length_in_20mK

A_300K = A_in_4K + A_in_100mK + A_in_20mK
A_4K = A_in_100mK + A_in_20mK
A_100mK = A_in_20mK
A_20mK = 0
```

```python
def log_expm1(x):  # avoid overflow in expm1 for very large x
    result = np.copy(x)

    mask = x < 20
    result[mask] = np.log(np.expm1(x[mask]))
    result[~mask] += np.log1p(-np.exp(-x[~mask]))

    return result


# PSD = 4kTR * (hf / kT) / expm1(hf / kT)
def logPSD(log_fpt, R, T):
    log_n = np.log10(sc.h / (sc.k * T)) + log_fpt
    return np.log10(4 * sc.k * T * R) + log_n - log_expm1(10**log_n) / np.log(10)


def find_eff_T(fpt, R, fpts, logSvv_total):
    # first use interpolation to find the effective PSD at the given frequency
    # then use opt.bisect to find the effective temperature
    logSvv = np.interp(fpt, fpts, logSvv_total)
    return opt.bisect(lambda T: logPSD(np.log10(fpt), R, T) - logSvv, 1e-6, 1e3)


def photonNum(T, f):
    return 1 / (np.exp((sc.h * f) / (sc.k * T)) - 1)
```

```python
fig, ax = plt.subplots(figsize=(9, 6))

logSvv_300K = logPSD(log_fpts, R, 300)
logSvv_4K = logPSD(log_fpts, R, 4)
logSvv_100mK = logPSD(log_fpts, R, 0.1)
logSvv_20mK = logPSD(log_fpts, R, 0.02)

logSvv_300K_attn = logSvv_300K - A_300K / 10
logSvv_4K_attn = logSvv_4K - A_4K / 10
logSvv_100mK_attn = logSvv_100mK - A_100mK / 10
logSvv_20mK_attn = logSvv_20mK - A_20mK / 10
logSvv_total = sp.logsumexp(
    [logSvv_300K_attn, logSvv_4K_attn, logSvv_100mK_attn, logSvv_20mK_attn], axis=0
)

ax.plot(fpts, logSvv_300K, label="300K")
ax.plot(fpts, logSvv_4K, label="4K")
ax.plot(fpts, logSvv_100mK, label="100mK")
ax.plot(fpts, logSvv_20mK, label="20mK")
ax.plot(fpts, logSvv_total, label="Effective")

if s_f is not None:
    eff_T = find_eff_T(s_f, R, fpts, logSvv_total)

    logSvv_eff = logPSD(log_fpts, R, eff_T)
    ax.vlines(
        s_f,
        -50,
        -10,
        colors="k",
        linestyles="dashed",
        label=f"freq = {s_f * 1e-9:.2f}GHz",
    )
    ax.plot(fpts, logSvv_eff, label=f"T_eff = {eff_T * 1e3:.1f}mK", linestyle="dashed")

    photonN = photonNum(eff_T, s_f)
    ax.set_title(f"T_eff = {eff_T * 1e3:.1f}mK, n_photon = {photonN:.3g}")


ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("PSD [V^2/Hz]")
ax.set_xscale("log")
ax.set_xlim(fpts[0], fpts[-1])
ax.set_ylim(-30.5, -17.5)
fig.legend(bbox_to_anchor=(1.00, 0.7))
fig.tight_layout()
fig.subplots_adjust(right=0.75)

fig.savefig(f"../../result/Eff_T/{s_f * 1e-9:.3f}GHz_01.png")

plt.show()
```

```python
t_fpts = np.linspace(10e6, 10e9, 1000)

T_effs = np.array([find_eff_T(f, R, fpts, logSvv_total) for f in t_fpts])

fig, ax = plt.subplots()
ax.plot(t_fpts, T_effs * 1e3)

ax.set_xscale("log")
ax.grid()

ax.set_title("Effective Temperature vs Frequency")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Effective Temp [mK]")

fig.savefig("../../result/Eff_T/Eff_T_vs_Freq.png")

plt.show()
```

```python
savepath = "../../result/Eff_T/Eff_T_vs_Freq.npz"
np.savez(savepath, fpts=t_fpts, T_effs=T_effs)
```

```python

```

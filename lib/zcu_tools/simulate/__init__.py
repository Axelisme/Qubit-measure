def mA2flx(mAs, mA_c, period):
    return (mAs - mA_c) / period + 0.5


def flx2mA(flxs, mA_c, period):
    return (flxs - 0.5) * period + mA_c

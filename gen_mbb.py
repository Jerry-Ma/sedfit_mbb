#! /usr/bin/env python


from mbb import MBB

import numpy as np

if __name__ == "__main__":
    # reff = 1.  # kpc
    lir = 11.
    tmbb = 30. # K
    beta = 1.5
    lam0 = 100.  # um

    z = 2.

    wl = np.linspace(100, 3000, 100)
    print(wl)

    flux = MBB.mbb_lir_z(wl, lir, tmbb, beta, lam0, z)

    f1100 = MBB.mbb_lir_z(1100., lir, tmbb, beta, lam0, z)

    import matplotlib.pyplot as plt

    plt.plot(wl, flux, label=f"MBB Log LIR/Lsun={lir} T={tmbb} K at z={z}")

    plt.axvline(
            1100., label=f'S_1100={f1100:.2f} mJy',
            linestyle='--', color='C2')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Flux (mJy)")
    plt.legend()
    plt.show()

"""
HFP Modeli v6 — Helisite Bozunmasına Dayalı Karanlık Enerji
===========================================================
- ρ_Λ(a) ∝ -dH_rel/dt,   H_rel(a) ∝ B(a)²
- Erken evren manyetik alanı: B(a) = B_start * (a / a_start)⁻²
- Friedman denklemi: H² = H₀² [ Ω_m a⁻³ + Ω_r a⁻⁴ + Ω_DE(a) ]
- Ω_DE(a) = Ω_DE0 * ( -dH_rel/dt / (-dH_rel/dt)|_{a=1} )
- Kod, modelin ΛCDM'den sapmasını sayısal olarak göstermektedir.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, cumulative_trapezoid
import os

class HFP_Parameters:
    def __init__(self, z_start=1100, B_start=None, H0=70.0, Om=0.3, Or=9e-5, Ode0=0.7):
        """
        z_start   : manyetik alanın tanımlandığı başlangıç redshift'i
        B_start   : z_start'daki manyetik alan şiddeti [Tesla]
        H0, Om, Or: kozmolojik parametreler
        Ode0      : bugünkü karanlık enerji yoğunluk parametresi
        """
        self.H0 = H0
        self.Omega_m = Om
        self.Omega_r = Or
        self.Ode0 = Ode0

        self.z_start = z_start
        self.a_start = 1 / (1 + z_start)
        if B_start is None:
            # Varsayılan: 1 nG bugün, flux korunumuyla geriye git
            B0 = 1e-9
            self.B_start = B0 * (1 + z_start)**2
        else:
            self.B_start = B_start

        # Diğer sabitler
        self.c = 299792.458   # km/s

    def B(self, a):
        """Manyetik alan: B(a) = B_start * (a/a_start)^{-2}"""
        return self.B_start * (a / self.a_start)**(-2)

    def H_rel(self, a):
        """Helisite yoğunluğu, manyetik alanın karesiyle orantılı"""
        return self.B(a)**2

    def dH_rel_dt(self, a, H):
        """dH_rel/dt = (dH_rel/da) * a * H(a)
           H(a) burada dışarıdan verilir (iteratif çözüm için)"""
        da = 1e-6 * a
        H_rel_a = self.H_rel(a)
        H_rel_ap = self.H_rel(a + da)
        dH_rel_da = (H_rel_ap - H_rel_a) / da
        return dH_rel_da * a * H

    def get_Omega_DE(self, a_array, H_array):
        """Karanlık enerji yoğunluk parametresini hesapla:
           Ω_DE(a) = Ω_DE0 * [ -dH_rel/dt(a) ] / [ -dH_rel/dt(1) ]"""
        # H_rel türevini hesapla (H_array kullanarak)
        dHdt_at_a = np.zeros_like(a_array)
        for i, ai in enumerate(a_array):
            dHdt_at_a[i] = self.dH_rel_dt(ai, H_array[i])
        # normalize et: a=1'deki değere böl
        idx1 = np.argmin(np.abs(a_array - 1.0))
        norm = dHdt_at_a[idx1]
        if norm == 0:
            norm = 1.0
        Omega_DE = self.Ode0 * (-dHdt_at_a) / (-norm)
        # negatif değerleri sıfırla (sayısal hata)
        Omega_DE = np.maximum(Omega_DE, 0)
        return Omega_DE

def solve_HFP(params, a_array):
    """Iteratif olarak H(a) ve Ω_DE(a) hesaplar."""
    # İlk tahmin: ΛCDM H(a)
    H_lcdm = params.H0 * np.sqrt(params.Omega_m * a_array**-3 +
                                 params.Omega_r * a_array**-4 +
                                 params.Ode0)
    # İlk Ω_DE için bu H kullan
    Omega_DE = params.get_Omega_DE(a_array, H_lcdm)
    # Yeni H²
    H2_new = params.H0**2 * (params.Omega_m * a_array**-3 +
                             params.Omega_r * a_array**-4 +
                             Omega_DE)
    H_new = np.sqrt(H2_new)
    # Bir kez daha iterasyon (genelde yeterli)
    Omega_DE = params.get_Omega_DE(a_array, H_new)
    H2_new = params.H0**2 * (params.Omega_m * a_array**-3 +
                             params.Omega_r * a_array**-4 +
                             Omega_DE)
    H_new = np.sqrt(H2_new)
    return H_new, Omega_DE

def growth_factor(a_array, H, Omega_m, H0):
    """Madde büyüme faktörü D(a) için ODE çözümü."""
    def odes(y, a):
        D, dD = y
        # H(a)'yı interpolate et
        H_a = np.interp(a, a_array, H)
        dHda = np.gradient(H, a_array)
        Hprime_over_H = dHda[np.argmin(np.abs(a_array - a))] / H_a
        src = 1.5 * Omega_m * H0**2 / (H_a**2 * a**3)
        ddD = -(3/a + Hprime_over_H) * dD + src * D
        return [dD, ddD]
    sol = odeint(odes, [a_array[0], 1.0], a_array, rtol=1e-9)
    D = sol[:,0]
    return D / D[-1]

def comoving_distance(z_array, H, a_array):
    """Vektörize eşzamanlı uzaklık."""
    z_fine = np.linspace(0, z_array.max(), 2000)
    a_fine = 1/(1+z_fine)
    H_interp = np.interp(a_fine, a_array, H)
    integrand = 299792.458 / H_interp   # c=299792 km/s
    chi = np.zeros_like(z_fine)
    chi[1:] = cumulative_trapezoid(integrand, z_fine)
    return np.interp(z_array, z_fine, chi)

def plot_results(params, a, H, Omega_DE):
    z = 1/a - 1
    w = np.zeros_like(a)
    for i, ai in enumerate(a):
        da = 1e-6*ai
        O1 = np.interp(ai, a, Omega_DE)
        O2 = np.interp(ai+da, a, Omega_DE)
        dOda = (O2 - O1)/da
        w[i] = -1 - (ai/(3*O1)) * dOda
    # Hubble karşılaştırması
    H_lcdm = params.H0 * np.sqrt(params.Omega_m * a**-3 + params.Omega_r * a**-4 + params.Ode0)

    plt.figure(figsize=(12, 10))
    plt.subplot(2,2,1)
    plt.semilogx(a, params.B(a), lw=2, color='steelblue')
    plt.xlabel('$a$'); plt.ylabel('$B(a)$ [T]')
    plt.title('Manyetik Alan')

    plt.subplot(2,2,2)
    plt.semilogx(a, w, lw=2, color='darkred')
    plt.axhline(-1, ls='--', color='k')
    plt.xlabel('$a$'); plt.ylabel('$w(a)$')
    plt.title('Denklem Durumu')
    plt.ylim([-1.05, -0.95])

    plt.subplot(2,2,3)
    plt.loglog(a, H/H[-1], lw=2, label='HFP')
    plt.loglog(a, H_lcdm/H_lcdm[-1], '--', lw=2, label='ΛCDM')
    plt.xlabel('$a$'); plt.ylabel('$H(a)/H_0$')
    plt.legend(); plt.title('Hubble Parametresi')

    plt.subplot(2,2,4)
    z_sn = np.linspace(0.01, 2.5, 200)
    mu_h = 5*np.log10((1+z_sn)*comoving_distance(z_sn, H, a)/3.086e13/10)
    mu_l = np.zeros_like(z_sn)
    for i, zi in enumerate(z_sn):
        chi_l, _ = cumulative_trapezoid(299792.458/H_lcdm[::-1], a[::-1], initial=0)
        chi_l = np.interp(zi, z_sn, chi_l)
        mu_l[i] = 5*np.log10((1+zi)*chi_l/3.086e13/10)
    plt.plot(z_sn, mu_h - mu_l, lw=2, color='teal')
    plt.axhline(0, ls='--', color='k')
    plt.xlabel('$z$'); plt.ylabel('$\Delta\mu$ [mag]')
    plt.title('Uzaklık Modülü Farkı')

    plt.tight_layout()
    plt.savefig('HFP_v6.png', dpi=150)
    plt.show()
    print("Grafik kaydedildi: HFP_v6.png")

def main():
    print("HFP Modeli v6 — Helisite Bozunmasına Dayalı Karanlık Enerji")
    params = HFP_Parameters(z_start=1100, B_start=None)  # B_start ~ 1.2e-6 T
    a = np.logspace(np.log10(params.a_start), 0, 1000)
    H, Omega_DE = solve_HFP(params, a)
    print(f"a_start = {params.a_start:.3e}, B_start = {params.B_start:.2e} T")
    print(f"Omega_DE(a=1) = {Omega_DE[-1]:.3f} (hedef: {params.Ode0})")
    plot_results(params, a, H, Omega_DE)

if __name__ == "__main__":
    main()

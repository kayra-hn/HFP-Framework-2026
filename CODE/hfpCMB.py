"""
HFP Modeli v5 — Erken Evren Manyetik Alanı Desteği
===================================================
- B(a) = B_start * (a / a_start)⁻², a_start = 1/(1+z_start)
- Erken evrendeki gözlemlenebilir sapmalar (CMB, 21 cm) için uygun.
- Bugünkü ΛCDM limitine otomatik geçiş.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, cumulative_trapezoid
import os

class HFP_Parameters:
    def __init__(self, z_start=1100, B_start=1e-9):  # B_start [Tesla], z_start CMB dönemi
        self.C          = 0.5
        self.alpha      = 0.1          # sabit, fiziksel anlamı KK ölçeğiyle ilgili
        self.z_start    = z_start
        self.B_start    = B_start
        self.a_start    = 1 / (1 + z_start)
        
        # Kozmolojik parametreler
        self.H0         = 70.0         # [km/s/Mpc]
        self.Omega_m    = 0.3
        self.Omega_r    = 9.0e-5
        self.Omega_DE   = 0.7
        self.c          = 299792.458   # [km/s]

    def B(self, a):
        """Erken evren başlangıçlı manyetik alan: B(a) = B_start * (a/a_start)^-2"""
        return self.B_start * (a / self.a_start)**(-2)

    def beta(self, a):
        denom = 1 + self.alpha * self.B(a)**2
        return self.C / denom

    def projection_factor(self, a):
        b1 = self.beta(1.0)
        return (1 + self.beta(a)**2) / (1 + b1**2)

    def Hubble(self, a):
        E2 = (self.Omega_m * a**-3 + self.Omega_r * a**-4 
              + self.Omega_DE * self.projection_factor(a))
        return self.H0 * np.sqrt(E2)

    def effective_w(self, a, da=1e-6):
        """w(a) = -1 - (a/3f) df/da (sayısal türev, stabilite için)"""
        f   = self.projection_factor(a)
        f2  = self.projection_factor(a + da)
        dfda= (f2 - f) / da
        return -1 - (a / (3*f)) * dfda

    def growth_factor(self, a_array):
        def odes(y, a):
            D, dD = y
            H = self.Hubble(a)
            # dHda sayısal türev
            dHda = (self.Hubble(a*1.001) - H) / (0.001*a)
            Hprime_over_H = dHda / H
            source = 1.5 * self.Omega_m * self.H0**2 / (H**2 * a**3)
            ddD = -(3/a + Hprime_over_H) * dD + source * D
            return [dD, ddD]
        a0 = a_array[0]
        sol = odeint(odes, [a0, 1.0], a_array, rtol=1e-9)
        D = sol[:,0]
        return D / D[-1]

# ------------------------------------------------------------
# Gözlemsel uzaklıklar (vektörize)
# ------------------------------------------------------------
def comoving_distance(z_array, params):
    z_fine = np.linspace(0, z_array.max(), 2000)
    a_fine = 1/(1+z_fine)
    integrand = params.c / params.Hubble(a_fine)
    chi = np.zeros_like(z_fine)
    chi[1:] = cumulative_trapezoid(integrand, z_fine)
    return np.interp(z_array, z_fine, chi)

def distance_modulus(z_array, params):
    chi = comoving_distance(z_array, params)
    dL_pc = (1+z_array) * chi / 3.086e13
    return 5 * np.log10(dL_pc / 10)

def distance_modulus_LCDM(z_array, H0=70, Om=0.3, Or=9e-5, OL=0.7):
    def H_LCDM(a):
        return H0 * np.sqrt(Om*a**-3 + Or*a**-4 + OL)
    z_fine = np.linspace(0, z_array.max(), 2000)
    a_fine = 1/(1+z_fine)
    integrand = 299792.458 / H_LCDM(a_fine)
    chi = np.zeros_like(z_fine)
    chi[1:] = cumulative_trapezoid(integrand, z_fine)
    chi_interp = np.interp(z_array, z_fine, chi)
    dL_pc = (1+z_array) * chi_interp / 3.086e13
    return 5 * np.log10(dL_pc / 10)

# ------------------------------------------------------------
# Görselleştirme
# ------------------------------------------------------------
def plot_HFP(params, out_dir="."):
    a = np.logspace(np.log10(params.a_start), 0, 500)
    z = 1/a - 1
    w_arr = np.array([params.effective_w(ai) for ai in a])
    H_hfp = params.Hubble(a)
    H_lcdm = 70 * np.sqrt(0.3*a**-3 + 9e-5*a**-4 + 0.7)
    
    fig, axes = plt.subplots(2,2, figsize=(12,10))
    ax = axes[0,0]
    ax.semilogx(a, params.B(a), lw=2, color='steelblue')
    ax.set_title('Manyetik Alan $B(a)$')
    ax.set_xlabel('$a$'); ax.set_ylabel('$B$ [T]')
    
    ax = axes[0,1]
    ax.semilogx(a, w_arr, lw=2, color='darkred')
    ax.axhline(-1, ls='--', color='k')
    ax.set_title('Denklem Durumu $w(a)$')
    ax.set_xlabel('$a$'); ax.set_ylabel('$w(a)$')
    ax.set_ylim([-1.02, -0.98])
    
    ax = axes[1,0]
    ax.loglog(a, H_hfp/H_hfp[-1], lw=2, label='HFP')
    ax.loglog(a, H_lcdm/H_lcdm[-1], ls='--', lw=2, label='ΛCDM')
    ax.set_title('Hubble $H(a)/H_0$'); ax.set_xlabel('$a$'); ax.legend()
    
    ax = axes[1,1]
    z_sn = np.linspace(0.01, 2.5, 100)
    mu_h = distance_modulus(z_sn, params)
    mu_l = distance_modulus_LCDM(z_sn)
    ax.plot(z_sn, mu_h - mu_l, lw=2, color='teal')
    ax.axhline(0, ls='--', color='k')
    ax.set_title(r'$\Delta\mu = \mu_{\rm HFP} - \mu_{\Lambda\rm CDM}$ [mag]')
    ax.set_xlabel('$z$'); ax.set_ylabel('Δμ [mag]')
    
    plt.tight_layout()
    path = os.path.join(out_dir, 'HFP_v5.png')
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Grafik: {path}")

# ------------------------------------------------------------
if __name__ == "__main__":
    # Erken evren başlangıcı: z=1100 (CMB), B_start = 1 nG * (1+z)^2 ≈ 1.2e-6 T
    params = HFP_Parameters(z_start=1100, B_start=1e-9 * (1101)**2)
    print(f"a_start = {params.a_start:.3e}, B_start = {params.B_start:.2e} T")
    plot_HFP(params)

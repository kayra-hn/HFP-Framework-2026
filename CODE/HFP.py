
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, odeint, cumulative_trapezoid

# ============================================================
# PARAMETRE SINIFI
# ============================================================

class HFP_Parameters:
    def __init__(self):
        self.C        = 0.5
        self.alpha    = 0.1
        self.B0       = 1.0e-9     # [Tesla]
        self.H0       = 70.0       # [km/s/Mpc]
        self.Omega_m  = 0.3
        self.Omega_r  = 9.0e-5
        self.Omega_DE = 0.7
        self.c        = 299792.458  # [km/s]

    def check_physics_consistency(self):
        regime = self.alpha * self.B0**2
        total  = self.Omega_m + self.Omega_r + self.Omega_DE
        print("\n[FİZİK TUTARLILIK KONTROLÜ]")
        print(f"  αB₀²  = {regime:.2e}  "
              f"({'<< 1 → ΛCDM ile özdeş' if regime < 1e-6 else '≥ 1 → anlamlı sapma'})")
        print(f"  ΣΩᵢ   = {total:.5f}  "
              f"({'düz evren ✓' if abs(total - 1) < 0.01 else 'UYARI: düzlük ihlali'})")
        # [#3] başlangıç koşulu notu
        a_eq = self.Omega_r / self.Omega_m
        print(f"  a_eq  = {a_eq:.2e}  "
              f"(a_start=1e-3 > a_eq → D≈a başlangıcı geçerli)")
        if regime < 1e-6:
            warnings.warn(
                f"\n  [FIZ-1] αB₀² = {regime:.2e} << 1.\n"
                "  Manyetik etki ihmal edilebilir → model ΛCDM ile özdeş.\n"
                "  Anlamlı sapma için α ~ 1/(B₀²) ≈ 1e18 gerekir.",
                UserWarning, stacklevel=2
            )
        print()


# ============================================================
# TEMEL FONKSİYONLAR
# ============================================================

def magnetic_field(a, B0):
    """B(a) = B₀ · a⁻²"""
    return B0 * a**(-2)


def beta(a, params):
    """β(a) = C / (1 + αB(a)²)"""
    denom = 1 + params.alpha * params.B0**2 * a**(-4)
    return params.C / denom


def _dbeta_da(a, params):
    """dβ/da — analitik türev"""
    d = 1 + params.alpha * params.B0**2 * a**(-4)
    return params.C * 4 * params.alpha * params.B0**2 * a**(-5) / d**2


def projection_factor(a, params):
    """f(a) = (1 + β²(a)) / (1 + β²(1))"""
    b1 = beta(1.0, params)
    return (1 + beta(a, params)**2) / (1 + b1**2)


def _df_da(a, params):
    """df/da — analitik türev"""
    b1 = beta(1.0, params)
    return 2 * beta(a, params) * _dbeta_da(a, params) / (1 + b1**2)


# ============================================================
# [#1][#2] ANALİTİK dH/da
# ============================================================

def dHubble_da(a, params):
    """
    dH/da analitik hesabı.

    H²(a) = H₀² [Ωₘ a⁻³ + Ωᵣ a⁻⁴ + Ω_DE f(a)]

    d(H²)/da = H₀² [-3Ωₘ a⁻⁴ - 4Ωᵣ a⁻⁵ + Ω_DE df/da]

    dH/da = d(H²)/da / (2H)
    """
    dE2_da = (- 3 * params.Omega_m  * a**(-4)
              - 4 * params.Omega_r  * a**(-5)
              + params.Omega_DE * _df_da(a, params))
    H = Hubble_HFP(a, params)
    return params.H0**2 * dE2_da / (2 * H)


def Hubble_HFP(a, params):
    """H(a) = H₀ √[Ωₘ a⁻³ + Ωᵣ a⁻⁴ + Ω_DE f(a)]"""
    E2 = (params.Omega_m  * a**(-3) +
          params.Omega_r  * a**(-4) +
          params.Omega_DE * projection_factor(a, params))
    return params.H0 * np.sqrt(E2)


def Hubble_LCDM(a, H0=70.0, Om=0.3, Or=9e-5, OL=0.7):
    return H0 * np.sqrt(Om * a**(-3) + Or * a**(-4) + OL)


# ============================================================
# DURUM DENKLEMİ
# ============================================================

def effective_w_exact(a, params):
    """
    w(a) = -1 - (a / 3f) · (df/da)   [tam analitik]
    """
    f   = projection_factor(a, params)
    dfd = _df_da(a, params)
    return -1 - (a / (3 * f)) * dfd


def w_paper_claim(z, eps=0.02):
    """
    [#4] Makalede iddia edilen form: w(z) = -1 + ε(1+z)^{-3/2}
    eps: belirsiz parametre, grafik birden fazla değeri gösteriyor.
    """
    return -1 + eps * (1 + z)**(-1.5)


# ============================================================
# [#5] VEKTÖRİZE UZAKLIK HESABI
# ============================================================

def comoving_distance_vec(z_array, params):
    """
    Eşzamanlı uzaklık χ(z) — cumulative_trapezoid ile O(n).
    Her z için ayrı quad() yerine tek geçişte hesaplar.
    """
    # Integrasyon noktaları: z=0'dan z_max'a kadar ince ızgara
    z_fine = np.linspace(0, z_array.max(), 2000)
    a_fine = 1.0 / (1.0 + z_fine)
    integrand = params.c / Hubble_HFP(a_fine, params)

    chi_fine = np.zeros_like(z_fine)
    chi_fine[1:] = cumulative_trapezoid(integrand, z_fine)

    # İstenen z noktalarına interpolasyon
    return np.interp(z_array, z_fine, chi_fine)


def luminosity_distance_vec(z_array, params):
    return (1 + z_array) * comoving_distance_vec(z_array, params)


def distance_modulus_vec(z_array, params):
    dL_pc = luminosity_distance_vec(z_array, params) / 3.086e13
    return 5 * np.log10(dL_pc / 10)


def distance_modulus_LCDM_vec(z_array, H0=70.0, Om=0.3, Or=9e-5, OL=0.7):
    z_fine    = np.linspace(0, z_array.max(), 2000)
    a_fine    = 1.0 / (1.0 + z_fine)
    integrand = 299792.458 / Hubble_LCDM(a_fine, H0, Om, Or, OL)
    chi_fine  = np.zeros_like(z_fine)
    chi_fine[1:] = cumulative_trapezoid(integrand, z_fine)
    chi       = np.interp(z_array, z_fine, chi_fine)
    dL_pc     = (1 + z_array) * chi / 3.086e13
    return 5 * np.log10(dL_pc / 10)


# ============================================================
# [#1][#2] BÜYÜME FAKTÖRÜ — ANALİTİK türevli ODE
# ============================================================

def growth_factor(params, a_array):
    """
    Madde yoğunluğu büyüme faktörü D(a).

    ODE: D'' + (3/a + H'/H) D' = (3/2) Ωₘ H₀² / (H² a³) · D

    [#1] dH/da artık analitik (dHubble_da). Küçük a'da sayısal
         kararsızlık yok.

    [#3] a_start = 1e-3 > a_eq ~ 3e-4: madde baskın bölgede
         D ≈ a, D' ≈ 1 başlangıç koşulu geçerli.
    """
    def odes(y, a):
        D, dD   = y
        H       = Hubble_HFP(a, params)
        dHda    = dHubble_da(a, params)          # [#1] analitik
        HpH     = dHda / H                       # H'/H
        source  = 1.5 * params.Omega_m * params.H0**2 / (H**2 * a**3)
        ddD     = -(3/a + HpH) * dD + source * D
        return [dD, ddD]

    a0  = a_array[0]
    y0  = [a0, 1.0]                              # D≈a, D'≈1
    sol = odeint(odes, y0, a_array, rtol=1e-9, atol=1e-11)
    D   = sol[:, 0]
    return D / D[-1]                             # D(a=1) = 1


# ============================================================
# GÖRSELLEŞTİRME
# ============================================================

def plot_HFP_results(params, output_dir="."):
    plt.rcParams.update({
        'font.size': 10, 'axes.titlesize': 10.5, 'axes.labelsize': 10,
        'figure.figsize': (17, 13), 'axes.grid': True, 'grid.alpha': 0.35,
        'font.family': 'serif'
    })
    clr = {'hfp': '#D62728', 'lcdm': '#1F77B4', 'aux': '#2CA02C',
           'paper': '#FF7F0E'}

    a   = np.logspace(-3, 0, 800)
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    fig.suptitle(
        'HFP Modeli v3 — Tüm Düzeltmeler Uygulandı\n'
        r'($\alpha B_0^2={:.1e}$)'.format(params.alpha * params.B0**2),
        fontsize=14, fontweight='bold'
    )

    # ── 1. Manyetik Alan ─────────────────────────────────────────
    ax = axes[0, 0]
    ax.loglog(a, magnetic_field(a, params.B0), color='steelblue', lw=2)
    ax.set_title(r'Manyetik Alan $B(a)\propto a^{-2}$')
    ax.set_xlabel('$a$'); ax.set_ylabel('$B(a)$ [T]')

    # ── 2. β(a) ──────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.semilogx(a, beta(a, params), color=clr['aux'], lw=2)
    ax.axhline(params.C, ls='--', color='k', alpha=0.5, label=r'$\beta\to C$')
    ax.set_title(r'Projeksiyon $\beta(a)$')
    ax.set_xlabel('$a$'); ax.set_ylabel(r'$\beta(a)$'); ax.legend(fontsize=8)

    # ── 3. w(a) ──────────────────────────────────────────────────
    ax = axes[0, 2]
    w_arr = effective_w_exact(a, params)
    ax.semilogx(a, w_arr, color=clr['hfp'], lw=2)
    ax.axhline(-1, color='k', ls='--', lw=1)
    ax.set_title('Denklem Durumu $w(a)$')
    ax.set_xlabel('$a$'); ax.set_ylabel('$w(a)$')
    ax.set_ylim([-1.0001, -0.9999])
    ax.annotate(r'$|\Delta w|\sim\mathcal{O}(10^{-20})$',
                xy=(0.05, 0.2), xycoords='axes fraction', fontsize=8,
                color='red', bbox=dict(fc='lightyellow', ec='orange', alpha=0.8))

    # ── 4. Hubble Karşılaştırması ────────────────────────────────
    ax = axes[1, 0]
    Hh = Hubble_HFP(a, params);   Hl = Hubble_LCDM(a)
    ax.loglog(a, Hh/Hh[-1], color=clr['hfp'],  lw=2.5, label='HFP')
    ax.loglog(a, Hl/Hl[-1], color=clr['lcdm'], ls='--', label='ΛCDM')
    ax.set_title('Hubble: HFP vs ΛCDM')
    ax.set_xlabel('$a$'); ax.set_ylabel('$H(a)/H_0$'); ax.legend()

    # ── 5. [#4] w(z): HFP vs makale iddiası (birden fazla ε) ─────
    ax   = axes[1, 1]
    z_p  = np.linspace(0, 3, 400)
    a_p  = 1 / (1 + z_p)
    w_hfp_z = effective_w_exact(a_p, params)
    ax.plot(z_p, w_hfp_z, color=clr['hfp'], lw=2.5, label='HFP gerçek $w(z)$')
    for eps, ls in [(0.005, ':'), (0.02, '-.'), (0.10, '--')]:
        ax.plot(z_p, w_paper_claim(z_p, eps), ls=ls, lw=1.5,
                label=rf'Makale iddiası $\varepsilon={eps}$')
    ax.axhline(-1, color='k', lw=0.8)
    ax.set_title(r'$w(z)$: HFP vs Makale İddiası (çoklu $\varepsilon$)')
    ax.set_xlabel('Redshift $z$'); ax.set_ylabel('$w(z)$')
    ax.legend(fontsize=7)

    # ── 6. [#1][#2] Büyüme Faktörü D(a) — analitik türevli ──────
    ax     = axes[1, 2]
    D_arr  = growth_factor(params, a)
    ax.semilogx(a, D_arr,     color='purple', lw=2, label='HFP $D(a)$')
    ax.semilogx(a, a / a[-1], color='gray',   ls=':', lw=1.5, label=r'$D\propto a$')
    ax.set_title('[#1] Büyüme Faktörü $D(a)$\n(analitik $dH/da$)')
    ax.set_xlabel('$a$'); ax.set_ylabel('$D(a)/D_0$'); ax.legend(fontsize=8)

    # ── 7. [#5] Süpernova Uzaklık Modülü — vektörize ─────────────
    ax    = axes[2, 0]
    z_sn  = np.linspace(0.01, 2.0, 200)         # artık 200 nokta, hızlı
    mu_h  = distance_modulus_vec(z_sn, params)
    mu_l  = distance_modulus_LCDM_vec(z_sn)
    ax.plot(z_sn, mu_h, color=clr['hfp'],  lw=2,   label='HFP')
    ax.plot(z_sn, mu_l, color=clr['lcdm'], ls='--', label='ΛCDM')
    ax.set_title('[#5] Süpernova $\\mu(z)$ — vektörize')
    ax.set_xlabel('$z$'); ax.set_ylabel(r'$\mu$ [mag]'); ax.legend()

    # ── 8. Δμ fark ───────────────────────────────────────────────
    ax = axes[2, 1]
    dmu = (mu_h - mu_l) * 1e3
    ax.plot(z_sn, dmu, color='teal', lw=2)
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set_title(r'$\Delta\mu_{\rm HFP-\Lambda CDM}$ [mmag]')
    ax.set_xlabel('$z$'); ax.set_ylabel(r'$\Delta\mu$ [×10⁻³ mag]')
    ax.annotate(f'max |Δμ| = {np.max(np.abs(dmu)):.4f} mmag',
                xy=(0.05, 0.85), xycoords='axes fraction', fontsize=9,
                bbox=dict(fc='lightcyan', ec='teal', alpha=0.8))

    # ── 9. [#6] Parametre Duyarlılığı — αB₀²≈1 görünür ──────────
    ax     = axes[2, 2]
    alphas = np.logspace(-3, 21, 100)            # üst sınır 1e21
    w_devs = []
    for alp in alphas:
        p_t       = HFP_Parameters()
        p_t.alpha = alp
        wv        = effective_w_exact(np.array([0.1, 0.5, 1.0]), p_t)
        w_devs.append(np.mean(np.abs(wv + 1)))
    ax.loglog(alphas, w_devs, color='royalblue', lw=2)
    ax.axvline(params.alpha, color='red',    ls='--', lw=1.5, label=f'Seçilen α={params.alpha}')
    critical = 1 / params.B0**2
    ax.axvline(critical,     color='orange', ls=':',  lw=1.5,
               label=r'$\alpha=1/B_0^2$' + f'\n(αB₀²=1)')
    ax.axhline(0.01,         color='green',  ls='-.', alpha=0.7, label='|w+1|=0.01')
    ax.set_title('[#6] Duyarlılık: αB₀²≈1 görünür')
    ax.set_xlabel(r'$\alpha$'); ax.legend(fontsize=7)

    plt.tight_layout()
    # [#9] Taşınabilir yol
    save_path = os.path.join(output_dir, 'hfp_model_v3.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Grafik kaydedildi: {save_path}")
    plt.show()
    return D_arr, mu_h, mu_l


# ============================================================
# ANA PROGRAM
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("HFP MODELİ v3 — TÜM DÜZELTMELERİ İÇERİR")
    print("=" * 70)

    params = HFP_Parameters()
    params.check_physics_consistency()

    # [#9] Çıktı dizini: script'in yanı
    out_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
    D_arr, mu_h, mu_l = plot_HFP_results(params, output_dir=out_dir)

    # ── Sayısal özet ────────────────────────────────────────────
    a_test = np.array([0.001, 0.1, 0.5, 1.0])

    print("\n[1] w(a) — TAM ANALİTİK TÜREV:")
    print(f"  {'a':<8} | {'w(a)':<22} | |w+1|")
    print("-" * 48)
    for ai in a_test:
        wi = effective_w_exact(ai, params)
        print(f"  {ai:<8.3f} | {wi:<22.12f} | {abs(wi+1):.4e}")

    print("\n[2] w(z): HFP vs Makale iddiası (ε=0.02):")
    z_chk = np.array([0, 0.5, 1.0, 2.0])
    print(f"  {'z':<6} | {'w_HFP':<20} | {'w_makale':<14} | Fark")
    print("-" * 58)
    for zi in z_chk:
        wh = effective_w_exact(np.array([1/(1+zi)]), params)[0]
        wp = w_paper_claim(zi, eps=0.02)
        print(f"  {zi:<6.2f} | {wh:<20.12f} | {wp:<14.6f} | {abs(wh-wp):.4e}")

    print("\n[3] Hubble sapması HFP vs ΛCDM:")
    print(f"  {'a':<8} | Sapma (%)")
    print("-" * 25)
    for ai in a_test:
        d = 100 * (Hubble_HFP(ai, params) - Hubble_LCDM(ai)) / Hubble_LCDM(ai)
        print(f"  {ai:<8.3f} | {d:+.8f}%")

    print("\n[4] Büyüme faktörü (analitik dH/da doğrulaması):")
    a_arr = np.logspace(-3, 0, 800)
    D     = growth_factor(params, a_arr)
    for ai, Di in zip([0.001, 0.1, 0.5, 1.0],
                      [D[0],
                       D[np.argmin(np.abs(a_arr - 0.1))],
                       D[np.argmin(np.abs(a_arr - 0.5))],
                       D[-1]]):
        print(f"  D(a={ai}) = {Di:.6f}")

    print("\n" + "=" * 70)
    print("ÖZET: mevcut parametrelerle HFP ≡ ΛCDM (sayısal olarak doğrulandı).")
    print("Anlamlı test için α ≈ 1/B₀² ≈ {:.1e} gerekir.".format(1/params.B0**2))
    print("=" * 70)

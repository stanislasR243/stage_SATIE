import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ==============================================================
#  PARAMÈTRES GÉNÉRAUX
# ==============================================================
A      = 1.0
phi    = 0.5
omega0_ref = 1.0  # Pulsation de référence
N      = 20
N_MC   = 1000
SNR_dB = 10.0

# Grille de recherche pour l'EMV (Périodogramme)
omega_grid = np.linspace(-np.pi, np.pi, 1000)

# ==============================================================
#  FONCTIONS
# ==============================================================

def generer_signal(N, omega_true, sigma):
    n     = np.arange(N)
    s     = A * np.exp(1j * (n * omega_true + phi))
    bruit = (sigma / np.sqrt(2)) * (np.random.randn(N) + 1j * np.random.randn(N))
    return s + bruit

def estimer_omega(x):
    n = np.arange(len(x))
    
    # -- Partie 1 : Recherche sur grille --
    # On cherche le meilleur point sur la grille de 1000 points
    vals = np.abs(np.dot(x, np.exp(-1j * np.outer(n, omega_grid))))**2
    idx_max = np.argmax(vals)
    omega_initial = omega_grid[idx_max]
    
    # -- Partie 2 : Raffinement local (Recherche continue) --
    # On définit la fonction de coût à MINIMISER (donc -Périodogramme)
    def objective(w):
        return -np.abs(np.sum(x * np.exp(-1j * n * w)))**2
    
    # On cherche le minimum de 'objective' autour de omega_initial
    # On définit un intervalle de recherche très petit (le pas de la grille)
    pas = (omega_grid[1] - omega_grid[0])
    res = minimize_scalar(objective, 
                          bounds=(omega_initial - pas, omega_initial + pas), 
                          method='bounded')
    
    return res.x

def BCR(N, sigma):
    return (sigma**2 / A**2) * 6 / (N * (N**2 - 1))

def monte_carlo(N, omega_true, sigma):
    estimations = []
    for _ in range(N_MC):
        x = generer_signal(N, omega_true, sigma)
        omega_hat = estimer_omega(x)
        estimations.append(omega_hat)

    estimations = np.array(estimations)
    biais       = np.mean(estimations) - omega_true
    variance    = np.var(estimations)
    mse         = np.mean((estimations - omega_true) ** 2)
    return biais, variance, mse

def tracer(ax, x, y, bcr, xlabel, ylabel, title):
    ax.semilogy(x, y,   'o-', label=ylabel)
    if np.isscalar(bcr):
        ax.axhline(y=bcr, color='r', linestyle='--', label='BCR')
    else:
        ax.semilogy(x, bcr, 'r--', label='BCR')
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which='both', alpha=0.4)

# ==============================================================
#  SWEEP 1 : SNR (N fixé à 20)
# ==============================================================
SNR_dB_values = np.linspace(-5, 25, 20)
biais_snr, variance_snr, mse_snr, bcr_snr = [], [], [], []

print("Sweep SNR...")
for snr_db in SNR_dB_values:
    snr   = 10 ** (snr_db / 10)
    sigma = np.sqrt(A**2 / snr)
    b, v, m = monte_carlo(N, omega0_ref, sigma)
    biais_snr.append(abs(b))
    variance_snr.append(v)
    mse_snr.append(m)
    bcr_snr.append(BCR(N, sigma))

# ==============================================================
#  SWEEP 2 : N (SNR fixé à 10 dB)
# ==============================================================
N_values = np.unique(np.geomspace(5, 200, 15).astype(int))
snr_ref_N  = 10 ** (0 / 10)
sigma_ref_N = np.sqrt(A**2 / snr_ref_N)
biais_N, variance_N, mse_N, bcr_N = [], [], [], []

print("Sweep N...")
for Nv in N_values:
    b, v, m = monte_carlo(Nv, omega0_ref, sigma_ref_N)
    biais_N.append(abs(b))
    variance_N.append(v)
    mse_N.append(m)
    bcr_N.append(BCR(Nv, sigma_ref_N))

# ==============================================================
#  SWEEP 3 : Pulsation Omega (SNR fixé à 5 dB, N=20)
# ==============================================================
# On crée une grille de valeurs de omega concentrée autour de omega0_ref
omega_values = np.concatenate([
    np.linspace(-np.pi, 0.8, 15),
    np.linspace(0.9, 1.1, 5), 
    np.linspace(1.2, np.pi, 15)
])
omega_values = np.sort(omega_values)
#########
snr_ref_W = 10 ** (5 / 10) # SNR fixé à 5 dB
sigma_ref_W = np.sqrt(A**2 / snr_ref_W)
biais_W, variance_W, mse_W, bcr_val = [], [], [], BCR(N, sigma_ref_W)

print("Sweep Omega (SNR = 5 dB)...")
for wv in omega_values:
    b, v, m = monte_carlo(N, wv, sigma_ref_W)
    biais_W.append(abs(b))
    variance_W.append(v)
    mse_W.append(m)

# ==============================================================
#  SAUVEGARDE DES FIGURES
# ==============================================================

# -- Figure 1 : SNR --
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
fig1.suptitle(f"Effet du SNR (N={N})", fontweight='bold')
axes1[0].plot(SNR_dB_values, biais_snr, 'o-')
axes1[0].set_title("|Biais|")
tracer(axes1[1], SNR_dB_values, variance_snr, bcr_snr, "SNR (dB)", "Variance", "Variance vs BCR")
tracer(axes1[2], SNR_dB_values, mse_snr, bcr_snr, "SNR (dB)", "MSE", "MSE vs BCR")
fig1.tight_layout()
fig1.savefig("resultats_snr.png", dpi=300)
plt.close(fig1)

# -- Figure 2 : N (Échelle Log-Log) --
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle(f"Effet de N (SNR={SNR_dB} dB)", fontweight='bold')

# Biais en log-log
axes2[0].loglog(N_values, biais_N, 'o-', label="|Biais|")
axes2[0].set_title("|Biais|")
axes2[0].set_xlabel("N")
axes2[0].grid(True, which='both', alpha=0.4)

# Variance vs BCR en log-log
tracer(axes2[1], N_values, variance_N, bcr_N, "N", "Variance", "Variance vs BCR")
axes2[1].set_xscale('log') # On force l'axe X en log car tracer() ne fait que semilogy

# MSE vs BCR en log-log
tracer(axes2[2], N_values, mse_N, bcr_N, "N", "MSE", "MSE vs BCR")
axes2[2].set_xscale('log') # On force l'axe X en log

fig2.tight_layout()
fig2.savefig("resultats_N_log.png", dpi=300)
plt.close(fig2)

# -- Figure 3 : Omega --
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle(f"Effet de la Pulsation (SNR=5 dB, N={N})", fontweight='bold')
axes3[0].plot(omega_values, biais_W, 'o-')
axes3[0].set_title("|Biais|")
axes3[0].set_xlabel(r"$\omega_0$")
tracer(axes3[1], omega_values, variance_W, bcr_val, r"$\omega_0$", "Variance", "Variance vs BCR")
tracer(axes3[2], omega_values, mse_W, bcr_val, r"$\omega_0$", "MSE", "MSE vs BCR")
fig3.tight_layout()
fig3.savefig("resultats_omega.png", dpi=300)
plt.close(fig3)

print("Toutes les simulations sont terminées. Images sauvegardées.")
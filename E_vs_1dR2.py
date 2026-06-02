# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:41:06 2026

@author: Miriam_Ucendo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# =========================
# Load Excel file
# =========================
file_path = "resultados.xlsx"
df = pd.read_excel(file_path)

E = df["E_PL (eV)"]
R_inf = df["R_infty (nm)"]
R_fin = df["R_finite (nm)"]

# =========================
# Transform variable
# =========================
X_inf = 1 / R_inf**2
X_fin = 1 / R_fin**2

# =========================
# Linear fits
# =========================
fit_inf, cov_inf = np.polyfit(X_inf, E, 1, cov=True)
fit_fin, cov_fin = np.polyfit(X_fin, E, 1, cov=True)

m_inf, b_inf = fit_inf
m_fin, b_fin = fit_fin

# Incertidumbres (1σ)
dm_inf, db_inf = np.sqrt(np.diag(cov_inf))
dm_fin, db_fin = np.sqrt(np.diag(cov_fin))

print("Infinite well:")
print(f"m = {m_inf:.4f} ± {dm_inf:.4f}")
print(f"b = {b_inf:.4f} ± {db_inf:.4f}")

print("\nFinite well:")
print(f"m = {m_fin:.4f} ± {dm_fin:.4f}")
print(f"b = {b_fin:.4f} ± {db_fin:.4f}")

X_line_inf = np.linspace(X_inf.min(), X_inf.max(), 300)
X_line_fin = np.linspace(X_fin.min(), X_fin.max(), 300)

E_fit_inf = np.polyval(fit_inf, X_line_inf)
E_fit_fin = np.polyval(fit_fin, X_line_fin)

# Predictions for R²
E_pred_inf = np.polyval(fit_inf, X_inf)
E_pred_fin = np.polyval(fit_fin, X_fin)

r2_inf = r2_score(E, E_pred_inf)
r2_fin = r2_score(E, E_pred_fin)

print(f"Infinite well:")
print(f"  E = {fit_inf[0]:.4f}(1/R²) + {fit_inf[1]:.4f}")
print(f"  R² = {r2_inf:.4f}")

print(f"\nFinite well:")
print(f"  E = {fit_fin[0]:.4f}(1/R²) + {fit_fin[1]:.4f}")
print(f"  R² = {r2_fin:.4f}")

# =========================
# Plot
# =========================
plt.figure(figsize=(7,5))

plt.scatter(X_inf, E, marker="o", label="Infinite well", color="blue")
plt.scatter(X_fin, E, marker="s", label="Finite well", color="orange")

plt.plot(X_line_inf, E_fit_inf, "--")
plt.plot(X_line_fin, E_fit_fin, "--")

plt.xlabel(r"$1/R^2$ (nm$^{-2}$)")
plt.ylabel(r"$E_{PL}$ (eV)")
plt.title(r"Emission Energy vs $1/R^2$")

plt.legend()
plt.tight_layout()
plt.show()
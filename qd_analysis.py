# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:28:20 2026

@author: Miriam_Ucendo
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# =========================
# LISTA DE ARCHIVOS Y RANGOS
# =========================

files = [
    "qd1_fort_orange_1412150E1.txt",
    "qd2_maple_red_orange_1412150E1.txt",
    "qd3_hops_yellow_2_1412150E1.txt",
    "qd4_lake_placid_blue_1412150E1.txt",
    "qd5_adirondack_green_1412150E1.txt",
    "qd6_libre_amarillo_chillon_1412150E1.txt",
    "qd7_rojo_liquido_1412150E1.txt"
]

titles = [
    "Fort Orange",
    "Maple Red Orange",
    "Hops Yellow",
    "Lake Placid Blue",
    "Adirondack Green",
    "Neon Yellow",
    "Liquid Red"
]

ranges = [
    (525, 660),
    (560, 660),
    (500, 590),
    (444, 520),
    (445, 560),
    (490, 620),
    (580, 665)
]

resultados=[]

# =========================
# CONSTANTES
# =========================
c = 3e8
h = 6.626e-34

# =========================
# FUNCIONES
# =========================

def load_data(file):
    x, y = [], []

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or not line[0].isdigit():
                continue

            line = line.replace(",", ".")
            parts = line.split(";")

            try:
                x.append(float(parts[0]))
                y.append(float(parts[1]))
            except:
                continue

    return np.array(x), np.array(y)


def interpolate_fwhm(x, y):
    y_max = np.max(y)
    half_max = y_max / 2

    def interp(x1, y1, x2, y2):
        return x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)

    lambdas = []

    for i in range(len(y) - 1):
        if (y[i] - half_max) * (y[i+1] - half_max) < 0:
            lambdas.append(interp(x[i], y[i], x[i+1], y[i+1]))

    if len(lambdas) < 2:
        return None, None, None, half_max

    l1, l2 = lambdas[0], lambdas[1]
    
    E1, E2 = 1240/l1, 1240/l2
    
    return l1, l2, E1, E2, l2 - l1, E2 - E1, half_max


def gaussian_fit(x, y):
    def gauss(x, A, mu, sigma):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

    A0 = np.max(y)
    mu0 = x[np.argmax(y)]
    sigma0 = 20

    try:
        popt, _ = curve_fit(gauss, x, y, p0=[A0, mu0, sigma0])
        A, mu, sigma = popt

        delta = np.sqrt(2 * sigma**2 * np.log(2))
        l1 = mu - delta
        l2 = mu + delta

        return l1, l2, l2 - l1, A, mu, sigma

    except:
        return None, None, None, None, None, None


# =========================
# LOOP PRINCIPAL
# =========================

for file, (xmin, xmax), title in zip(files, ranges, titles):

    print(f"\n==============================")
    print(f"Analizando: {file}")
    print(f"Rango: {xmin} - {xmax} nm")
    print(f"==============================")

    x, y = load_data(file)

    # cortar rango
    mask = (x >= xmin) & (x <= xmax)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        print("⚠️ No hay datos en este rango")
        continue

    # pico
    x_max = x[np.argmax(y)]
    y_max = np.max(y)

    print(f"Pico: x = {x_max:.2f} nm, y = {y_max:.2f}")

    # =========================
    # FWHM INTERPOLACIÓN
    # =========================
    l1_i, l2_i, E1_i, E2_i, fwhm_i, Delta_E_i, half_max = interpolate_fwhm(x, y)
    
    resultados.append([
    file,
    l1_i,
    l2_i,
    fwhm_i,
    E1_i,
    E2_i,
    Delta_E_i
    ])

    print("\nInterpolación:")
    print(f"lambda1 = {l1_i}")
    print(f"lambda2 = {l2_i}")
    print(f"E1 = {E1_i}")
    print(f"E2 = {E2_i}")
    print(f"Delta_E = {Delta_E_i}")
    print(f"FWHM = {fwhm_i}")
    
    df = pd.DataFrame(resultados, columns=[
    "archivo",
    "lambda1",
    "lambda2",
    "FWHM",
    "E1",
    "E2",
    "Delta_E"
    ])
    

    df.to_excel("resultados.xlsx", index=False)

    print("\n✔ Excel guardado como 'resultados.xlsx'")

    # =========================
    # FWHM GAUSSIANO
    # =========================
    l1_g, l2_g, fwhm_g, A, mu, sigma = gaussian_fit(x, y)

    print("\nGaussiano:")
    print(f"lambda1 = {l1_g}")
    print(f"lambda2 = {l2_g}")
    print(f"FWHM = {fwhm_g}")

    # =========================
    # PLOT
    # =========================
    plt.figure()

    plt.plot(x, y, label="Data", color="blue")

    # interpolación (negro)
    if l1_i is not None:
        plt.scatter([l1_i, l2_i], [half_max, half_max],
                    color="black", label="FWHM interp")

    # gaussiano (amarillo)
    if l1_g is not None:
        plt.scatter([l1_g, l2_g], [A/2, A/2],
                    color="red", edgecolors="black",
                    label="FWHM gauss")

        # curva ajustada
        x_fit = np.linspace(min(x), max(x), 1000)
        y_fit = A * np.exp(-(x_fit - mu)**2 / (2 * sigma**2))
        plt.plot(x_fit, y_fit, "--", label="Gaussian fit", color="orange")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (counts)")
    
    plt.title(title)
    plt.legend()

    plt.show()
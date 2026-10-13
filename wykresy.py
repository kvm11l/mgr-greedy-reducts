import matplotlib.pyplot as plt
import numpy as np

# konfiguracja ogólna wykresów
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# --- DANE Z LOGÓW ---
datasets = ['Mushroom', 'Cars', 'House-votes', 'Breast-cancer', 'Lymphography', 'Balance-scale']
x = np.arange(len(datasets))
width = 0.25  # szerokość słupka

# wykres 1: Długości
len_base = [4.0, 6.0, 10.7, 8.4, 6.0, 4.0]
len_base_err = [0.0, 0.0, 1.1, 0.49, 0.0, 0.0]
len_mod1 = [4.0, 6.0, 10.2, 7.9, 6.0, 4.0]
len_mod1_err = [0.0, 0.0, 0.87, 0.30, 0.0, 0.0]
len_mod2 = [4.0, 6.0, 10.6, 8.3, 6.0, 4.0]
len_mod2_err = [0.0, 0.0, 1.02, 0.46, 0.0, 0.0]

# wykres 2: Accuracy
acc_baseline = [1.000, 0.982, 0.957, 0.699, 0.888, 0.829]
acc_base = [1.000, 0.987, 0.932, 0.715, 0.812, 0.829]
acc_base_err = [0.000, 0.005, 0.052, 0.081, 0.114, 0.043]
acc_mod1 = [1.000, 0.987, 0.932, 0.726, 0.812, 0.829]
acc_mod1_err = [0.000, 0.005, 0.052, 0.083, 0.114, 0.043]
acc_mod2 = [1.000, 0.987, 0.932, 0.707, 0.812, 0.829]
acc_mod2_err = [0.000, 0.005, 0.052, 0.067, 0.114, 0.043]

# ==========================================
# WYKRES 1: DŁUGOŚĆ REDUKTÓW
# ==========================================
fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, len_base, width, yerr=len_base_err, label='Algorytm bazowy', color='#4c72b0', capsize=5)
rects2 = ax.bar(x, len_mod1, width, yerr=len_mod1_err, label='Modyfikacja 1 (Opt. wsteczna)', color='#dd8452', capsize=5)
rects3 = ax.bar(x + width, len_mod2, width, yerr=len_mod2_err, label='Modyfikacja 2 (Rozgałęzianie)', color='#55a868', capsize=5)

ax.set_ylabel('Średnia długość reduktu (liczba atrybutów)')
ax.set_title('Porównanie średniej długości reduktów dla poszczególnych algorytmów', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()
plt.tight_layout()
plt.savefig('wykres_dlugosci.png', dpi=300)
plt.close()

# ==========================================
# WYKRES 2: DOKŁADNOŚĆ (ACCURACY) + BASELINE
# ==========================================
fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, acc_base, width, yerr=acc_base_err, label='Algorytm bazowy', color='#4c72b0', capsize=5)
rects2 = ax.bar(x, acc_mod1, width, yerr=acc_mod1_err, label='Modyfikacja 1 (Opt. wsteczna)', color='#dd8452', capsize=5)
rects3 = ax.bar(x + width, acc_mod2, width, yerr=acc_mod2_err, label='Modyfikacja 2 (Rozgałęzianie)', color='#55a868', capsize=5)

# czerwona linia referencyjna (Baseline) nad każdym zbiorem osobno
for i in range(len(datasets)):
    if i == 0:
        ax.hlines(y=acc_baseline[i], xmin=i - width*1.5, xmax=i + width*1.5, color='red', linestyle='--', linewidth=2, label='Baseline (Pełen zbiór)')
    else:
        ax.hlines(y=acc_baseline[i], xmin=i - width*1.5, xmax=i + width*1.5, color='red', linestyle='--', linewidth=2)

ax.set_ylabel('Trafność klasyfikacji (Accuracy)')
ax.set_title('Jakość predykcyjna modeli w porównaniu do pełnej przestrzeni atrybutów', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
# ograniczenie osi Y dla lepszej czytelności różnic (np. od 0.4 do 1.1)
ax.set_ylim([0.5, 1.15]) 
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('wykres_dokladnosci.png', dpi=300)
plt.close()

print("Wykresy zostały wygenerowane i zapisane jako 'wykres_dlugosci.png' oraz 'wykres_dokladnosci.png'.")
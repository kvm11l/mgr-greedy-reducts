import subprocess
import time
import sys

# nazwy plików z funkcjami
skrypty = [
    "basic_greedy.py",
    "modified_greedy_1.py",
    "modified_greedy_2.py"
]

print("========== ROZPOCZĘCIE ZBIORCZEGO URUCHAMIANIA ALGORYTMÓW ==========")
start_total = time.time()

for skrypt in skrypty:
    print(f"\n---> Uruchamianie skryptu: {skrypt}...")
    start_script = time.time()
    
    try:
        # uruchomienie skryptu i oczekiwanie na jego pełne zakończenie
        # sys.executable gwarantuje, że użyty zostanie ten sam interpreter Pythona
        subprocess.run([sys.executable, skrypt], check=True)
        
        czas_script = time.time() - start_script
        print(f"---> Zakończono: {skrypt} (Czas: {czas_script / 60:.2f} minut)")
        
    except subprocess.CalledProcessError as e:
        print(f"BŁĄD: Skrypt {skrypt} przerwał działanie (kod błędu: {e.returncode}).")
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku {skrypt}. Sprawdź nazwy plików w liście 'skrypty'.")

czas_total = time.time() - start_total
print("\n========== WSZYSTKIE OBLICZENIA ZAKOŃCZONE ==========")
print(f"Całkowity czas działania pakietu: {czas_total / 60:.2f} minut ({czas_total / 3600:.2f} godzin).")
print("Szczegółowe wyniki z każdej iteracji zostały zapisane w dedykowanych plikach .txt.")
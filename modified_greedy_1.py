import os
import pandas as pd
import numpy as np
import time
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# logi
logger = logging.getLogger(__name__)

def get_discernibility_pairs(X, y):
    """
    Tworzy zbiór par obiektów (wierszy), które należą do różnych klas decyzyjnych.
    Reprezentuje to główny zbiór konfliktów P(T).
    
    Wszystkie przypadki których system musi się nauczyć odróżniać od siebie aby nie popełnić błędu (np. pacjent chory vs zdrowy).
    """
    pairs = set()
    y_vals = y.values
    n = len(y_vals)
    
    # przejście przez wszystkie możliwe kombinacje obiektów (każdy z każdym)
    for i in range(n):
        for j in range(i + 1, n):
            # jeśli obiekty mają różną klasę, tworzą parę konfliktową
            if y_vals[i] != y_vals[j]:
                pairs.add((i, j))
    return pairs

def get_distinguished_pairs(X, pairs, attr):
    """
    Zwraca zbiór par, które dany atrybut potrafi odseparować (przyjmuje dla nich różne wartości).
    
    Sprawdza, jak wiele konfliktów potrafi "rozwiązać" badana cecha.
    """
    distinguished = set()
    col_vals = X[attr].values
    
    # weryfikacja tylko par otrzymanych w argumencie 'pairs'
    for i, j in pairs:
        if col_vals[i] != col_vals[j]:
            distinguished.add((i, j))
    return distinguished

def basic_greedy_super_reduct(X, y, all_pairs, fold_num):
    """
    Faza 1: Podstawowy algorytm zachłanny rozwiązujący problem pokrycia zbioru.
    Generuje pierwotny super-redukt, który może zawierać cechy nadmiarowe.
    """
    Q = [] # tabela na wybrany wstępnie redukt
    U_P = set(all_pairs) # kopia wszystkich par do odseparowania
    A = list(X.columns) # lista wszystkich dostępnych atrybutów
    step = 1

    logger.info(f"    [Fold {fold_num}] --- Faza 1: Generowanie super-reduktu ---")
    logger.info(f"    [Fold {fold_num}] Do pokrycia: {len(U_P)} par.")

    # pętla działa do momentu odróżnienia wszystkich par
    while len(U_P) > 0:
        best_attr = None
        max_cover = -1
        best_cover_set = set()

        # szukany atrybutu, który na ten moment odróżnia najwięcej par
        for a in A:
            if a in Q:
                continue
            cover_set = get_distinguished_pairs(X, U_P, a)
            if len(cover_set) > max_cover:
                max_cover = len(cover_set)
                best_attr = a
                best_cover_set = cover_set

        # zabezpieczenie przed nierozróżnialnymi danymi
        if best_attr is None or max_cover == 0:
            logger.info(f"    [Fold {fold_num}] UWAGA: Brak możliwości dalszego rozróżniania par.")
            break

        # dodanie najlepszego w tej iteracji atrybutu i usunięcie pokrytych przez niego par
        Q.append(best_attr)
        U_P = U_P - best_cover_set
        
        logger.info(f"      Dodano atrybut '{best_attr}' (pokrył {max_cover} nowych par). Pozostało par: {len(U_P)}")
        step += 1

    return Q

def check_unseparated_pairs(X, all_pairs, subset):
    """
    Symuluje pokrycie par dla zadanego podzbioru atrybutów (subset).
    Zwraca liczbę par, które pozostały nierozróżnione.
    
    Funkcja kluczowa dla Fazy 2 - sprawdza, czy po wyrzuceniu jakiejś cechy
    system nadal potrafi bezbłędnie rozróżnić wszystkie przypadki.
    """
    unseparated = set(all_pairs)
    for attr in subset:
        distinguished = get_distinguished_pairs(X, unseparated, attr)
        unseparated = unseparated - distinguished
        # jeśli uda się rozróżnić wszystko przed sprawdzeniem wszystkich cech, przerwanie
        if len(unseparated) == 0:
            break
    # zwróc ile par NIE UDAŁO SIĘ rozróżnić
    return len(unseparated)

def backward_elimination_reduct(X, y, initial_reduct, all_pairs, fold_num):
    """
    Faza 2 (Modyfikacja 1): sekwencyjna eliminacja wsteczna, usuwanie wsteczne.
    Testowanie usunięcia każdej z cech super-reduktu. Jeśli po jej usunięciu
    zbiór nadal rozróżnia 100% par, cecha jest usuwana na stałe (uznana za nadmiarową).
    """
    Q_opt = list(initial_reduct)
    
    logger.info(f"    [Fold {fold_num}] --- Faza 2: Optymalizacja wsteczna (Modyfikacja 1) ---")
    logger.info(f"    [Fold {fold_num}] Weryfikacja niezbędności {len(initial_reduct)} cech...")

    # iterujemy po kolei przez każdą cechę w super-redukcie wygenerowanym w Fazie 1
    for attr in initial_reduct:
        # konstruowanie Q_temp poprzez tymczasowe usunięcie badanego atrybutu z listy
        Q_temp = [a for a in Q_opt if a != attr]
        
        # weryfikacja czy uszczuplony zbiór nadal posiada pełną zdolność rozróżniania
        unseparated_count = check_unseparated_pairs(X, all_pairs, Q_temp)
        
        # jeśli liczba nierozróżnionych par = 0, oznacza to że atrybut był zbędny
        if unseparated_count == 0:
            Q_opt = Q_temp
            logger.info(f"      [-] Usunięto: '{attr}' (był nadmiarowy, system zachowuje pełne pokrycie).")
        else:
            # jeśli straciliśmy zdolność rozróżniania atrybut musi zostać w redukcie
            logger.info(f"      [+] Zatrzymano: '{attr}' (niezbędny - bez niego {unseparated_count} par staje się nierozróżnialnych).")
            
    return Q_opt

def evaluate_mod1_cv(csv_path, target_col=None):
    # dynamiczna nazwa pliku logowania na podstawie nazwy pliku CSV
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    log_filename = f"log_mod1_greedy_{base_name}.txt"
    
    # konfiguracja loggera (zabezpieczenie przed dublowaniem wpisów przy uruchamianiu z run_all.py)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # uruchomienie licznika czasu do analizy wydajności
    start_time = time.time()
    
    logger.info(f"========== ROZPOCZĘCIE ANALIZY ==========")
    logger.info(f"Plik wejściowy: {csv_path}")
    logger.info(f"Wersja algorytmu: Modyfikacja 1 (Z optymalizacją wsteczną)")
    
    # wczytanie pliku CSV i oczyszczenie nazw kolumn
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    if target_col is None:
        target_col = df.columns[-1]

    # usuwanie metadanych i błędnych wierszy
    # pozbycie się klas, które występują rzadziej niż 10 razy tak by walidacja krzyżowa zadziałała poprawnie
    counts = df[target_col].value_counts()
    valid_classes = counts[counts >= 10].index
    df = df[df[target_col].isin(valid_classes)].copy()
    df = df.dropna()

    # kodowanie z wymuszonym utworzeniem nowej macierzy numerycznej
    # algorytmy wymagają liczb dlatego zamieniamy teksty na wartości numeryczne
    X_raw = df.drop(columns=[target_col])
    y_raw = df[target_col]

    # OrdinalEncoder koduje dane atrybutów warunkowych
    encoder_X = OrdinalEncoder()
    X_encoded_array = encoder_X.fit_transform(X_raw.astype(str))
    X = pd.DataFrame(X_encoded_array, columns=X_raw.columns, index=X_raw.index)

    # LabelEncoder koduje klasę decyzyjną
    encoder_y = LabelEncoder()
    y = pd.Series(encoder_y.fit_transform(y_raw.astype(str)), index=y_raw.index, name=y_raw.name)

    logger.info(f"Liczba poprawnych obiektów skierowanych do walidacji: {len(X)}")
    logger.info(f"Początkowa przestrzeń atrybutów warunkowych: {len(X.columns)}\n")

    # ustawienie 10-krotnej walidacji krzyżowej i klasyfikatora
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # listy do przechowywania wyników z 10 foldów
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    super_reduct_lengths = []
    opt_reduct_lengths = []
    baseline_accuracies = [] # dla pełnego zbioru atrybutów bez redukcji

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        logger.info(f"--- Rozpoczęcie Fold {fold}/10 ---")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Faza 0: obliczenie wartości referencyjnej (Baseline) na pełnym zbiorze atrybutów
        clf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_baseline.fit(X_train, y_train)
        preds_baseline = clf_baseline.predict(X_test)
        acc_baseline = accuracy_score(y_test, preds_baseline)
        baseline_accuracies.append(acc_baseline)

        # inicjalizacja głównego zbioru par P(T) DLA DANYCH TRENINGOWYCH
        all_pairs = get_discernibility_pairs(X_train, y_train)

        # Faza 1: zgrubne wyszukiwanie (Algorytm Zachłanny)
        super_reduct = basic_greedy_super_reduct(X_train, y_train, all_pairs, fold)
        super_reduct_lengths.append(len(super_reduct))

        # Faza 2: redukcja zbioru cech metodą eliminacji wstecznej
        opt_reduct = backward_elimination_reduct(X_train, y_train, super_reduct, all_pairs, fold)
        opt_reduct_lengths.append(len(opt_reduct))

        # zastosowanie zoptymalizowanego reduktu do ucięcia obu zbiorów danych
        X_train_red = X_train[opt_reduct]
        X_test_red = X_test[opt_reduct]

        # Faza 3: klasyfikacja przy użyciu Lasu Losowego
        clf.fit(X_train_red, y_train)
        preds = clf.predict(X_test_red)
        
        # obliczenie skuteczności (ile procent z wycinka testowego zgadł poprawnie) oraz miar dodatkowych
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='macro', zero_division=0)
        rec = recall_score(y_test, preds, average='macro', zero_division=0)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

        logger.info(f"    [ZAKOŃCZONO Fold {fold}] Redukcja cech: {len(super_reduct)} -> {len(opt_reduct)}")
        logger.info(f"    [ZAKOŃCZONO Fold {fold}] Redukt końcowy: {opt_reduct} | Accuracy (Redukt): {acc:.3f} | Accuracy (Baseline - pełne dane): {acc_baseline:.3f}\n")

    # obliczenie czasu działania po zamknięciu głównej pętli
    end_time = time.time()
    total_time = end_time - start_time

    # raport końcowy
    logger.info("========== PODSUMOWANIE EKSPERYMENTU ==========")
    logger.info(f"Wartość referencyjna (Baseline Accuracy - pełen zbiór): {np.mean(baseline_accuracies):.3f} ± {np.std(baseline_accuracies):.3f}")
    logger.info(f"Średnia długość super-reduktu (przed optymalizacją): {np.mean(super_reduct_lengths):.3f} ± {np.std(super_reduct_lengths):.3f}")
    logger.info(f"Średnia długość zoptymalizowanego reduktu: {np.mean(opt_reduct_lengths):.3f} ± {np.std(opt_reduct_lengths):.3f}")
    logger.info(f"--- Miary jakości dla modelu ze zredukowanymi atrybutami ---")
    logger.info(f"Średnia dokładność klasyfikatora (Accuracy): {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
    logger.info(f"Średnia precyzja (Precision): {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
    logger.info(f"Średnia czułość (Recall): {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
    logger.info(f"Średnia miara F1 (F1-score): {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
    logger.info(f"Całkowity czas obliczeń: {total_time:.3f} sekund ({total_time / 60:.3f} minut)")
    logger.info(f"Pełne logi z eksperymentu zapisano w pliku: {log_filename}")

if __name__ == '__main__':
    datasets = [
#        'data/modified_mushroom.csv',
        'data/modified_house-votes.csv',
        'data/modified_breast-cancer.csv',
        'data/modified_lymphography.csv',
        'data/modified_cars.csv',
        'data/balance-scale.csv'
    ]
    
    for dataset in datasets:
        print(f"\n---> Uruchamiam obliczenia dla: {dataset}")
        # jeśli zbiór balance-scale, kolumna decyzyjna 'class'
        if 'balance-scale' in dataset:
            evaluate_mod1_cv(dataset, target_col='class')
        # dla pozostałych zbiorów kolumna decyzyjna domyślnie ostatnia
        else:
            evaluate_mod1_cv(dataset)
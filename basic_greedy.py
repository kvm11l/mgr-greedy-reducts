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
    Tworzy zbiór par obiektów (wierszy) które należą do różnych klas decyzyjnych.
    Budowa głównego zbioru konfliktów P(T).
    
    Wszystkie przypadki których system musi się nauczyć odróżniać od siebie aby nie popełnić błędu
    """
    pairs = set()
    y_vals = y.values
    n = len(y_vals)
    
    # przejscie przez wszystkie możliwe kombinacje obiektów (każdy z każdym)
    for i in range(n):
        for j in range(i + 1, n):
            # jeśli obiekty mają różną klasę (jadalny vs trujący) tworzą parę konfliktową
            if y_vals[i] != y_vals[j]:
                pairs.add((i, j))
    return pairs

def get_distinguished_pairs(X, pairs, attr):
    """
    weryfikuje, które z przekazanych par obiektów posiadają różne wartości na badanym atrybucie warunkowym
    zwraca zbiór par rozróżnianych P(T, a)
    sprawdza, jak wiele konfliktów potrafi "rozwiązać" dana cecha
    """
    distinguished = set()
    col_vals = X[attr].values
    
    # przejście tylko przez te pary które wciąż wymagają rozróżnienia
    for i, j in pairs:
        # jeśli na danej cesze (attr) obiekty mają inną wartość, cecha ta je rozróżnia
        if col_vals[i] != col_vals[j]:
            distinguished.add((i, j))
    return distinguished

def basic_greedy_super_reduct(X, y, fold_num):
    """
    Podstawowy algorytm zachłanny rozwiązujący problem pokrycia zbioru.
    Iteracyjnie dobiera atrybut, który odseparowuje największą liczbę wciąż
    nierozróżnionych par konfliktowych (lokalne maksimum)
    """
    Q = [] # pusta lista do której dodawane będą najlepsze cechy (przyszły redukt)
    
    logger.info(f"    [Fold {fold_num}] Inicjalizacja zbioru konfliktów P(T)...")
    
    # generowanie wszystkich par obiektów z różnych klas decyzyjnych
    U_P = get_discernibility_pairs(X, y)
    initial_pairs = len(U_P)
    logger.info(f"    [Fold {fold_num}] Liczba par wymagających odseparowania: {initial_pairs}")
    
    # lista wszystkich dostępnych cech warunkowych w zbiorze
    A = list(X.columns)
    step = 1

    # główna pętla wykonuje się do momentu rozróżnienia wszystkich par (aż zbiór U_P będzie pusty)
    while len(U_P) > 0:
        best_attr = None
        max_cover = -1
        best_cover_set = set()

        # poszukiwanie lokalnego maksimum pokrycia w danej iteracji
        # pętla sprawdza "wynik punktowy" dla każdego dostępnego atrybutu
        for a in A:
            # pomiń atrybuty, które już wcześniej zostały dodane do reduktu
            if a in Q:
                continue
            
            # ile z pozostałych par zostanie rozróżnionych przez ten atrybut (symulacja)
            cover_set = get_distinguished_pairs(X, U_P, a)
            
            # jeśli wynik jest lepszy niż dotychczasowy rekord, zapisujemy ten atrybut jako lidera
            if len(cover_set) > max_cover:
                max_cover = len(cover_set)
                best_attr = a
                best_cover_set = cover_set

        # zabezpieczenie przed sprzecznymi danymi (sytuacja, gdy dwa obiekty z różnych klas 
        # mają identyczne wartości na WSZYSTKICH cechach - brak możliwości podziału)
        if best_attr is None or max_cover == 0:
            logger.info(f"    [Fold {fold_num}] UWAGA: Wykryto pary nierozróżnialne. Przerwanie pętli.")
            break

        # aktualizacja stanu: dodanie najlepszej cechy do reduktu
        Q.append(best_attr)
        
        # usunięcie ze zbioru U_P tych par, które właśnie zostały odseparowane
        U_P = U_P - best_cover_set
        
        logger.info(f"      Krok {step}: Wybrano atrybut '{best_attr}' (pokrył {max_cover} par). Pozostało par do pokrycia: {len(U_P)}")
        step += 1

    return Q

def evaluate_basic_cv(csv_path, target_col=None):
    # dynamiczna nazwa pliku logowania na podstawie nazwy pliku CSV
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    log_filename = f"log_basic_greedy_{base_name}.txt"
    
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

    # uruchomienie licznika czasu do analizy wydajności algorytmu
    start_time = time.time()
    
    logger.info(f"========== ROZPOCZĘCIE ANALIZY ==========")
    logger.info(f"Plik wejściowy: {csv_path}")
    logger.info(f"Wersja algorytmu: Podstawowy Algorytm Zachłanny")
    
    # wczytanie pliku CSV do struktury DataFrame (tabeli)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip() # zabezpieczenie przed spacjami w nazwach kolumn
    
    # jeśli brak kolumny decyzyjnej, domyślnie bierzemy ostatnią
    if target_col is None:
        target_col = df.columns[-1]

    # 1. usuwanie metadanych i błędnych wierszy
    # pozbycie się klas decyzyjnych, które występują rzadziej niż 10 razy, 
    # zakłóciłyby one 10-krotną walidację krzyżową (wymaga min. 10 obiektów)
    counts = df[target_col].value_counts()
    valid_classes = counts[counts >= 10].index
    df = df[df[target_col].isin(valid_classes)].copy()
    df = df.dropna() # usunięcie wierszy z brakującymi danymi (NaN)

    # 2. kodowanie z wymuszonym utworzeniem nowej macierzy numerycznej
    # modele (jak Las Losowy) wymagają danych liczbowych
    # teksty (np. 'red', 'blue') muszą zostać zamienione na liczby (np. 0, 1)
    X_raw = df.drop(columns=[target_col])
    y_raw = df[target_col]

    # ordinalEncoder rygorystycznie zamienia całą macierz atrybutów na liczby zmiennoprzecinkowe
    encoder_X = OrdinalEncoder()
    X_encoded_array = encoder_X.fit_transform(X_raw.astype(str))
    # ponowne złożenie tabeli zapobiega błędom typowania "object"
    X = pd.DataFrame(X_encoded_array, columns=X_raw.columns, index=X_raw.index)

    # labelEncoder zamienia tekstowe etykiety klas (np. 'jadalny', 'trujący') na numery (0, 1)
    encoder_y = LabelEncoder()
    y = pd.Series(encoder_y.fit_transform(y_raw.astype(str)), index=y_raw.index, name=y_raw.name)

    logger.info(f"Liczba poprawnych obiektów (wierszy) skierowanych do walidacji: {len(X)}")
    logger.info(f"Początkowa przestrzeń atrybutów warunkowych: {len(X.columns)}\n")

    # ustawienie walidacji krzyżowej (Stratified zapewnia równe proporcje klas we wszystkich foldach)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # deklaracja klasyfikatora lasu losowego ze stałym ziarnem losowości (powtarzalność eksperymentu)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    reduct_lengths = []
    baseline_accuracies = [] # dla pełnego zbioru atrybutów bez redukcji

    # 10-krotna walidacja: w każdej iteracji 90% danych to trening (X_train), a 10% to test (X_test)
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

        # Faza 1: wyznaczenie super-reduktu na danych trenujących
        # Algorytm nigdy nie widzi danych testowych przed klasyfikacją, zapobiega zjawisku wycieku danych (data leakage)
        reduct = basic_greedy_super_reduct(X_train, y_train, fold)
        reduct_lengths.append(len(reduct))

        # Faza 2: redukcja przestrzeni informacyjnej obu zbiorów
        # z matryc wyrzucane są wszystkie kolumny, których algorytm nie umieścił w redukcie
        X_train_red = X_train[reduct]
        X_test_red = X_test[reduct]

        # Faza 3: klasyfikacja przy użyciu lasu Losowego
        # uczymy model TYLKO na odnalezionych, kluczowych cechach
        clf.fit(X_train_red, y_train)
        # model przewiduje klasy dla ukrytego wycinka testowego
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

        logger.info(f"    [ZAKOŃCZONO Fold {fold}] Redukt końcowy: {reduct}")
        logger.info(f"    [ZAKOŃCZONO Fold {fold}] Długość: {len(reduct)} atrybutów | Accuracy (Redukt): {acc:.3f} | Accuracy (Baseline - pełne dane): {acc_baseline:.3f}\n")

    # obliczenie całkowitego czasu wykonania po zakończeniu wszystkich 10 iteracji
    end_time = time.time()
    total_time = end_time - start_time

    # raportowanie wyników zbiorczych z całego eksperymentu
    logger.info("========== PODSUMOWANIE EKSPERYMENTU ==========")
    logger.info(f"Wartość referencyjna (Baseline Accuracy - pełen zbiór): {np.mean(baseline_accuracies):.3f} ± {np.std(baseline_accuracies):.3f}")
    logger.info(f"Średnia długość super-reduktu: {np.mean(reduct_lengths):.3f} ± {np.std(reduct_lengths):.3f}")
    logger.info(f"--- Miary jakości dla modelu ze zredukowanymi atrybutami ---")
    logger.info(f"Średnia dokładność klasyfikatora (Accuracy): {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
    logger.info(f"Średnia precyzja (Precision): {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
    logger.info(f"Średnia czułość (Recall): {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
    logger.info(f"Średnia miara F1 (F1-score): {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
    logger.info(f"Całkowity czas obliczeń: {total_time:.3f} sekund ({total_time / 60:.3f} minut)")
    logger.info(f"Pełne logi z eksperymentu zapisano w pliku: {log_filename}")

# wejście
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
            evaluate_basic_cv(dataset, target_col='class')
        # dla pozostałych zbiorów kolumna decyzyjna domyślnie ostatnia
        else:
            evaluate_basic_cv(dataset)
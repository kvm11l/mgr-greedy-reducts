import os
import pandas as pd
import numpy as np
import time
import logging
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# zwiększenie limitu rekurencji dla rozbudowanych drzew rozgałęzień
# w modyfikacji rekurencja do badania różnych ścieżek wyboru
# domyślny limit w Pythonie to 1000, co przy dużej liczbie cech mogłoby wywołać błąd RecursionError
sys.setrecursionlimit(5000)

# logi
logger = logging.getLogger(__name__)

def get_discernibility_pairs(X, y):
    """
    Tworzy zbiór par wierszy o różnych decyzjach. 
    Tak samo jak w algorytmie bazowym - wszystkie pary obiektów, które system musi umieć rozróżnić.
    """
    pairs = set()
    y_vals = y.values
    n = len(y_vals)
    for i in range(n):
        for j in range(i + 1, n):
            if y_vals[i] != y_vals[j]:
                pairs.add((i, j))
    return pairs

def get_distinguished_pairs(X, pairs, attr):
    """
    Sprawdza, które pary ze zbioru głównego są rozróżniane przez konkretny atrybut.
    """
    distinguished = set()
    col_vals = X[attr].values
    for i, j in pairs:
        if col_vals[i] != col_vals[j]:
            distinguished.add((i, j))
    return distinguished

def filter_search_space(X, all_pairs, attributes, fold_num):
    """
    Faza 1: Wstępne oczyszczanie przestrzeni poszukiwań.
    Usuwa z pamięci atrybuty, których zbiory par rozróżnianych w całości zawierają się
    w zbiorach innych atrybutów (są słabsze lub stanowią duplikaty).

    Atrybut A jest słabszą wersją Atrybutu B (jest jego podzbiorem), dlatego możemy
    go wyrzucić na samym starcie, żeby nie spowalniał algorytmu w dalszej części.
    """
    logger.info(f"    [Fold {fold_num}] --- Faza 1: Wstępne oczyszczanie przestrzeni ---")
    
    # wstępne wyliczenie i zapamiętanie zbiorów rozróżnianych par dla każdej cechy, by uniknąć wielokrotnego przeliczania
    attr_pairs = {a: get_distinguished_pairs(X, all_pairs, a) for a in attributes}
    A_cand = list(attributes) # kopia listy atrybutów, z której będziemy usuwać słabsze atrybuty
    removed_attrs = []
    
    # podwójna pętla - porównujemy każdy atrybut z każdym innym
    for a_i in attributes:
        if a_i not in A_cand:
            continue
        for a_j in attributes:
            if a_i == a_j or a_j not in A_cand:
                continue
            
            # weryfikacja relacji zawierania się zbiorów P(T, a_i) ⊆ P(T, a_j)
            if attr_pairs[a_i].issubset(attr_pairs[a_j]):
                if attr_pairs[a_i] == attr_pairs[a_j]:
                    # usunięcie ścisłego duplikatu przy zachowaniu naturalnej kolejności
                    # jeśli dwa atrybuty są identyczne pod kątem rozróżniania jeden jest wyrzucamy
                    if attributes.index(a_i) > attributes.index(a_j):
                        A_cand.remove(a_i)
                        removed_attrs.append(a_i)
                        break
                else:
                    # usunięcie atrybutu bezwzględnie słabszego
                    A_cand.remove(a_i)
                    removed_attrs.append(a_i)
                    break
                    
    if removed_attrs:
        logger.info(f"      Usunięto {len(removed_attrs)} atrybutów jako podzbiory/duplikaty: {removed_attrs}")
    else:
        logger.info(f"      Brak atrybutów nadmiarowych do usunięcia na tym etapie.")
        
    # funkcja zwraca oczyszczoną listę kandydatów (A_cand) i ich wyniki
    return A_cand, attr_pairs

def find_shortest_reduct(Q_curr, U_P, A_cand, attr_pairs_cache, depth=0):
    """
    Faza 2: Główna funkcja rozgałęziająca (Branching).
    Rozwiązuje problem remisów poprzez systematyczne przeszukiwanie ścieżek.
    
    W podejściu zachłannym, w przypadku remisu (tyle samo pokrytych par) 
    wybierany był jeden atrybut (ten o najmniejszym indeksie). 
    W modyfikacji 2. algorytm tworzy niezależne gałęzie dla każdego z remisujących atrybutów, 
    doprowadza każdą z tych ścieżek do końca, a następnie zwraca tę, która  wygenerowała najkrótszy redukt.
    """
    if len(U_P) == 0:
        return Q_curr
        
    # korzystamy z globalnego cache (wyników z pruningu) aby nie przeliczać macierzy wielokrotnie
    covers = {a: len(attr_pairs_cache[a].intersection(U_P)) for a in A_cand}
    max_cover = max(covers.values()) if covers else 0
    
    # zabezpieczenie przed niespójnościami w danych
    if max_cover == 0:
        return Q_curr 
        
    # zbiór atrybutów remisujących - wyciągamy wszystkie, które osiągnęły "max_cover"
    best_attrs = [a for a, cov in covers.items() if cov == max_cover]
    
    # informacja o rozgałęzieniu (tylko na samej górze drzewa aby uniknąć tysięcy wierszy w pliku log.txt przy głębokim rozgałęzianiu)
    if depth == 0 and len(best_attrs) > 1:
        logger.info(f"      [!] Wykryto remis. Start niezależnych rozgałęzień dla atrybutów: {best_attrs}")
        
    global_best = None
    
    # przejscie przez KAŻDEGO kandydata z grupy remisujących
    for a_best in best_attrs:
        # usunięcie ze zbioru par pokrytych przez aktualnie testowaną cechę
        new_U_P = U_P - attr_pairs_cache[a_best]
        # tworzymy nową listę dostępnych atrybutów bez wybranego kandydata
        new_A_cand = [a for a in A_cand if a != a_best]
        
        # wywołanie rekurencyjne - funkcja uruchamia samą siebie dla nowej rzeczywistości
        result = find_shortest_reduct(Q_curr + [a_best], new_U_P, new_A_cand, attr_pairs_cache, depth + 1)
        
        if result is not None:
            # porównanie wyników z poszczególnych ścieżek
            # zapisujemy wynik tylko wtedy gdy ta ścieżka wygenerowała krótszy redukt
            if global_best is None or len(result) < len(global_best):
                global_best = result
                
    return global_best

def evaluate_mod2_cv(csv_path, target_col=None):
    # dynamiczna nazwa pliku logowania na podstawie nazwy pliku CSV
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    log_filename = f"log_mod2_greedy_{base_name}.txt"
    
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

    # uruchomienie licznika czasu
    start_time = time.time()
    
    logger.info(f"========== ROZPOCZĘCIE ANALIZY ==========")
    logger.info(f"Plik wejściowy: {csv_path}")
    logger.info(f"Wersja algorytmu: Modyfikacja 2 (Oczyszczanie przestrzeni + Rozgałęzianie)")
    
    # wczytanie pliku i czyszczenie nazw kolumn
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    if target_col is None:
        target_col = df.columns[-1]

    # usuwanie metadanych i błędnych wierszy
    counts = df[target_col].value_counts()
    valid_classes = counts[counts >= 10].index
    df = df[df[target_col].isin(valid_classes)].copy()
    df = df.dropna()

    # kodowanie z wymuszonym utworzeniem nowej macierzy numerycznej
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

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    reduct_lengths = []
    removed_counts = []
    baseline_accuracies = [] # dla pełnego zbioru atrybutów bez redukcji

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        logger.info(f"--- Rozpoczęcie Fold {fold}/10 ---")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Faza 0: Obliczenie wartości referencyjnej (Baseline) na pełnym zbiorze atrybutów
        clf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_baseline.fit(X_train, y_train)
        preds_baseline = clf_baseline.predict(X_test)
        acc_baseline = accuracy_score(y_test, preds_baseline)
        baseline_accuracies.append(acc_baseline)

        # inicjalizacja głównego zbioru par na odseparowanym zbiorze trenującym
        logger.info(f"    [Fold {fold}] Inicjalizacja zbioru konfliktów P(T)...")
        all_pairs = get_discernibility_pairs(X_train, y_train)

        # -------------------------------------------------------------
        # 2. Faza 1: Oczyszczanie przestrzeni przed wejściem do pętli
        # -------------------------------------------------------------
        attributes = list(X_train.columns)
        # otrzymujemy czystą listę (A_cand) i wyliczoną z góry macierz (cache)
        A_cand, attr_pairs_cache = filter_search_space(X_train, all_pairs, attributes, fold)
        
        # monitorowanie skuteczności oczyszczania (ile cech wyrzuciliśmy na starcie)
        removed_count = len(attributes) - len(A_cand)
        removed_counts.append(removed_count)

        # -------------------------------------------------------------
        # 3. Faza: Znajdowanie najkrótszego reduktu (Drzewo)
        # -------------------------------------------------------------
        logger.info(f"    [Fold {fold}] --- Faza 2: Algorytm Rozgałęziający ---")
        best_reduct = find_shortest_reduct([], all_pairs, A_cand, attr_pairs_cache, depth=0)
        
        # zabezpieczenie na wypadek ekstremalnie zepsutego pliku w celu uniknięcia błędu
        if best_reduct is None:
            best_reduct = list(X_train.columns)
            
        reduct_lengths.append(len(best_reduct))

        # redukcja zbioru trenującego i testowego z wykorzystaniem znalezionych cech
        X_train_red = X_train[best_reduct]
        X_test_red = X_test[best_reduct]

        # uruchomienie modelu predykcyjnego (Las Losowy)
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

        logger.info(f"    [ZAKOŃCZONO Fold {fold}] Redukt końcowy: {best_reduct}")
        logger.info(f"    [ZAKOŃCZONO Fold {fold}] Długość: {len(best_reduct)} cech | Accuracy (Redukt): {acc:.3f} | Accuracy (Baseline - pełne dane): {acc_baseline:.3f}\n")

    # obliczenie czasu wykonania
    end_time = time.time()
    total_time = end_time - start_time

    # raportowanie wyników zbiorczych do konsoli i pliku TXT
    logger.info("========== PODSUMOWANIE EKSPERYMENTU ==========")
    logger.info(f"Wartość referencyjna (Baseline Accuracy - pełen zbiór): {np.mean(baseline_accuracies):.3f} ± {np.std(baseline_accuracies):.3f}")
    logger.info(f"Średnia długość wygenerowanego reduktu: {np.mean(reduct_lengths):.3f} ± {np.std(reduct_lengths):.3f}")
    logger.info(f"Średnia liczba usuniętych atrybutów w fazie wstępnego oczyszczania: {np.mean(removed_counts):.3f}")
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
            evaluate_mod2_cv(dataset, target_col='class')
        # dla pozostałych zbiorów kolumna decyzyjna domyślnie ostatnia
        else:
            evaluate_mod2_cv(dataset)
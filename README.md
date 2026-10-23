# Selekcja cech z wykorzystaniem algorytmu zachłannego

Repozytorium zawiera implementację algorytmów, skrypty eksperymentalne oraz logi wynikowe opracowane w ramach pracy magisterskiej pt. *„Selekcja cech z wykorzystaniem algorytmu zachłannego”*. Projekt dotyczy redukcji wymiarowości (wyznaczania reduktów i super-reduktów) w tablicach decyzyjnych przy wykorzystaniu **Teorii Zbiorów Przybliżonych (ang. *Rough Set Theory*)**.

## Zaimplementowane algorytmy

W projekcie zaimplementowano i porównano trzy warianty zachłannego wyznaczania reduktów dla problemu pokrycia zbioru (*Set Cover Problem*):

1. **Podstawowy algorytm zachłanny (`basic_greedy.py`)** – klasyczna heurystyka zachłanna, która w każdej iteracji dołącza do reduktu atrybut rozróżniający największą liczbę dotychczas nieodseparowanych par obiektów z różnych klas decyzyjnych.
2. **Modyfikacja 1 – Optymalizacja wsteczna (`modified_greedy_1.py`)** – rozszerzenie algorytmu bazowego o fazę sekwencyjnej eliminacji wstecznej (*backward elimination*), która weryfikuje wyznaczony super-redukt i usuwa z niego atrybuty nadmiarowe bez utraty pełnej zdolności rozróżniania par.
3. **Modyfikacja 2 – Oczyszczanie przestrzeni i rozgałęzianie (`modified_greedy_2.py`)** – dwuetapowe rozszerzenie obejmujące wstępną filtrację przestrzeni poszukiwań (usuwanie duplikatów oraz atrybutów będących podzbiorami innych cech przed startem pętli głównej) oraz rekurencyjne badanie alternatywnych gałęzi w sytuacjach remisowych.

## Metodyka badawcza i zbiory danych

* **Zbiory danych (`data/`):** Sześć zbiorów z repozytorium UCI Machine Learning Repository: `Mushroom`, `Cars`, `House-votes`, `Breast-cancer`, `Lymphography` oraz `Balance-scale`.
* **Przygotowanie danych:** Usunięcie braków w danych, odfiltrowanie klas decyzyjnych o liczności poniżej 10 obiektów oraz kodowanie zmiennych nominalnych (`OrdinalEncoder`, `LabelEncoder`).
* **Walidacja:** 10-krotna walidacja krzyżowa (`StratifiedKFold`, `random_state=42`). Konstrukcja reduktów odbywa się wyłącznie na podzbiorach uczących, co eliminuje zjawisko wycieku danych (*data leakage*).
* **Ewaluacja:** Ocena wyznaczonych podzbiorów atrybutów przy użyciu klasyfikatora lasu losowego (`RandomForestClassifier`) w zestawieniu z modelem referencyjnym (*Baseline* – pełny zbiór cech). Mierzone parametry obejmują długość reduktu, `Accuracy`, `Precision`, `Recall`, `F1-score` (w wariancie `macro`) oraz czas wykonania.

## Struktura repozytorium

* `basic_greedy.py` – implementacja i ewaluacja podstawowego algorytmu zachłannego.
* `modified_greedy_1.py` – implementacja i ewaluacja Modyfikacji 1 (optymalizacja wsteczna).
* `modified_greedy_2.py` – implementacja i ewaluacja Modyfikacji 2 (oczyszczanie przestrzeni + rozgałęzianie).
* `wykresy.py` – skrypt generujący wykresy porównawcze (`wykres_dlugosci.png`, `wykres_dokladnosci.png`).
* `data/` – katalog ze zbiorami danych w formacie `.csv`.
* `log_*.txt` – szczegółowe logi z przebiegu 10-krotnej walidacji krzyżowej dla każdego wariantu i zbioru danych.
* `requirements.txt` – lista wymaganych bibliotek środowiska Python.

## Instalacja i uruchomienie

1. Sklonuj repozytorium i zainstaluj wymagane pakiety:
```bash
git clone [https://github.com/kvm11l/mgr-greedy-reducts.git](https://github.com/kvm11l/mgr-greedy-reducts.git)
cd mgr-greedy-reducts
pip install -r requirements.txt
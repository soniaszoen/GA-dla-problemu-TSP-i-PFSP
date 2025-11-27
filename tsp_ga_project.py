import random
import time
import json
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv

import random
import numpy as np


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def read_distance_matrix_excel(path):
    df = pd.read_excel(path, header=0, index_col=0)
    return df.values



def tour_length(tour, dist_matrix):
    """
    Oblicza długość zamkniętej trasy (suma odległości + powrót do startu).
    - tour: lista indeksów miast 0..n-1
    - dist_matrix: numpy array n x n
    """
    s = 0.0
    n = len(tour)
    for i in range(n):
        a = tour[i]
        b = tour[(i+1) % n]
        s += dist_matrix[a, b]
    return s

def random_tour(n):
    """Generuje losową permutację miast długości n."""
    t = list(range(n))
    random.shuffle(t)
    return t

# ------------------------------
# III. Local moves (ruchy lokalne)
# ------------------------------

def move_swap(tour):
    """Swap: zamiana dwóch losowych pozycji."""
    a,b = random.sample(range(len(tour)), 2)
    tr = tour.copy()
    tr[a], tr[b] = tr[b], tr[a]
    return tr

def move_two_opt(tour):
    """2-opt: odwrócenie losowego fragmentu."""
    n = len(tour)
    a,b = sorted(random.sample(range(n),2))
    tr = tour.copy()
    tr[a:b+1] = list(reversed(tr[a:b+1]))
    return tr

def move_insertion(tour):
    """Insertion: wycięcie jednego elementu i wstawienie w inne miejsce."""
    n = len(tour)
    a,b = random.sample(range(n), 2)
    tr = tour.copy()
    city = tr.pop(a)
    tr.insert(b, city)
    return tr

# Prosty lokalny improver (próbuje kilka losowych ruchów i zachowuje poprawę)
def local_improve(tour, dist_matrix, moves=['swap','two_opt','insertion'], tries=10):
    """
    Lokalna poprawa: próbuje losowych ruchów (swap/2opt/insertion) na danej trasie.
    Zwraca najlepszą znalezioną w próbach (lub oryginał).
    """
    best = tour.copy()
    best_len = tour_length(best, dist_matrix)
    for _ in range(tries):
        mv = random.choice(moves)
        if mv == 'swap':
            cand = move_swap(best)
        elif mv == 'two_opt':
            cand = move_two_opt(best)
        else:
            cand = move_insertion(best)
        l = tour_length(cand, dist_matrix)
        if l < best_len:
            best, best_len = cand, l
    return best

# ------------------------------
# IV. Crossover operators
# ------------------------------

# w rodzicach krzyżowaniu losujemy dwa punkty które będą punktem cięcia, środek jednego z nich wpada do drugiego, to co jest pomiędzy tymi punktami, wpada do potomka
# wielkość populacji pomiędzy każdymi pokoleniami musi być stała
# silna losowość, rozsądne może być aby 10% z popoulacji rodzicó (najlepszych) weszło do kolejnej populaci (z potomkami)
# 

# 
def pmx(parent1, parent2):
    """Partially Mapped Crossover (PMX)."""
    n = len(parent1)
    a,b = sorted(random.sample(range(n),2))
    child = [-1]*n
    # copy slice from p1
    child[a:b+1] = parent1[a:b+1]
    # mapping using parent2
    p2_pos = {val:i for i,val in enumerate(parent2)}
    for i in range(a,b+1):
        val = parent2[i]
        if val in child:
            continue
        cur = i
        mapped = val
        while True:
            val1 = parent1[cur]
            cur = p2_pos[val1]
            if child[cur] == -1:
                child[cur] = mapped
                break
    # fill remaining from parent2 order
    it = (v for v in parent2 if v not in child)
    for i in range(n):
        if child[i] == -1:
            child[i] = next(it)
    return child

def ox(parent1, parent2):
    """Order Crossover (OX)."""
    n = len(parent1)
    a,b = sorted(random.sample(range(n),2))
    child = [-1]*n
    child[a:b+1] = parent1[a:b+1]
    present = set(child[a:b+1])
    idx = (b+1) % n
    for v in parent2:
        if v in present:
            continue
        child[idx] = v
        idx = (idx+1) % n
    return child

def cycle_crossover(parent1, parent2):
    """Cycle Crossover (CX)."""
    n = len(parent1)
    child = [-1]*n
    remaining = set(range(n))
    pos1 = {val:i for i,val in enumerate(parent1)}
    cycle_num = 0
    while remaining:
        start = next(iter(remaining))
        idx = start
        indices = []
        while True:
            indices.append(idx)
            remaining.remove(idx)
            val = parent2[idx]
            idx = pos1[val]
            if idx == start:
                break
        if cycle_num % 2 == 0:
            for i in indices:
                child[i] = parent1[i]
        else:
            for i in indices:
                child[i] = parent2[i]
        cycle_num += 1
    return child

# ------------------------------
# V. Mutation
# ------------------------------

def mutate_swap(child):
    """Mutacja: swap na potomku."""
    return move_swap(child)

# ------------------------------
# VI. Selection methods (pracujemy na 'length' = mniejsze lepsze)
# ------------------------------

def select_tournament(population, lengths, k=3):
    """
    Tournament selection: wybiera k losowych i zwraca tego z najmniejszą długością.
    - population: lista permutacji
    - lengths: lista długości (mniejsze=lepsze)
    """
    idxs = random.sample(range(len(population)), k)
    best_idx = min(idxs, key=lambda i: lengths[i])
    return deepcopy(population[best_idx])

def select_roulette_from_length(population, lengths, eps=1e-9):
    """
    Roulette selection przystosowana do minimalizacji długości:
    wagi = max(lengths) - length + eps
    """
    arr = np.array(lengths, dtype=float)
    M = arr.max()
    weights = (M - arr) + eps
    s = weights.sum()
    if s <= 0:
        return deepcopy(random.choice(population))
    probs = weights / s
    idx = np.random.choice(len(population), p=probs)
    return deepcopy(population[idx])

def select_rank_from_length(population, lengths):
    """
    Rank selection: lepszy (mniejsza długość) ma wyższą rangę.
    """
    arr = np.array(lengths)
    order = np.argsort(arr)  # ascending
    ranks = np.empty_like(order)
    # assign ranks so best gets N, worst gets 1
    ranks[order] = np.arange(len(arr), 0, -1)
    probs = ranks / ranks.sum()
    idx = np.random.choice(len(population), p=probs)
    return deepcopy(population[idx])

# ------------------------------
# VII. GA main loop (run_ga)
# ------------------------------

def run_ga(dist, n_pop=100, n_gen=300, p_mut=0.05, p_cx=0.9,
           selection_method='tournament', crossover_method='pmx',
           local_search_tries=0, local_moves=['swap','two_opt','insertion'],
           elite=1, tournament_k=3, seed=None):

    n = dist.shape[0]

    # initial population
    population = [random_tour(n) for _ in range(n_pop)]
    history = []

    # map strings to functions
    crossover_map = {'pmx': pmx, 'ox': ox, 'cx': cycle_crossover}
    selection_map = {'tournament': select_tournament,
                     'roulette': select_roulette_from_length,
                     'rank': select_rank_from_length}

    crossover_fn = crossover_map[crossover_method]
    selection_fn = selection_map[selection_method]

    for gen in range(n_gen):
        # evaluate
        lengths = [tour_length(t, dist) for t in population]
        best_idx = int(np.argmin(lengths))
        best_len = lengths[best_idx]
        history.append(best_len)

        # prepare next generation
        new_pop = []
        # elitism: copy top 'elite'
        ranked = sorted(zip(lengths, population), key=lambda x: x[0])
        for i in range(min(elite, len(ranked))):
            new_pop.append(deepcopy(ranked[i][1]))

        while len(new_pop) < n_pop:
            # select parents
            if selection_method == 'tournament':
                p1 = selection_fn(population, lengths, tournament_k)
                p2 = selection_fn(population, lengths, tournament_k)
            else:
                p1 = selection_fn(population, lengths)
                p2 = selection_fn(population, lengths)
            # crossover
            if random.random() < p_cx:
                child = crossover_fn(p1, p2)
            else:
                child = deepcopy(p1)

            # mutation
            if random.random() < p_mut:
                child = mutate_swap(child)

            # local improvement (opcjonalnie)
            if local_search_tries > 0:
                child = local_improve(child, dist, moves=local_moves, tries=local_search_tries)

            new_pop.append(child)

        population = new_pop

    # final evaluation
    lengths = [tour_length(t, dist) for t in population]
    idx = int(np.argmin(lengths))
    return {'tour': population[idx], 'length': lengths[idx], 'history': history}

# ------------------------------
# VIII. Experiment runner (one-factor-at-a-time)
# ------------------------------

def generate_one_factor_grid(baseline, param_name, values):
    """
    Tworzy listę konfiguracji (słowników) zmieniając tylko param_name na kolejne values,
    reszta pozostaje zgodnie z baseline.
    """
    grid = []
    for v in values:
        cfg = baseline.copy()
        cfg[param_name] = v
        grid.append(cfg)
    return grid


# --------------------------------------------
# Run OFAT experiments, save results to Excel,
# generate boxplot, mean±std and time vs best scatter
# --------------------------------------------
import os
import time
import math
import random
import pandas as pd
import matplotlib.pyplot as plt

# Ustalony seed bazowy (możesz zmienić)
SEED = 42

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def run_experiments_to_excel_and_plots(dist, dataset_name,
                                       baseline_params, param_tests,
                                       repetitions=5,
                                       out_dir='results_excel'):
    """
    OFAT runner:
      - dist: macierz odległości (numpy array)
      - dataset_name: string, np. 'Dane_TSP_127'
      - baseline_params: dict z wartościami bazowymi
      - param_tests: dict param_name -> list(values)
      - repetitions: ile powtórzeń na konfigurację (tu: 5)
      - out_dir: katalog wyjściowy (zawiera plik Excel i wykresy)
    Zapisuje:
      - Excel: out_dir/<dataset_name>_results.xlsx (arkusz 'results')
      - Pliki PNG w: out_dir/plots/<dataset_name>/<param>/
    """
    _ensure_dir(out_dir)
    results_rows = []

    run_id = 0
    total_runs = sum(len(v) for v in param_tests.values()) * repetitions
    print(f"Starting OFAT for dataset {dataset_name}: {len(param_tests)} params, ~{total_runs} runs")

    for param_name, values in param_tests.items():
        print(f"\n>>> Testing parameter: {param_name} (values: {values})")
        for val in values:
            # konfiguracja dla tego eksperymentu
            cfg = baseline_params.copy()
            cfg[param_name] = val

            for r in range(repetitions):
                seed = SEED + run_id
                t0 = time.time()
                # uruchom GA - zakładam, że masz zdefiniowaną funkcję run_ga(dist, ...)
                out = run_ga(dist,
                             n_pop=cfg['n_pop'],
                             n_gen=cfg['n_gen'],
                             p_mut=cfg['p_mut'],
                             p_cx=cfg['p_cx'],
                             selection_method=cfg['selection_method'],
                             crossover_method=cfg['crossover_method'],
                             local_search_tries=cfg.get('local_search_tries', 0),
                             local_moves=cfg.get('local_moves', ['swap','two_opt','insertion']),
                             elite=cfg.get('elite', 1),
                             tournament_k=cfg.get('tournament_k', 3),
                             seed=seed)
                elapsed = time.time() - t0

                # przygotuj wiersz z wszystkimi parametrami + wynikami
                row = cfg.copy()
                # standardowe pola eksperymentalne
                row.update({
                    'dataset': dataset_name,
                    'param_name': param_name,
                    'param_value': val,
                    'repetition': r,
                    'seed': seed,
                    'best_length': out['length'],
                    'time_s': round(elapsed, 4)
                })
                results_rows.append(row)

                print(f" run {run_id:04d}/{total_runs-1} | {param_name}={val} | rep={r} | best={out['length']:.2f} time={elapsed:.1f}s")
                run_id += 1

    # Zapis do Excela
    df = pd.DataFrame(results_rows)
    excel_path = os.path.join(out_dir, f"{dataset_name}_results.xlsx")
    print("Saving Excel:", excel_path)
    # zapisujemy cały DataFrame do jednego arkusza
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='results')

    # Generacja wykresów
    plots_base = os.path.join(out_dir, 'plots', dataset_name)
    _ensure_dir(plots_base)
    generate_plots_from_df(df, plots_base, dataset_name)

    print("\nDone. Excel and plots saved.")
    return excel_path

def generate_plots_from_df(df, out_base, dataset_name):
    """
    Dla każdego parametru w df tworzy:
      - boxplot (best_length per param_value)
      - mean ± std (errorbar)
      - scatter time_s vs best_length (kolor=param_value label)
    Pliki zapisywane w: out_base/<param>/
    """
    params = df['param_name'].unique()
    for p in params:
        dfp = df[df['param_name'] == p].copy()
        if dfp.empty:
            continue
        out_dir = os.path.join(out_base, str(p))
        _ensure_dir(out_dir)

        # Upewnij się, że param_value jest traktowane jako string przy grupowaniu (łatwiej porządkować)
        dfp['param_value_str'] = dfp['param_value'].astype(str)

        # --- 1) BOXPLOT ---
        groups = [ grp['best_length'].values for _, grp in dfp.groupby('param_value_str') ]
        labels = [ str(k) for k, _ in dfp.groupby('param_value_str') ]
        plt.figure(figsize=(10,6))
        plt.boxplot(groups, labels=labels, showfliers=True)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('best_length')
        plt.title(f"{dataset_name} | {p} — boxplot best_length")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'boxplot_best_length.png'))
        plt.close()

        # --- 2) MEAN ± STD (errorbar) ---
        # spróbuj konwersji param_value na liczby, jeśli możliwe (do sortowania)
        try:
            dfp['param_value_num'] = pd.to_numeric(dfp['param_value'])
            stats = dfp.groupby('param_value_num')['best_length'].agg(['mean','std']).reset_index().sort_values('param_value_num')
            xs = stats['param_value_num'].values
            ys = stats['mean'].values
            yerr = stats['std'].values
            plt.figure(figsize=(8,5))
            plt.errorbar(xs, ys, yerr=yerr, fmt='-o', capsize=4)
            plt.xlabel(str(p))
            plt.ylabel('mean best_length ± std')
            plt.title(f"{dataset_name} | {p} — mean ± std")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'mean_std.png'))
            plt.close()
        except Exception:
            # categorical fallback
            stats = dfp.groupby('param_value_str')['best_length'].agg(['mean','std']).reset_index()
            labels = stats['param_value_str'].values
            xs = range(len(labels))
            plt.figure(figsize=(9,5))
            plt.errorbar(xs, stats['mean'].values, yerr=stats['std'].values, fmt='-o', capsize=4)
            plt.xticks(xs, labels, rotation=45, ha='right')
            plt.xlabel(str(p))
            plt.ylabel('mean best_length ± std')
            plt.title(f"{dataset_name} | {p} — mean ± std")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'mean_std.png'))
            plt.close()

        # --- 3) SCATTER time_s vs best_length colored by param_value ---
        plt.figure(figsize=(8,6))
        unique_vals = dfp['param_value_str'].unique()
        for val in unique_vals:
            sub = dfp[dfp['param_value_str']==val]
            plt.scatter(sub['time_s'], sub['best_length'], alpha=0.7, label=str(val))
        plt.xlabel('time_s')
        plt.ylabel('best_length')
        plt.title(f"{dataset_name} | {p} — time vs best_length")
        plt.legend(title=str(p), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'time_vs_best_scatter.png'))
        plt.close()

        print(f" Plots saved for param {p} -> {out_dir}")

# Lista Twoich plików TSP
files = [
    "Dane_TSP_127.xlsx",
    "Dane_TSP_76.xlsx",
    "Dane_TSP_48.xlsx"
]
 

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})  # unikamy warningów przy dużej liczby wykresów

def ensure_dir(path):
    """Upewnij się że katalog istnieje."""
    os.makedirs(path, exist_ok=True)

def load_results(csv_path):
    """Wczytuje CSV z wynikami do DataFrame."""
    df = pd.read_csv(csv_path)
    # upewnij się, że kolumny kluczowe istnieją
    required = {'param_name', 'param_value', 'repetition', 'best_length', 'time_s'}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Brakujące kolumny w CSV: {miss}")
    return df

def safe_str(x):
    """Konwertuje wartość parametru na bezpieczny string do nazwy pliku."""
    s = str(x)
    s = s.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")
    if len(s) > 80:
        s = s[:80]
    return s

def numeric_or_categorical(series):
    """Sprawdza czy seria jest numeryczna (można sortować numerycznie)"""
    try:
        pd.to_numeric(series)
        return True
    except:
        return False

# lista plików
files = [
    ("Dane_TSP_127.xlsx","Dane_TSP_127"),
    ("Dane_TSP_76.xlsx","Dane_TSP_76"),
    ("Dane_TSP_48.xlsx","Dane_TSP_48")
]

# parametry do przetestowania (zgodnie z Twoją prośbą)
param_tests = {
    'n_pop': [50, 100, 200, 400],
    'n_gen': [100, 300, 600, 1000],
    'p_mut': [0.01, 0.03, 0.07, 0.15],
    'p_cx' : [0.6, 0.75, 0.9, 1.0],
    'elite' : [1, 3, 5, 10]
}

# baseline (proponowany)
baseline = {
    'n_pop': 100,
    'n_gen': 300,
    'p_mut': 0.03,
    'p_cx' : 0.9,
    'selection_method': 'tournament',
    'crossover_method': 'pmx',
    'local_search_tries': 5,
    'local_moves': ['swap','two_opt','insertion'],
    'elite': 3,
    'tournament_k': 3
}

repetitions = 5  # dokładnie tyle chcesz

for path, dataset_name in files:
    if not os.path.exists(path):
        print(f"File not found: {path} - skipping")
        continue
    print(f"\n=== Running dataset {dataset_name} from file {path} ===")
    dist = read_distance_matrix_excel(path)  # użyj Twojej funkcji wczytującej
    out_excel = run_experiments_to_excel_and_plots(dist, dataset_name, baseline, param_tests, repetitions=repetitions, out_dir='results_excel')
    print("Saved Excel:", out_excel)


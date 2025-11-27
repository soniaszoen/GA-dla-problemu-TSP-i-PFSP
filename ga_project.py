import random
import time
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

plt.rcParams.update({'figure.max_open_warning': 0}) 

def read_distance_matrix_excel(path):
    df = pd.read_excel(path, header=0, index_col=0)
    return df.values


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def tour_length(tour, dist_matrix):
    s = 0.0
    n = len(tour)
    for i in range(n):
        a = tour[i]
        b = tour[(i+1) % n]
        s += dist_matrix[a, b]
    return s


def random_tour(n):
    t = list(range(n))
    random.shuffle(t)
    return t

def move_swap(tour):
    a,b = random.sample(range(len(tour)), 2)
    tr = tour.copy()
    tr[a], tr[b] = tr[b], tr[a]
    return tr


def move_two_opt(tour):
    n = len(tour)
    a,b = sorted(random.sample(range(n),2))
    tr = tour.copy()
    tr[a:b+1] = list(reversed(tr[a:b+1]))
    return tr


def move_insertion(tour):
    n = len(tour)
    a,b = random.sample(range(n), 2)
    tr = tour.copy()
    city = tr.pop(a)
    tr.insert(b, city)
    return tr


def local_improve(tour, dist_matrix, moves=['swap','two_opt','insertion'], tries=10):
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


def pmx(parent1, parent2): 
    n = len(parent1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[a:b+1] = parent1[a:b+1]
    p2pos = {v:i for i,v in enumerate(parent2)}
    for i in range(a, b+1):
        val = parent2[i]
        if val in child:
            continue
        cur = i
        mapped = val
        while True:
            val1 = parent1[cur]
            cur = p2pos[val1]
            if child[cur] == -1:
                child[cur] = mapped
                break
    it = (v for v in parent2 if v not in child)
    for i in range(n):
        if child[i] == -1:
            child[i] = next(it)
    return child

def ox(parent1, parent2):
    n = len(parent1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
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
    n = len(parent1)
    child = [-1] * n
    remaining = set(range(n))
    pos1 = {v:i for i,v in enumerate(parent1)}
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


def mutate_swap(child):
    return move_swap(child)


def select_tournament(population, lengths, k=3):
    idxs = random.sample(range(len(population)), k)
    best_idx = min(idxs, key=lambda i: lengths[i])
    return deepcopy(population[best_idx])


def select_roulette_from_length(population, lengths, eps=1e-9):

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
    arr = np.array(lengths)
    order = np.argsort(arr)  # ascending (najlepszy pierwszy)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(arr), 0, -1)  # best -> N, worst -> 1
    probs = ranks / ranks.sum()
    idx = np.random.choice(len(population), p=probs)
    return deepcopy(population[idx])


# Główny GA (run_ga) 

def run_ga(dist, n_pop=100, n_gen=300, p_mut=0.05, p_cx=0.9,
           selection_method='tournament', crossover_method='pmx',
           local_search_tries=0, local_moves=['swap','two_opt','insertion'],
           elite=1, tournament_k=3, seed=None,
           # mechanizm wnuka (grandchild)
           p_make_grandchild=0.0, grandchild_method='mutate', grandchild_local_tries=None):
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n = dist.shape[0]  # liczba miast
    population = [random_tour(n) for _ in range(n_pop)] # populacja początkowa o rozmiarze n_pop
    history = [] # historia najlepszych długości w każdej generacji

    crossover_map = {'pmx': pmx, 'ox': ox, 'cx': cycle_crossover}
    selection_map = {'tournament': select_tournament,
                     'roulette': select_roulette_from_length,
                     'rank': select_rank_from_length}

    crossover_fn = crossover_map[crossover_method]
    selection_fn = selection_map[selection_method]

    for gen in range(n_gen):

        # ocena populacji
        lengths = [tour_length(t, dist) for t in population] # dlugosc trasy dla kazdego osobnika
        best_idx = int(np.argmin(lengths)) # indeks najlepszego osobnika
        best_len = lengths[best_idx] # dlugosc najlepszego osobnika
        history.append(best_len) # zapisz do historii

        # przygotowanie następnej generacji
        new_pop = []
        # elityzm: kopiujemy top 'elite'
        ranked = sorted(zip(lengths, population), key=lambda x: x[0]) # sortowanie po dlugosci, rosnaco, najlepszy pierwszy
        for i in range(min(elite, len(ranked))): # kopiujemy najlepszych 'elite' osobnikow
            new_pop.append(deepcopy(ranked[i][1])) # dodajemy do nowej populacji, deepcopy zeby nie bylo referencji

        while len(new_pop) < n_pop: # dopóki nie uzyskamy pełnej populacji
            # selekcja rodziców
            if selection_method == 'tournament': # tournament wymaga parametru k
                p1 = selection_fn(population, lengths, tournament_k) # wybierz rodzica 1
                p2 = selection_fn(population, lengths, tournament_k) # wybierz rodzica 2
            else: # inne metody selekcji nie wymagają dodatkowych parametrów
                p1 = selection_fn(population, lengths) # wybierz rodzica 1
                p2 = selection_fn(population, lengths) # wybierz rodzica 2
            # crossover
            if random.random() < p_cx: # z prawdopodobieństwem p_cx wykonaj crossover
                child = crossover_fn(p1, p2) # utwórz dziecko przez crossover
            else: # bez crossovera, kopiuj rodzica
                child = deepcopy(p1)
            # mutacja
            if random.random() < p_mut: # z prawdopodobieństwem p_mut wykonaj mutację
                child = mutate_swap(child)
            # lokalne ulepszenie 
            if local_search_tries > 0: # jeśli dozwolone są próby lokalnego ulepszenia
                child = local_improve(child, dist, moves=local_moves, tries=local_search_tries) # lokalne ulepszenie dziecka

            if p_make_grandchild > 0 and random.random() < p_make_grandchild: # mechanizm wnuka, z prawdopodobieństwem p_make_grandchild
                grandchild = deepcopy(child) 
                if grandchild_method == 'mutate': 
                    grandchild = mutate_swap(grandchild)
                elif grandchild_method == 'local_improve': 
                    tries = grandchild_local_tries if grandchild_local_tries is not None else local_search_tries # użyj podanej liczby prób lub domyślnej
                    if tries and tries > 0: 
                        grandchild = local_improve(grandchild, dist, moves=local_moves, tries=tries) 
                    else:
                        grandchild = mutate_swap(grandchild)
                # wybierz lepszego z pary
                if tour_length(grandchild, dist) < tour_length(child, dist):
                    child = grandchild

            new_pop.append(child) # dodaj dziecko do nowej populacji

        population = new_pop # przejdź do nowej populacji

    lengths = [tour_length(t, dist) for t in population] # ostateczna ocena populacji
    idx = int(np.argmin(lengths)) # indeks najlepszego osobnika
    return {'tour': population[idx], 'length': lengths[idx], 'history': history} # zwróć najlepszy tour, jego długość i historię


# Eksperymenty OFAT 

def run_experiments_to_excel_and_plots(dist, dataset_name,
                                       baseline_params, param_tests,
                                       repetitions=5,
                                       out_dir='results_excel'): 

    _ensure_dir(out_dir) # upewnij się, że katalog wyjściowy istnieje
    results_rows = [] # wiersze do arkusza 'results'
    tours_rows = [] # wiersze do arkusza 'tours_expanded'

    run_id = 0 # unikalny identyfikator uruchomienia
    total_runs = sum(len(v) for v in param_tests.values()) * repetitions # całkowita liczba uruchomień
    print(f"Starting OFAT for dataset {dataset_name}: {len(param_tests)} params, ~{total_runs} runs")

    n_cities = dist.shape[0]  # ile kolumn city_0..city_{n-1} stworzyć w arkuszu 'tours_expanded'

    for param_name, values in param_tests.items(): # dla każdego parametru do przetestowania
        print(f"\n>>> Testing parameter: {param_name} (values: {values})") # rozpocznij testowanie parametru
        for val in values:
            cfg = baseline_params.copy() # zacznij od parametrów bazowych, następnie zmień jeden parametr
            cfg[param_name] = val # ustaw testowaną wartość parametru
            # upewnij się, że wszystkie klucze baseline istnieją w cfg
            for k in baseline_params.keys(): # jeśli klucz nie istnieje w cfg, dodaj go z baseline
                if k not in cfg: # dodaj klucz z baseline
                    cfg[k] = baseline_params[k]
            for r in range(repetitions): # powtórzenia dla danej konfiguracji
                seed = SEED + run_id # unikalne ziarno dla każdego uruchomienia
                t0 = time.time() # czas rozpoczęcia
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
                             seed=seed,
                             p_make_grandchild=cfg.get('p_make_grandchild', 0.0),
                             grandchild_method=cfg.get('grandchild_method', 'mutate'),
                             grandchild_local_tries=cfg.get('grandchild_local_tries', None)
                             )
                elapsed = time.time() - t0

                tour_list = out.get('tour', [])
                tour_str = '-'.join(map(str, tour_list))

                row = cfg.copy()
                row.update({
                    'run_id': run_id,
                    'dataset': dataset_name,
                    'param_name': param_name,
                    'param_value': val,
                    'repetition': r,
                    'seed': seed,
                    'best_length': out['length'],
                    'time_s': round(elapsed, 4),
                    'tour': tour_str
                })
                results_rows.append(row)

                # wiersz do arkusza 'tours_expanded' (meta + city_0..city_{n-1})
                tour_expanded = {
                    'run_id': run_id,
                    'dataset': dataset_name,
                    'param_name': param_name,
                    'param_value': val,
                    'repetition': r,
                    'seed': seed,
                    'best_length': out['length'],
                    'time_s': round(elapsed, 4),
                    'tour': tour_str
                }
                # wypełnij city_0..city_{n-1}; jeśli lista jest krótsza/nieprawidłowa -> None
                for i in range(n_cities):
                    try:
                        tour_expanded[f'city_{i}'] = int(tour_list[i])
                    except Exception:
                        tour_expanded[f'city_{i}'] = None
                tours_rows.append(tour_expanded)

                print(f" run {run_id:04d}/{total_runs-1} | {param_name}={val} | rep={r} | best={out['length']:.2f} time={elapsed:.1f}s")
                run_id += 1


    df_results = pd.DataFrame(results_rows)
    df_tours = pd.DataFrame(tours_rows)

    # Zapis do Excela
    excel_path = os.path.join(out_dir, f"{dataset_name}_results.xlsx")
    print("Saving Excel:", excel_path)
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_results.to_excel(writer, index=False, sheet_name='results')
        df_tours.to_excel(writer, index=False, sheet_name='tours_expanded')

    # generacja wykresów (używa df_results)
    plots_base = os.path.join(out_dir, 'plots', dataset_name)
    _ensure_dir(plots_base)
    generate_plots_from_df(df_results, plots_base, dataset_name)

    print("\nDone. Excel and plots saved.")
    return excel_path


def generate_plots_from_df(df, out_base, dataset_name):

    params = df['param_name'].unique()
    for p in params:
        dfp = df[df['param_name'] == p].copy()
        if dfp.empty:
            continue
        out_dir = os.path.join(out_base, str(p))
        _ensure_dir(out_dir)
        dfp['param_value_str'] = dfp['param_value'].astype(str)

        # BOXPLOT
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

        # MEAN ± STD 
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

        # SCATTER time_s vs best_length
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



if __name__ == "__main__":
    files = [
        ("Dane_TSP_127.xlsx","Dane_TSP_127"),
        ("Dane_TSP_76.xlsx","Dane_TSP_76"),
        ("Dane_TSP_48.xlsx","Dane_TSP_48")
    ]

    # parametry do przetestowania (OFAT)
    param_tests = {
        'n_pop': [50, 100, 200, 400],
        'n_gen': [100, 300, 600, 1000],
        'p_mut': [0.01, 0.03, 0.07, 0.15],
        'p_cx' : [0.6, 0.75, 0.9, 1.0],
        'elite' : [1, 3, 5, 10],
        'p_make_grandchild': [0.1, 0.3, 0.5],
        'local_search_tries': [0, 5, 10, 20],
        'crossover_method': ['pmx', 'ox', 'cx'],
        'selection_method': ['tournament', 'roulette', 'rank']
    }


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
        'tournament_k': 3,
        'p_make_grandchild': 0.0,
        'grandchild_method': 'mutate',
        'grandchild_local_tries': None
    }

    repetitions = 10  # ile powtórzeń na konfiguracja

    for path, dataset_name in files:
        if not os.path.exists(path):
            print(f"File not found: {path} - skipping")
            continue
        print(f"\n=== Running dataset {dataset_name} from file {path} ===")
        dist = read_distance_matrix_excel(path)
        out_excel = run_experiments_to_excel_and_plots(dist, dataset_name, baseline, param_tests, repetitions=repetitions, out_dir='results_excel')
        print("Saved Excel:", out_excel)


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

def wczytaj_macierz_przetwarzania_excel(path):
    df = pd.read_excel(path, header=0, index_col=0)  
    return df.values.astype(float)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def makespan(perm, czas_proc):
    n = len(perm)
    m = int(czas_proc.shape[1])
    prev = [0.0] * m
    for job in perm:
        cur = [0.0] * m
        for machine in range(m):
            tproc = float(czas_proc[job, machine])
            if machine == 0:
                cur[machine] = prev[machine] + tproc
            else:
                cur[machine] = max(cur[machine-1], prev[machine]) + tproc
        prev = cur
    return prev[-1]

def losowa_permutacja(n):
    perm = list(range(n))
    random.shuffle(perm)
    return perm

def ruch_swap(perm):
    a, b = random.sample(range(len(perm)), 2)
    p = perm.copy()
    p[a], p[b] = p[b], p[a]
    return p

def ruch_two_opt(perm):
    n = len(perm)
    a, b = sorted(random.sample(range(n), 2))
    p = perm.copy()
    p[a:b+1] = list(reversed(p[a:b+1]))
    return p

def ruch_insertion(perm):
    n = len(perm)
    a, b = random.sample(range(n), 2)
    p = perm.copy()
    val = p.pop(a)
    p.insert(b, val)
    return p

def local_improve(perm, czas_proc, moves=['swap','two_opt','insertion'], tries=10):

    best = perm.copy()
    best_val = makespan(best, czas_proc)
    for _ in range(tries):
        mv = random.choice(moves)
        if mv == 'swap':
            cand = ruch_swap(best)
        elif mv == 'two_opt':
            cand = ruch_two_opt(best)
        else:
            cand = ruch_insertion(best)
        v = makespan(cand, czas_proc)
        if v < best_val:
            best, best_val = cand, v
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


def mutate_swap(perm):
    return ruch_swap(perm)

def select_tournament(pop, values, k=3):
    idxs = random.sample(range(len(pop)), k)
    best_idx = min(idxs, key=lambda i: values[i])
    return deepcopy(pop[best_idx])

def select_roulette_from_value(pop, values, eps=1e-9):
    arr = np.array(values, dtype=float)
    M = arr.max()
    weights = (M - arr) + eps
    s = weights.sum()
    if s <= 0:
        return deepcopy(random.choice(pop))
    probs = weights / s
    idx = np.random.choice(len(pop), p=probs)
    return deepcopy(pop[idx])

def select_rank_from_value(pop, values):
    arr = np.array(values)
    order = np.argsort(arr)  # ascending (najlepszy pierwszy)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(arr), 0, -1)
    probs = ranks / ranks.sum()
    idx = np.random.choice(len(pop), p=probs)
    return deepcopy(pop[idx])

def run_ga(czas_proc, n_pop=100, n_gen=100, p_mut=0.05, p_cx=0.9,
           selection_method='tournament', crossover_method='pmx',
           local_search_tries=0, local_moves=['swap','two_opt','insertion'],
           elite=1, tournament_k=3, seed=None,
           p_make_grandchild=0.0, grandchild_method='mutate', grandchild_local_tries=None):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_jobs = int(czas_proc.shape[0])
    population = [losowa_permutacja(n_jobs) for _ in range(n_pop)]
    history = []

    crossover_map = {'pmx': pmx, 'ox': ox, 'cx': cycle_crossover}
    selection_map = {'tournament': select_tournament,
                     'roulette': select_roulette_from_value,
                     'rank': select_rank_from_value}

    crossover_fn = crossover_map[crossover_method]
    selection_fn = selection_map[selection_method]

    for gen in range(n_gen):

        values = [makespan(p, czas_proc) for p in population]
        best_idx = int(np.argmin(values))
        best_val = values[best_idx]
        history.append(best_val)

        new_pop = []
        ranked = sorted(zip(values, population), key=lambda x: x[0])
        for i in range(min(elite, len(ranked))):
            new_pop.append(deepcopy(ranked[i][1]))

        while len(new_pop) < n_pop:
            # selekcja rodziców
            if selection_method == 'tournament':
                p1 = selection_fn(population, values, tournament_k)
                p2 = selection_fn(population, values, tournament_k)
            else:
                p1 = selection_fn(population, values)
                p2 = selection_fn(population, values)

            # crossover
            if random.random() < p_cx:
                child = crossover_fn(p1, p2)
            else:
                child = deepcopy(p1)

            # mutacja
            if random.random() < p_mut:
                child = mutate_swap(child)

            # lokalne ulepszenie
            if local_search_tries > 0:
                child = local_improve(child, czas_proc, moves=local_moves, tries=local_search_tries)

            # wnuk (grandchild)
            if p_make_grandchild > 0 and random.random() < p_make_grandchild:
                grandchild = deepcopy(child)
                if grandchild_method == 'mutate':
                    grandchild = mutate_swap(grandchild)
                elif grandchild_method == 'local_improve':
                    tries = grandchild_local_tries if grandchild_local_tries is not None else local_search_tries
                    if tries and tries > 0:
                        grandchild = local_improve(grandchild, czas_proc, moves=local_moves, tries=tries)
                    else:
                        grandchild = mutate_swap(grandchild)
                if makespan(grandchild, czas_proc) < makespan(child, czas_proc):
                    child = grandchild

            new_pop.append(child)

        population = new_pop

    values = [makespan(p, czas_proc) for p in population]
    idx = int(np.argmin(values))
    return {'perm': population[idx], 'makespan': values[idx], 'history': history}


def run_experiments_to_excel_and_plots(czas_proc, dataset_name,
                                       baseline_params, param_tests,
                                       repetitions=10,
                                       out_dir='results_excel_pfsp'):
    _ensure_dir(out_dir)
    results_rows = []
    perms_rows = []

    run_id = 0
    total_runs = sum(len(v) for v in param_tests.values()) * repetitions
    print(f"Start OFAT dla {dataset_name}: {len(param_tests)} parametrow, ~{total_runs} uruchomien")

    n_jobs = int(czas_proc.shape[0])

    for param_name, values in param_tests.items():
        print(f"\n>>> Test parametru: {param_name} -> {values}")
        for val in values:
            cfg = baseline_params.copy()
            cfg[param_name] = val
            for k in baseline_params.keys():
                if k not in cfg:
                    cfg[k] = baseline_params[k]
            for r in range(repetitions):
                seed = SEED + run_id
                t0 = time.time()
                out = run_ga(czas_proc,
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

                perm = out['perm']
                perm_str = '-'.join(map(str, perm))

                row = cfg.copy()
                row.update({
                    'run_id': run_id,
                    'dataset': dataset_name,
                    'param_name': param_name,
                    'param_value': val,
                    'repetition': r,
                    'seed': seed,
                    'best_makespan': out['makespan'],
                    'time_s': round(elapsed, 4),
                    'perm': perm_str
                })
                results_rows.append(row)

                perm_row = {
                    'run_id': run_id,
                    'dataset': dataset_name,
                    'param_name': param_name,
                    'param_value': val,
                    'repetition': r,
                    'seed': seed,
                    'best_makespan': out['makespan'],
                    'time_s': round(elapsed, 4),
                    'perm': perm_str
                }
                for i in range(n_jobs):
                    try:
                        perm_row[f'job_{i}'] = int(perm[i])
                    except Exception:
                        perm_row[f'job_{i}'] = None
                perms_rows.append(perm_row)

                print(f" run {run_id:04d}/{total_runs-1} | {param_name}={val} | rep={r} | makespan={out['makespan']:.2f} time={elapsed:.1f}s")
                run_id += 1

    df_results = pd.DataFrame(results_rows)
    df_perms = pd.DataFrame(perms_rows)

    # Zapis do Excela
    excel_path = os.path.join(out_dir, f"{dataset_name}_results.xlsx")
    print("Saving Excel:", excel_path)
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_results.to_excel(writer, index=False, sheet_name='results')
        df_perms.to_excel(writer, index=False, sheet_name='permutations')

    # wykresy
    plots_base = os.path.join(out_dir, 'plots', dataset_name)
    _ensure_dir(plots_base)
    generate_plots_from_df(df_results, plots_base, dataset_name)

    print("Gotowe. Excel i wykresy zapisane.")
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
        groups = [ grp['best_makespan'].values for _, grp in dfp.groupby('param_value_str') ]
        labels = [ str(k) for k, _ in dfp.groupby('param_value_str') ]
        plt.figure(figsize=(10,6))
        plt.boxplot(groups, labels=labels, showfliers=True)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('best_makespan')
        plt.title(f"{dataset_name} | {p} — boxplot best_makespan")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'boxplot_best_makespan.png'))
        plt.close()

        # MEAN ± STD
        try:
            dfp['param_value_num'] = pd.to_numeric(dfp['param_value'])
            stats = dfp.groupby('param_value_num')['best_makespan'].agg(['mean','std']).reset_index().sort_values('param_value_num')
            xs = stats['param_value_num'].values
            ys = stats['mean'].values
            yerr = stats['std'].values
            plt.figure(figsize=(8,5))
            plt.errorbar(xs, ys, yerr=yerr, fmt='-o', capsize=4)
            plt.xlabel(str(p))
            plt.ylabel('mean best_makespan ± std')
            plt.title(f"{dataset_name} | {p} — mean ± std")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'mean_std.png'))
            plt.close()
        except Exception:
            stats = dfp.groupby('param_value_str')['best_makespan'].agg(['mean','std']).reset_index()
            labels = stats['param_value_str'].values
            xs = range(len(labels))
            plt.figure(figsize=(9,5))
            plt.errorbar(xs, stats['mean'].values, yerr=stats['std'].values, fmt='-o', capsize=4)
            plt.xticks(xs, labels, rotation=45, ha='right')
            plt.xlabel(str(p))
            plt.ylabel('mean best_makespan ± std')
            plt.title(f"{dataset_name} | {p} — mean ± std")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'mean_std.png'))
            plt.close()

        # SCATTER time_s vs best_makespan
        plt.figure(figsize=(8,6))
        unique_vals = dfp['param_value_str'].unique()
        for val in unique_vals:
            sub = dfp[dfp['param_value_str']==val]
            plt.scatter(sub['time_s'], sub['best_makespan'], alpha=0.7, label=str(val))
        plt.xlabel('time_s')
        plt.ylabel('best_makespan')
        plt.title(f"{dataset_name} | {p} — time vs best_makespan")
        plt.legend(title=str(p), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'time_vs_best_scatter.png'))
        plt.close()

        print(f" Plots saved for param {p} -> {out_dir}")



if __name__ == "__main__":

    files = [
        ("Dane_PFSP_200_10.xlsx", "PFSP_200_10"),
        ("Dane_PFSP_100_10.xlsx", "PFSP_100_10"),
        ("Dane_PFSP_50_20.xlsx", "PFSP_50_20")
    ]

    # parametry do testowania (OFAT)
    param_tests = {
        'n_pop': [50, 100, 200, 400],
        'n_gen': [100, 300, 600, 1000],
        'p_mut': [0.01, 0.03, 0.07, 0.15],
        'p_cx': [0.6, 0.75, 0.9, 1.0],
        'elite': [1, 3, 5, 10],
        'local_search_tries': [0, 3, 7, 15],
        'p_make_grandchild': [0.0, 0.1, 0.3, 0.5],
        'crossover_method': ['pmx', 'ox', 'cx'],
        'selection_method': ['tournament', 'roulette', 'rank']
    }


    baseline = {
        'n_pop': 100,
        'n_gen': 100,
        'p_mut': 0.03,
        'p_cx': 0.9,
        'selection_method': 'tournament',
        'crossover_method': 'pmx',
        'local_search_tries': 3,
        'local_moves': ['swap','two_opt','insertion'],
        'elite': 3,
        'tournament_k': 3,
        'p_make_grandchild': 0.0,
        'grandchild_method': 'local_improve',
        'grandchild_local_tries': 3
    }

    repetitions = 10

    for path, dataset_name in files:
        if not os.path.exists(path):
            print(f"Plik nie znaleziony: {path}")
            continue
        print(f"\n Uruchamiam dataset {dataset_name} z pliku {path}")
        proc = wczytaj_macierz_przetwarzania_excel(path)
        out = run_experiments_to_excel_and_plots(proc, dataset_name, baseline, param_tests, repetitions=repetitions, out_dir='results_excel_pfsp')
        print("Zapisano Excel:", out)

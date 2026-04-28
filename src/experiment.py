"""
experiment.py — координация многопрогонного эксперимента
для сравнения политик управления запасами.
"""
import numpy as np
import pandas as pd
from policy import assign_policy
from simulation import run_simulation


# ============================================================
# Конфигурации сравниваемых политик
# ============================================================
DEFAULT_POLICIES = [
    {'name': 'P0', 'policy': 'uniform', 'csl_override': 0.95,
     'description': 'Единый CSL=0.95'},
    {'name': 'P1', 'policy': 'abcxyz', 'csl_override': None,
     'description': 'ABC-XYZ-сегментация'},
    {'name': 'P2', 'policy': 'multilevel', 'csl_override': None,
     'description': 'Многоуровневая (предложенная)'},
    {'name': 'P3', 'policy': 'uniform', 'csl_override': 0.99,
     'description': 'Единый CSL=0.99'},
]


def run_policy_experiment(skus_df, policy_config, n_runs=30,
                            sim_days=395, warmup_days=30,
                            base_seed=42, verbose=True):
    """
    Многократный прогон одной политики.

    Возвращает pd.DataFrame с агрегированными метриками по прогонам.
    """
    skus_p = assign_policy(skus_df,
                            policy=policy_config['policy'],
                            csl_override=policy_config.get(
                                'csl_override'))

    runs = []
    for k in range(n_runs):
        seed = base_seed + k * 100  # разные seed для разных прогонов
        result = run_simulation(skus_p, sim_days=sim_days,
                                  warmup_days=warmup_days, seed=seed)

        weighted_fr = ((result['fill_rate'] * skus_p['annual_demand']).sum()
                        / skus_p['annual_demand'].sum())
        inv_value = (result['avg_inventory'] * skus_p['price']).sum()

        # Издержки за период наблюдения (sim_days - warmup_days дней)
        days_observed = sim_days - warmup_days

        # Издержки хранения (тыс. руб./год)
        # На основе среднего запаса и стоимости хранения
        holding = ((result['avg_inventory'] * skus_p['price']
                     * skus_p['holding_cost_rate']).sum()) / 1000

        # Издержки заказа (тыс. руб./год) — годовое количество * 1500
        annual_orders = (result['orders_count']
                          * 365 / days_observed)
        ordering = (annual_orders * 1500).sum() / 1000

        # Издержки дефицита (тыс. руб./год)
        # Условная цена = 5x маржи (5x от цены * 0.2 как маржа -> 1.0 от цены)
        # Используем 5-кратную маржу: 5 * 0.2 * price = price
        annual_lost = result['lost_demand'] * 365 / days_observed
        stockout_cost = (annual_lost * skus_p['price']).sum() / 1000

        # Дефицитных циклов на 100 SKU
        n_skus = len(skus_p)
        stockout_cycles_per_100 = ((1 - result['cycle_service_level']).sum()
                                       * 100 / n_skus)

        agg = {
            'policy': policy_config['name'],
            'run': k,
            'seed': seed,
            'fill_rate_pct': float(weighted_fr * 100),
            'avg_inventory_k': float(inv_value / 1000),
            'holding_k': float(holding),
            'ordering_k': float(ordering),
            'stockout_k': float(stockout_cost),
            'tc_k': float(holding + ordering + stockout_cost),
            'stockout_cycles_per_100': float(stockout_cycles_per_100),
            'total_lost': int(result['lost_demand'].sum()),
            'total_orders': int(result['orders_count'].sum()),
        }
        runs.append(agg)

        if verbose and (k + 1) % 5 == 0:
            print(f"    {policy_config['name']}: run {k+1}/{n_runs} done")

    return pd.DataFrame(runs)


def compare_policies(skus_df, configs=None, n_runs=30,
                      sim_days=395, warmup_days=30, base_seed=42,
                      verbose=True):
    """
    Сравнить несколько политик и вернуть сводку + детали по прогонам.

    Возвращает (summary_df, all_runs_df).
    """
    if configs is None:
        configs = DEFAULT_POLICIES

    all_runs = []
    for cfg in configs:
        if verbose:
            print(f"  Running {cfg['name']} ({cfg['description']}) "
                  f"x {n_runs} runs ...")
        df = run_policy_experiment(skus_df, cfg, n_runs=n_runs,
                                     sim_days=sim_days,
                                     warmup_days=warmup_days,
                                     base_seed=base_seed,
                                     verbose=verbose)
        all_runs.append(df)

    full = pd.concat(all_runs, ignore_index=True)

    # Сводка: среднее ± стд по прогонам
    metrics = ['fill_rate_pct', 'avg_inventory_k', 'holding_k',
                'ordering_k', 'stockout_k', 'tc_k',
                'stockout_cycles_per_100']
    summary = full.groupby('policy', sort=False)[metrics].agg(['mean', 'std'])
    summary = summary.round(2)

    return summary, full


def format_comparison_table(summary, full=None):
    """Красивое текстовое представление сводной таблицы."""
    lines = []
    lines.append("=" * 88)
    lines.append("  Сравнение политик управления запасами "
                  "(среднее ± std по прогонам)")
    lines.append("=" * 88)
    header = (f"{'Policy':<8} {'Fill Rate, %':<14} {'Inv, k.rub':<14} "
              f"{'TC, k.rub':<14} {'Stockout cycles':<18}")
    lines.append(header)
    lines.append("-" * 88)
    for policy in summary.index:
        fr_m = summary.loc[policy, ('fill_rate_pct', 'mean')]
        fr_s = summary.loc[policy, ('fill_rate_pct', 'std')]
        inv_m = summary.loc[policy, ('avg_inventory_k', 'mean')]
        inv_s = summary.loc[policy, ('avg_inventory_k', 'std')]
        tc_m = summary.loc[policy, ('tc_k', 'mean')]
        tc_s = summary.loc[policy, ('tc_k', 'std')]
        sc_m = summary.loc[policy, ('stockout_cycles_per_100', 'mean')]
        sc_s = summary.loc[policy, ('stockout_cycles_per_100', 'std')]
        lines.append(
            f"{policy:<8} {fr_m:>5.2f} ± {fr_s:<5.2f} "
            f" {inv_m:>6.0f} ± {inv_s:<5.0f} "
            f" {tc_m:>6.0f} ± {tc_s:<5.0f} "
            f" {sc_m:>5.1f} ± {sc_s:<5.1f}"
        )
    lines.append("=" * 88)

    # Преимущества P2 над P1
    if 'P1' in summary.index and 'P2' in summary.index:
        p1_fr = summary.loc['P1', ('fill_rate_pct', 'mean')]
        p2_fr = summary.loc['P2', ('fill_rate_pct', 'mean')]
        p1_inv = summary.loc['P1', ('avg_inventory_k', 'mean')]
        p2_inv = summary.loc['P2', ('avg_inventory_k', 'mean')]
        p1_tc = summary.loc['P1', ('tc_k', 'mean')]
        p2_tc = summary.loc['P2', ('tc_k', 'mean')]
        p1_sc = summary.loc['P1', ('stockout_cycles_per_100', 'mean')]
        p2_sc = summary.loc['P2', ('stockout_cycles_per_100', 'mean')]

        lines.append("\nПреимущество P2 над P1 (предложенная над ABC-XYZ):")
        lines.append(f"  Fill Rate :  {p2_fr - p1_fr:+.2f} п.п.")
        lines.append(f"  Запас     :  {(p2_inv - p1_inv) / p1_inv * 100:+.1f} %")
        lines.append(f"  TC        :  {(p2_tc - p1_tc) / p1_tc * 100:+.1f} %")
        lines.append(f"  Дефициты  :  "
                      f"{(p2_sc - p1_sc) / p1_sc * 100:+.1f} %")
    lines.append("=" * 88)
    return "\n".join(lines)


if __name__ == '__main__':
    from data_generator import generate_skus_dataset
    from abc_xyz import abc_xyz_matrix
    from clusterer import cluster_all_groups

    skus, demand = generate_skus_dataset(n=100, T=365, seed=2025)
    skus = abc_xyz_matrix(skus, demand)
    skus['cluster'] = cluster_all_groups(skus, k=2)

    print("Starting comparison experiment with 30 runs per policy...")
    summary, full = compare_policies(skus, n_runs=30)

    print("\n" + format_comparison_table(summary, full))

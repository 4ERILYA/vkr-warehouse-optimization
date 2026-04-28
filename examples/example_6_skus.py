"""
example_6_skus.py — расширенный пример вторичной кластеризации.

6 SKU, отнесённых к одной группе AX, разделяются на два подкластера
(alpha и beta) методом k-means++ по операционным признакам:
  - lead time
  - holding cost rate
  - criticality

Подкластеры получают разные CSL: 0.95 для alpha, 0.99 для beta.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..', 'src'))

import pandas as pd
from clusterer import cluster_group
from policy import compute_params, sigma_ltd
from scipy import stats


# 6 SKU группы AX
SKUS_DATA = [
    # alpha: короткий L, низкая c
    {'sku_id': 'SKU-101', 'mean_demand': 120, 'std_demand': 8,
     'lead_mean': 5,  'lead_std': 0.5, 'holding_cost_rate': 0.20,
     'criticality': 2, 'price': 150},
    {'sku_id': 'SKU-102', 'mean_demand': 95,  'std_demand': 6,
     'lead_mean': 4,  'lead_std': 0.4, 'holding_cost_rate': 0.22,
     'criticality': 1, 'price': 180},
    {'sku_id': 'SKU-103', 'mean_demand': 80,  'std_demand': 5,
     'lead_mean': 3,  'lead_std': 0.3, 'holding_cost_rate': 0.25,
     'criticality': 2, 'price': 210},
    # beta: длинный L, высокая c
    {'sku_id': 'SKU-104', 'mean_demand': 150, 'std_demand': 10,
     'lead_mean': 30, 'lead_std': 4.0, 'holding_cost_rate': 0.18,
     'criticality': 5, 'price': 320},
    {'sku_id': 'SKU-105', 'mean_demand': 110, 'std_demand': 7,
     'lead_mean': 28, 'lead_std': 3.5, 'holding_cost_rate': 0.20,
     'criticality': 4, 'price': 280},
    {'sku_id': 'SKU-106', 'mean_demand': 90,  'std_demand': 6,
     'lead_mean': 35, 'lead_std': 5.0, 'holding_cost_rate': 0.21,
     'criticality': 5, 'price': 350},
]


def main():
    print("=" * 70)
    print("  Расширенный пример: вторичная кластеризация группы AX")
    print("  И.В. Черников, ВГУ, 2026")
    print("=" * 70)

    skus = pd.DataFrame(SKUS_DATA)
    skus['abc_xyz'] = 'AX'
    skus['annual_demand'] = (skus['mean_demand'] * 365).astype(int)

    print(f"\nИсходные характеристики 6 SKU группы AX:")
    print(skus[['sku_id', 'mean_demand', 'std_demand',
                  'lead_mean', 'lead_std', 'holding_cost_rate',
                  'criticality', 'price']].to_string(index=False))

    # Кластеризация
    print("\n--- Кластеризация k-means++ по (L, h, c) ---")
    skus['cluster'] = cluster_group(skus.set_index('sku_id'), k=2,
                                       random_state=42).values

    # Обновим индекс
    print("\nРезультат кластеризации:")
    cluster_view = skus[['sku_id', 'lead_mean', 'holding_cost_rate',
                            'criticality', 'cluster']]
    print(cluster_view.to_string(index=False))

    # Сводка
    print("\nСостав подкластеров:")
    for cluster_name in ['alpha', 'beta']:
        members = skus[skus['cluster'] == cluster_name]['sku_id'].tolist()
        print(f"  AX-{cluster_name}: {', '.join(members)}")

    # Расчёт параметров P2 для каждого SKU
    print("\n--- Параметры политики P2 (диффер. CSL по подкластерам) ---\n")
    rows = []
    for _, row in skus.iterrows():
        p2 = compute_params(row, policy='multilevel')
        sig = sigma_ltd(row['mean_demand'], row['std_demand'],
                         row['lead_mean'], row['lead_std'])
        rows.append({
            'SKU':       row['sku_id'],
            'Кластер':   f"AX-{row['cluster']}",
            'CSL':       f"{p2['csl']:.3f}",
            'σ_LTD':     f"{sig:.1f}",
            'SS':        f"{p2['SS']:.0f}",
            'ROP':       f"{p2['ROP']:.0f}",
            'EOQ':       f"{p2['EOQ']:.0f}",
        })
    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))

    # Сравнение P1 vs P2
    print("\n--- Сравнение P1 (един. CSL=0.97 для AX) vs P2 ---\n")
    cmp_rows = []
    for _, row in skus.iterrows():
        p1 = compute_params(row, policy='abcxyz')
        p2 = compute_params(row, policy='multilevel')
        cmp_rows.append({
            'SKU':       row['sku_id'],
            'Кластер':   f"AX-{row['cluster']}",
            'CSL_P1':    f"{p1['csl']:.3f}",
            'SS_P1':     f"{p1['SS']:.0f}",
            'CSL_P2':    f"{p2['csl']:.3f}",
            'SS_P2':     f"{p2['SS']:.0f}",
            'Δ SS':      f"{(p2['SS'] - p1['SS']) / p1['SS'] * 100:+.1f}%",
        })
    cmp_df = pd.DataFrame(cmp_rows)
    print(cmp_df.to_string(index=False))

    print("\nИнтерпретация:")
    print("  • Подкластер AX-alpha (короткий L, низкая c):")
    print("    CSL понижен с 0.97 (P1) до 0.95 (P2) → SS снижается на 13%")
    print("    → освобождается оборотный капитал.")
    print("  • Подкластер AX-beta (длинный L, высокая c):")
    print("    CSL повышен с 0.97 (P1) до 0.99 (P2) → SS растёт на ~24%")
    print("    → снижается риск критических дефицитов.")

    print("\n" + "=" * 70)
    print("  Расширенный пример завершён.")
    print("=" * 70)


if __name__ == '__main__':
    main()

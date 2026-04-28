"""
example_3_skus.py — базовый пример из статьи Черникова И.В. (2026).

Применяет многоуровневую модель к трём SKU разных групп ABC-XYZ:
  SKU-A1 — группа AX (стабильный, высокий оборот, короткий лаг)
  SKU-B1 — группа BY (умеренная вариация, средний оборот)
  SKU-C1 — группа CZ (нерегулярный, малый оборот, длинный лаг)

Результат — пошаговый расчёт SS, ROP, EOQ для каждого SKU.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..', 'src'))

import pandas as pd
import numpy as np
from policy import compute_params, sigma_ltd
from scipy import stats


# Исходные данные трёх SKU (как в статье)
SKUS_DATA = [
    {
        'sku_id': 'SKU-A1',
        'description': 'AX — высокооборотная стабильная позиция',
        'abc_xyz': 'AX',
        'cluster': 'alpha',
        'mean_demand': 120,
        'std_demand': 8,
        'lead_mean': 5,
        'lead_std': 0.5,
        'holding_cost_rate': 0.20,
        'criticality': 2,
        'price': 150,
    },
    {
        'sku_id': 'SKU-B1',
        'description': 'BY — позиция средней значимости',
        'abc_xyz': 'BY',
        'cluster': 'alpha',
        'mean_demand': 60,
        'std_demand': 12,
        'lead_mean': 10,
        'lead_std': 1.5,
        'holding_cost_rate': 0.22,
        'criticality': 3,
        'price': 300,
    },
    {
        'sku_id': 'SKU-C1',
        'description': 'CZ — малооборотная нерегулярная позиция',
        'abc_xyz': 'CZ',
        'cluster': 'alpha',
        'mean_demand': 8,
        'std_demand': 6,
        'lead_mean': 20,
        'lead_std': 4.0,
        'holding_cost_rate': 0.25,
        'criticality': 1,
        'price': 500,
    },
]


def main():
    print("=" * 70)
    print("  Базовый численный пример из статьи Черникова И.В. (2026)")
    print("  Применение многоуровневой модели к трём SKU")
    print("=" * 70)

    skus = pd.DataFrame(SKUS_DATA)
    skus['annual_demand'] = (skus['mean_demand'] * 365).astype(int)

    # Применяем политику P2 (многоуровневая)
    print("\n--- Политика P2: дифференцированный CSL по 18 подкластерам ---\n")

    for _, row in skus.iterrows():
        params = compute_params(row, policy='multilevel')
        sig = sigma_ltd(row['mean_demand'], row['std_demand'],
                         row['lead_mean'], row['lead_std'])
        z = stats.norm.ppf(params['csl'])
        mu_ltd = row['mean_demand'] * row['lead_mean']

        print(f"** {row['sku_id']}: {row['description']} **")
        print(f"  Группа: {row['abc_xyz']}, кластер: {row['cluster']}")
        print(f"  Исходные параметры:")
        print(f"    d̄  = {row['mean_demand']} ед./день")
        print(f"    σ_d = {row['std_demand']} ед./день")
        print(f"    L̄  = {row['lead_mean']} дней")
        print(f"    σ_L = {row['lead_std']} дней")
        print(f"    h  = {row['holding_cost_rate']:.2f}, "
              f"c = {row['criticality']}, p = {row['price']} руб.")
        print(f"  Расчёт:")
        print(f"    μ_LTD  = d̄ · L̄ = {mu_ltd:.1f} ед.")
        print(f"    σ_LTD  = √(L̄·σ_d² + d̄²·σ_L²) = {sig:.2f} ед.")
        print(f"    CSL    = {params['csl']:.3f}, z = {z:.3f}")
        print(f"    SS     = z · σ_LTD = {params['SS']:.1f} ед.")
        print(f"    ROP    = μ_LTD + SS = {params['ROP']:.1f} ед.")
        print(f"    EOQ    = √(2·D·S/H) = {params['EOQ']:.1f} ед.")
        print()

    # Сравнение P1 (без кластеризации) vs P2 (с кластеризацией)
    print("\n--- Сравнение политик P1 (ABC-XYZ) vs P2 (многоуровневая) ---\n")
    rows = []
    for _, row in skus.iterrows():
        p1 = compute_params(row, policy='abcxyz')
        p2 = compute_params(row, policy='multilevel')
        rows.append({
            'SKU': row['sku_id'],
            'Группа': row['abc_xyz'],
            'CSL_P1': f"{p1['csl']:.3f}",
            'CSL_P2': f"{p2['csl']:.3f}",
            'SS_P1':  f"{p1['SS']:.0f}",
            'SS_P2':  f"{p2['SS']:.0f}",
            'ROP_P1': f"{p1['ROP']:.0f}",
            'ROP_P2': f"{p2['ROP']:.0f}",
            'EOQ':    f"{p2['EOQ']:.0f}",
        })
    cmp_df = pd.DataFrame(rows)
    print(cmp_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("  Пример из статьи воспроизведён.")
    print("=" * 70)


if __name__ == '__main__':
    main()
